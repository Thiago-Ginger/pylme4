"""Shared parallel-execution helpers for pylme4.

This module contains **no numerical code**. It exists so that every
embarrassingly-parallel workload in the package (``bootMer``, the profile
grids, ``confint``) goes through one well-tested dispatcher instead of each
growing its own ad-hoc pool.

Design constraints
------------------

*No new dependencies.* Everything here is standard library
(:mod:`concurrent.futures`, :mod:`multiprocessing`, :mod:`os`).

*Process-based by default.* The workloads we parallelize are full model
refits: they spend most of their time in Python, patsy and ``scipy.sparse``
kernels that do **not** release the GIL, so a thread pool would not scale.
A ``"thread"`` backend is still exposed for callers who know their payload is
BLAS-bound.

*Order-preserving.* :func:`parallel_map` always returns results in the same
order as the inputs, so a parallel run produces exactly the same output
object as a serial one.

*No shared mutable state.* Workers are separate processes; each task builds
its own model state. Isolation is structural, not lock-based, so no race
condition is possible on ``PLSState`` / ``GLMMState`` / the fitting
dataframe.

*Always falls back to serial.* If a process pool cannot be created or a
payload cannot be pickled (a Windows script without an
``if __name__ == "__main__":`` guard, a user-defined ``Family`` built from
closures, a nested worker), we warn and run the same tasks in-process. The
result is identical — only the speed-up is lost.

*Parallel is opt-in, not the default.* ``n_jobs=None`` resolves to serial
(see :func:`resolve_n_jobs`) unless the caller has opted in via an explicit
``n_jobs=``, :func:`set_n_jobs`, or ``PYLME4_N_JOBS``. Measured on real
models, spinning up a process pool "just in case" is frequently a net loss:
Windows pays a multi-second fixed cost per pool (fresh interpreter +
reimporting the scientific stack per worker), and pinning BLAS to one
thread per worker perturbs the finite-difference gradient of the
``scipy L-BFGS-B`` fallback (active whenever ``nlopt`` is not installed)
enough to change the optimizer's iteration count — both effects can erase,
or invert, the benefit of splitting work across cores. Parallelism still
pays off for workloads with a high per-task cost (e.g. GLMM bootstraps with
many PIRLS iterations); it just needs to be requested, not assumed.
"""
from __future__ import annotations

import os
import pickle
import multiprocessing as _mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Optional, Sequence

try:  # BrokenProcessPool moved around across versions; keep it optional.
    from concurrent.futures.process import BrokenProcessPool  # type: ignore
except Exception:  # pragma: no cover
    class BrokenProcessPool(Exception):  # type: ignore
        pass


__all__ = [
    "set_n_jobs", "get_n_jobs", "resolve_n_jobs", "resolve_parallel",
    "effective_n_jobs", "parallel_map", "in_worker_process",
]


# Environment variables that control the number of threads spawned by the
# BLAS/LAPACK backend behind numpy/scipy. When we fan out over N processes we
# pin these to 1 in the children, otherwise N processes x N BLAS threads
# oversubscribe the machine and the speed-up evaporates.
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Process-pool failures we treat as "cannot parallelize here" rather than as a
# real error from the user's task. Every one of these is re-checked by running
# the same tasks serially afterwards, so a genuine error is never swallowed.
_POOL_FAILURES = (
    BrokenProcessPool,
    pickle.PicklingError,
    AttributeError,
    TypeError,
    OSError,
    RuntimeError,
    ImportError,
    EOFError,
    NotImplementedError,
)

_GLOBAL_N_JOBS: Optional[int] = None


# ---------------------------------------------------------------------------
# n_jobs resolution
# ---------------------------------------------------------------------------

def _cpu_count() -> int:
    """Number of CPUs usable by this process (respects cgroup/affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):  # pragma: no cover - Windows/macOS
        return max(1, os.cpu_count() or 1)


def in_worker_process() -> bool:
    """True when running inside a multiprocessing worker.

    Used to force nested :func:`parallel_map` calls to run serially — spawning
    a pool from inside a pool either deadlocks or explodes the process count.
    """
    try:
        return _mp.parent_process() is not None
    except AttributeError:  # pragma: no cover - Python < 3.8
        return _mp.current_process().name != "MainProcess"


def set_n_jobs(n_jobs: Optional[int]) -> None:
    """Set the package-wide default worker count.

    ``None`` restores the default (serial, unless ``PYLME4_N_JOBS`` is set —
    see :func:`resolve_n_jobs`). Explicit ``n_jobs=`` arguments passed to
    individual functions always win over this global.
    """
    global _GLOBAL_N_JOBS
    _GLOBAL_N_JOBS = None if n_jobs is None else int(n_jobs)


def get_n_jobs() -> Optional[int]:
    """Return the package-wide default set by :func:`set_n_jobs` (or ``None``)."""
    return _GLOBAL_N_JOBS


def resolve_n_jobs(n_jobs: Optional[int] = None, default: int = 1) -> int:
    """Turn a user-supplied ``n_jobs`` into a concrete positive worker count.

    Resolution order: explicit argument -> :func:`set_n_jobs` global ->
    ``PYLME4_N_JOBS`` environment variable -> ``default``.

    ``default`` is what applies when none of the three sources above gave an
    answer. It follows the same positive/negative convention as ``n_jobs``
    itself (see below); the module-level default of ``1`` is what every
    existing caller gets when it doesn't pass ``default=``, so this is a
    backward-compatible extension, not a behaviour change. :func:`resolve_parallel`
    is the only caller that passes ``default=-1``.

    Parallel execution is **opt-in**, not the default. Spinning up a process
    pool has a measured fixed cost on the order of several seconds on
    Windows (fresh interpreter + reimporting numpy/scipy/pandas/patsy per
    worker), and — separately — pinning BLAS to one thread per worker
    changes the rounding of intermediate sums, which the finite-difference
    gradient of the ``scipy L-BFGS-B`` fallback (used whenever ``nlopt`` is
    not installed) can amplify into materially more optimizer iterations
    per task. Net effect, measured on real models: unqualified "use every
    CPU" defaults have produced 2-3x *slower* wall-clock time than plain
    serial execution for ``bootMer``/``profile``/``confint``, not a
    speed-up. Callers who know their workload benefits from parallelism
    (large per-task cost, e.g. GLMM refits with many PIRLS iterations) can
    still opt in explicitly via ``n_jobs=``, :func:`set_n_jobs`, or
    ``PYLME4_N_JOBS`` — benchmark on your own model before doing so.

    Follows the familiar joblib convention for non-positive values:
    ``-1`` means "all CPUs", ``-2`` means "all but one", and so on — these
    only apply once you've opted in via one of the three routes above.
    """
    if n_jobs is None:
        n_jobs = _GLOBAL_N_JOBS
    if n_jobs is None:
        env = os.environ.get("PYLME4_N_JOBS")
        if env not in (None, ""):
            try:
                n_jobs = int(env)
            except ValueError:
                warnings.warn(
                    f"PYLME4_N_JOBS={env!r} is not an integer; ignoring.",
                    RuntimeWarning, stacklevel=2,
                )
                n_jobs = None
    if n_jobs is None:
        n_jobs = default
    n_jobs = int(n_jobs)
    if n_jobs > 0:
        return n_jobs
    return max(1, _cpu_count() + 1 + n_jobs)


def resolve_parallel(parallel: bool, n_jobs: Optional[int] = None) -> int:
    """Combine the public ``parallel``/``n_jobs`` pair into one worker count.

    This is what every public function (``bootMer``, ``profile``,
    ``confint``, ...) calls first, and it is the single place that decides
    whether a run is serial or not:

    - ``parallel=False`` (the default everywhere) **always** means serial,
      regardless of ``n_jobs``, :func:`set_n_jobs`, or ``PYLME4_N_JOBS`` --
      those only take effect once ``parallel=True`` has opted in. Passing
      ``n_jobs`` without ``parallel=True`` is almost certainly a mistake
      (you asked for a worker count but not for parallelism), so it warns.
    - ``parallel=True`` with ``n_jobs=None`` means "every CPU" -- unlike
      :func:`resolve_n_jobs` on its own, which would resolve a bare
      ``None`` to serial. ``n_jobs`` still lets you pick a specific worker
      count (or the joblib ``-1``/``-2``/... convention) instead of every
      core.

    Functions that forward to another public function internally (e.g.
    ``confint(method='boot')`` -> ``bootMer``) should resolve once, then
    forward with ``parallel=True, n_jobs=<resolved value>`` -- the callee's
    own ``resolve_parallel`` re-resolves that concrete value idempotently
    without re-triggering the "parallel=False default" warning.
    """
    if not parallel:
        if n_jobs not in (None, 1):
            warnings.warn(
                f"n_jobs={n_jobs!r} was given but parallel=False (the "
                "default) -- running serially and ignoring n_jobs. Pass "
                "parallel=True to actually use it.",
                RuntimeWarning, stacklevel=3,
            )
        return 1
    return resolve_n_jobs(n_jobs, default=-1)


def effective_n_jobs(n_jobs: Optional[int], n_tasks: int) -> int:
    """Worker count actually used for ``n_tasks`` independent tasks."""
    if n_tasks <= 1 or in_worker_process():
        return 1
    return max(1, min(resolve_n_jobs(n_jobs), n_tasks))


# ---------------------------------------------------------------------------
# BLAS oversubscription guard
# ---------------------------------------------------------------------------

@contextmanager
def _blas_threads_pinned(enabled: bool):
    """Temporarily pin BLAS thread counts to 1 in the *environment*.

    Child processes inherit the environment at creation time, so this caps
    threads inside the workers. It does not affect the parent, whose BLAS
    backend already read these variables at import time.

    On start methods that use ``fork`` the child inherits an already-initialised
    BLAS runtime, so the environment variable may be ignored; installing the
    optional ``threadpoolctl`` package lets the worker initializer pin the
    limits at runtime instead. Neither is required for correctness.
    """
    if not enabled:
        yield
        return
    saved = {k: os.environ.get(k) for k in _BLAS_ENV_VARS}
    try:
        for k in _BLAS_ENV_VARS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _worker_bootstrap(limit_blas: bool, initializer, initargs):
    """Runs once per worker process before any task."""
    if limit_blas:
        for k in _BLAS_ENV_VARS:
            os.environ[k] = "1"
        try:  # optional; only helps on fork-based start methods
            import threadpoolctl  # type: ignore
            threadpoolctl.threadpool_limits(1)
        except Exception:
            pass
    if initializer is not None:
        initializer(*initargs)


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------

def parallel_map(func: Callable[[Any], Any],
                 items: Iterable[Any],
                 *,
                 n_jobs: Optional[int] = None,
                 backend: Optional[str] = None,
                 initializer: Optional[Callable[..., None]] = None,
                 initargs: Sequence[Any] = (),
                 limit_blas_threads: bool = True,
                 chunksize: Optional[int] = None) -> list:
    """Apply ``func`` to every item, in parallel, preserving input order.

    Parameters
    ----------
    func : callable
        Must be importable by name (module-level function or an instance of a
        module-level class) so it can be pickled for the process backend.
    items : iterable
        One entry per independent task.
    n_jobs : int or None
        Worker count; see :func:`resolve_n_jobs`. ``1`` runs in-process.
    backend : {"process", "thread"} or None
        ``None`` means ``"process"``. Use ``"thread"`` only for payloads that
        release the GIL.
    initializer, initargs : callable, tuple
        Called once per worker (and once in-process for the serial path) — the
        place to ship large read-only context such as the fitting dataframe,
        so it is pickled once per worker instead of once per task.
    limit_blas_threads : bool
        Pin BLAS threads to 1 inside workers to avoid oversubscription.
    chunksize : int or None
        Tasks handed to a worker at a time. ``None`` picks a value that keeps
        every worker busy while limiting scheduling overhead.

    Returns
    -------
    list
        ``[func(item) for item in items]`` — same values, same order, whether
        it ran serially or in parallel.
    """
    items = list(items)
    n_tasks = len(items)
    if n_tasks == 0:
        return []

    jobs = effective_n_jobs(n_jobs, n_tasks)
    backend = (backend or "process").lower()
    if backend not in ("process", "thread"):
        raise ValueError(
            f"backend must be 'process' or 'thread', got {backend!r}")

    if jobs <= 1:
        return _run_serial(func, items, initializer, initargs)

    if chunksize is None:
        # A couple of chunks per worker: enough to smooth out uneven task
        # durations without paying per-task IPC on every single item.
        chunksize = max(1, n_tasks // (jobs * 4))

    if backend == "thread":
        if initializer is not None:
            initializer(*initargs)  # threads share the address space
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(func, items, chunksize=chunksize))

    try:
        with _blas_threads_pinned(limit_blas_threads):
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_worker_bootstrap,
                initargs=(limit_blas_threads, initializer, tuple(initargs)),
            ) as ex:
                return list(ex.map(func, items, chunksize=chunksize))
    except _POOL_FAILURES as exc:
        warnings.warn(
            f"pylme4: parallel execution unavailable ({type(exc).__name__}: "
            f"{exc}); falling back to serial. On Windows, wrap your script "
            f"entry point in `if __name__ == \"__main__\":`.",
            RuntimeWarning, stacklevel=2,
        )
        return _run_serial(func, items, initializer, initargs)


def _run_serial(func, items, initializer, initargs) -> list:
    if initializer is not None:
        initializer(*initargs)
    return [func(item) for item in items]
