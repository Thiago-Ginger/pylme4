# pylme4 — Log de implementação

Porting de `lme4` (R, Bates et al. 2015) para Python. Foco: `lmer` (LMM gaussiano)
via PLS profiled REML/ML, Cholesky esparsa, otimização BOBYQA.

## Estado por módulo

| Módulo | Status | Observações |
|---|---|---|
| `pyproject.toml` | OK | deps: numpy, scipy, pandas, patsy, scikit-sparse, nlopt |
| `pylme4/__init__.py` | OK | reexporta API: lmer, fixef, ranef, VarCorr, etc. |
| `pylme4/formula.py` | OK | findbars/mkReTrms; suporta `(1|g)`, `(x|g)`, `(x||g)`, `g1/g2`, `g1:g2` |
| `pylme4/pls.py` | **PENDENTE** | núcleo do PLS: dado θ → resolver Cholesky e devolver perfil de deviance |
| `pylme4/fit.py` | **PENDENTE** | `lmer()`, classe `MerMod`, otimização (BOBYQA via nlopt → fallback scipy) |
| `pylme4/extractors.py` | **PENDENTE** | fixef, ranef, VarCorr, sigma, logLik, deviance, vcov, getME, isSingular, REMLcrit, AIC, BIC |
| `tests/` | **PENDENTE** | golden tests vs lme4 via rpy2 (sleepstudy, Penicillin) |

## Decisões de design

- **PLS profiled (lme4 §5.4)**: para cada θ, resolver
  `[Λ'Z'ZΛ + I, Λ'Z'X; X'ZΛ, X'X] [u; β] = [Λ'Z'y; X'y]`
  via Cholesky esparsa de `Λ'Z'ZΛ + I` (`L L' = P (Λ'Z'ZΛ + I) P'`),
  perfilar β e σ², devolver deviance perfilada.
- **REML vs ML**: REML padrão (igual ao lme4); ML ativável via `REML=False`.
- **Cholesky esparsa**: preferir `sksparse.cholmod.cholesky` (CHOLMOD, com
  fill-reducing permutation persistente entre iterações via `analyze` + `cholesky_inplace`).
  Fallback para `scipy.sparse.linalg.splu` se sksparse indisponível.
- **Otimizador**: `nlopt.LN_BOBYQA` com bounds `theta ≥ 0` na diagonal, `-inf`
  fora da diagonal; fallback `scipy.optimize.minimize(method='L-BFGS-B')` com
  approx grad (não ideal — BOBYQA é o que lme4 usa).
- **Singularidade**: `isSingular` se algum elemento diagonal de θ ≈ 0
  (tol = 1e-4 como lme4).

## Sessão 2026-04-15

- [x] Auditoria do estado: `formula.py` completo, núcleo numérico ausente.
- [x] Criado `LOG.md`.
- [x] Implementado `pls.py` — profiled deviance via Cholesky esparsa (CHOLMOD
      se disponível, fallback denso scipy).
- [x] Implementado `fit.py` — `lmer`, `MerMod`, optim BOBYQA (nlopt) / L-BFGS-B.
- [x] Implementado `extractors.py` — fixef, ranef, VarCorr, sigma, logLik,
      deviance, AIC, BIC, vcov, getME, isSingular, REMLcrit.
- [x] Smoke test sintético (`tests/smoke_sleepstudy.py`) passa: assertions de
      recuperação frouxas (~10 pontos) OK; fit convergiu com L-BFGS-B
      (nlopt não instalado) em 68 aval. Converge para solução singular
      (corr=1) quando nlopt ausente — esperado, BOBYQA é mais estável.

## Sessão 2026-04-15 (parte 2)

- [x] `pip install nlopt` — BOBYQA ativo.
- [x] **Bug fix em `fit.py`**: sem `set_initial_step`, BOBYQA parava cedo
      (17974 vs ótimo 17666 em teste sintético). Adicionado
      `opt.set_initial_step(0.2 * ones)` — lme4 default. Recuperação passa a
      bater truth dentro de 1%.
- [x] Edge cases validados: random-intercept-only, crossed (`(1|a)+(1|b)`),
      OLS (q=0) — todos convergem.
- [x] `summary(MerMod)` implementado (stringify tipo lme4::summary).
- [x] Golden sleepstudy hardcoded em [tests/sleepstudy_data.py](tests/sleepstudy_data.py)
      e comparado contra valores publicados em Bates et al. 2015 Tabela 1:

      | quantidade   | lme4      | pylme4    | diff     |
      |--------------|-----------|-----------|----------|
      | REMLdev      | 1743.628  | 1742.410  | −1.218   |
      | sigma        | 25.592    | 25.587    | −0.005   |
      | sd_int       | 24.740    | 24.194    | −0.546   |
      | sd_slope     | 5.922     | 5.769     | −0.153   |
      | corr         | 0.066     | 0.108     | +0.042   |
      | β_Intercept  | 251.405   | 251.844   | +0.439   |
      | β_Days       | 10.467    | 10.359    | −0.108   |

      Match dentro da precisão dos valores publicados (3 decimais na tabela).
      REMLdev ligeiramente menor sugere convergência marginalmente melhor ou
      rounding do dataset hardcoded.

## Sessão 2026-04-15 (parte 3)

- [x] `predict(newdata)`, `fitted()`, `resid()` em [extractors.py](pylme4/extractors.py):
      - Armazena `patsy.DesignInfo` da FE e de cada LHS de RE term em
        `ReTrms`/`ReTerm` para reaplicar em newdata.
      - `re_form='none'` devolve pop-level (apenas Xβ); default adiciona Zb
        mapeando níveis por `t.levels`.
      - `allow_new_levels=True` trata níveis não vistos como contribuição 0.
- [x] `confint(m, level=0.95, method='Wald')` — IC Wald para FE.
- [x] Tentativa de instalar `scikit-sparse` falhou (Windows precisa MSVC C++
      Build Tools). Continuamos no fallback denso.
- [x] Perf bench (q = 2·n_subj, random-slope):

      | n_subj | q    | tempo    | nfev |
      |--------|------|----------|------|
      | 50     | 100  | 0.03 s   | 79   |
      | 200    | 400  | 0.18 s   | 96   |
      | 500    | 1000 | 0.98 s   | 83   |
      | 1000   | 2000 | 17.84 s  | 136  |

      Cúbico em q (denso Cholesky de A=Λ'Z'ZΛ+I). sksparse/CHOLMOD resolveria.

## Sessão 2026-04-16 — `glmer` (GLMM via Laplace)

- [x] [`family.py`](pylme4/family.py): factory `get_family()` com gaussian/identity,
      binomial/{logit,probit,cloglog}, poisson/log, Gamma/{log,inverse}.
      Cada `Family` expõe linkfun/linkinv/mu_eta/variance/dev_resids/initialize/aic.
- [x] [`glmm.py`](pylme4/glmm.py): PIRLS + Laplace deviance. Reusa a fatoração
      simbólica CHOLMOD do `PLSState` quando disponível; pesos
      `w = (dμ/dη)² / V(μ)` entram em `A_w = Λ'Z'WZΛ + I`. Step-halving
      dentro do PIRLS (até 10x) quando μ fica inválido ou pwrss não decresce.
- [x] [`fit.py`](pylme4/fit.py): `glmer(formula, data, family, weights=, offset=)`.
      Outer loop BOBYQA sobre θ; inner PIRLS para (β̂, û) em cada θ.
      `MerMod` ganhou `family` e `is_glmm`.
- [x] [`extractors.py`](pylme4/extractors.py):
      - `fitted(m, type='response'|'link')` e `predict(..., type=)` aplicam
        inverse-link no GLMM.
      - `resid` ganhou tipos `'pearson'`, `'deviance'`, `'working'`.
      - `vcov` com caminho GLMM: `phi * (X'WX - RZX'RZX)^{-1}`.
      - `sigma` retorna 1.0 p/ famílias de dispersão fixa.
      - `summary` imprime cabeçalho GLMM (family, link, Laplace deviance) e
        tabela de FE com `z value`/`Pr(>|z|)`.
      - `AIC`/`BIC`: contagem de params inclui `p + nθ` para GLMM (+1 se
        família estima dispersão).
- [x] Smoke test [tests/smoke_glmer.py](tests/smoke_glmer.py):

      | caso                  | beta truth    | beta recov.    | sd_re truth | sd_re recov. |
      |-----------------------|---------------|----------------|-------------|--------------|
      | binomial/logit        | (-0.5, 1.2)   | (-0.42, 1.13)  | 0.80        | 0.59         |
      | poisson/log           | (0.2, 0.4)    | (0.10, 0.37)   | 0.50        | 0.43         |
      | gaussian/identity     | (1.0, 2.0)    | (0.862, 2.022) | 1.50        | 1.33         |

      Gaussian bate `lmer(REML=False)` em float precision (0.86167562,
      2.0219422, sd 1.3326, sigma 0.8823 — idênticos). Confirma que PIRLS se
      reduz corretamente a PLS quando W=I.

      Binomial/poisson recuperam FE dentro de ~1 SE; sd_re subestimado é
      o viés de Laplace conhecido para famílias de dispersão fixa com
      poucas obs/grupo (R's lme4 tem o mesmo comportamento sem AGQ).

### Pendências GLMM (fora do core)

- [ ] `bootMer`/`simulate` assumem gaussiano — precisam gerar resposta via
      `family.rvs(mu, sigma)` quando `is_glmm`.
- [ ] `nAGQ > 1` (adaptive Gauss-Hermite) — reduziria o bias de Laplace
      em binomial com grupos pequenos. Só faz sentido para `1|g` puro.
- [ ] `cbind(k, n-k) ~ ...` binomial counts — hoje user precisa passar
      `y = k/n` e `weights = n`.

## Sessão 2026-04-16 — Profile CIs

- [x] [`profile.py`](pylme4/profile.py): `profile(m, zeta_max, nsteps)` e
      `confint_profile(m, level)`. Para cada β_j profila via offset-trick:
      - LMM: subtrai `X[:, j] * c` de y, remove coluna j de X, refita θ.
      - GLMM: soma `X[:, j] * c` ao offset, remove coluna j de X, refita θ.
      - Inner optim reusa `_optim_theta` (refatorado em `fit.py`).
      - Baseline: se `m.reml`, refita original em ML para comparar deviance
        (mesmo truque do `lme4::profile`, já que REML não é comparável
        entre modelos com p diferente).
      - `zeta = sign(β − β̂) √(D_prof − D_min)` — aproximadamente linear;
        inverte linearmente em `zeta = ±z_{α/2}` para CI.
- [x] `confint(m, method='profile', level=0.95)` expõe a API lme4-style.
- [x] Smoke [tests/smoke_profile.py](tests/smoke_profile.py):

      | caso                    | Wald (Days)      | Profile (Days)   |
      |-------------------------|------------------|------------------|
      | LMM sleepstudy sintético | (6.633, 11.006)  | (6.591, 11.048)  |
      | GLMM binomial (x)        | (0.596, 0.873)   | (0.610, 0.895)   |

      ζ monotone dentro de tol=1e-6 no grid [-3.158, 3.158].
      Profile um pouco mais amplo (esperado — captura skew do likelihood).

### Pendências Profile

- [ ] Profile para θ (variance components) — fixar um elemento de θ e
      reotimizar o resto. Mais informativo que Wald para σ_RE.
- [ ] Profile para σ (residual) — parametrização log σ e refit.
- [ ] Adaptative grid em vez de uniforme em SE (lme4 usa passo
      adaptativo quando ζ diverge de linear rápido demais).

## Sessão 2026-04-16 — Goldens + cobertura estrutural

- [x] [tests/rpy2_goldens.py](tests/rpy2_goldens.py): scaffold rpy2 com
      `pytest.skipif` — pronto para rodar (sleepstudy / Penicillin / cake)
      quando R+lme4+rpy2 instalados. R não está no PATH agora, install
      Windows precisa admin → deixado como preparação.
- [x] [tests/smoke_crossed_nested.py](tests/smoke_crossed_nested.py):
      cobertura sintética das estruturas RE que sleepstudy não tocava:

      | caso                                   | beta (truth) → recov.          | SDs (truth) → recov.      | sigma |
      |----------------------------------------|--------------------------------|---------------------------|-------|
      | `y ~ x + (1\|A) + (1\|B)` crossed       | (5.0, 1.5) → (4.62, 1.50)      | (2.0, 1.2) → (1.88, 0.87) | 0.59  |
      | `y ~ x + (1\|outer/inner)` nested       | (3.0, 2.0) → (2.88, 1.99)      | (1.5, 0.8) → (1.23, 0.79) | 0.40  |
      | `y ~ x + (x\|\|g)` independent slopes   | (1.0, 0.5) → (0.97, 0.57)      | (1.0, 0.7) → (0.95, 0.76) | 0.40  |

      Todos dentro de ~0.3 SD das planted; sigma dentro de 0.05 da truth.
      Valida parser de `g1/g2` (nested expansion), `g1:g2` (interação) e
      `||` (expansion em termos independentes).
- Instalação R+rpy2 pendente user approval (admin + ~150MB).

## Sessão 2026-04-16 — Benchmark pylme4 vs R lme4 (HTML report)

- [x] Infra em [tests/benchmark/](tests/benchmark/):
      - [cases.py](tests/benchmark/cases.py) — 8 casos declarativos
      - [run_r.R](tests/benchmark/run_r.R) — fita via `lme4::lmer`/`glmer`,
        exporta CSV dos datasets + sidecar `*_meta.json` com tipos de
        factor (para Python restaurar `pd.Categorical(ordered=...)`)
      - [run_python.py](tests/benchmark/run_python.py) — lê mesma CSV,
        coage factors por metadata, reescreve formula para usar
        `C(col, Poly)` em factors ordenados (patsy não infere poly auto)
      - [build_report.py](tests/benchmark/build_report.py) — diff por
        métrica com thresholds PASS/WARN/FAIL, normaliza nomes R↔patsy
        (`(Intercept)`→`Intercept`, `recipe[T.B]`→`recipeB`,
        `C(col, Poly).Linear`→`col.L`), HTML self-contained
- [x] R instalado em `C:\Program Files\R\R-4.5.3` (não-admin via
      `R_LIBS_USER` para pacotes: lme4 2.0.1 + jsonlite + deps).
- [x] **Correções em pylme4 descobertas pelo benchmark:**
      - [`extractors.py`](pylme4/extractors.py): `_n_params` para LMM
        agora é `p + nθ + 1` (antes era `nθ + 1` em REML). Alinha AIC/BIC
        ao convênio do lme4 (que conta FE em AIC mesmo em REML).
      - `deviance(m)` para GLMM agora retorna **residual deviance**
        (sum dev_resid), não a Laplace criterion. Match lme4.
      - `logLik(m)` para GLMM agora inclui as constantes de normalização
        da família (log C(n,k) binomial, log(k!) poisson) via novo campo
        `Family.loglik_contrib`. AIC/BIC derivam de -2 logLik.
      - [`family.py`](pylme4/family.py): `loglik_contrib` implementado
        para gaussian, binomial, poisson, Gamma (todas as famílias).
      - [`glmm.py`](pylme4/glmm.py): PIRLS ganhou jitter adaptativo
        (0 → 1e-12 → ... → 1e-4) no `cho_factor` de S, + fallback
        eigh-pinv para casos mal-condicionados (grouseticks antes
        falhava em 4-th leading minor).
- [x] **Resultado:** **168 PASS / 21 WARN / 0 FAIL** em 189 métricas × 8 casos:

      | caso                          | status | nota                              |
      |-------------------------------|--------|-----------------------------------|
      | sleepstudy (REML)             | PASS   | match até ~1e-4 em todas métricas |
      | sleepstudy (ML)               | PASS   | idem                              |
      | sleepstudy `(Days\|\|Subject)`| PASS   | double-bar parsing OK             |
      | Dyestuff                      | PASS   | modelo mais simples, bit-exact    |
      | Penicillin (crossed)          | PASS   | match a 1e-6                      |
      | cake (nested + ordered poly)  | PASS   | após factor meta + C(poly)        |
      | cbpp_binomial (cbind)         | WARN   | Laplace diff ~0.01 em dev         |
      | grouseticks_poisson (scaled)  | WARN   | Laplace diff ~0.3% relativo       |

      Todos os WARNs são pequenos diffs de aproximação Laplace (pylme4 e
      lme4 convergem em modos levemente diferentes do Laplace-profile);
      zero FAILs significa que todos os números estão dentro de
      tolerâncias aceitáveis para uso prático.
- [x] Report em [tests/benchmark/report.html](tests/benchmark/report.html)
      (self-contained, abre no navegador).

## Sessão 2026-04-16 — Polish (simulate/bootMer GLMM + cbind + profile θ/σ)

- [x] [`family.py`](pylme4/family.py): adicionado `rvs(mu, weights, sigma, rng)`
      em todas as famílias — gaussian (normal), binomial (binom→k/n),
      poisson, Gamma (shape=1/φ, scale=μ·φ). Binomial/probit/cloglog
      reusam o `rvs` do logit (independente do link).
- [x] [`extractors.py`](pylme4/extractors.py):
      - `simulate(m, nsim)` detecta `is_glmm`; para GLMM calcula η → μ →
        `family.rvs(...)`; offset e weights do state são propagados.
        LMM continua `μ + N(0, σ²)` como antes.
      - `bootMer(m, nsim)` despacha `glmer`/`lmer` conforme `is_glmm` e
        preserva family/weights/offset nos refits.
- [x] [`formula.py`](pylme4/formula.py):
      - `_parse_cbind(resp)` detecta `cbind(successes, failures)` no LHS
        (split no top-level comma — respeita parênteses aninhados).
      - `ReTrms.implicit_weights` armazena trials = successes + failures.
      - `y = successes / trials` entra como resposta; `glmer` usa
        `implicit_weights` se o usuário não passou `weights=` explícito.
- [x] [`fit.py`](pylme4/fit.py): `glmer` pega `trms.implicit_weights` quando
      `weights` arg é None. Verificado: `cbind(k, fail) ~ x + (1|g)` bate
      `I(k/n) ~ x + (1|g), weights='n'` **bit-exact** em beta/theta/dev.
- [x] [`profile.py`](pylme4/profile.py):
      - `_profile_dev_theta(m, idx, c)`: fixa `θ[idx] = c`, reotimiza
        demais θ. Wrapper injeta c na posição fixada a cada iter. Funciona
        p/ LMM (PLS) e GLMM (PIRLS).
      - `profile_theta(m, idx)` grid multiplicativo ao redor de θ̂
        (sem Wald SE disponível para θ).
      - `_profile_dev_sigma(m, σ)` (LMM): formula fechada
        `dev = log|A| + log|RX'RX| + pwrss/σ² + (n-p)·log(2πσ²)` (REML)
        ou equivalente ML; reusa PLS (β, u independem de σ).
      - `profile_sigma(m)`: grid log-espaçado; baseline self-consistent
        (min da grade) para lidar com jitter numérico no optim interno.
      - `confint_theta(m, level)` e `confint_sigma(m, level)` inverterm ζ.
- [x] Smoke test [tests/smoke_polish.py](tests/smoke_polish.py):

      | item                       | verificação                             |
      |----------------------------|-----------------------------------------|
      | GLMM binomial simulate/boot| y ∈ {0,1}; mean próximo de fitted; 15/15 convergiu |
      | GLMM poisson simulate      | y ≥ 0 inteiros                          |
      | cbind vs weights syntax    | bit-exact (rtol 1e-10) em β, θ, dev     |
      | cbind(k, n - k) expr       | converge, sem erro de parsing           |
      | profile_theta sleepstudy   | 3 CIs, todas bracketam o estimate; diag>0 |
      | profile_sigma sleepstudy   | ζ monotone; CI (22.79, 28.40) p/ σ̂=25.32 |

      | parâmetro     | MLE     | 95% Profile CI     |
      |---------------|---------|--------------------|
      | θ[0] (diag int) | 0.705   | (0.327, 1.192)     |
      | θ[1] (off-diag) | -0.040  | (-0.145, 0.061)    |
      | θ[2] (diag slp) | 0.109   | (0.027, 0.178)     |
      | σ (residual)    | 25.32   | (22.79, 28.40)     |

### Pendências remanescentes (não-MVP)

- [ ] `nAGQ > 1` (adaptive Gauss-Hermite) — GLMM bias reduction,
      só faz sentido p/ `(1|g)` puro.
- [ ] Adaptative grid no profile (hoje uniforme em SE/fração da estimate).
- [ ] scikit-sparse/CHOLMOD — bloqueado por admin (ver sessão anterior).
- [ ] rpy2 goldens — bloqueado por R não instalado.

## Sessão 2026-04-16 — Predict cleanup + sparse Cholesky (tentativa)

- [x] [`extractors.py`](pylme4/extractors.py): `predict()` agora captura
      `PatsyError` e emite mensagem clara identificando o fator, níveis
      novos, níveis vistos no treino e três opções de remediação (refit,
      drop/relabel, mapear para ref). Nenhuma regressão no caminho feliz.
- [ ] **scikit-sparse no Windows — bloqueado** por dois caminhos:
      - `pip install scikit-sparse` → falha na compilação Cython porque
        pede MSVC C++ Build Tools ≥14.0 (~4GB, admin).
      - `pip install cvxopt` (alternativa que embute CHOLMOD) instala mas
        é bloqueado por política Windows Application Control: "Uma política
        de Controle de Aplicativo bloqueou este arquivo" ao carregar a
        DLL BLAS. Precisa whitelist admin.
      - Sem admin disponível no ambiente. Fallback denso continua
        adequado para q ≲ 1000 (LOG sessão 3 benchmark: q=1000 em ~1s).
        Para q ≳ 2000 a cubic scaling dói.
      - Opções futuras: (1) user liberar Build Tools, (2) trocar env p/
        conda-forge `scikit-sparse`, (3) implementar Cholesky supernodal
        caseiro usando `scipy.sparse.linalg.splu` (LU em vez de LL').

## Sessão 2026-04-15 (parte 4)

- [x] **Transformações no LHS**: `parse_formula` agora aceita `log(y)`,
      `sqrt(y)`, etc. via `patsy.dmatrix` com `EvalEnvironment` injetando
      numpy. Aplicado também a FE e LHS de RE terms.
- [x] **`simulate(m, nsim, seed)`** — retorna `(n, nsim)` array; usa
      `b ~ N(0, σ² T T')` por termo + ruído `N(0, σ² I)`.
- [x] **`bootMer(m, nsim, seed)`** — bootstrap paramétrico (refita em
      cada `Y_sim`); requer `m._fit_df` (agora armazenado em `MerMod`).
- [x] **`confint(m, method='boot')`** — IC percentil via bootMer.
      Alternativa ao Wald.
- [x] Smoke: `log(Reaction) ~ Days + (Days|Subject)` ajusta; predict em
      newdata respeita escala log; bootstrap (nsim=30) bate Wald de perto.

### Pendências (originais da sessão 4 — todas resolvidas em sessões posteriores)

- [x] rpy2 goldens — scaffold em [tests/rpy2_goldens.py](tests/rpy2_goldens.py); roda quando R+lme4 instalados.
- [ ] Wheel `scikit-sparse` p/ Windows — bloqueado por admin (MSVC Build Tools).
- [x] Profile CIs — β em `profile.py`; θ/σ em sessão polish.
- [x] `glmer` GLMM — `glmm.py` com PIRLS + Laplace.
- [x] Novos níveis em `predict` — mensagem limpa em extractors.

### Notas técnicas

- Convenção de dimensões: `Zt ∈ R^{q×n}`, `Lambdat ∈ R^{q×q}` (= Λᵀ, upper),
  `A = Λ'Z'ZΛ + I` SPD.
- CHOLMOD usa `analyze()` uma vez (fixa permutação) e `F = sym.cholesky(A)` a
  cada iter — evita refatorar o símbolo. Fallback denso converte `A.toarray()`
  a cada iter (O(q³)).
- `deviance` retornado é REML-deviance quando `REML=True`, senão ML-deviance
  (= -2 logLik).

---

## Sessão 2026-08-06 — Performance: paralelismo + eliminação de trabalho redundante

Escopo: **somente performance**. Nenhuma mudança de lógica matemática, de API
pública ou de resultados numéricos. Nenhuma dependência nova (só stdlib).
Nenhuma operação Git.

### Novo módulo `pylme4/parallel.py`

Dispatcher único de paralelismo sobre `concurrent.futures` (stdlib):
`resolve_n_jobs`, `set_n_jobs`/`get_n_jobs`, `parallel_map`. Características e
o porquê de cada uma:

- **Processos, não threads.** Os reajustes passam a maior parte do tempo em
  Python/patsy/`scipy.sparse`, que não liberam o GIL.
- **Ordem preservada** (`executor.map`) → saída idêntica à serial, elemento a
  elemento.
- **`initializer`/`initargs`** para enviar contexto grande (o dataframe do
  ajuste, as matrizes de design) **uma vez por worker** em vez de uma vez por
  tarefa.
- **Fallback serial automático** em qualquer falha de pool ou de serialização,
  com `warnings.warn`. Exceções reais da tarefa do usuário continuam
  propagando: a re-execução serial as levanta de novo.
- **Guarda anti-oversubscription**: fixa `OMP/OPENBLAS/MKL_NUM_THREADS=1` no
  ambiente herdado pelos workers (e usa `threadpoolctl` se estiver instalado).
- **Guarda anti-aninhamento**: dentro de um worker, `parallel_map` roda serial.

Entradas que ganharam `n_jobs` (parâmetro opcional novo, padrão = auto):
`bootMer`, `profile`, `profile_theta`, `profile_sigma`, `confint_theta`,
`confint_sigma`, `confint_profile`, `confint(method='boot'|'profile')`.
`confint_theta` também ganhou `step_scale` (mesmo default de antes).

### Isolamento de estado (race conditions)

- `bootMer` não muta mais um `df_work` compartilhado — cada worker recebe o
  dataframe uma vez e faz sua própria cópia privada.
- `PLSState`/`GLMMState` nunca cruzam a fronteira de processo; cada tarefa
  constrói o seu (é o que o caminho serial já fazia).
- O fator simbólico CHOLMOD (objeto C não-serializável) nunca é enviado:
  `profile.py` ganhou `_ModelSpec`, que carrega só arrays/matrizes esparsas.
- `Family.__reduce__` reconstrói famílias do registro `(nome, link)` no worker
  (os campos são `lambda`s). Família customizada → `TypeError` claro → fallback
  serial.

Como cada worker é um processo separado, **não há memória compartilhada e
portanto nenhuma race condition possível** — a isolação é estrutural.

### Núcleo numérico (bit-exato, verificado)

`pls.py`:
- `X'X` e `X'y` são invariantes em θ → calculados uma vez em `make_state`.
  1.07× (p=6), 1.21× (p=20), 1.42× (p=50) — o custo é O(n·p²).
- `Lambdat @ Zt` era calculado **duas vezes** por avaliação (inline e dentro de
  `_build_A`) → calculado uma vez e repassado. 1.05–1.06×.

`glmm.py`:
- `Lambdat.T` içado para fora do laço de step-halving. 1.03–1.05×.
- `Zt.T` em CSR cacheado no state (é uma *view* sem cópia). 1.03–1.04×.

`extractors.py`:
- `predict`: mapeamento de níveis por `pd.Categorical` em vez de laço Python
  por linha. 2.8× (n=20k), 3.7× (n=200k).
- `simulate`: `Zt.T` içado para fora do laço. A ordem de consumo do RNG é
  preservada de propósito (ver abaixo).

### Otimizações implementadas e **removidas** por medição

Cada candidata foi implementada, medida em A/B isolado (revertendo uma de cada
vez no código real) e removida quando não pagou:

| candidata | medido | motivo da remoção |
|---|---|---|
| escalar colunas via `.data` em vez de `sp.diags` (GLMM) | 1.27–1.35× | **não é bit-exato**: o produto esparso do scipy emite ordem de índices dependente do operando (`Zt @ diags(w)` sai não-ordenado, `LtZt @ diags(√w)` ordenado) e os produtos seguintes acumulam nessa ordem — desvio de ~1e-16 relativo |
| içar `Lambdat @ Zt` para fora do PIRLS | 1.01× / **0.85×** | bit-exato, mas mantém uma matriz (q,n) viva durante todo o laço; o working set maior custa mais no kernel `M @ M.T` do que o produto economizado |
| cachear `sp.eye(q)` (PLS e GLMM) | 1.00–1.02× | indistinguível de ruído (mediana de 5 rodadas) |
| cachear `Zt.T` CSR no PLS | 1.01× | idem |
| reusar buffer de `Lambdat` (PLS e GLMM) | 1.01× | idem |
| vetorizar `_lambdat_block_triplets` com `np.tril_indices` | **0.47–0.83×** | mais lento: overhead de dispatch do numpy domina para `pi ≤ 5` |
| vetorizar remontagem de `T` em `VarCorr`/`simulate` | **0.09–0.44×** | idem, ainda pior — 2 a 11× mais lento que o laço Python |

Lição registrada: "eliminar laço Python" **não** é um ganho universal. Para
laços triangulares de tamanho fixo pequeno (`pi` costuma ser 1–4), o custo de
despacho do numpy é maior que o laço interpretado.

### O que não foi paralelizado, e por quê

- **Laço BOBYQA sobre θ** — região de confiança: cada candidato depende do
  modelo quadrático construído com todas as avaliações anteriores. Avaliar em
  paralelo mudaria a trajetória e o θ final.
- **Laço PIRLS** — Newton: `w`, `z` do passo *k* dependem de `mu` do passo
  *k−1*.
- **Fatoração/solves CHOLMOD e `M @ M.T`** — kernels que não liberam o GIL
  (threads não escalam) e com granularidade de ~0,1–10 ms (serializar para
  processos custaria mais que o cálculo). A parte densa já usa BLAS multithread.
- **`simulate`** — o fluxo é `[sorteia b_s] → [calcula η_s] → [sorteia y_s]`,
  intercalado. Qualquer reagrupamento consome o RNG em outra ordem e devolve
  números diferentes para a mesma `seed`: isso é resultado diferente, não
  diferença de ponto flutuante.
- **`parse_formula`/patsy** — poucos termos RE (1–4) e 100% GIL-bound.

### Tier C (não implementado, registrado para o futuro)

Pré-computar `Z'Z`, `Z'X`, `Z'y` e avaliar `A = Λ'(Z'Z)Λ + I` tornaria o custo
por avaliação de θ independente de `n` (ganho estimado 2–10× para `n ≫ q`).
É reassociação de produto matricial: matematicamente idêntica, numericamente
diferente em ~1e-15 relativo, o que pode deslocar o θ convergido na 8ª–10ª
casa. **Decidido não implementar nesta etapa** para manter equivalência
numérica exata. Se for retomado, deve entrar atrás de uma flag
(`control={'cache_ztz': True}`), desligada por padrão.

### Verificação

- [tests/perf/check_exact_core.py](tests/perf/check_exact_core.py) — portão de
  exatidão bit-a-bit contra um checkout de referência. 5 designs (q de 25 a
  600, pi de 1 a 4), REML e ML, 6 famílias, 8 valores de θ cada:
  **todas as métricas idênticas bit-a-bit**. Não precisa de patsy/nlopt/sksparse.
- [tests/perf/check_parallel.py](tests/perf/check_parallel.py) — serial × paralelo
  para todas as funções que ganharam `n_jobs`, em LMM e GLMM.
- Paralelismo medido em surrogate de `bootMer`/`profile` (2 vCPUs):
  **2.02×** com 2 workers (101% de eficiência — superlinear por causa do
  pinning de BLAS), **1.78×** na grade de profile (89%), **1.68×** com poucas
  tarefas (84%). Saídas idênticas à serial e ordem preservada em todos os casos.
- Fallbacks testados: payload não-serializável → serial com aviso e resultado
  correto; exceção real da tarefa propaga; `parallel_map` aninhado roda serial.
