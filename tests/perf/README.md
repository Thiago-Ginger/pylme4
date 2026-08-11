# tests/perf — verificação da camada de performance

Dois scripts, com propósitos diferentes.

## `check_exact_core.py` — portão de exatidão bit-a-bit

As otimizações em `pls.py` / `glmm.py` são de eliminação de trabalho redundante,
não de reformulação algébrica. Elas devem produzir **exatamente os mesmos bits**.
Este script prova isso rodando o núcleo atual lado a lado com uma cópia de
referência sobre designs sintéticos, exigindo `assert_array_equal` (igualdade
exata, não `allclose`) em `beta`, `u`, `b`, `eta`, `mu`, `pwrss`, `logdet`,
`sigma2`, `deviance` e `pirls_iter`.

Constrói `PLSState` / `GLMMState` direto, então **não precisa de patsy, nlopt
nem scikit-sparse** — exercita o caminho de Cholesky densa.

```bash
git worktree add /tmp/pylme4-ref <commit-anterior-a-otimizacao>
python tests/perf/check_exact_core.py /tmp/pylme4-ref/pylme4
```

Cobertura: 5 designs (q de 25 a 600, pi de 1 a 4), REML e ML, 6 famílias,
8 valores de θ cada. Rode isto depois de qualquer mexida no núcleo numérico.

`gamma(inverse)` pode aparecer como `skip`: nesses designs sintéticos a
*própria referência* estoura com matriz não positiva-definida. O script exige
que as duas versões falhem com o mesmo tipo de exceção antes de pular.

## `check_parallel.py` — equivalência serial × paralelo

Toda função que ganhou `n_jobs` precisa devolver os mesmos números com
`n_jobs=1` e com vários workers. Cobre `profile`, `profile_theta`,
`profile_sigma`, `confint_theta`, `confint_sigma`, `bootMer` e
`confint(method='boot')`, em LMM e GLMM — o caso GLMM também verifica que o
objeto `Family` sobrevive à ida para os workers.

```bash
python tests/perf/check_parallel.py
```

Precisa do ambiente completo (patsy, nlopt, scikit-sparse).

## O que já foi medido

Metodologia dos benchmarks que embasaram as decisões:

- **Temporização pareada e intercalada.** As duas variantes são chamadas
  alternadamente e guarda-se o *mínimo* de cada chamada isolada. Medir "A e
  depois B" sequencialmente é dominado por deriva de frequência da CPU — no
  laboratório isso produziu oscilações de ±12% em ambas as direções e chegou a
  inverter o veredito de duas otimizações.
- **Trabalho determinístico por chamada.** `glmm.update` faz warm start a
  partir de `st.beta`/`st.u`, então chamadas consecutivas fazem menos iterações
  PIRLS que a primeira. O warm start é resetado antes de cada chamada
  cronometrada.
- **Estado independente por variante** — senão uma variante contamina o warm
  start da outra e a comparação vira ruído.
- **Critério de corte:** speedup ≥ 1.03× em pelo menos uma carga, confirmado
  por mediana de 5 rodadas de alta repetição. O que não passou foi removido.
