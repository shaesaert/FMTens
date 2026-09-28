
## Noise-scale fix in Gaussian gridding (2026-09-28)

**Bug.** `src/abstraction/utils/abstraction_transition_builder.py` passed the noise
variance, eig(Bw·Σ·Bwᵀ), to `norm.cdf(..., scale=...)`, which expects the standard
deviation. Affected: 1D path (`_transition_matrix_1d`) and 2D path
(`_transition_matrix_2d_separable`). The same pattern exists in SySCoRe's
`GridSpace_nd.m` and `GridSpace_nonlin_tensor.m`, which FMTens follows.

**Effect.** Results are unchanged when Bw·Σ·Bwᵀ = I. Otherwise the abstraction used
std = variance: too little noise when the variance is below 1 (optimistic
certificates), too much when above 1 (conservative).

**Fix.** Pass `sqrt(eig(Bw·Σ·Bwᵀ))` as the scale in both paths.
Regression test: `tests/test_noise_scale.py`.

**Reproducing earlier results.** Tag `pre-noise-fix` marks the last commit before this fix.
