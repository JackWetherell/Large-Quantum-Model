# iDEA (iDEA-latest 1.1.0): what it does and how it works

Source read: the 1.1.0 sdist from PyPI (installed copy at `.venv/lib/python3.12/site-packages/iDEA/`, about 3.7k lines). Units are atomic (Hartree, Bohr) throughout. Line links point at the installed copy.

## 1. Purpose

iDEA solves **1D model systems of a few electrons exactly** (the full many-body Schrödinger equation on a real-space grid, static and time-dependent). It also solves the same systems with **the standard approximations**: non-interacting, Hartree, HF, LDA, hybrid and KS-SCE. That lets you see exactly where and why the approximations fail. **Reverse engineering** turns an exact density into the exact Kohn–Sham potential, which gives you the exact v_xc for that system.

For a density-model project this is the key property: iDEA is a **ground-truth generator** for pairs such as v_ext → n, n → v_s (hence v_xc), and n → E.

## 2. Architecture and data flow

```
System(x, v_ext, v_int, electrons="ud", stencil=13)
   │
   ├── method.solve(system, k=0)            -> State   (ManyBodyState | SingleBodyState)
   ├── method.propagate(system, state, v_ptrb[t,x], t) -> Evolution
   │
   ├── observables.density / density_matrix / *_energy (state | evolution)
   └── reverse_engineering.reverse(system, target_n, method) -> fictitious System (its v_ext reproduces target_n)
```

| Module | Role |
|---|---|
| [system.py](../.venv/lib/python3.12/site-packages/iDEA/system.py) | `System` is a plain container plus a `check()` method. The spin configuration is a string (`"ud"`, `"uud"`). Two built-in systems: `systems.atom` (300 pts on [−20, 20], v_ext = −2/(\|x\|+1), `"ud"`) and `systems.qho` (ω = 0.25, `"uu"`). |
| [interactions.py](../.venv/lib/python3.12/site-packages/iDEA/interactions.py) | Softened Coulomb interaction v_int(x, x′) = 1/(\|x−x′\| + 1), stored as a dense N_x×N_x matrix. |
| [state.py](../.venv/lib/python3.12/site-packages/iDEA/state.py) | `ManyBodyState` holds `space` ψ(x₁…x_N), `spin` χ(σ₁…σ_N), `full` Ψ(x₁σ₁…) and `energy`. `SingleBodyState` holds `up`/`down` energies, orbitals, occupations and occupied indices. Plus the matching `*Evolution` classes. |
| [observables.py](../.venv/lib/python3.12/site-packages/iDEA/observables.py) | Density, density matrix, Hartree and exchange potentials and energies, from any state or evolution. |
| `methods/*.py` | One module per method. Every module exposes the same duck-typed interface (`hamiltonian`, `total_energy`, `solve`, `propagate`, `propagate_step`, …), so `iDEA.iterate_methods` can loop over them. The modules act as strategy objects; there are no classes. |
| [reverse_engineering.py](../.venv/lib/python3.12/site-packages/iDEA/reverse_engineering.py) | Static and time-dependent potential inversion. |

## 3. Discretisation

- **Grid.** Uniform; `dx = x[1] − x[0]`. Uniformity is assumed, not checked.
- **Kinetic energy.** −½ d²/dx² uses 3- to 13-point central finite differences (default 13, error O(dx¹²)). The stencil is truncated at the box edges, which amounts to **hard walls** (ψ = 0 just outside the grid).
- **Normalisation.** Orbitals satisfy ∫\|φ\|² dx = 1: the `eigh` eigenvectors are divided by √dx. So the density n = Σ fᵢ\|φᵢ\|² needs no dx factor.

## 4. Exact interacting solver ([methods/interacting.py](../.venv/lib/python3.12/site-packages/iDEA/methods/interacting.py))

**Hamiltonian** ([L82](../.venv/lib/python3.12/site-packages/iDEA/methods/interacting.py#L82)). The solver works in the full N_x^N spatial tensor-product space with no symmetry reduction:

H = Σᵢ 1⊗…⊗hᵢ⊗…⊗1 + U, &nbsp;&nbsp; U(x₁…x_N) = Σ_{i<j} v_int(xᵢ, xⱼ)

The Kronecker sums are built by folding `scipy.sparse.kron` (DIA format on CPU, CSR on GPU). `U` is diagonal. Because `einsum` can only multiply, the code computes **U = log(einsum(exp v_int, …))**, which turns a sum into a product ([L119](../.venv/lib/python3.12/site-packages/iDEA/methods/interacting.py#L119)).

**Eigenproblem** ([L249](../.venv/lib/python3.12/site-packages/iDEA/methods/interacting.py#L249)). `eigsh(which="SA")` (ARPACK/Lanczos) finds the `level` lowest eigenstates of the **distinguishable-particle** spatial Hamiltonian. Fermionic statistics are imposed afterwards:

1. Multiply each spatial eigenvector by the fixed spin product state χ built from the electrons string (`"ud"` gives \|↑↓⟩).
2. Apply the antisymmetriser A = Σ_P sgn(P) P over joint (x, σ) permutations, using `np.moveaxis` on a (N_x, 2)^N tensor.
3. Discard states that vanish, i.e. those with the wrong exchange symmetry. For `"uu"` the symmetric spatial states drop out; for 3 electrons the bosonic ground state drops out.
4. Normalise.
5. Remove duplicates: degenerate spatial partners can map to the same fermionic state, for example the 2D irrep of S₃ for 3 electrons.

Because so many states are discarded, `level` is heuristically over-requested: (\|N↑−N↓\|+1)²·N·(k+1).

This design works because H is spin-independent and commutes with particle permutations, so the eigenstates can be classified after solving. It is simple and fully general in N and spin. The cost is wasted eigenpairs and N_x^N memory. Two electrons on 300 points gives a 9×10⁴-dimensional H; three electrons gives 2.7×10⁷.

**Observables.** The density is n(x) = N Σ_σ ∫\|Ψ(xσ, x₂σ₂, …)\|² d(rest), computed from `full`, the antisymmetrised state.

**Time evolution.** Ψ(t+dt) = exp(−i H(t+dt) dt) Ψ(t) via `expm_multiply`. On CPU that is Al-Mohy–Higham's method. On GPU it is CuPy's version, or else a scaled truncated Taylor series. Only the spatial part is propagated; antisymmetrisation is redone whenever an observable is evaluated. The perturbation is sampled at the end of each step, so the scheme is first-order for time-dependent v_ptrb and exact for a static H.

**GPU.** The same code runs on either backend by swapping modules (`xp = cupy | numpy`, `sp = cupyx.scipy.sparse | scipy.sparse`). Results come back as NumPy arrays.

## 5. Mean-field methods (shared SCF engine)

[non_interacting.solve](../.venv/lib/python3.12/site-packages/iDEA/methods/non_interacting.py#L243) is the SCF loop for every approximate method. Each method module supplies only `hamiltonian(s, n↑, n↓, p↑, p↓)` and `total_energy`.

- **Each step.** Dense `scipy.linalg.eigh` of the N_x×N_x matrix H_σ, then linear mixing (default 0.5) of both n and p. Convergence is reached when ∫\|n − n_old\| dx < 10⁻¹⁰.
- **Spin symmetry breaking.** H↓ gets +10⁻¹²·x added ([L303](../.venv/lib/python3.12/site-packages/iDEA/methods/non_interacting.py#L303)).
- **Excited states.** Occupations stay aufbau throughout the SCF. State `k` is chosen *after* convergence by enumerating occupation patterns and ranking them by Σε ([L358](../.venv/lib/python3.12/site-packages/iDEA/methods/non_interacting.py#L358)). So excited states are **not** self-consistent (this is not ΔSCF).

| Method | Added to K + V_ext |
|---|---|
| `non_interacting` | (nothing) |
| `hartree` | v_H(x) = ∫ v_int(x, x′) n(x′) dx′ |
| `hartree_fock` | Nonlocal exchange −p_σ(x, x′) v_int(x, x′) dx, applied per spin |
| `lda` | v_xc[n](x) from the 1D HEG parametrisation for this softened interaction (class `HEG` in [lda.py](../.venv/lib/python3.12/site-packages/iDEA/methods/lda.py#L22)). The **spin-polarised** form is used for all densities; there is no LSDA. |
| `hybrid` | α × HF exchange + (1−α) × LDA xc, with α = 0.8 by default |
| `kssce` | KS strictly-correlated electrons: co-motion functions from the cumulant N_e(x). Experimental (see the issues section). |

**Total energy.** Computed as Σε minus double counting. I checked the LDA formulas: v_x = d(n·e_x)/dn and v_c = e_c − r_s·de_c/dr_s are implemented correctly. The hybrid energy α·E_HF + (1−α)·E_LDA, evaluated on hybrid orbitals, is also exactly consistent with the hybrid Hamiltonian, because the eigenvalue sums cancel.

**Time evolution.** Each orbital is propagated with `expm_multiply`, using H built from the density at the **previous** step. This is explicit and not self-consistent within a step, so the nonlinear term is first-order in dt. Orbitals are renormalised every step, which hides any norm drift.

## 6. Reverse engineering

- **Static** ([L16](../.venv/lib/python3.12/site-packages/iDEA/reverse_engineering.py#L16)). Iterates v ← v + μ(n^p − n_target^p), with p = 0.1 and μ = 1. The small power compresses the density's dynamic range, so the exponentially small tails still drive the update. Each SCF is warm-started from the previous density. With `method=non_interacting` and an exact target density, the result is the **exact KS potential** v_s, and then v_xc = v_s − v_ext − v_H.
- **Time-dependent** ([L116](../.venv/lib/python3.12/site-packages/iDEA/reverse_engineering.py#L116)). At each step, `scipy.optimize.root(method="hybr")` solves for the full vector v(x, t_j) that reproduces n(t_j): N_x equations in N_x unknowns. The Jacobian is singular in at least two directions: a constant shift of v only changes the phase, and ∫n = N is conserved. It works in practice but is not well-posed in the strict sense.

## 7. Issues found (verified numerically unless marked)

Script: `check_idea.py` in the session scratchpad; results are in the chat write-up. None of the SCF methods has a test in `tests/`, which is probably why issues 1 and 2 survived.

| # | Issue | Evidence | Impact |
|---|---|---|---|
| 1 | `hartree.total_energy` subtracts **½E_H** instead of E_H ([hartree.py:83](../.venv/lib/python3.12/site-packages/iDEA/methods/hartree.py#L83)). `hartree_energy` already includes the ½. | Atom: E_lib − (T + E_ext + E_H) = +0.4987 = E_H/2. HF and LDA agree to 10⁻¹⁰. | Hartree total energies are wrong. Densities are unaffected. |
| 2 | Restricted HF uses exchange from the **total** p instead of p_σ ([hartree_fock.py:67](../.venv/lib/python3.12/site-packages/iDEA/methods/hartree_fock.py#L67); same in [hybrid.py:57](../.venv/lib/python3.12/site-packages/iDEA/methods/hybrid.py#L57)). | 2-electron atom: the RHF density equals the **non-interacting** density to 10⁻¹⁴, because exchange cancels Hartree exactly. RHF energy is −1.6813 vs UHF −1.6965. | `restricted=True` is wrong for HF and hybrid. The default (unrestricted) is fine. |
| 3 | After de-duplication, `spaces` is truncated (`spaces[..., :M]`) rather than index-selected ([interacting.py:228](../.venv/lib/python3.12/site-packages/iDEA/methods/interacting.py#L228)). | 3-electron `"uud"`: `space`/`energy`/`full` are misaligned in 16 of 17 states. Only k = 0 is aligned. | `full`, `energy` and density are fine. `state.space` is wrong for k ≥ 1 whenever degeneracies exist (in practice N ≥ 3). **`propagate()` uses `state.space`**, so time evolution of excited states starts from the wrong state. |
| 4 | KS-SCE does not run. | Built-in atom: `ValueError: x must be strictly increasing` (the cumulant N_e(x) is flat in the tails). | The code also calls `s.interaction`, `s.dinteraction` and `s.ddinteraction`, which `System` does not define. *(Inferred, not run:)* the constant shift at [kssce.py:206](../.venv/lib/python3.12/site-packages/iDEA/methods/kssce.py#L206) gives ∫v n dx = E_sce·dx where E_sce is presumably intended. Untested and absent from `__all__`. |
| 5 | The log/exp interaction trick overflows. | Relative error ≈ 4×10⁻¹⁵ at strength 1 and 50; `inf` at strength 800 (3 electrons). The limit is roughly strength × max pair sum ≲ 709. | Only matters for strongly scaled interactions. |
| 6 | `observables.kinetic_energy(ManyBodyState)` raises `NotImplementedError`. | Run directly. | The API advertises it but it is not implemented. |
| 7 | Minor issues. | Read only. | `softened_interaction_alternative` docstring says s/(√(x−x′)+a)², but the code computes s/√((x−x′)²+a). `System.check` has an error message missing its `f` prefix. The `test_time_evolution` wavefunction metric is √(−2 + 2\|⟨ψ₁\|ψ₂⟩\|), which is ≤ 0 by Cauchy–Schwarz, so it can only return 0 or NaN (it printed 0.0). `reverse_propagation`'s `_residual` does not forward `**kwargs`, so hybrid would raise `KeyError: 'alpha'`. |
