# Wetherell (2018): *Electron correlation and screening in model nanostructures*

PhD thesis, University of York, Physics, November 2018. 121 pages; PDF page index = printed page number, cited as [p. N]. Source: `background/jw1294_thesis.pdf`. These notes are a reference digest written in my own words. Where something is read off a figure rather than the text, it is marked *(from figure)*.

## 1. Overview

The thesis builds 1D model systems of 1–3 electrons with a softened Coulomb interaction, solves them exactly with iDEA, and compares MBPT (HF, many flavours of GW), KS-DFT (LDA) and hybrids against the exact densities, Kohn–Sham potentials and quasiparticle energies. It has four main results:

1. **The flavour of GW matters a lot for the density.** One-shot G₀W₀ from HF orbitals is the best of the dynamically screened variants. A new static "inertial" screening, P(t = 0) instead of P(ω = 0), beats both dynamic GW and COHSEX [pp. 53–70].
2. **Self-screening error.** It is the systematic GW error affecting the density and IP, and a three-parameter local density functional, fitted on one-electron slab systems, removes it [pp. 71–80].
3. **Nearsightedness.** Nonlocal MBPT potentials (Fock operator, GW Σ) are nearsighted: the operator for two distant subsystems is the sum of the subsystem operators. The local exact KS potential instead needs an ultranonlocal step [pp. 81–93]. Hybrids inherit this property, and with α fixed by Koopmans-type constraints they give near-exact densities and gaps [pp. 94–103].
4. **Time-dependent MBPT.** Propagating orbitals under HF exchange plus a statically (inertially) screened correlation term clearly outperforms the adiabatic LDA for driven and colliding-electron systems, at TDDFT-like cost [pp. 104–112].

**Publications** [pp. 9–10]:
- Ch. 4: in preparation at the time, "Electron densities and Kohn–Sham potentials from MBPT" (with Talirz, Hodgson, Godby).
- Ch. 5: PRB 97, 121102(R) (2018).
- Ch. 6: PRB 99, 045129 (2019).
- Ch. 7: Phys. Rev. Materials 2, 040801(R) (2018), joint first author with Elmaslmane.
- Ch. 8: in preparation, "Time-dependent electron densities from MBPT".
- Also a co-author of Entwistle *et al.*, "LDAs from finite systems", PRB 94, 205134 (2016), which is used for the LDA throughout.

## 2. Conventions used throughout

- **Units and interaction.** Hartree atomic units. Softened interaction 1/(|x−x′|+1), modelling two charged disks in a nanowire [p. 21, p. 43].
- **Like-spin electrons.** Electrons are **like-spin** unless stated otherwise: the spatial wavefunction is made exchange-antisymmetric. The stated reason is to get closer to the exchange and correlation of larger systems for a given computational cost [p. 43, p. 54].
- **Model systems:**
  - "Exchange-dominated" (HF is accurate): QHO, ½ω²x² with ω = 0.25, for 1, 2 and 3 electrons [p. 57].
  - "Correlated" (HF fails): atom-like −1/(a|x|+1), with a = 0.05 for 1 and 2 electrons and a = 0.02 for 3 electrons. p. 57 prints the formula without the minus sign; pp. 72 and 99 give the sign explicitly, so p. 57 is a typo.
- **Density error.** ∫\|n_exact − n_approx\| dx, normalised by N in Fig. 4.1 (maximum 2 means no overlap) [p. 57–58].
- **Not stated anywhere:** grid sizes, box lengths (other than what figure axes show), time steps, imaginary-time and frequency grids.

## 3. Chapter-by-chapter

### Ch. 2: Theory [pp. 14–42]
Standard material: one- and many-particle QM, the Pauli principle, Hartree (local and self-interacting) vs HF (nonlocal Fock operator Σ_x(x,x′) = −Σ_occ φ_n(x)φ_n*(x′)U(x,x′), self-interaction-free), and FCI. MBPT covers:
- G = −i⟨T ψ(x′,t′)ψ†(x,t)⟩, with n(x) = −iG(x,x,0⁻);
- the Dyson equation G = G₀ + G₀ΣG; the RPA P = −iGG; W = v + vPW; Σ = V_H δ + iGW;
- Hedin's equations, with vertex Γ; GW corresponds to δΣ/δG = 0 and BSE to δΣ/δG ≈ iW.

DFT: HK, KS (V_KS = V_ext + V_H + V_xc), LDA, and TDDFT, where v_xc depends on the density history (memory).

### Ch. 3: iDEA as used in the thesis [pp. 43–52]
- **Exact ground state:** imaginary-time Crank–Nicolson propagation with renormalisation. Real-time Crank–Nicolson for dynamics [p. 45].
- **Ground-state reverse engineering:** V_KS ← V_KS + μ(n_KS^p − n^p) with **μ = 0.01, p = 0.05**. The small p emphasises low-density regions [p. 45].
- **SCF mixing:**
  - LDA: linear mixing of V_KS with α = 0.2 (Alg. 1).
  - HF: linear mixing of the Hamiltonian with α = 0.9 (Alg. 2) [pp. 46–47].
- **LDA:** fitted to 1-, 2- and 3-electron finite "density slabs" (Entwistle 2016). A 3D-HEG option also exists.
- **GW module (space-time method)** [pp. 47–51]:
  1. G₀(x,x′,iτ) is built from starting orbitals on an imaginary-time grid (occupied for τ ≤ 0, unoccupied for τ > 0, factor e^{−ε τ}, chemical potential set to 0). Orbitals are real in 1D.
  2. P(iτ) = −iG(iτ)G(−iτ). At τ = 0 an alternative G ("G_ALT", which treats τ = 0 as electron addition) is used.
  3. W(iω) = (I − vP(iω))⁻¹v.
  4. Σ_xc(iτ) = iG(iτ)W(iτ). V_H is added on the diagonal. If the starting orbitals came from a mean-field method, its V_Hxc is subtracted.
  5. G(iω) = (I − G₀Σ)⁻¹G₀.
  6. A "Hedin shift" Σ(iω) → Σ(iω) − ½[⟨φ_L\|Σ(0)\|φ_L⟩ + ⟨φ_H\|Σ(0)\|φ_H⟩]·I conserves particle number, which is monitored.
  7. n(x) = −iG(x,x,iτ = 0).
- **Illustrations:** for the 2e QHO, W is weaker than v at short range and stronger at long range (Fig. 3.3). For two 1D H atoms at separations 0–8 a.u., like-spin electrons give no bond, while opposite-spin electrons give a binding curve (Fig. 3.6).

### Ch. 4: Densities from GW flavours [pp. 53–70]
- **Self-energy flavours:**
  - G₀W₀@LDA and G₀W₀@HF;
  - GW₀ (G and Σ updated, W₀ fixed);
  - fully self-consistent GW, which is starting-point independent.
- **Polarisability options:**
  - dynamic RPA, P(ω);
  - static COHSEX, P(ω = 0), with Σ_COH = ½δ(x,x′)W_c(x,x′,ω = 0);
  - new **inertial**, P(t = 0). It fixes the screening at the instant of addition, the intuition being heavier, slower plasmons, and is intended for stronger correlation [pp. 55–56].
- **QP energies.** For dynamic GW: ε^QP = ε⁰ + ⟨φ⁰\|Σ(ω = 0)\|φ⁰⟩ (first order). For the static variants they are the eigenvalues [p. 57].
- **Results.** These are given as colour charts only (Fig. 4.1 for density, Fig. 4.2 for IP, EA and gap), with no tables.
  - **Starting point:** with dynamic P, one-shot beats self-consistent (error cancellation between missing vertex and missing self-consistency). HF orbitals beat LDA orbitals for both density and QP energies. G₀W₀@LDA gives a spurious central bump, wrong density decay and an oscillating V_Hxc [pp. 60–62].
  - **Full self-consistency:** fully self-consistent GW over-diffuses the 1e density (self-screening) and over-screens the 2e atom, pulling density inward. HF pushes it outward [pp. 63–64].
  - **COHSEX vs dynamic:** COHSEX densities are nearly indistinguishable from dynamic ones and inherit the self-screening error [pp. 66–67].
  - **Inertial screening** is best for all systems: it roughly halves the 1e density error and gives accurate decay and IPs. It is nearly exact for 2e in GW₀@LDA. The remaining error sits in delocalised central regions, attributed to electron–hole interaction missing from the RPA. Its EA is slightly worse [pp. 66–70].
- **KS potentials of GW densities.** Reverse-engineered KS potentials of GW densities reproduce features of the exact KS potential when the density is good [p. 65, p. 70].

### Ch. 5: Local self-screening correction [pp. 71–80]
- **Diagnostic.** An orbital effective potential V^eff(x) = φ(x)⁻¹ ∫ Σ(x,x′,ε) φ(x′) dx′, evaluated at Σ(ε_F) because only imaginary frequencies are available. For a 1e atom (−1/(0.05\|x\|+1)), V_H and V_x^eff cancel exactly but V_c^eff ≠ 0. That residual, the self-screening potential V_ss, accounts for the whole GW error and pushes density outward (Fig. 5.1) [pp. 72–74].
- **Training set.** One-electron slabs with density n₀·exp(−10⁻¹¹(mx)¹²) + 10⁻⁴·exp(−0.007\|x\|), plateau n₀ ∈ [0.03, 0.58]. The external potential comes from the single-orbital approximation, which is exact for 1e. Then ε_ss = E_GW − E_exact per electron, with E_GW the lowest eigenvalue of V_ext + V_Hxc^eff. ε_ss(0) = 0 is imposed. Points get sparse at low n because the imaginary-frequency grids grow [pp. 74–75].
- **Fit:** ε_ss(n) = −a·n·exp(−b nᶜ), with a = 4.09268, b = 9.20609, c = 0.53652 (eq. 5.3).
- **Correction:** V_ssc(n) = −d(nε_ss)/dn = a·n·e^{−bnᶜ}(2 − bc·nᶜ) (eq. 5.5; I checked it is consistent with 5.3), added to Σ (eq. 5.6). It reproduces the exact training energies (not the densities, which are very sensitive to the potential) [p. 76].
- **Tests** [pp. 76–79]:
  - 1e atom: GW+ssc density essentially exact, and V_ssc cancels V_c^eff.
  - 2e (a = 0.05) and 3e (a = 0.02): decay and IP corrected; central-peak errors remain (e–h interaction, would need vertex corrections in P).
  - G₀W₀@LDA + ssc: density error reduced by 16–50%.
- **Table 5.1** (IP in a.u.):

| N | KS: GW | KS: GW+ssc | QP: GW | QP: GW+ssc | Exact |
|---|---|---|---|---|---|
| 1 | 0.908 | 0.900 | 0.908 | 0.900 | 0.900 |
| 2 | 0.624 | 0.610 | 0.577 | 0.577 | 0.611 |
| 3 | 0.662 | 0.641 | 0.675 | 0.654 | 0.642 |

  "KS" means the IP was taken as −(HOMO) of the KS potential reverse-engineered from the GW density; this route is preferred to fitting the density's asymptotic decay, I = ¼(∂ ln n/∂x)².

### Ch. 6: Nearsightedness of MBPT vs KS [pp. 81–93]
- **System:** 2 like-spin electrons in an asymmetric double well. *(From Fig. 6.1:)* wells at x ≈ ±15, the left one shallower (≈ −0.9) than the right (≈ −1.2), box x ∈ [−30, 30]. Exact: one electron per well. Non-interacting: both electrons in the right (deeper) well [pp. 82–83].
- **Exact KS potential.** It has a **step** S_xc between the wells that lifts the right well, so the second KS orbital localises on the left. The step sits where the local effective IP changes (Hodgson's analysis). It is about 0.45 a.u. *(from figure)*; the text says its exact height is numerically limited but must be ≥ 0.03 a.u. V_xc ≈ −V_H plus the step [pp. 82–83].
- **HF.** Near-exact density. The Fock operator has no step-like feature and is nonlocal only on the scale of a subsystem.
  - The effective exchange potential for the left orbital cancels its own Hartree bump (SIC) but not the other electron's, so each electron feels the other's repulsion. The local KS potential cannot remove self-interaction orbital by orbital, so it needs the step instead [pp. 84–86]. The node of φ₁ makes the ratio in eq. 6.1 singular and needed careful numerical handling.
- **Nearsightedness, quantified.** F ≈ F_L + F_R, with the subsystem operators computed independently, to about 0.03 a.u. (about 2% of F's scale). The HF density from F_L + F_R differs by about 0.01 (about 2%), and both go to zero with separation. In contrast V_xc = V_xc^L + V_xc^R + S_xc (eqs. 6.2–6.3) [p. 87].
- **GW.** G₀W₀ from *non-interacting* orbitals, which put both electrons in one well, still gets one electron per well: the self-energy swaps HOMO and LUMO in the first iteration. So G₀W₀ accuracy depends on the unoccupied starting orbitals too. Fully self-consistent GW is better, with a small self-screening error (Fig. 6.6) [pp. 89–91].
- **Derivative discontinuity.** Right well with 1 + 10⁻⁴ electrons. The exact KS potential shows a plateau of height Δ bounded by steps that move outward as δ → 0⁺. The Fock operator shows nothing similar: the fractional electron simply sees the full electron's Hartree potential, so the IP changes without any discontinuity in the operator. The same argument covers hybrids, generalised KS and GW [pp. 91–93].

### Ch. 7: Koopmans-compliant hybrids [pp. 94–103]
- **Forms:**
  - Main: V_xc^HYB = α·V_x^HF + (1−α)·V_xc^LDA (eq. 7.1).
  - Alternative, PBE0-like: α·V_x^HF + (1−α)·V_x^LDA + V_c^LDA (eq. 7.2).
  - The LDA is the slab-fitted one; the author found HEG LDAs give indistinguishable results [p. 97].
- **Fixing α.** α is chosen by one of three internal-consistency conditions, motivated by piecewise linearity of E(N) and by hybrids having (and needing) no derivative discontinuity, since a hybrid acts as a screened-exchange self-energy:
  - (A) ε_N(N−1) = −A(N−1), with A(N−1) = E(N) − E(N−1) [the LUMO–A condition];
  - (B) ε_N(N−1) = ε_N(N) [the LUMO–HOMO condition];
  - (C) ε_N(N) = −I(N) = E(N) − E(N−1) [the HOMO–I condition, i.e. the generalised Koopmans' theorem].

  [pp. 95–97]
- **Results** [pp. 98–103]:
  - **3e QHO** (ω = 0.25): all conditions give α ≈ 1. The conditions can't be satisfied simultaneously, but the three crossing points are close and densities between (A) and (C) are excellent.
  - **3e atom** (−1/(0.05\|x\|+1)): the α from (A) and (C) differ by about 10%, but both give density error < 0.03. (A) gives a slightly better density than (C). Exchange-only mixing (7.2) is slightly worse: minimum error about 0.05 vs about 0.025. ∂E/∂N tracks the exact result between 2 and 3 electrons.
  - **Asymmetric double well:** the LDA puts about 0.1 e too much in the deeper well. HF and the GKT hybrid (α ≈ 1) are near-exact. Even α = 0.2 gives the correct charge per well, though with the wrong shape inside each well, which is attributed to nearsighted exact exchange.
- **Table 7.1** (2-electron systems, a.u.):

| | LDA | HF | Hybrid | Exact |
|---|---|---|---|---|
| QP gap, harmonic | 0.222 | 0.491 | 0.472 | 0.469 |
| QP gap, atom-like | 0.037 | 0.172 | 0.152 | 0.141 |
| IP, harmonic | −0.761 | −0.620 | −0.629 | −0.628 |
| IP, atom-like | 0.551 | 0.620 | 0.608 | 0.612 |

  The harmonic IPs are printed as negative; I left the sign convention as printed. Hybrid errors: gap 1% and 8%, IP 0.2% and 0.5%.

### Ch. 8: Time-dependent MBPT [pp. 104–112]
- **TDHF.** Propagate occupied orbitals under −½∂² + V_ext(t) + V_H(t) + Σ_x(t), using Crank–Nicolson.
- **TDGW (new).** Add a static correlation term from inertial screening:
  - P_inertial(x,x′) = −[Σ_occ φ_k(x)φ_k*(x′)]·[Σ_unocc φ_k(x′)φ_k*(x)]. It needs orbitals only, no energies, and no two-time G.
  - W = v + vPW.
  - Σ_c = −Σ_occ φ_k(x,t)φ_k*(x′,t)·[W(x,x′) − v(x,x′)].

  Σ_c is Hermitian, so orbital orthonormality is preserved; the operators become complex because the orbitals do (eqs. 8.4–8.6) [pp. 105–106].
- **Electric field.** Driven 2e (like-spin) correlated atom, field V = −0.01x switched on at t = 0, followed to t = 100. Numerical stability was checked with the harmonic potential theorem. The ALDA is poor in both the ground state (spurious central bump, wrong decay) and over time. TDHF is much better but has the wrong central shape. TDGW is best for most of the run, degrading at late times. At t = 100, Σ_c screens Σ_x by about 5.3% [pp. 107–109].
- **Electron collision.** Initial potential V_atom + V_bump holds one electron on each side, so HF is nearly exact for the ground state. At t = 0 the bump is removed and the electrons fall in and collide, so correlation grows with time. TDGW gives the best shape at t = 100. TDHF and the ALDA put too much charge in the centre. TDHF beats TDGW early on (the inertial GW ground state carries self-screening error), and TDGW pulls ahead as correlation builds [pp. 109–111]. The bump's functional form is not given.

### Ch. 9: Conclusions [pp. 113–114]
These restate the results above. Future work: put the ssc into large-scale codes, add electron–hole vertex terms in P for delocalised regions, test Koopmans hybrids in extended systems, and improve TDGW with dynamic correlation and vertex terms.

## 4. Glossary
- **Self-screening error:** the part of the self-interaction error left when exchange is screened by W without vertex corrections; each electron screens itself [p. 72].
- **Inertial screening:** P_RPA evaluated at t = 0 rather than ω = 0 [p. 56, p. 106].
- **Hedin shift:** constant shift of Σ(iω) to conserve particle number in self-consistent GW [p. 51].
- **Orbital effective potential:** V^eff_m = φ_m⁻¹ ∫ Σ φ_m. It is orbital-dependent and singular at nodes [p. 73, p. 85].
- **Nearsightedness:** additivity of operators for distant subsystems, F ≈ F_L + F_R [p. 87].
- **Density slab:** finite system with a uniform central density, used to fit local functionals [p. 74].
- **Conditions A, B, C and GKT:** Koopmans-type constraints used to fix α [p. 96].

## 5. Relevance to learning functionals and nonlocal operators with ML

1. **Target choice matters more than the architecture.** The exact local v_xc[n] has steps whose height and position depend on distant density, while the nonlocal Fock and GW operators are nearsighted (Ch. 6). A model that outputs a **nonlocal operator** K(x,x′), or orbital-dependent potentials, should learn far more easily and transfer between subsystems than a model regressing local v_xc(x). An attention kernel is exactly such a learned operator.
2. **Additivity is a testable inductive bias.** F ≈ F_L + F_R for separated fragments gives a concrete held-out test: train on single fragments, test on composites. A local-v_xc model must also learn the step S_xc, which is a function of *global* information (the local effective IP of each fragment).
3. **Step regions are ill-conditioned targets.** The step lives where the density is exponentially small, so an L2 loss on n is almost blind to it. Reverse engineering uses p = 0.05–0.1 power updates for the same reason, and the step height is only bounded from below (≥ 0.03 a.u.). Losses on v_xc need density-aware weighting, or one should use quantities like the electric field or ∇v that are better behaved.
4. **Orbital-based representations.** Inertial P and Σ_c are closed-form bilinear functions of occupied and unoccupied orbitals (eq. 8.4). That is structurally the same as attention scores built from φφ* products. Orbital inputs carry information the density alone hides (e.g. G₀W₀'s dependence on unoccupied orbitals, p. 89).
5. **Precedent for training on synthetic finite systems.** Ch. 5 fits a local functional to a designed family of slabs (n₀ ∈ [0.03, 0.58]) and then tests it on atoms. The training-set design (coverage in n, enforcing ε(0) = 0) and the "exact on training energies, not training densities" caveat both carry over to an ML functional.
6. **Physics constraints as training signals.** These include piecewise linearity of E(N), the Koopmans conditions A, B and C, Janak's theorem, asymptotic decay I = ¼(∂ ln n)², and the exact 1e limit (SIC). They can be used as auxiliary losses or architectural constraints.
7. **Like-spin data.** The thesis uses like-spin electrons by default. For data generation that is a choice to make deliberately: `"uu"` vs `"ud"` changes the physics (exchange-dominated vs correlated singlet).
8. **Time-dependent targets.** The ALDA fails badly once correlation is dynamic (the collision system). Memory and nonlocality matter, and a Hermitian static nonlocal operator already captures much of it.

## 6. Reproducible with current iDEA (1.1.0)

| Item | Current iDEA |
|---|---|
| Exact ground states: QHO ω = 0.25 (1–3e), atom −1/(a\|x\|+1) (a = 0.05 for 1–2e, 0.02 for 3e), like-spin `"u"`, `"uu"`, `"uuu"` | Yes (`interacting.solve`; the current solver uses ARPACK, not imaginary-time CN) |
| Double well (Ch. 6/7), collision potential (Ch. 8) | Yes if the potentials are re-specified. The thesis gives no exact forms: wells only from the figure, bump form not given. |
| Exact KS potential by reverse engineering | Yes (`reverse_engineering.reverse`; defaults μ = 1, p = 0.1 vs the thesis's 0.01 and 0.05) |
| HF, LDA, hybrid of the eq. 7.1 form (α·HF + (1−α)·LDA xc) | Yes. The current `hybrid` uses exactly eq. 7.1. The current LDA is a HEG parametrisation, not the slab LDA (the thesis says they give indistinguishable results). |
| Koopmans α (conditions A, B, C) | Buildable by scanning α with N and N−1 calculations |
| TDHF, adiabatic LDA, exact TD (field −0.01x) | Yes (`propagate`) |
| GW (all flavours, space-time method, Hedin shift), ssc, inertial P, TDGW | **No.** No MBPT module in the current release. Inertial P/W/Σ_c (eqs. 8.4–8.6) are simple enough to add from orbitals. |
| Fractional electron numbers (1 + 10⁻⁴) | No direct support |

## 7. Limitations and observations
- **Author-stated limitations:**
  - Electron–hole vertex terms are still missing in P, which explains the errors in delocalised regions.
  - TDGW degrades at long times.
  - Koopmans hybrids are untested in extended systems.
  - The step height is numerically limited.
- **My observations:**
  - The sign of the atom-like potential is missing on p. 57.
  - Ch. 4 has no numerical tables (colour charts only).
  - Grids, time steps and frequency grids are not documented, and neither are the double-well and bump potential forms, so exact reproduction needs the original input files.
  - In Ch. 8 the text says "TDFT begins by outperforming TDGW" (p. 110); from context this should read TDHF.
