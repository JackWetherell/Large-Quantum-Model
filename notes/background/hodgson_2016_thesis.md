# Hodgson (2016), *Electrons in model nanostructures*: study notes

**Source.** `background/MJPHodgson_PhD_thesis.pdf` (137 pp.), read in full: front matter, chapters 1–9, appendices A–B and the bibliography. Bulk text came from `pdftotext -layout`. Equations, figures and tables were checked against the rendered pages.

**Page convention.** [p. N] is the PDF page index. In this PDF it equals the printed page number (checked: the title page is p. 1 and the abstract is p. 2).

**Markers.** "**Obs.:**" marks my own observations, inferences or checks, as opposed to claims in the thesis. Numbers marked "≈ (read off fig.)" are my readings of plots, not values the thesis states.

**Units.** Atomic units throughout (ħ = m_e = e = 4πε₀ = 1) [p. 18]. All model systems are 1D.

---

## 1. Metadata and overview

| | |
|---|---|
| Title | *Electrons in model nanostructures* |
| Author | Matthew James Paul Hodgson |
| Degree / institution | PhD, Physics, University of York; dated September 2016 [p. 1] |
| Supervisor | Rex Godby (named as "my supervisor" in the acknowledgements [p. 12]) |
| Length | 137 pages: 9 chapters plus appendices A (iDEA) and B (functional development) |
| Funding | EPSRC [p. 12] |
| Underlying papers [p. 13] | Hodgson et al., PRB 88, 241102(R) (2013): Ch. 4.1.1 and Ch. 6 (exact TD potentials for tunnelling). Entwistle, Hodgson, Wetherell, Longstaff, Ramsden, Godby, "Local density approximations from finite systems", submitted 2016: Ch. 5, with most calculations by Entwistle. Hodgson, Ramsden, Durrant, Godby, PRB 90, 241107(R) (2014): Ch. 7 (MLP). Durrant et al., arXiv:1505.07687: §§7.2–7.3 (localisation). Hodgson, Ramsden, Godby, PRB 93, 155146 (2016): Ch. 8 (origin of steps). |

**Overview.** The thesis uses an early version of iDEA to study (TD)DFT with no approximations, for 1D systems of one to three spinless electrons interacting through a softened Coulomb repulsion. iDEA computes the exact many-body wavefunction and density (and current), then reverse-engineers the exact Kohn–Sham (KS) potential. The thesis then asks which features of the exact v_xc are missing from local approximations and why they matter [pp. 14–16, 38–45].

The main results are as follows:

1. **LDAs built from finite "slab" systems** of 1, 2 or 3 electrons are almost identical to each other and to the exact-exchange curve of the 1D HEG. They give a surprisingly good self-interaction correction (SIC), but they fail when correlation is strong, for weakly confined systems, and for tunnelling when applied adiabatically [Ch. 5].
2. **The exact TD KS potential of tunnelling electrons has dynamic spatial steps.** These steps form at density minima where the velocity field u = j/n peaks and the density has been modified by the interaction. The ALDA lacks them and so gets the timing of tunnelling reversal wrong [Ch. 6].
3. **Localisation is high in these finite systems.** This motivates the **single orbital approximation (SOA)**, v_SOA = ∇²n/(4n) − (∇n)²/(8n²), which is exact for one occupied orbital. The SOA is then mixed with a reference potential in proportion to the localisation: the **mixed localisation potential (MLP)**, v_MLP = f v_SOA + (1−f) v_ref. With a constant, hand-tuned f, the MLP gives accurate self-consistent densities for dissociated molecules, polarised chains, single wells and TD tunnelling, largely because it can produce steps [Ch. 7].
4. **Steps in v_xc come from a change in the "local effective ionisation energy"**, the decay rate of the density coming from different subsystems. They also come from its KS counterpart, the crossover of the dominant KS orbital densities. These give S_xc = (I_R − I_L) + (ε_R − ε_L). Steps sit at density minima that coincide with *integer electron points*. Delocalisation softens them, and in symmetric systems they superpose into bumps. The thesis also argues that steps are *not* a manifestation of the derivative discontinuity [Ch. 8].

---

## 2. Chapter-by-chapter summary

### Front matter [pp. 1–13]

- **Abstract** [p. 2]: summarised above. Its key phrasing is: steps "form due to a change in the 'local effective ionisation energy' of the electrons: this concept is well defined for strongly localised electrons". The MLP uses "the analytical form of the Kohn–Sham potential in the limit of complete localisation".
- **Acknowledgements** [p. 12]: Entwistle led Ch. 5 (following Longstaff) and Durrant led §§7.2–7.3. Jack Wetherell is credited with optimising the many-body solver [p. 38, fn. 2] and with iDEA development.
- **Table 4.1** [p. 45] maps each class of system to the chapter that uses it:
  - homogeneous slabs → LDA (Ch. 5);
  - tunnelling electrons → exact TD KS potential (Ch. 6);
  - localised electrons → role in functionals (Ch. 7);
  - molecules → origin of steps (Ch. 8).

### Ch. 1 Introduction [pp. 14–16]

- **Strategy.** Model few-electron systems exactly so that TDDFT can be studied "in the absence of approximation". Each system is designed to isolate a general phenomenon [pp. 14–15].
- **Name.** iDEA = "interacting dynamic electrons approach" [p. 15, fn. 3]. It allows any external potential plus any static perturbing field, with real-time propagation [p. 15].

### Ch. 2 Established theory [pp. 17–29] (standard background; only items reused later)

- **KS energy:** $E=\sum_i\varepsilon_i+E_{xc}-\int n\,(v_{xc}+\tfrac12 v_H)$ (Eq. 2.12 [p. 21]). The xc hole obeys ∫n_xc dr′ = −1 [p. 21].
- **Exact properties** [pp. 22–23]: $v_{xc}^{(1)}=-v_H^{(1)}$ and $E_{xc}^{(1)}=-E_H^{(1)}$ (SIC); a derivative discontinuity Δ_xc that is a uniform jump at integer N; v_xc → −1/r; Lieb–Oxford, which takes a different form in 1D.
- **Author's stance** [pp. 22, 28]: exact constraints should not be enforced at the cost of density accuracy.
- **GGA** [pp. 23–24]: $s=|\nabla n|n^{-4/3}$ diverges in exponential tails; PBE $F_x=1+as^2/(1+bs^2)$ (Eq. 2.20).
- **HF as correlation detector:** "correlation is defined as anything beyond the capability of HF" [p. 25, fn. 2]. Hartree theory (v_xc = 0) is the SIC benchmark [p. 25].
- **TDDFT** [pp. 25–28]: Runge–Gross and van Leeuwen; zero-force theorem ∫n∇v_xc = 0 (Eq. 2.25); $dE_{xc}/dt=\int (dn/dt)v_{xc}$ (Eq. 2.31); adiabatic approximation $v_{xc}[n](r,t)=v_{xc}[n(t)](r)$; linear-response Dyson equation (Eq. 2.33).

### Ch. 3 Limitations of the established theory [pp. 30–37]

- **DFT failures** [pp. 30–32]: delocalisation and static-correlation errors (figures from Cohen, Mori-Sánchez & Yang [7]). Local functionals put non-integer charge on the atoms of a dissociated molecule because they lack the SIC [p. 31].
- **Transport** [pp. 32–34]: Landauer G = G₀ΣT_n. The variant "frozen t = 0 KS potential plus field" misses dynamic interaction. Computed I–V curves for organic molecules are off by 1–2 orders of magnitude.
- **Burke et al.'s TDDFT error sources** [pp. 35–36]:
  - Er. 1: ground-state error.
  - Er. 2: missing nonlocality.
  - Er. 3: missing memory.
  - Er. 4: non-density observables.

  The thesis targets Er. 1–3. It probes Er. 1 by propagating with the frozen exact ground-state KS potential.
- **HOMO** [pp. 36–37]: the exact KS HOMO equals −I. Local potentials decay too fast and under-bind, by up to 5 eV for N₂.

### Ch. 4 Method: the iDEA code (thesis-era version) [pp. 38–45]

**Many-body solver** [pp. 38–39]

- **System size.** N = 2 or 3 electrons for interacting runs; "it is not possible … to model more than 3 electrons exactly" [p. 38, fn. 3].
- **Ground state.** Imaginary-time propagation of an antisymmetric trial wavefunction, typically for ~10² a.u. of imaginary time [p. 38].
- **Antisymmetry.** Kept and exploited by "cut-down" matrices that store only the antisymmetric part, allowing finer grids (due to J. Ramsden) [p. 38].
- **Spinless electrons** [p. 39]: every electron occupies its own KS orbital, which maximises exchange and correlation per unit cost. A singlet pair, by contrast, occupies one KS orbital.
- **Interaction** [p. 39]: $u(x_i,x_j)=(|x_i-x_j|+\gamma)^{-1}$, with γ set by the wire width (App. A.2). γ = 1 everywhere except the main Ch. 6 system, which uses γ = 0.1 [pp. 67, 121]. Ch. 6's re-run of the Ch. 5 system uses γ = 1 [p. 72].
- **Propagator.** Crank–Nicolson for both imaginary and real time (Eq. 4.1 [p. 39]):
  $$\left(\hat I+\tfrac{i}{2}\delta t\,\hat H\right)\phi(t+\delta t)=\left(\hat I-\tfrac{i}{2}\delta t\,\hat H\right)\phi(t).$$
  The thesis states it is "numerically stable, provided one abides by $\delta t\lesssim \frac{1}{N}\delta x^{N}$" [p. 39].
- **Density** (Eq. 4.2): $n'(x,t)=N\int dx_2\ldots dx_N|\Psi|^2$ [p. 39].
- **Grid.** Converged results are generally found for δx ≤ 0.1 [p. 42]. Box lengths are not stated for any system.

**Ground-state reverse engineering** (Eq. 4.3 [p. 40]):

$$v_s\;\to\;v_s+\mu\left[n(x)^p-n'(x)^p\right],\qquad 0<\mu\le1,$$

- Notation: n is the KS density from the current v_s, and n′ is the exact target.
- **Parameters.** The fixed point n = n′ holds for any p and μ. Convergence is "especially rapid" for p ≈ 0.05 and μ = 1. A small p puts the emphasis on low-density regions and avoids oscillatory instability in high-density regions [p. 40].
- **Convergence metric** (Eq. 4.4): $f_n=\int|n-n'|\,dx$. Getting below 10⁻¹¹ takes ~1500 iterations [p. 40].
- **p dependence** (App. A.1.1 [p. 119]): for two interacting electrons in a harmonic well, going from p = 1 to p = 0.05 cuts the iteration count by ~10³ and improves accuracy by ~10³. Some systems need p and/or μ changed before they converge.

**Time-dependent reverse engineering** [pp. 40–41, 119–120]

- **Basis.** A variant of the Ramsden–Godby method [10].
- **Gauge choice.** The method temporarily works in a gauge where the scalar potential is frozen at its ground-state value and all time dependence goes into a vector potential:
  $$v_s'(x,t>0)=v_{ext}(x)+v_{Hxc}(x,0),\qquad A_s'(x,t>0)=\int_{0^+}^{t}dt'\,\partial_x v_{Hxc}(x,t').$$
  This removes a spatial integral and time derivative per iteration and improves stability.
- **Restriction.** The thesis-era iDEA assumed v_ext is constant in time for t > 0 [p. 120].
- **Update at each time step** (Eq. 4.5): $A_s\to A_s+\nu\,(j-j')/n'$, typically with ν = 0.5.
- **Current.** Both currents come from their densities through the continuity equation, so matching j also enforces matching n.
- **Output.** The result is transformed back to the A = 0 gauge, giving a purely scalar v_s(x,t) [p. 41].
- **Algorithm 1** [p. 42] lists the pipeline:
  1. Choose v_ext.
  2. Imaginary-time propagation to the ground state.
  3. Apply the perturbation.
  4. Real-time propagation.
  5. Compute n and j.
  6. Ground-state reverse engineering.
  7. TD reverse engineering.

**Tests** [pp. 42–45]

| Test | Setup | Result |
|---|---|---|
| Energy | Two non-interacting electrons in v = ½ω²x², with E₀ = 2ω = 0.4, so ω = 0.2 | E₀ converges exponentially in imaginary time (Fig. 4.3) [p. 42] |
| TD density | Three non-interacting electrons in ½ω²x² − ε₀x, solved analytically as a shifted oscillator (App. A.3.2) | Agrees with iDEA and obeys the harmonic potential theorem (Fig. 4.4) [p. 43] |
| Current | iDEA's j (from continuity) vs analytic j from wavefunctions (Eq. A.14) | Agree [p. 44] |
| Coulomb | Two interacting spinless electrons in a harmonic well vs Hooke's atom (Taut [111]) | Agree (Fig. 4.6) [p. 44] |
| Reverse engineering | Zero interaction gives v_Hxc = 0. A singlet density gives the analytic KS potential. | Both reproduced [pp. 44–45] |

- **Caveat** [p. 45]: "tiny variations in the density (possibly smaller than our 10⁻¹¹ error) may lead to large features in the KS potential". Results must therefore be converged in grid spacing.
- Numerical error in the TD potential grows with time [p. 45].

### Ch. 5 Local density approximation from finite systems [pp. 46–66]

**Aim** [pp. 46–47]. Build 1D LDAs from *exact finite* systems ("slabs") instead of the infinite HEG, then map where local approximations work and where they fail. Interaction γ = 1 throughout, with spinless electrons.

**Slab construction** [pp. 47–48]

- An "optimisation code" iteratively finds the v_ext whose *interacting* ground-state density equals a target. It uses the same update as Eq. 4.3.
- **Target density:**
  $$n_T(x)=n_0\exp\!\left[-10^{-11}(mx)^{12}\right],$$
  with m scaled so that ∫n = N (N = 1, 2 or 3). Plateau densities go up to ~0.6 a.u.
- The required v_ext is non-trivial. For a 2e slab with n₀ ≈ 0.33, it has deep minima near the edges (≈ −1.5 at |x| ≈ 3.2) and a central dip (≈ −0.7) (≈ read off Fig. 5.1(b)) [p. 48].

**Fitting** [pp. 49–50]

- Reverse-engineering each slab gives the exact E_xc, and hence ε_xc = E_xc/N, which is assigned to the plateau density n₀.
- **Fit form:** $\varepsilon_{xc}=(A+Bn+Cn^2)n^D$.
- **Refinement for inhomogeneous edges:** the LDA is applied to the slabs and corrected by ε_xc → ε_xc − ΔE_xc/N. This reduces ΔE_xc from 2–3% to below 0.5%.
- **The fitted LDAs:**

| LDA | ε_xc(n) | v_xc(n) = ε_xc + n dε_xc/dn |
|---|---|---|
| 1e (Eqs. 5.8–5.9) | (−0.803 + 0.82n − 0.47n²) n^0.638 | (−1.315 + 2.16n − 1.71n²) n^0.638 |
| 2e (Eqs. 5.2, 5.5) | (−0.74 + 0.68n − 0.38n²) n^0.604 | (−1.19 + 1.77n − 1.37n²) n^0.604 |
| 3e (Eqs. 5.3, 5.6) | (−0.77 + 0.79n − 0.48n²) n^0.61 | (−1.24 + 2.1n − 1.7n²) n^0.61 |

**Obs.:** I checked that each v_xc row follows from its ε_xc row through v = [(1+D)A + (2+D)Bn + (3+D)Cn²]n^D, to the quoted precision.

- **1e LDA.** E_xc = −E_H exactly: $\varepsilon_{xc}=E_{xc}=-\frac12\iint n(x)n(x')/(|x-x'|+1)$ (Eq. 5.7 [p. 50]). It is built from pure SIC, with no physical correlation.
- **Comparison** [pp. 50–52]: the three LDAs nearly coincide. They deviate "numerically significantly" only at high density, with a 1e → 2e → 3e progression (Fig. 5.3).
- **1D HEG exact exchange** (Eq. 5.10 [p. 51]), with k_F = πn for spinless electrons:
  $$\varepsilon_x=-\frac{1}{8\pi^2 n}\int_{-\pi n}^{\pi n}dk\int_{-\pi n}^{\pi n}dk'\,u(k-k').$$
  This is "surprisingly close" to all three ε_xc curves (Fig. 5.4).
- **Correlation size.** Using Helbig et al.'s QMC [126] (slightly different softening), ε_c is a few percent of ε_xc except in the low-density limit. The conclusion: in 1D an LDA from small finite systems ≈ an HEG LDA [p. 51].
- **3D test.** A 1e-built 3D LDA is "qualitatively similar" to the QMC LDA [p. 52].

**Physics of the slabs** [pp. 52–54]

- **Correlation strength.** HF densities are accurate for high-density slabs and break down at low density. The correlation energy is < a few % in all slabs, "common to all our 1D test systems". Yet low-density slabs are "systems of relatively strong correlation" in the sense that the density is very sensitive to it [p. 52].
- **Self-consistent LDA on the slabs.** Energies come out within < 1%, but the densities show spurious dips and bumps on the plateau. The exact v_xc is "highly nonlocal" even in the slab centre. Energy errors cancel; the potential is "less forgiving" [p. 53].
- **ELF.** Dobson's exact ELF is computed from the many-body wavefunction for 2e slabs (n₀ ≈ 0.16 and ≈ 0.51; Fig. 5.5).
  - Electrons are extremely localised (ELF near 1) toward the edges, with strong delocalisation at the interface between them.
  - The dip is deeper and wider at high density.
  - In contrast to the HEG (ELF ≡ 0.5), the ELF varies with position and density.
- **Author's interpretation:** the HEG has "incipient Wigner crystallisation" [pp. 53–54].

**Exchange- or SIC-dominated tests** [pp. 54–58]

- **2e triple well** (deep centre, two shallow side wells; v_ext parameters **not given**; Fig. 5.6) [pp. 54–56]:
  - Non-interacting electrons both sit in the centre. Exact and 2e-LDA densities agree remarkably well, and HF performs very well.
  - The exact v_xc has large dips at the density peaks (SIC), which the LDA reproduces. The LDA misses nonlocal features in the low-density interfaces and decays exponentially instead of as −1/x.
  - The v_xc constant is fixed so that v_xc → 0 as |x| → ∞ [p. 54, fn. 2].
  - Exact E = −0.690 and E_xc = −0.467.
  - Table 5.1 (self-consistent): E_LDA = −0.698 / −0.697 / −0.698 (1e/2e/3e) and E_xc = −0.474 / −0.472 / −0.472, i.e. about 1% error.
  - "In the majority of systems" the SIC is described best by the 1e LDA, then 2e, then 3e. The N-electron LDA is not necessarily best for N electrons [p. 56].
- **1e harmonic well** (ω not given; Fig. 5.7) [pp. 56–58]:
  - The LDA density is close to exact, far better than Hartree.
  - Exact v_xc minimum ≈ −0.53 vs LDA ≈ −0.37 (read off fig.).
  - Exact E = 0.127 and E_xc = −0.237.
  - Table 5.2: E_LDA = 0.138 / 0.139 / 0.137 (+8–9%) and E_xc = −0.225 / −0.223 / −0.224 (5–6%).
  - The text says ΔE is what matters, since % errors depend on the v_ext offset.

**Strong correlation** [pp. 58–64]

- **2e harmonic, ω = 0.4** (strongly confined) [pp. 58–60]: HF is almost exact and the LDA is good. The LDA misses the central *bump* in exact v_xc, a "superposition of two steps" that pushes the electrons apart. The E_xc error is ~2% [p. 59].
- **2e harmonic, ω = 0.01** (weakly confined, strongly correlated) [pp. 59–61]:
  - The LDA "completely breaks down", giving three density peaks (slab-like). Its v_xc has a central *dip* where the exact one has a bump, and it greatly underestimates the two side dips (Fig. 5.8).
  - Exact E = 0.068 and E_xc = −0.215.
  - Table 5.3: E_LDA = 0.072 / 0.066 / 0.063 (6 / −3 / −7%) but E_xc = −0.182 / −0.186 / −0.191 (15 / 13 / 11% error).
  - Good total energies despite bad densities, which the thesis attributes to cancellation of errors.
- **Tunnelling, TD** [pp. 61–63]:
  - Two spinless electrons, $v_{ext}=\alpha x^{10}-\beta x^4$ with α = 5×10⁻¹¹ and β = 5×10⁻⁵, plus $v_{pert}=-\varepsilon x$ with ε = 0.01 for t > 0 (Fig. 5.9). Wells at x ≈ ±7.5 (read off fig.).
  - HF shows this ground state is exchange-dominated. The exact v_xc has a central bump (two superposed steps) that the LDA misses.
  - The ALDA, $v^{ALDA}_{xc}[n](x,t)=v^{LDA}_{xc}[n(t)](x)$, gives a density at t = 40 that is good except in the central barrier region.
  - The tunnelling rate (−dn_L/dt, n_L = ∫_{x<0} n) under ALDA is "on average, nearly twice" the exact rate up to t = 40. Non-interacting is worse still (Fig. 5.10).
  - At t = 40: exact n_L ≈ 0.93, ALDA ≈ 0.88, non-interacting ≈ 0.80 (read off fig.).
  - Early-time errors come mainly from the ground state; later errors from growing correlation [p. 63].

**Landauer-style comparison** [pp. 64–65] (Fig. 5.11, same tunnelling system, to t = 160)

- **Exact:** tunnelling reverses because of Coulomb repulsion ("after ~60 a.u." in the caption).
- **ALDA:** reverses, but ~80 a.u.
- **"Exact Landauer"** (exact ground-state KS potential, frozen, plus the field): no reversal in 160 a.u. It does eventually reverse, much later [p. 64, fn. 5].
- **LDA-Landauer:** ≈ non-interacting.
- **Conclusion:** nonlocal *ground-state* features of the exact v_xc matter for early dynamics, and a TD functional must get both ground-state and TD nonlocality right [p. 65].

**Ch. 5 conclusions** [pp. 65–66]. Finite-system LDAs give a good SIC even from one electron. They fail under strong correlation and adiabatic TD use, because nonlocal features are missing.

### Ch. 6 Many-electron tunnelling: exact TDKS [pp. 67–73]

**System** [pp. 67–68]

| Parameter | Value |
|---|---|
| Electrons | 2, spinless |
| Grid | δx = 0.05, δt = 0.002 |
| Interaction | **(|x−x′| + 0.1)⁻¹**, i.e. a wire of diameter 1 a.u. |
| v_ext | αx¹⁰ − βx⁴, with α = 5×10⁻¹¹ and **β = 1.3×10⁻⁴** |
| Perturbation | **−εx, ε = 0.1**, for t > 0 (Fig. 6.1) |

- The stronger field (vs Ch. 5) makes features appear *before* reverse-engineering noise builds up.
- Wells at x ≈ ±10 (read off fig.).

**Dynamics** [pp. 68–69]

- The field starts oscillations in each well, then the left electron tunnels.
- The density in the right well develops standing-wave "ripples" from interference. These are not an interaction effect: they also appear for one electron.
- By t = 5.3, a new density minimum has appeared in the tunnelling region (Fig. 6.1 inset).
- Coulomb repulsion suppresses the current on the right side of the central region. The places where j/n is high are marked (Fig. 6.2, densities ~10⁻⁶).

**Ground-state features** [pp. 70–71]

- v_Hxc = v_s − v_ext shows a highly non-LDA "bump" between the wells. Within each well there are oppositely signed xc electric fields, |ε_xc| ≈ 0.016 a.u. once the SIC is removed [p. 70, fn. 2]. Together these push the KS density peaks apart.
- v_xc is dominated by the SIC, with wells ≈ −1.9 deep at t = 0 (read off Fig. 6.4).

**TD steps** [pp. 71–72]

- A step forms at the central density minimum x_min. It moves right as current builds and changes sign when the current reverses.
- Further steps appear in the right well at the high-j/n points of the interference pattern (Fig. 6.3: v_Hxc at t = 0, 3.5, 5.3).
- **Mechanism:** features of u = j/n go straight into A_s through Eq. 4.5. A narrow peak in u at a density minimum becomes a narrow peak in A_s. The gauge transform to A = 0 integrates its time derivative over space, producing a **step** in v_s.
- A density minimum is *not sufficient*. The density near the minimum must be modified by the interaction; otherwise j = j′ needs no correction.
- Fig. 6.5 shows u(x, 5.3), which is identical for the many-body and KS systems, against v_Hxc(5.3) − v_Hxc(0). Peaks align with steps, and the largest steps are central.
- The same analysis explains the TD steps of Ramsden & Godby [10].
- **Recommendation:** account for u = j/n in TD functionals, since j and n are always available [p. 72].

**Ch. 5 system, exact** [pp. 72–73]

- Setup: ε = 0.01, β = 5×10⁻⁵, and **γ = 1** "for consistency".
- **Accuracy limit:** reverse engineering out to t ≈ 50 has integrated absolute errors ~10⁻² in j and ~10⁻¹ in n.
- **Result:** a dynamic step forms at t ≈ 40 and develops over ~10 a.u. (Fig. 6.6: t = 45, 46.5, 48). It lowers the left well relative to the right and drives the reversal.
- **Conclusion:** "the nonlocal feature absent from the ALDA is a spatial step" [pp. 72–73].

### Ch. 7 Role of electron localisation; the MLP [pp. 74–93]

**Localisation** [pp. 74–76]

- Defined as the tendency of an electron to exclude same-spin electrons from its vicinity.
- It is driven by Pauli exclusion and enhanced by the Coulomb repulsion, and opposed by kinetic energy.
- It is a genuine many-body property, not readable from any single KS orbital [p. 75].

**Measures** [pp. 75–76]

- The ELF (Becke–Edgecombe) is normalised so that ELF = 1 is complete localisation and ELF = 0.5 is the HEG of the same density.
- Dobson's version is computed directly from the wavefunction and is used here as the "exact" ELF.
- **Accuracy** (Durrant et al. [148]): both are good for ground states. ELF from *exact KS orbitals* degrades once the system is strongly perturbed.
- A practical functional would need an ELF that stays reliable under strong correlation.
- The thesis gives **no explicit ELF formula**.

**SOA** (Eq. 7.1 [p. 76])

- **Derivation:** in any region dominated by one KS orbital φ_k with n ≈ |φ_k|², the KS equation $(-\tfrac12\nabla^2+v_s)\sqrt n=\varepsilon_k\sqrt n$ gives
  $$v_{SOA}=\frac{\nabla^2 n}{4n}-\frac{(\nabla n)^2}{8n^2}\qquad(\text{zero of energy at }\varepsilon_k).$$
- **Origin:** the exact singlet two-electron result (Smith et al. [112]; Hessler et al. [82]). It is exact for any one-electron system.
- **Symmetric systems:** delocalised degenerate orbitals can be rotated into localised ones [p. 76, fn. 3].
- **Not usable alone in SCF:** it approximates all of v_s, not just v_xc, so it is "unanchored" to v_ext [p. 77].

**MLP** (Eq. 7.2 [p. 77]): $v_{MLP}=f\,v_{SOA}+g\,v_{ref}$, with f(x), g(x) set by the localisation.

- At ELF = 1: f = 1, g = 0.
- At ELF = 0.5 with v_ref = v_LDA: f = 0, g = 1.
- From the xc-hole sum rule, f + g ≈ 1, so g = 1 − f.
- **With v_ref = v_ext:** $v_{MLP}=v_{ext}+f\,v^{Hxc}_{SOA}$, so f simply scales the SOA's Hxc part [p. 79, fn. 5].

**Test systems** [pp. 77–90] (2 or 3 spinless electrons, γ = 1, "wire diameter 6 a.u."). **f is a hand-chosen constant in every case.**

| System | Setup (as given) | Key result |
|---|---|---|
| 1. Asymmetric double well with a step *in v_ext* [pp. 77–79] | Potential formula not given. Fig. 7.1 spans x ≈ ±35. Non-interacting electrons would both sit in the left well; the exact system has one per well. | The exact v_s has a new xc step at the density minimum, which the SOA (on the exact density) reproduces well. The delicate density features behind the steps (n ~10⁻⁶) are at x = −3 and x = 10. The SOA reproduces the in-well SIC. Self-consistent MLP with v_ref = v_ext: accurate density for f ∈ [0.6, 0.99], with f = 0.6 best, including both steps and the well alignment. It shows a spurious narrow peak where n is tiny. Steps always form at the density minimum [p. 78, fn. 4]. |
| 2. Dissociated molecule [pp. 79–81] | $v_{ext}=-\frac{1}{|x-5|+1}-\frac{4}{|x+5|+2}$. Non-interacting: both electrons on the left atom. | LDA: N_left = 1.25, N_right = 0.75 (Figs. 7.2–7.3). MLP with v_ref = v_ext and f = 0.35 (range 0.35–0.99) matches the exact density, one electron per atom, via the step. Lower f reflects more delocalisation (wells closer). |
| 3. Single well [pp. 81–82] | Two electrons in one well; strong central delocalisation, with an ELF dip to ≈ 0.1 (read off fig.) | The exact v_Hxc has a central bump plus long-range fields, and the raw SOA overestimates both. MLP with f = 0.2–0.3 is accurate in the centre; f = 0.25 ≈ the mean ELF in the central region. With constant f, the long-range fields at the edges are suppressed. |
| 4. Polarised three-atom chain [pp. 82–85] | Three wells, one electron each, plus a ground-state field; formula not given | The exact v_xc partly screens the field through two steps. The SOA predicts *complete* screening (it aligns the lowest eigenvalue of each well). MLP with v_ref = v_ext and f = 0.17 (range 0.17–0.99) gives one electron per site. An f(x) rising to 0.99 at the edges improves the outer-well polarisation. The LDA gives non-integer occupations, with N(x) ≈ 0.75 after the first site (Fig. 7.7, read off). |
| 5 and 5′. 1D atoms [pp. 85–88] | $v_{ext}=-Z/(|x|+\alpha)$. System 5: Z = −1.5, α = 1 (weak correlation, HF accurate). System 5′: Z = −0.02, α = 20 (strong correlation, HF density wrong). | The SOA on the exact density reproduces the *form* of v_Hxc but overestimates the central peak: ≈ 0.8 vs 0.33 for System 5 and ≈ 0.34 vs 0.22 for System 5′ (read off figs). SOA accuracy tracks **localisation, not correlation**: the more strongly correlated atom is more localised and the SOA is *better* there. This argues for a spatially varying f. |
| 6. TD molecule [pp. 88–89] | Symmetric double well, field −0.1x for t ≥ 0 | TD SOA (Eq. 7.3) below. With f = 0.2 (range 0.1–0.3), the MLP reproduces the dynamic step at t = 5 (the latest time, limited by reverse-engineering noise). Protocol: exact ground-state v_s + [v_MLP(t) − v_MLP(0⁺)], with orbitals propagated and v_MLP recomputed every step. The time evolution of the step differs in detail, especially early on. |
| Tunnelling system of Ch. 5 [pp. 89–90] | Fully self-consistent ground state + TD, with v_ref = v_LDA (ALDA) | Ground state: f = 0.5 gives an accurate density and barrier potential (any f in 0–0.99 gives a good ground-state density). The same f is kept for all times, justified because localisation hardly changes. The TD MLP reproduces the exact n_L(t) including reversal starting at t ≈ 40. ALDA reverses only at ~80, and non-interacting never. Noise dominates from t ≈ 55 (Fig. 7.11). |

**TD extension and gauge** [pp. 88–89, 125]

- **TD SOA** (single-orbital TDKS; Eq. 7.3):
  $$v_{SOA}(r,t)=\frac{\nabla^2 n}{4n}-\frac{(\nabla n)^2}{8n^2}-\int_{-\infty}^{r}\frac{\partial u}{\partial t}\cdot dr'-\tfrac12 u^2,\qquad u=j/n.$$
- **Gauge change:** adding $\chi=-\int^r\partial_t u\cdot dr'$ removes the numerically troublesome integral and leaves a vector potential **A_SOA = −u = −j/n** [pp. 89, 125]. The author highlights this "remarkable" simplicity: it is "capable of reproducing the effects of nonlocal, dynamic steps" [p. 125].
- **Numerical issue:** the 1/n in u is delicate at the system edges [p. 89].

**SOA exact properties** [pp. 90–92]

- **SIC with several orbitals:** in a triple well where one electron is central and the other is split 50/50 between the outer wells, the SOA (exact density) reproduces the exact v_xc dip in the outer well (≈ −0.24). The self-consistent LDA overshoots (≈ −0.35) (Fig. 7.12, read off).
- **Asymptotics:** the SOA gives the correct −1/x decay of v_xc far from a single-well two-electron system; the LDA decays too fast (Fig. 7.13).

**Refined MLP** (Eq. 7.4 [p. 92]). Since $n\propto e^{-2\sqrt{2I}|x|}$ asymptotically, v_SOA(|x| → ∞) = I. A spatially varying f therefore needs
$$v_{MLP}=f(x)\,(v_{SOA}-I)+(1-f(x))\,v_{ref},$$
otherwise the constant I enters as an f-weighted spatial shape.

**Ch. 7 conclusions** [pp. 92–93]. The SOA captures steps and the SIC for reasonably localised electrons ("unlike any other functional", in the author's words). The MLP works with the crudest choices (constant f; v_ref = v_ext or LDA). The future work needed is f(x, t) from a localisation measure; see Torelli's MSc thesis [156].

### Ch. 8 The origin of steps in the KS potential [pp. 94–116]

System labels restart in this chapter; I write them as Ch8-S1, Ch8-S2 and so on. All runs use spinless electrons and γ = 1 [p. 95]. Explicit potential formulas are **not given** for any Ch. 8 system, only figures.

**Almbladh–von Barth (AvB) thought experiment** [pp. 95–97]

- Two **spin-half** electrons in two separated open-shell "atoms", with the right well lower. Non-interacting electrons would both go right; the interacting system has one per well.
- **Step definition:** $S_{xc}\equiv v^{xc}_R-v^{xc}_L=v^s_R-v^s_L$, the plateau levels of v_s in each subsystem's region. This is exact at infinite separation but "holds well for … a few ångströms" [p. 96].
- **Result:** the reverse-engineered step equals I_R − I_L and aligns the subsystem ground-state energies (Fig. 8.1). The step also has a sharp peak at its foot.
- **Additive constant:** reverse-engineered potentials are fixed only up to a constant [p. 96, fn. 2].
- **Second derivation:**
  - Far from an atom, $n\propto e^{-2\sqrt{2I}x}$, and for one orbital the SOA is exact.
  - At the density minimum the decay rate switches from the left atom's to the right atom's, so the SOA potential jumps from I_L to I_R over a short distance.
  - This is the "**change in the local effective ionisation energy**" [p. 97].

**General step height** (Eq. 8.1 [pp. 97–98])

- **Many-body decay:** $n'_L\propto e^{-2\sqrt{2I_L}x}$ and $n'_R\propto e^{+2\sqrt{2I_R}x}$.
- **KS decay:** $n_L\propto e^{-2\sqrt{2(v^s_L-\varepsilon_L)}x}$, where ε_L is the highest occupied KS orbital dominating the left subsystem's tail.
- Matching n = n′ gives $v^s_{L,R}=I_{L,R}+\varepsilon_{L,R}$, so
  $$S_{xc}=(I_R-I_L)+(\varepsilon_R-\varepsilon_L)=S^I_{xc}+S^\varepsilon_{xc}.$$
- Exact at infinite separation and "accurate for separations of a few ångströms".
- Not predictive on its own, because the ε are partly set by the step.
- **Spin-half with odd electron count per site:** the HOMO spans both sites, so ε_R = ε_L and S_xc = I_R − I_L (AvB).
- **Not the derivative discontinuity** [p. 98]: the step depends only on how the density decays from each subsystem. It has "no direct association" with subsystem or system electron affinities, and in the AvB system the insensitivity to affinity is complete.

**Allowed range** (Eq. 8.2 [pp. 99–101])

- Setup: well-separated wells, M KS states filled on the left and N on the right, with $v^s_R=0$.
- **Window:**
  $$\varepsilon^L_M(M)-\varepsilon^R_{N+1}(N)<S_{xc}<\varepsilon^L_{M+1}(M)-\varepsilon^R_N(N).$$
  Any step in this window gives a "fairly accurate" density (Fig. 8.2, M = N = 1).
- Also holds for spin-half electrons, except with an odd number per site, where S = I_R − I_L exactly.
- **Ch8-S1** (two spinless electrons; the right well's first excited state lies below the left ground state): artificially changing the step height within the window changes the density only slightly near the minimum (ln n ≈ −21.6 to −21.7; Fig. 8.3).
- **Conclusion:** "the step height ultimately determines the degree to which the left-hand electron occupies the right well and vice versa". Local and semilocal approximations would therefore need extreme sensitivity to the density at the step, or full nonlocality [p. 101].

**Step position** [pp. 101–105]

- **Second step far out:** far from a molecule the density must decay with the *whole* molecule's ionisation energy, the lowest I among the subsystems. A subsystem whose tail decays faster forces a second change of local I, and so a second step far from the molecule. This was seen by Perdew [49] and by Makmal et al. in EXX for LiF [164] [p. 101].
- **Local effective ionisation energy** [p. 102]:
  $$\tilde I(x)=\frac{1}{8n^2}\left(\frac{\partial n}{\partial x}\right)^2.$$
  It is the second term of v_SOA, and equals I wherever n decays as $n_0e^{-2\sqrt{2I}x}$.
- **Ch8-S2** (two-electron molecule, Fig. 8.4):
  - Two steps: one at the density minimum and one far out at x ≈ +18 (read off fig.).
  - Both lie where Ĩ(x) changes *and* where the KS orbital densities |φ₁|², |φ₂|² cross. Far out, the slower-decaying excited orbital must overtake the ground-state orbital.
  - The two contributions superpose. A change of dominant KS density implies a change in local I, but not vice versa: AvB has one orbital, hence no crossing.
- **Della Sala–Görling:** consistent with their result that v_xc tends to a non-zero constant along a HOMO nodal plane, since no crossing occurs there [p. 103].
- **Orbital dependence:** the author links the role of orbitals here to meta-GGAs, the ELF and the MLP [p. 103].
- **Time-dependent (Ch8-S3)** [pp. 103–105]:
  - Asymmetric double well, perturbation +0.1|x| for t ≥ 0, which pushes the electrons together.
  - At t = 4.6 the KS orbital densities have not crossed and there is no step. By t = 5 they have crossed and a step forms there (Fig. 8.5).
  - The TD step also coincides with a velocity-field peak. In Ch. 6, density minima *without* steps were confirmed to lack orbital-density crossings.
  - Interference ripples make crossings likely at dynamic minima. Step magnitudes in TD are not given by Eq. 8.1, except approximately in the adiabatic limit.

**Density minima and the integer electron point (IEP)** [pp. 105–107]

- **Which minima can carry steps:** a minimum *inside* a one-electron subsystem cannot carry a step, since there is only one occupied level and so no change of local I.
- **IEP definition:** the boundary at which $\int_a^b n\,dx=N\in\mathbb Z$.
- **Rule:** density minima that are also IEPs indicate step positions in ground states.
- **Interaction aligns IEP and minimum** (Ch8-S4, crafted potential, Fig. 8.6):
  - Non-interacting: IEP ≈ −2.8 vs minimum ≈ 0.13.
  - Interacting: IEP ≈ −0.39 vs minimum ≈ −0.26.
  - Argument: stronger repulsion drives the density at the IEP toward 0, which must then be a minimum.
- **Counter-example** (Ch8-S5, crafted to delocalise): IEP ≈ −1.2 vs minimum ≈ 0.24. The main v_Hxc feature sits at the IEP and is a peak, not a step, because local I is ill-defined on either side (Fig. 8.7).
- **Time dependence:** in TD the IEP is *not* a good indicator, because orbitals can cross anywhere [p. 106].

**Sharpness** [pp. 107–109]

- A more abrupt switch of dominant orbital or local I (i.e. more localisation) gives a sharper step. This agrees with Tempel et al. [167].
- **Ch8-S6 vs S6′** (wells moved closer): the step becomes less sharp. In S6′ a second, very diffuse counter-step at x ≈ 11 cancels the net step (Fig. 8.8).
- The MLP reproduces these steps for a range of f [pp. 108–109].
- **HF comparison:** reverse-engineering the HF density gives an "HF-KS" potential. Its step is almost perfect when the KS HOMO and LUMO are well separated, and worse when they are close (stronger correlation). Exchange and correlation both shape steps [p. 109].

**Bumps as superposed steps** [pp. 109–114]

- **Linear-response argument:** if systems A and B are similar enough that δv_s/δv_ext is about the same for both (Eq. 8.3), then
  $$v_{ext}^C=\tfrac12(v^A_{ext}+v^B_{ext})\;\Rightarrow\;v_s^C\approx\tfrac12(v^A_s+v^B_s)\quad\text{(Eq. 8.4)}.$$
- **Ground state (Ch8-S7A/B/C):** A is an asymmetric double well, B its mirror image, C the symmetric average, with minima aligned at x = 0.
  - The exact v_xc^C matches ½(v_xc^A + v_xc^B) very closely, including the central bump and the SIC wells (Fig. 8.9).
  - A symmetric system cannot have a net step, so a bump forms instead, pushing the electrons apart.
  - Unequal steps superpose to a step-plus-peak (as in Fig. 8.1(c)).
- **Time-dependent (Ch8-S8A/B/C):** at t = 0 the left well of A is deepened (B mirrored). The dynamic parts of v_Hxc superpose: a dip at t = 1.25 becomes a bump at t = 2.5 (Fig. 8.10).

**Step theory applied to approximate functionals** [pp. 114–115]

- **LDA** of the form (α + βn + γn²)n^p goes to 0 in exponential tails and is insensitive to I, so it cannot form steps.
- **"1D PBE"**, with $s_{1D}=|\partial_x n|/n^2$ (dimensionless in 1D): in the tails, $F_x=1+\frac{8aIe^{4\sqrt{2I}x}}{1+8bIe^{4\sqrt{2I}x}}\to1+a/b$, independent of I. A step is therefore "very unlikely". The author notes this is not a proof; one would have to run PBE on step-forming systems.
- **SOA** (Eq. 8.5): on $n\sim e^{-2\sqrt{2I}x}$, the two terms give 2I − I = I. So S_SOA = I_R − I_L, at the right position.
- **What the SOA lacks** (§8.9 [p. 115]):
  - It has no knowledge of KS orbital-density crossings (the ε_R − ε_L part). It therefore gets step and bump heights wrong when several orbitals matter, which explains its overestimated bumps in Figs. 7.8–7.9.
  - A correcting term needs the KS eigenvalues, so the thesis calls it impractical.

**Ch. 8 conclusions** [pp. 115–116]:

- Steps arise where decaying densities from different subsystems meet (a change in local I), and where the dominant KS orbital density changes.
- These superpose, and in symmetric systems they form bumps; the same holds in TD.
- Step positions are indicated by density minima at IEPs, and an allowed window bounds the heights.
- Delocalisation softens steps.

### Ch. 9 Conclusions [pp. 117–118]

- Restates the iDEA approach, the SOA and the MLP.
- **What is still needed:**
  - the relationship between f and the localisation;
  - an accurate localisation measure computable from quantities available in a DFT calculation;
  - new reference potentials;
  - dynamic-step magnitudes from first principles.
- A constant f(x, t) worked only "for our small systems, simulated for relatively short times" [p. 118].

### Appendix A: iDEA [pp. 119–124]

- **A.1:** the p-speed-up and gauge details (covered above).
- **A.2 Interaction** [pp. 120–121]:
  - Model: two charged disks of radius a in a 3D wire, each with transverse charge profile $n(r)=\frac{2\pi}{a^2(\pi^2-4)}\cos^2\frac{\pi r}{2a}$.
  - Disk–disk energy:
    $$v(x)=\frac{8\pi^3}{(\pi^2-4)^2a^4}\int_0^{2\pi}\!d\theta\int_0^a\!dr_1\int_0^a\!dr_2\,\frac{r_1r_2\cos^2(\pi r_1/2a)\cos^2(\pi r_2/2a)}{\sqrt{r_1^2+r_2^2-2r_1r_2\cos\theta+x^2}}\quad(\text{A.1}).$$
  - For width 6 a.u. ("≈ a gold nano-lead"), 1/(|x|+1) fits v(x) better than 1/√(x²+1) or 1/|x| (Fig. A.2).
  - 1/(|x|+0.1) corresponds to a 1 a.u. diameter and is used only in Ch. 6.
- **A.3:** the ground-state energy is read off the norm decay in imaginary time [pp. 121–122]. Closed-form coefficients are given for three non-interacting electrons in a shifted harmonic well (Eqs. A.7–A.13, $\alpha=\varepsilon_0/2\sqrt{\omega^3}$), together with the current formula (A.14) [pp. 122–124].

### Appendix B: Density functional development [p. 125]

- **B.1:** any KS equation can be rearranged into an expression for v_s in terms of one orbital. This is a guide only, since orbitals are unknown. (See §6 for an apparent typo in the printed formula.)
- **B.2:** the TD-SOA gauge change giving A_SOA = −j/n, recommended as a template for TD functionals.

### Bibliography [pp. 126–137]

167 references. Worth following up:

- In-group: [10] Ramsden & Godby PRL 2012 (TD steps); [156] Torelli MSc 2016 (MLP f).
- Steps: [11] Elliott et al. PRL 2012 (dynamical steps); [151] Almbladh & von Barth; [157] van Leeuwen et al. (shell steps); [158] Helbig et al. 2009; [164] Makmal et al. (LiF EXX).
- Other: [126] Helbig et al. 2011 (1D QMC LDA); [135] Dobson (ELF).

---

## 3. Glossary of concepts introduced or used

| Term | Meaning (thesis) | Ref. |
|---|---|---|
| Spinless electrons | Same-spin electrons; each occupies its own KS orbital, giving maximal exchange and correlation per electron | [p. 39] |
| Softened Coulomb | (\|x−x′\| + γ)⁻¹. γ = 1 ↔ a 6 a.u. wire (default); γ = 0.1 ↔ 1 a.u. (Ch. 6 only) | [pp. 39, 120–121] |
| Slab system | Finite system with a flat density plateau n₀; target n₀exp[−10⁻¹¹(mx)¹²] | [p. 47] |
| 1e/2e/3e LDA | LDA fitted to exact slab ε_xc(n₀) for N = 1, 2, 3, then refined | [pp. 49–50] |
| Exact Landauer | Frozen exact ground-state KS potential plus the field, with no TD update | [pp. 64, 66] |
| v_Hxc | v_s − v_ext; shows remaining nonlocal features once the SIC is cancelled | [p. 70] |
| Velocity field u | j/n; its peaks at density minima produce dynamic steps through A_s → gauge → v_s | [pp. 71–72] |
| Step | A jump in the level of v_xc over a short distance | [p. 94] |
| Bump | A symmetric peak equal to superposed opposite steps | [pp. 109–111] |
| Electron localisation | Tendency to exclude same-spin electrons nearby; a many-body property | [pp. 74–75] |
| ELF | 0–1 localisation index; 0.5 for the HEG. Dobson (exact, from Ψ) vs Becke–Edgecombe (from orbitals). | [pp. 53, 75] |
| SOA | v = ∇²n/4n − (∇n)²/8n². Exact for one orbital; correct steps I_R − I_L and −1/x asymptotics | [pp. 76, 90–92, 114–115] |
| MLP | f v_SOA + (1−f) v_ref, with f set by localisation; refined to f(v_SOA − I) + (1−f)v_ref | [pp. 77, 92] |
| A_SOA | −j/n: TD-SOA vector potential in a convenient gauge | [pp. 89, 125] |
| Local effective ionisation energy | I implied by the local density decay; $\tilde I(x)=(\partial_xn)^2/(8n^2)$ | [pp. 97, 102] |
| S^I_xc, S^ε_xc | Step parts from the many-body decay change (I_R − I_L) and the KS dominant-orbital change (ε_R − ε_L) | [p. 98] |
| Allowed step window | Eq. 8.2 bounds on S_xc that preserve correct well occupations | [p. 99] |
| Integer electron point (IEP) | Boundary at which the enclosed charge is an integer; steps form at density minima that are IEPs (ground state) | [pp. 105–107] |
| HF-KS potential | Local potential reverse-engineered from the HF density | [p. 109] |
| Er. 1–4 | Burke et al.'s TDDFT error sources: ground state, nonlocality, memory, observables | [pp. 35–36] |

---

## 4. Relevance to learning density functionals with ML

These are my inferences from the thesis's findings, with the supporting pages cited. They are not claims made by the thesis.

**Structure a learned map must capture**

1. **Strong spatial nonlocality in n → v_xc.**
   - Steps depend on the decay rates of densities from *other* subsystems (Eq. 8.1 [p. 98]) and on how far the molecule extends (the second far-field step [pp. 101–103]).
   - In a polarised chain, field screening needs "ultra-nonlocal" dependence [p. 82].
   - LDA and PBE forms provably (LDA) or very likely (PBE) cannot produce steps [p. 114].
   - So the model's receptive field must cover the whole system. Global attention is a natural fit, local convolutions are not, unless the inputs are already nonlocal.
2. **The signal lives in exponentially small densities.**
   - Steps are set by n ≈ 10⁻⁶ (Fig. 7.1 inset [p. 78]) or ln n ≈ −21 (Figs. 8.1, 8.3 [pp. 96, 101]); far-field steps appear at ln n ≈ −50 to −75 (Fig. 8.4 [p. 102]).
   - Tiny density variations (≲ 10⁻¹¹) can produce large potential features [p. 45].
   - Inputs should therefore include **log-density and its derivatives**. A loss on n alone is blind in the tails, while a loss on v is ill-conditioned. Mind float32 underflow and precision: e⁻⁷⁵ ≈ 3×10⁻³³.
   - The reverse-engineering update with p ≈ 0.05 (Eq. 4.3) is itself a "low-density-emphasising" metric. A loss on n^p or ln n is the analogous choice for training.
3. **Physically motivated input features.** The thesis supplies cheap nonlocal descriptors:
   - **Ĩ(x) = (∂n)²/(8n²)**, the local effective ionisation energy [p. 102];
   - **v_SOA = ∇²n/(4n) − (∇n)²/(8n²)** [p. 76];
   - the **cumulative charge N(x) = ∫_{−∞}^x n**, whose integer crossings (IEPs) mark step positions [pp. 80, 105];
   - for TD, the velocity field **u = j/n** [p. 71] or **A_SOA = −j/n** [p. 125].
   - A residual design, v_s = v_SOA-based baseline + learned correction, or *learning f(x) in the MLP* is a natural, low-dimensional target. The thesis explicitly leaves f(x, t) open [pp. 92, 117–118].
4. **Localisation as a latent variable.**
   - SOA accuracy tracks localisation, not correlation strength [p. 88].
   - The exact (Dobson) ELF needs Ψ; orbital-based ELF degrades under strong perturbation [p. 75].
   - iDEA can supply the exact ELF as an auxiliary training target or diagnostic. A pure density model would have to infer localisation implicitly.
5. **Orbital information matters.**
   - The KS-side part of a step, ε_R − ε_L, comes from crossings of KS orbital densities, which the SOA cannot see [pp. 103, 115].
   - So pure n → v maps may need to implicitly reconstruct orbital structure, or the model could take KS orbitals or eigenvalues as inputs (meta-GGA-like) [p. 103].
6. **Degeneracy and tolerance in the n ↔ v map.**
   - The density is insensitive to the step height inside the window of Eq. 8.2 [pp. 99–101]. A v-space loss may therefore penalise errors that barely change n, while missing a step entirely changes the occupations completely.
   - Evaluate learned potentials by **self-consistent densities**, as the thesis does throughout (e.g. integrated well occupations, Figs. 7.3, 7.7).
7. **Offsets, gauge and asymptotics.**
   - Reverse-engineered potentials are defined only up to a constant [p. 96, fn. 2]; the thesis fixes v_xc → 0 at |x| → ∞ [p. 54, fn. 2]. TD data also need a fixed gauge [pp. 41, 119–120].
   - The exact v_xc decays as −1/x, while the LDA decays exponentially [pp. 22, 91]. Hard-wall boxes can contaminate the tails.
   - The tail n ∝ e^{−2√(2I)|x|} encodes I = −ε_HOMO [pp. 92, 97].
8. **Energy vs potential.**
   - LDA total energies are within ~1% (slabs, triple well), or a few % even when densities are qualitatively wrong (ω = 0.01: E within 3–7%, but E_xc off 11–15%) [pp. 53, 56, 61]. Energy-only training will not constrain v or n.
   - In 1D, exchange and SIC dominate E_xc (ε_c is a few %; the 1e, 2e and 3e LDAs nearly coincide) [pp. 50–52]. Yet small energetic correlation can change densities strongly [pp. 52, 59–60]. The learning signal is mostly in v_xc's nonlocal structure.
9. **Time dependence.**
   - Dynamic steps build over ~10 a.u. at u = j/n peaks and orbital-density crossings, and they control tunnelling reversal [pp. 71–73, 103–105]. Adiabatic local maps get the timing qualitatively wrong (~80 vs ~40–60 a.u.) [pp. 65, 90].
   - A TD model needs j (or u) and memory. Early-time TD error is dominated by the ground-state functional [pp. 63–65].
10. **Symmetry and linearity.**
    - The approximate superposition v_s^C ≈ ½(v_s^A + v_s^B) for linear-response-related systems holds closely in the examples (Eqs. 8.3–8.4 [pp. 109–112]).
    - This suggests mirror-pair augmentation and a consistency test for a learned functional. Symmetric systems show bumps rather than steps.
11. **Pitfalls in the reference data.**
    - TD reverse-engineering noise grows with time: ~10⁻² (j) and ~10⁻¹ (n) integrated error by t ~ 50 in the ε = 0.01 run [p. 72]. Late-time TD targets are unreliable unless regenerated with better numerics.
    - Ground-state targets must be grid-converged [p. 45].
    - Delocalised systems give peaks rather than clean steps (Ch8-S5, S6′) [pp. 107–108].

---

## 5. Reproducible with iDEA

**Common settings** (thesis-era):

- 1D grid with δx ≤ 0.1 [p. 42] (Ch. 6 used δx = 0.05, δt = 0.002 [p. 67]).
- Spinless electrons; in current iDEA these correspond to same-spin strings such as `"uu"` or `"uuu"`.
- Softened Coulomb with γ = 1 unless stated.
- Ground-state reverse engineering with p ≈ 0.05, μ = 1, to f_n < 10⁻¹¹ [p. 40]. Current iDEA 1.1.0 uses p = 0.1 (per `notes/idea_overview.md`).
- v_xc offset: → 0 at |x| → ∞.
- **Box lengths are never stated**; the plot ranges below are only guides.

**Differences from current iDEA** (Obs., per `notes/idea_overview.md`): the TD reverse engineering differs (thesis: vector potential fitted to current; iDEA 1.1.0: per-step root solve for the scalar potential). iDEA 1.1.0's built-in LDA is an HEG parametrisation, not the thesis's slab LDAs.

| Calculation | Parameters available | Missing / to reconstruct |
|---|---|---|
| Harmonic energy test [p. 42] | Two non-interacting electrons, ½ω²x², ω = 0.2 (E₀ = 0.4) | Grid and box |
| Shifted-oscillator TD test [pp. 43, 122–124] | Three non-interacting electrons, ½ω²x² − ε₀x, analytic c_{k,n} | ω and ε₀ not stated |
| Hooke's atom test [p. 44] | Two interacting spinless electrons in a harmonic well vs Taut [111] | ω and interaction form for the analytic comparison not stated |
| Slab LDAs [pp. 47–50] | Target n₀exp[−10⁻¹¹(mx)¹²], N = 1–3, n₀ ≤ 0.6; fit form and final coefficients (table in Ch. 5) | Needs an *interacting* inverse (v_ext from target n). The n₀ grid and refinement details are not given. The final LDAs can simply be implemented from Eqs. 5.2–5.9. |
| 1D HEG exchange [p. 51] | Eq. 5.10 with u = FT of the softened Coulomb, k_F = πn | Fit form for ε_x not given |
| 2e triple well [pp. 54–56] | Exact E = −0.690, E_xc = −0.467; Table 5.1 | v_ext not given (read shape from Fig. 5.6) |
| 1e harmonic [pp. 56–58] | Exact E = 0.127, E_xc = −0.237; Table 5.2 | ω not given |
| 2e harmonic [pp. 58–61] | ω = 0.4 and ω = 0.01; for ω = 0.01, exact E = 0.068, E_xc = −0.215, Table 5.3 | Box (Fig. 5.8 spans x ≈ ±40 for ω = 0.01) |
| Tunnelling, Ch. 5 [pp. 61–65] | Two spinless electrons, γ = 1, v = 5×10⁻¹¹x¹⁰ − 5×10⁻⁵x⁴, field −0.01x for t > 0, run to t = 160. Observable n_L(t) = ∫_{x<0}n. | Grid and box (Fig. 5.9 spans ≈ ±12) |
| Tunnelling, Ch. 6 [pp. 67–73] | Two spinless electrons, **γ = 0.1**, v = 5×10⁻¹¹x¹⁰ − 1.3×10⁻⁴x⁴, field −0.1x, δx = 0.05, δt = 0.002; snapshots t = 3.5, 5.3 | Box (Fig. 6.1 spans ≈ ±12) |
| Ch7 S2 dissociated molecule [p. 79] | v = −1/(\|x−5\|+1) − 4/(\|x+5\|+2), two spinless electrons. Targets: LDA N_L = 1.25; MLP f = 0.35, v_ref = v_ext. | Box (Fig. 7.2 spans ±15) |
| Ch7 S5 and S5′ atoms [pp. 85–87] | v = −Z/(\|x\|+α); (Z, α) = (−1.5, 1) and (−0.02, 20); two spinless electrons | Sign/scale inconsistent with the figures (see §6) |
| Ch7 S6 TD molecule [p. 88] | Symmetric double well, field −0.1x, t ≤ 5, f = 0.2 | Well shape not given |
| TD-MLP on tunnelling [pp. 89–90] | v_ref = ALDA (2e LDA), f = 0.5, A_SOA = −j/n gauge | Implementation details of the MLP SCF and propagation |
| Ch7 S1, S3, S4; all Ch. 8 systems | Qualitative designs: double wells with steps or offset depths, single well, three-well polarised chain, crafted IEP potentials, mirror-pair superpositions. Ch8-S3 uses perturbation +0.1\|x\|. | No formulas; reconstruct from Figs. 7.1, 7.4, 7.5, 8.1–8.10 |
| Analyses reusable on any iDEA dataset | Ĩ(x), v_SOA, N(x), IEPs, KS orbital-density crossings, u = j/n; superposition test (Eq. 8.4); step-height window (Eq. 8.2); HF-KS potential by reverse-engineering the HF density | — |

---

## 6. Open questions, limitations, and possible inconsistencies

**Limitations and open questions stated by the author**

- Exact treatment is limited to ≤ 3 electrons [p. 38].
- TD reverse engineering accumulates noise, which limits times to ~5 a.u. for some systems and ~50 a.u. for others [pp. 68–69, 72, 89–90].
- f is a hand-tuned constant. The functional relation f ↔ localisation, a DFT-computable localisation measure, and the time dependence of f are all open [pp. 79–93, 117–118].
- The SOA cannot be used alone self-consistently [p. 77].
- The SOA misses the ε_R − ε_L part of steps, and the corrective term needs KS eigenvalues [p. 115].
- TD step magnitudes are not given by Eq. 8.1 [p. 105]. Determining dynamic-step magnitudes from first principles is open [p. 118].
- The PBE no-step argument is asymptotic and not a proof [p. 114].
- 1D forms of some exact constraints (e.g. Lieb–Oxford) differ from 3D [pp. 22–23].

**Possible inconsistencies or unclear points (Obs.)**

1. **Sign of the KS-side step term.** Eq. 8.1 gives S_xc = (I_R − I_L) + (ε_R − ε_L) [p. 98]. §8.9 says that adding a correcting step of magnitude ε_R − ε_L to the SOA's I_R − I_L makes the height exact as "I_R − I_L − (ε_R − ε_L)" [p. 115]. These disagree in sign. My own re-derivation from the stated decay forms reproduces Eq. 8.1.
2. **Appendix B.1 formula as printed:** $v_s=-\frac12\phi_i^{-1/2}\nabla^2\phi_i^{1/2}$ [p. 125]. The KS equation gives $v_s=\varepsilon_i+\frac{1}{2}\phi_i^{-1}\nabla^2\phi_i$. Equivalently, for the orbital density n_i = |φ_i|², $v_s=\varepsilon_i+\frac12 n_i^{-1/2}\nabla^2 n_i^{1/2}$, which equals v_SOA + ε_i (I checked this algebra). The printed version appears to have the wrong sign, to omit ε_i, and probably to write φ where n_i is meant.
3. **Ch7 Systems 5 and 5′:** v_ext = −Z/(|x|+α) with Z = −1.5 would be *repulsive* [p. 85]. Z = −0.02, α = 20 gives |v| ≲ 10⁻³, inconsistent with the ~0.4–0.7 a.u. confining shapes plotted in Figs. 7.8–7.9 [pp. 86–87]. Either the sign convention or the values (or the plotted scaling) are off; the parameters cannot be used verbatim.
4. **Crank–Nicolson "stability" condition** δt ≲ δx^N/N [p. 39]:
   - CN is unconditionally stable for a linear Hamiltonian, so this is presumably an empirical accuracy or implementation constraint.
   - Ch. 6 uses δx = 0.05 and δt = 0.002 for N = 2 [p. 67], whereas the rule gives δt ≲ 0.00125.
5. **Tunnelling reversal time.** Fig. 5.11's caption says the electron is forced back "after ~60 a.u." [p. 65]. §7.7.7 and the Fig. 7.11 caption say the reversal begins at t ~40 [p. 90], and Ch. 6 says the step forms at t ~40 [p. 72]. These are probably onset vs minimum of n_L, but this is not stated.
6. **Interaction change not mentioned.** Ch. 6 calls its system "the same as the tunnelling system in Chapter 5 except" for the field and barrier [p. 68], but the interaction also differs (γ = 0.1 vs 1) [pp. 67, 121].
7. **Harmonic tests:**
   - The density test formula is printed as $n(x)=\sqrt{\pi/\omega}\,(2x^2+1)e^{-\omega x^2}$ [p. 43]. For two spinless fermions in ½ω²x² the result is $\sqrt{\omega/\pi}\,(1+2\omega x^2)e^{-\omega x^2}$; the prefactor appears inverted and an ω is missing.
   - The "Hooke's atom" check cites Taut [111], which treats 3D Coulomb electrons, yet iDEA uses a 1D softened interaction [p. 44]. How the analytic 1D reference was obtained (interaction, ω) is not explained.
8. **Conductance quantum** [p. 33]: the text says conductance rises "in discrete steps of 4π" and refers to G₀ "defined above" without defining it. In a.u. G₀ = 2e²/h = 1/π.
9. **Minor:** "Table 4.2" is referenced but labelled 4.1 [p. 45]. Imaginary-time propagation is cited to Metropolis & Ulam [p. 38, ref. 106]. "System" numbering restarts between Ch. 7 and Ch. 8.
10. **Missing reproducibility data:**
    - Box lengths, most grids/time steps, and most external potentials (triple well, Ch7 S1/S3/S4/S6, all Ch. 8 systems) are not given.
    - Slab fitting data and the n₀ list are not given.
    - The ELF formula used is not written out.
    - The MLP self-consistency details (mixing, convergence) are only described as "standard potential-mixing".
11. **Strong claims to treat with care:**
    - The SOA reproduces steps "unlike any other functional" [p. 93]. Orbital-dependent potentials such as exact exchange do show steps, as the thesis itself cites for LiF [164].
    - The HEG "incipient Wigner crystallisation" interpretation [p. 54] is suggested by the similarity of the curves, not demonstrated.
    - f + g ≈ 1 is justified only by a brief sum-rule argument [p. 77].
    - The "derivative discontinuity is not the origin of steps" argument [p. 98] concerns ground-state steps between separated subsystems at fixed integer N.
12. **Spin-half calculations.** Fig. 8.1 uses *spin-half* electrons [p. 96], although the code is described as spinless. Presumably it is a singlet in one doubly occupied KS orbital, but this is not described.
