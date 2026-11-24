# Ramsden (2013): Properties of exact density functionals for electronic quantum transport

Study notes, written for the density_model project. Page references `[p. N]` are **PDF page indices** (the number you pass to a PDF reader's `pages`). For the main matter, PDF page = printed page + 17 (for example, printed p. 1 is PDF p. 18). Equation numbers are the thesis's own. Where a number was read off a plot and not stated in the text, it says "read off figure". Comments marked **[obs]** are my own observations, not claims made by the thesis.

---

## 1. Metadata and overview

- **Title:** *Properties of exact density functionals for electronic quantum transport*
- **Author:** James David Ramsden
- **Degree, institution, date:** PhD, Department of Physics, University of York, May 2013 [p. 1]
- **Supervisor:** Rex Godby [p. 16]. Others acknowledged: H. Ness, L. Dash, P. Bokes, P. Hasnip, R. van Leeuwen, G. Vignale and M. Hodgson [p. 16].
- **Length:** 201 PDF pages. Chapter 1 starts on p. 18, Ch. 2 on p. 48, Ch. 3 on p. 79, Ch. 4 on p. 106, Ch. 5 on p. 126, Ch. 6 on p. 161, Appendix A on p. 169 and the references on p. 187.
- **Associated publications** [p. 17]:
  - J. D. Ramsden and R. W. Godby, PRL **109**, 036402 (2012), "Exact density-functional potentials for time-dependent quasiparticles". This is Ch. 5; the thesis says "a modified version of this work appeared" there [p. 128].
  - J. D. Ramsden and R. W. Godby, PRB **88**, 195115 (2013), "Intrinsic exchange-correlation magnetic fields in exact current density functional theory for degenerate systems". This is Ch. 4.
  - Closely related: Hodgson, Ramsden, Chapman, Lillystone and Godby, PRB **88**, 241102(R) (2013) [ref. 141]. That paper used exact continuum solutions for 2–3 electrons together with the time-dependent reverse-engineering algorithm from this thesis [p. 167]. It is the paper that leads to iDEA.
- **The thesis never mentions "iDEA" by name.** None of its "interacting" data comes from a many-body wavefunction (see below).

**Overview.** The thesis asks what exact Kohn–Sham (KS) potentials look like for current-carrying systems, both steady-state and time-dependent. It also builds the tools needed to compute them. It has three parts:

1. **A formal part on ground-state current-DFT (CDFT)** (Chs 2–3, Appendix). The thesis argues that neither paramagnetic CDFT (Vignale–Rasolt) nor physical-current CDFT (Pan–Sahni, Diener) has a valid Hohenberg–Kohn (HK) theorem. It then proves uniqueness for the basic variables $(n,\mathbf j_p,\mathbf j_d)$ (and for $(n,\mathbf j_d)$ in the Appendix). It gives a KS scheme in which the KS system reproduces the *physical* $n$ and $\mathbf j$.
2. **Reverse-engineering (RE) algorithms** (Ch. 3). They produce exact KS scalar and vector potentials from target $n$ and $\mathbf j$:
   - steady state: iterate $\mathbf A_{KS}\leftarrow\mathbf A_{KS}+(\mathbf j-\mathbf j_{KS})/n$, with van Leeuwen–Baerends (VLB) updates for $v$;
   - time-dependent: the same correction applied one time step at a time;
   - an "electric-field" variant for 1D.
3. **Applications to model "interacting" systems** (Chs 4–5). These systems are *quasiparticle (QP)* models: one extra electron, governed by a **model nonlocal self-energy** $\Sigma(x,x')$, added to a periodic model-silicon nanowire. They are not exact many-body solutions.

**Main conclusions:**
- (i) A 1D steady-state current-carrying system generally *cannot* be reproduced by a KS system, because a 1D vector potential is pure gauge. In 3D the exact KS system needs an **XC magnetic field** even when there is no external magnetic field. That field can induce an XC Stern–Gerlach coupling [pp. 113, 120–124].
- (ii) The exact time-dependent KS scalar potential for a propagating QP wavepacket contains a **time-dependent potential step across the wavepacket**. The step is spatially *ultranonlocal* in its dependence on the densities, so no local or adiabatic functional can produce it [pp. 139–141].
- (iii) The corresponding **KS electric field** is localised. Within one history it is close to a single-valued, near-planar function of $(\partial_x n)/n$ and $(\partial_t j)/n$ [p. 145]. Future functionals should therefore target XC *fields* (E and B), not potentials [pp. 158, 166–167].
- (iv) QP decay (non-Hermitian $\Sigma$) needs very large, short-lived KS potentials. The memory of the initial current distribution is short [pp. 154–159].

---

## 2. Chapter-by-chapter summary

### Front matter [pp. 1–17]

The abstract [p. 2] summarises the work: the exact charge and current densities of model transport devices with nonlocal interactions, RE of exact TDDFT and steady-state KS potentials, and the proposal that improved functionals should "describ[e] not potentials but electric and magnetic fields, and have a strong dependence on the local and semilocal charge and current density".

### Chapter 1: Introduction (background on quantum transport) [pp. 18–47]

A standard literature review of transport methods. The project-relevant points:
- **Notation.** TDSE in atomic units, $n=|\psi|^2$, $\mathbf j=\frac{1}{2i}(\psi^*\nabla\psi-\psi\nabla\psi^*)$, continuity (Eqs 1.1–1.5) [pp. 19–20].
- **Methods reviewed:**
  - Landauer and Büttiker, $G=G_0\sum|t_{mn}|^2$ (Eq. 1.23) [pp. 22–27]. Landauer–Büttiker leaves out many-body effects that cannot be folded into transmissions (Vignale–Di Ventra) [p. 28].
  - Boltzmann/RTA (Eq. 1.40) and Liouville–von Neumann with relaxation, $i\partial_t\hat\rho=[\hat H_0,\hat\rho]-(\hat\rho-\hat\rho_0)/\tau$ (Eq. 1.51), both reused in Ch. 5 [pp. 31–34].
  - Kubo (Eq. 1.48); tight-binding and Hubbard [pp. 33–35].
  - Green's functions: Dyson, and $\Sigma$ "nonlocal in space, energy-dependent and non-Hermitian" [p. 38]; GW [p. 39]; the QP equation $\varepsilon\psi=\hat H_0\psi+\int\Sigma(\mathbf r,\mathbf r',E)\psi(\mathbf r')d\mathbf r'$ (Eq. 1.67) [p. 39]; lead embedding; Kadanoff–Baym and Keldysh [pp. 40–46].
- **Dynamical Coulomb blockade (Kurth et al., PRL 104, 236801 (2010)).** The dot never reaches a steady state. The exact TDDFT KS potential has **discontinuous steps whenever the dot's integer electron number changes**, which the thesis attributes to the derivative discontinuity (Fig. 1.5) [pp. 35–36].
- **Stance.** QP and Green's-function calculations on small or model systems "can provide a useful basis by which to construct and test density-functionals". This is the approach the thesis adopts [p. 47].

### Chapter 2: Density and current-density functional theories (review) [pp. 48–78]

**DFT and TDDFT** [pp. 48–59]. This is standard review: HK, Levy–Lieb, KS and LDA (Eqs 2.1–2.22); Runge–Gross with its Taylor-expandability condition; later extensions (Maitra et al.'s $\nabla\cdot[n\nabla v]=\partial_t^2n+\nabla\cdot\mathbf a[\Psi]$, Eq. 2.32; Ruggenthaler–van Leeuwen); the action functional; van Leeuwen's TD KS mapping [pp. 54–57]. Statements worth keeping:
- "Generally, the XC potential is not a local or semilocal functional of the ground-state density, except in the limit of … the homogeneous electron gas, and for high densities at short wavelengths" [p. 53].
- $\mathbf j=\mathbf j_L+\mathbf j_T$; through continuity, $n(t)$ fixes only $\mathbf j_L$ [p. 54].
- KS initial states must match both $n(0)$ and $\nabla\cdot\mathbf j(0)$ (Eqs 2.37–2.38) [pp. 57–58].
- $v_{xc}(t)$ is "nonlocal … in both space and time, not least because of its intrinsic and important dependence on the initial state" [p. 58].
- The adiabatic approximation and ALDA (Eqs 2.41–2.42) are local in time, so they have no memory [pp. 58–59].

**DFT for transport** [pp. 59–62]:
- Lang's scheme (Eqs 2.43–2.44).
- **Derivative discontinuity.** LDA/GGA are continuous in $n$ and $\nabla n$, but $E_{xc}$ depends discontinuously on local electron number (Fig. 2.1; Toher et al.; SIC Eq. 2.45; Perdew–Parr–Levy–Balduz piecewise-linear $E(N)$).
- Even exact ground-state KS gives wrong band structures, e.g. "metallic band structures for semiconductor densities" (Godby–Needs) [pp. 59–60].
- Kurth et al. 2005 TDDFT transport with partitioned leads and memory terms (Eqs 2.46–2.48) [pp. 60–62].

**TDCDFT** [pp. 62–67]:
- **Moving-slab argument (Fig. 2.2):** in the middle of a translating slab $\partial_tn=0$ but $\mathbf j\neq0$. Any local or semilocal functional of $n$ (ALDA, AGGA) "will be unable to see the direction of the current"; "a practical functional of the charge density must therefore be very long-ranged", whereas "a local functional of the current density will capture the correct physics" [pp. 62–63].
- Ghosh–Dhara uniqueness and the current equation of motion $\partial_t\mathbf j=-i\langle[\hat{\mathbf j}_p,\hat H_0]\rangle+\mathbf E n+\mathbf B\times\mathbf j$ (Eq. 2.52) [pp. 63–64].
- KS TDCDFT: $\mathbf A_{KS}=\mathbf A_{ext}+\mathbf A_{xc}$ (with Biot–Savart folded into $\mathbf A_{xc}$) and $v_{KS}=v_{ext}+v_H+v_{xc}+\tfrac12(A_{KS}^2-A_{ext}^2)$ (Eqs 2.55–2.60) [p. 65]. Vignale–Kohn (Eq. 2.61) [p. 66].
- **Gauge trick:** a scalar potential can be moved into a vector potential, and $\mathbf A=(\mathbf j-\langle\hat{\mathbf j}_p\rangle)/n$ (Eq. 2.66) [p. 66]. This is the seed of the RE algorithm in Ch. 3.

**Ground-state CDFT** [pp. 67–78]:
- Current decomposition: $\mathbf j=\mathbf j_p+\mathbf j_d+\mathbf j_m$, with $\mathbf j_d=\mathbf An$ and $\mathbf j_m=\nabla\times\mathbf m$ (Eqs 2.72–2.75) [p. 68].
- **Vignale–Rasolt (VR)** uses $(n,\mathbf j_p)$. The Capelle–Vignale counterexample shows two $(v,\mathbf A)$ can share one ground state, *also in the KS system*. The difference lives in $\mathbf j_d$ [pp. 68–71].
- **Pan–Sahni** uses $(n,\mathbf j)$. Tellgren et al. showed the proof does not close: it gives $E_1+E_2<E_1+E_2+\int n(\mathbf A_2-\mathbf A_1)^2$ (Eq. 2.99) [pp. 71–73]. Diener's version is shown in Ch. 3 to reduce to the same inequality [pp. 73–75].
- Any HK-type CDFT therefore needs current variables beyond $\mathbf j_p$ and an intermediate quantity beyond $|\Psi_0\rangle$, and it cannot use the physical $\mathbf j$ directly [pp. 75–76].
- $E_{xc}$ is gauge invariant, depending only on $n$ and $\nabla\times(\mathbf j_p/n)$ [pp. 76–77]. VR functional: $\mathbf A_{xc}=\frac{b}{n}\nabla\times\nabla\times(\mathbf j_p/n)$ (Eq. 2.120) [p. 77].
- In VR-CDFT **the KS physical current is wrong**, because the KS diamagnetic current differs (Eqs 2.122–2.123) [pp. 77–78].

**Scope statement** [p. 78]. The thesis studies "exact KS potentials for excited and degenerate nanowires without an applied electric potential and for time-independent electrochemical potentials". Wires are infinite and coupled to reservoirs only at infinity, so **periodic boundary conditions** replace transparent ones and the isolated QP equation (Eq. 1.67) can be used.

### Chapter 3: Unique and exact KS fields in current-carrying systems [pp. 79–105]

**Aims.** Fix the foundations of ground-state CDFT. Then build RE algorithms for exact KS $(v,\mathbf A)$ in the steady state and out of equilibrium.

**Theory:**
- **Diener's proof does not hold** (Sec. 3.1). His "stricter inequality" is Rayleigh–Ritz with a positive semidefinite term subtracted from both sides, $E_0'=E_0-\tfrac12\int n(\mathbf A-\mathbf a)^2$ (Eq. 3.5). Done correctly it gives Eq. 3.12, the same as Tellgren's. Physical-current CDFT therefore "cannot be justified by a Hohenberg–Kohn proof", because the HK argument cannot tell gauge changes from genuine magnetic-field changes [pp. 80–83].
- **Choice of variables** (Sec. 3.2). Take the intermediate set $\mathcal G=\{(|\Psi_0\rangle,\mathbf A)\}$. The densities must be gauge-variant, so the diamagnetic current has to be included. Candidates are $(n,\mathbf j_d)$ (balanced; Appendix) and $(n,\mathbf j_p,\mathbf j_d)$ (main text, more practical) [pp. 83–85].
- **Uniqueness** (Sec. 3.3): $(v,\mathbf A)\leftrightarrow(n,\mathbf j_p,\mathbf j_d)$ via $(|\Psi_0\rangle,\mathbf A)$. The key step is trivial: $\mathbf A=\mathbf j_d/n$ (Eq. 3.21). After that, ordinary HK at fixed $\mathbf A$ (Eqs 3.22–3.30) gives $(v,\mathbf A)\leftrightarrow(n,\mathbf j_p,\mathbf j_d)$ (Eq. 3.31) [pp. 85–87].
- **Variational theorem** (Sec. 3.4): $E_0[n_0,\mathbf j_{p,0}]\le E_{v,\mathbf A}[n,\mathbf j_p]$ (Eq. 3.37), with constrained-search and ensemble extensions [pp. 87–90].
- **Practical KS scheme** (Sec. 3.5) [pp. 90–96]:
  - With $N=(\mathbf j,n)$ and $V=(\mathbf A,v-\tfrac12A^2)$, the conjugate pair is $\delta E_V/\delta V=N$ (Eq. 3.57).
  - The KS system is required to reproduce the *physical* $N$ (Eq. 3.63). The result is $\mathbf A_{KS}=\mathbf A+\mathbf A_{xc}$ and $v_{KS}=v+v_H+v_{xc}+\tfrac12(A_{KS}^2-A^2)$ (Eqs 3.73–3.75), the Pan–Sahni form, not the VR form.
  - The gauge is fixed by $\nabla\cdot\mathbf A_{xc}=0$ with $v_{xc}\to0$ at infinity (Eq. 3.81).
  - $V_{KS}$ is unique as a functional of the KS gauge-variant densities but **not as a functional of the physical $N$**.

**Steady-state RE algorithm** (Sec. 3.6, Fig. 3.1) [pp. 97–100]:
1. Start from a KS system that reproduces $n$, using $v^{(0)}_{KS}$ from VLB or similar and $\mathbf A^{(0)}_{KS}$ (e.g. the external $\mathbf A$), but with $\mathbf j_{KS}\neq\mathbf j$ (Eqs 3.82–3.84).
2. Update $\mathbf A^{(i+1)}_{KS}=\mathbf A^{(i)}_{KS}+\frac{\mathbf j-\mathbf j^{(i)}_{KS}}{n}$ (Eq. 3.86). If the paramagnetic current did not respond, this would be exact in one step (Eq. 3.85).
3. After each $\mathbf A$ update, recompute the eigenstates, $n_{KS}$ and $\mathbf j_{KS}=\sum_k\mathbf j_{p,k}+\mathbf A_{KS}n$. Whenever $n_{KS}\neq n$, recompute $v_{KS}$ (VLB). Stop when $\mathbf j_{KS}=\mathbf j$.

The correction has a longitudinal part $\nabla\cdot\Delta\mathbf A=\Delta\mathbf j\cdot\nabla n/n^2$, described as "inconsequential" (Eq. 3.87). It is "guaranteed" to have a transverse part, $\nabla\times\Delta\mathbf A=(n\nabla\times\Delta\mathbf j-\nabla n\times\Delta\mathbf j)/n^2$ (Eq. 3.88). The reason: steady-state currents are divergence-free, so $\Delta\mathbf j$ is purely transverse, and the correction changes $\mathbf B_{KS}$ [p. 100].

**Nonequilibrium RE algorithm** (Sec. 3.7, Figs 3.2–3.3) [pp. 100–105]:
- Propagate the KS orbitals one step: $\psi_{KS}(t+\Delta t)=e^{-i\hat H_{KS}\Delta t}\psi_{KS}(t)$ (Eq. 3.90). Compute $n^{(i)}$, $\mathbf j_p^{(i)}$, $\mathbf j_d^{(i)}=\mathbf A n$ and $\mathbf j^{(i)}$ (Eqs 3.91–3.94).
- Correct $\mathbf A^{(i+1)}_{KS}(t)=\mathbf A^{(i)}_{KS}(t)+\Delta\mathbf j/n$ with $\Delta\mathbf j=\mathbf j-\mathbf j^{(i)}$. Re-propagate from $t-\Delta t$ and iterate until $\mathbf j^{(i)}=\mathbf j$, then move to $t+\Delta t$.
- Justification: if $\mathbf j(t)$ is right at every step and $n(0)$ is right, continuity makes $n(t)$ right [p. 101].
- All the time dependence, Hartree and XC included, can be put into $\mathbf A$ by gauge choice (Eqs 3.96–3.97) [p. 102].
- **Electric-field variant (Fig. 3.3; Eqs 3.100–3.102)** [pp. 103–104]. Keep an auxiliary $\mathbf A'$, updated by $\Delta\mathbf j/n$. Wherever $\Delta\mathbf j/n$ is longitudinal, gauge-transform it into a scalar-potential update $v^{(i+1)}=v^{(i)}+\int^{\mathbf r}d\mathbf r'\cdot\partial_t\mathbf A'^{(i+1)}$. Only $v$ is used in the propagation. The thesis calls this "particularly useful for 1D calculations where there can be no KS magnetic fields" [p. 104].
- The algorithm "has the correct stop-condition and is, in principle, applicable to all systems in the absence of magnetic fields" [p. 104].
- **[obs]** The thesis does not give a discretised form of $\partial_t\mathbf A'$ (e.g. $\Delta\mathbf A'/\Delta t$). It gives no convergence proof or mixing parameter; convergence is shown only empirically (Chs 4–5).

### Chapter 4: Steady-state quantum transport [pp. 106–125]

**Aims.** Use the steady-state RE algorithm on current-carrying QP states of a periodic model nanowire, in 1D and in 3D.

**Model self-energy** (Sec. 4.1) [pp. 107–109]. Godby–Schlüter–Sham found that the LDA–GW difference in Si and diamond comes mainly from the **spatial nonlocality** of $\Sigma$, not its energy dependence. Their model has the form $\Sigma(\mathbf r,\mathbf r',\omega)=\frac{f(\mathbf r,\omega)+f(\mathbf r',\omega)}{2}g(|\mathbf r-\mathbf r'|)$ (Eq. 4.4), with separable energy dependence $h(\omega)$ (Eq. 4.5). The reciprocal-space matrix elements are $\langle\mathbf k|\Sigma|\mathbf k'\rangle=f(\mathbf k-\mathbf k')\frac{g(|\mathbf k|)+g(|\mathbf k'|)}{2}$ (Eq. 4.6). Throughout Ch. 4, $\Sigma$ is taken as **Hermitian and energy-independent** [pp. 108–109].

**1D nanowire** (Sec. 4.2) [pp. 109–114]:
- **Model:**
  - $\Sigma(z,z')=\tfrac12(f(z)+f(z'))g(|z-z'|)$
  - $f(z)=-F_0[1-\cos(2\pi z/a)]$
  - $g(x)=e^{-x^2/w^2}/(\sqrt\pi w)$, a normalised Gaussian (Eqs 4.7–4.9)
- **Parameters:**
  - $F_0=4.1$ eV (= 0.151 Ha, my conversion), $w=2$ a.u., $a=4$ a.u.
  - Supercell of **10 unit cells**, $L=40$ a.u., periodic boundary conditions, Γ-point sampling of the supercell (primitive $k$ at multiples of $2\pi/L$) [pp. 109–110].
  - $v_{ext}+v_H=0$ in both the QP and the KS system, because they are "the same in both" [p. 109].
- **QP equation:** $(-\tfrac12\partial_z^2-\varepsilon_i)\psi_i+\int\Sigma(z,z')\psi_i(z')dz'=0$ (Eq. 4.10).
- **Electron count:** ground state of $N=22$ electrons (each cell gives one up and one down electron, plus 2 electrons in the standing wave at the conduction-band bottom, so it is "metallic", Fig. 4.1). One QP electron is added in the lowest *right-moving* unoccupied state, giving 23 electrons. That state is one of a degenerate pair (right- and left-moving). All of the current is carried by the QP [pp. 110–111].
- Fig. 4.1 shows the band structure. Its y-axis is mislabelled "n(z)" [p. 110]. Fig. 4.2 shows the ground-state density, which oscillates between about 0.47 and 0.63 with period $a$ [p. 111].
- **Nonlocal current.** With a nonlocal $\Sigma$, continuity gives $j(z)=\frac{1}{2i}(\psi^*\partial_z\psi-\psi\partial_z\psi^*)+\int_{-\infty}^zdz'\int dz''\,[i\psi^*(z')\Sigma(z',z'')\psi(z'')+c.c.]$ (Eqs 4.11–4.12). The second term is "unique to the self-energy operator". The QP current is $j=0.0234$ a.u. [p. 111].
- **VLB procedure** (Eqs 4.13–4.15) [p. 112]: start from a guess $v^{(0)}_{KS}>0$ everywhere, then iterate $v^{(i+1)}_{KS}=\frac{n^{(i)}}{n}v^{(i)}_{KS}$. The result is $v_{KS}=\Delta v_H+v_{xc}$ (Eq. 4.16). After 250 iterations the density error is below $5\times10^{-5}\%$ at every point (Fig. 4.3). The exact KS potential oscillates with amplitude ±2 eV (read off figure) [pp. 112–113].
- **Key result:** the exact density-matching DFT current is **2.5% too low**, "very large compared with the typical $10^{-5}\%$ error in the charge density". The cause is that exact DFT does not reproduce the QP band structure and hence the group velocities [p. 112]. The error falls to zero as $w\to0$ (Fig. 4.5; read off figure: ~0.9% at $w=0.5$, ~2% at $w=1$, ~2.5% at $w=2$). At $w=0$ "the DFT KS system becomes exact" and the bands agree [pp. 112, 115]. **[obs]** Because $g$ is a normalised Gaussian, $g\to\delta$ and $\Sigma\to f(z)\delta(z-z')$, i.e. a local potential.
- **No 1D fix.** Adding a VR-CDFT vector potential reproduces the QP *paramagnetic* current. $A_{KS}$ is negative and oscillatory, about $-0.0009$ to $-0.0012$ a.u. (read off Fig. 4.4). But the *physical* KS current does not change, because in 1D $A_{KS}(z)=-\partial_z\lambda(z)$ is always pure gauge (Eq. 4.17). Hence "steady-state Kohn–Sham systems in one dimension are unlikely to be accurate when there is any significant difference between the KS band structure and that of the interacting system" [pp. 113–114].

**3D nanowire** (Sec. 4.3) [pp. 114–125]:
- **Model:** the same $\Sigma$ form with $g(r)=e^{-r^2/w^2}/(\sqrt\pi w)$ (Eqs 4.18–4.20). $F_0=4.1$ eV and $a=4$ a.u. as before, but **$w=0.5$ a.u.**, roughly $r_s$ of the wire [p. 114].
- Cylindrical symmetry, 10 cells, $N=22$ plus the QP [p. 114].
- Radial confinement $v_{ext}+v_H=Hr^6$ with $H=2$ eV/a.u.$^6$, so that only the lowest subband is occupied. The analysis focuses on $r\le2$ a.u. (Fig. 4.6) [pp. 115–116].
- **Exact DFT potential:** VLB starting from the $w=0$ local limit of $\Sigma$. VLB stalls very early (around $i\approx10$) because $n\to0$ radially, so from $i=133$ it switches to the additive update
  $$v^{(i+1)}_{KS}=v^{(i)}+\mu\,(n^{(i)}_{KS}-n)$$
  (Eq. 4.21; the caption of Fig. 4.8 says $i=130$). The final maximum density error is 0.005% (Fig. 4.7). The error metric is $\epsilon_n=\sqrt{\int(n-n^{(i)}_{KS})^2/V}$ (Eq. 4.22) [pp. 116–118].
- **Current:** the exact-DFT current is **5% too low**, again because of different band slopes at the QP state (Fig. 4.9) [pp. 117–118].
- **CDFT RE** from $i=150$: $\epsilon_j$ (Eq. 4.23) sits at about $10^{-6}$ throughout the DFT stage, then falls "suddenly and exponentially" to about $10^{-10}$–$10^{-11}$ (Fig. 4.10). The final current accuracy is $10^{-4}\%$. The scalar potential barely needs re-tuning ($\epsilon_n=1.3\times10^{-7}$ vs $2.6\times10^{-8}$ for exact DFT) [pp. 117, 119].
- **XC magnetic field (Fig. 4.11).** Only $A_{xc,z}$ is nonzero, up to ~0.003 a.u. (read off figure). It gives a purely **azimuthal $B_{xc}$**, which looks like a Biot–Savart field but points the *opposite* way and grows radially outward. The thesis attributes the radial growth to the radially increasing confining electric field [p. 120].
- **Degeneracy.** The KS state is degenerate with a left-moving KS state, but that partner is *not* the KS image of the interacting left-mover. The current-dependent $\mathbf A_{KS}$ lifts the interacting degeneracy and creates a new KS one (Fig. 4.12) [pp. 119, 121].
- **VR functional fails (Fig. 4.13).** With the fitted susceptibility $b=0.35$ a.u., the VR $\mathbf A_{xc}$ is parallel to $\mathbf j$ but is "qualitatively very different": far more radial dependence, which means a much stronger $B_{xc}$. It also gives the wrong density. The thesis's explanation is that the functional is HEG-based and has a $1/n^2$-type dependence that blows up as $n\to0$ radially [pp. 119, 122].
- **KS current structure (Fig. 4.14).** The QP current is purely paramagnetic and carried by one electron. The KS current is partly paramagnetic and partly diamagnetic, and **all 22 lower KS orbitals carry some paramagnetic current** [pp. 120, 123].
- **XC Stern–Gerlach.** The QP carries the net magnetisation and there is no external $\mathbf B$, so the interacting system is spin-degenerate. The KS system has both $\mathbf m$ and $\mathbf B_{KS}$, so a Stern–Gerlach term $\mu_B\mathbf B_{KS}\cdot\hat{\boldsymbol\sigma}$ appears in the KS equations (Eqs 4.24–4.27). Cancelling it needs a spin- and current-dependent extra $v_{KS}$. For an illustrative $s_{KS,\theta}=\tfrac12$ that extra potential is small, of order $10^{-2}$ eV (Fig. 4.15, read off figure). The thesis argues it is potentially large in more strongly magnetised systems [pp. 121–124].

**Chapter conclusions** [pp. 124–125]:
- The exact KS system that reproduces $(n,\mathbf j)$ "is qualitatively and strikingly different" from the interacting system.
- It generally needs an **effective magnetic field that depends on charge and current density**.
- Interacting degeneracies are not preserved in the KS system.
- $v_{xc}$ acquires a functional dependence on $\mathbf j$ and $\mathbf m$.
- Mapping gauge-variant densities onto KS potentials can give purely longitudinal (unphysical) $\mathbf A_{xc}$. Mapping gauge-invariant densities avoids this but leaves a gauge ambiguity, analogous to the constant in $v$. Whether different KS fields could give the same physical densities but different wavefunctions is still open.

### Chapter 5: Time-dependent quantum transport [pp. 126–160]

**Aims.** Compute exact time-dependent KS potentials for a QP wavepacket moving freely, with no applied fields, through a model semiconducting wire. This is done first with infinite lifetime (Hermitian $\Sigma$) and then with decay (non-Hermitian $\Sigma$, RTA). The chapter also studies initial-state dependence. Only the longitudinal current is considered, because 1D steady currents cannot be reproduced (Ch. 4) [p. 126].

**Initial-state philosophy** [p. 127]. Following Maitra–Burke, the state at $t_1$ encodes all earlier history, so one can start from an excited state with the correct density. The cost is an explicit initial-state dependence of the potentials.

**System** (Sec. 5.1) [pp. 128–133]:
- $\Sigma$ as in Ch. 4: $F_0=4.1$ eV, $a=4$ a.u., $w=2$ a.u. (Eqs 5.1–5.3).
- Supercell of **20 unit cells** ($L=80$ a.u.), periodic boundary conditions, Γ-point sampling.
- **One spinless electron per cell**, so the 20 ground-state electrons exactly fill the valence band (semiconducting). $v_{ext}+v_H=0$ again (Eq. 5.4). The ground-state density oscillates between about 0.21 and 0.29 (Fig. 5.1) [pp. 129–130].
- **QP wavepacket:** $\psi_{QP}(x,t_0)=\sum_kc_k\phi_k(x)$ (Eq. 5.5), built from the right-moving states of the first conduction band; there are **9** of them for this supercell. The $c_k$ come from Powell's conjugate-direction method, minimising the wavepacket density in the first two and last two cells (Fig. 5.2). The packet is centred at $x=40$ a.u., peak density ~0.085 (Fig. 5.3) [pp. 129–131].
- **Propagation:** only the wavepacket is propagated, since the valence states are eigenstates. It uses the TD QP equation (Eq. 5.6) with **Crank–Nicolson**, $(1+\tfrac12i\Delta t\hat H_{QP})\psi(t+\Delta t)=(1-\tfrac12i\Delta t\hat H_{QP})\psi(t)$ (Eq. 5.7), and **$\Delta t=4\times10^{-3}$ a.u.** The current is Eq. 5.8 (including the nonlocal-$\Sigma$ term) [pp. 131–132].
- **Spatial grid spacing for this calculation is not stated.**
- **Dynamics (Fig. 5.4).** The density does not translate rigidly; it grows ahead of the centre of mass and shrinks behind it. The current density (peak ~0.09 a.u.) translates nearly rigidly [p. 132].
- The supercell always holds 21 electrons [p. 133].
- The thesis expects possible **local derivative-discontinuity steps as the electron number per unit cell passes through integers**. It argues that the KS potential therefore need not be periodic even though all physical quantities are periodic [p. 133].

**Initial KS state** (Sec. 5.2) [pp. 133–135]:
- The 20-electron ground-state KS potential comes from VLB; it oscillates with amplitude about ±1 eV (Fig. 5.5).
- All of the initial QP density and current is assigned to one KS orbital: $\psi_{KS,WP}=\sqrt{\Delta n(x,t_0)}\,e^{i\lambda(x)}$ with $\lambda(x)=\int^x j/\Delta n\,dx'$ (Eqs 5.9–5.10). Its overlap with the valence band is 0.0002, so it is effectively a sum of conduction states of the KS Hamiltonian, but with different weights from the QP packet (Eq. 5.11).
- **Under the static ground-state KS potential, the KS packet lags the QP packet** (Fig. 5.6). The KS band slope near the packet is lower (Fig. 5.9).
- So "even when the interacting system being modelled is free from time-dependent fields, the KS system … cannot reproduce its time-dependent charge and current density without time-dependent KS potentials" [p. 135].

**Time-dependent RE** (Sec. 5.3) [pp. 135–143]:
- Coulomb gauge with $A_{KS}=0$ in the propagation (1D, so no $B$) [p. 137].
- At most 1000 RE iterations per time step. "Generally the correct potential to within around 0.001% was found within 20 iterations." The algorithm "remained convergent throughout" [p. 137].
- **Error metrics** (Eqs 5.12–5.13): $\epsilon_n(t)=\sqrt{\int_0^L(n-n_{KS})^2dx/L}$, and likewise $\epsilon_j$. Over 150 steps (read off Fig. 5.7):
  - $\epsilon_n$ grows smoothly from about $10^{-8}$ to $10^{-6}$;
  - $\epsilon_j$ grows in sawtooth jumps from about $1.5\times10^{-8}$ to $5\times10^{-7}$.
  
  The text attributes this to accumulated discretisation and CN error. It notes a "pathological" effect: the algorithm absorbs propagation errors into the potential whenever doing so improves $j$ [pp. 137–138].
- **Three features of the extra potential $\Delta v_{KS}(x,t)$ (Fig. 5.8)** [pp. 138–139]:
  1. a small barrier that follows the wavepacket, peak ~0.2 eV (read off figure), which the thesis identifies with the QP Hartree potential;
  2. a time-dependent periodic component on the barrier that locally "tunes" the band structure and hence the group velocity;
  3. **a time-dependent potential step between the left and right sides of the packet**, "equivalent to having a time-dependent potential bias across the entire (infinite) wire". Away from the packet the density is time-independent and equal to the ground-state density, yet the potential ahead and behind differ: "a dynamic effective exchange-correlation electrochemical potential". At $\Delta t$, 75$\Delta t$ and 150$\Delta t$ the step is about 0, −0.07 and −0.09 eV (read off figure; potential fixed to zero on the left).
- **Long-time step (Fig. 5.10).** After transients, the step amplitude is roughly periodic, consistent with the packet hopping cell to cell. It neither grows without bound nor settles, i.e. it is "not a transient effect but a genuine dynamic one". Read off figure: it runs from about +0.06 eV at $t\approx350\Delta t$ through a minimum near −0.13 eV at $\approx520\Delta t$ and back up to about +0.02 eV at $\approx670\Delta t$. The thesis cannot say whether the slight attenuation is numerical or physical [pp. 139–140].
- The three features together set the correct group velocity [p. 141].
- **ALDA and AGGA fail:** "a functional such as the ALDA or the adiabatic GGA would predict that the regions ahead of and behind the wavepacket would have exactly the same potential and fail to predict the necessary time-dependent potential step" [p. 141].
- **Boundary conditions.** Periodic densities can have non-periodic KS potentials, and "enforcing periodic boundary conditions in the KS description … would destroy the time-dependent potential step" [p. 141]. The fix is a **mixed real-space approach**. Propagate on a larger grid in which the supercell is extended at each step by left and right **buffer regions**, filled by extrapolating with the Bloch condition $\phi(x+a)=e^{i\alpha}\phi(x)$ (Eq. 5.14). Then discard the buffers, which absorb the finite-difference/CN boundary errors. This works while the packet is zero at the supercell edges [pp. 141–142].
- **Gauge trade-off.** The RE actually works in the velocity gauge: it stores $\Delta A'_{KS}$, forms $\mathcal E_{KS}=-\partial_tA'$, and converts to $\Delta v_{KS}=-\int_{-\infty}^x\mathcal E_{KS}dx'$ (Eq. 5.15). In the velocity gauge the spatial ultranonlocality disappears, but **temporal ultranonlocality appears**. At $x=40$ a.u., $A_{KS}(x,t)$ grows steadily while $n$ stays almost constant (Fig. 5.11), because a static field gives $A\approx\partial_xv\,t$ (Eq. 5.16). "Thus we have traded one ultranonlocality for another" [pp. 142–143].

**The KS electric field** (Sec. 5.4) [pp. 143–146]:
- Both potentials are integrals of $\mathcal E_{KS}=-\partial_xv_{KS}-\partial_tA_{KS}$ (Eq. 5.17).
- The extra field $\Delta\mathcal E_{KS}$ is **localised with the wavepacket**, magnitude ≲0.02 eV/a.u. (Fig. 5.12; read off figure). The step in $v$ arises because this localised field "has a time-dependent nonzero average", so its spatial integral across the packet is nonzero [pp. 143–144].
- Motivation from the 1D current equation of motion, $\partial_tj-\mathcal E n+P[n,\Psi(t_0)]=0$ (Eq. 5.18), where $P$ is the stress-momentum (unknown). This suggests $(1/n)\partial_tj$ as an ingredient, and a GGA-like guess for $P$ suggests $\nabla n$ [p. 144].
- **Result (Fig. 5.13).** Plot the *total* KS field $-\partial_xv$ against $(\partial_xn)/n$ (range about ±0.4) and $(\partial_tj)/n$ (about ±0.04; ranges read off figure) at 25, 50, 75 and 100$\Delta t$. The field "is single-valued and lies almost on a plane", and "is largely independent of time".
  - Away from the packet the $(\partial_xn)/n$ dependence dominates.
  - Inside the packet the $(\partial_tj)/n$ dependence becomes visible.
  - Elsewhere the thesis puts it as "still strongly dominated by the charge density, having a smaller but certainly nonzero semilocal functional dependence on the current density" [p. 144].
- Conclusion: for this history the field is "localised in both space and time" and has "to good approximation, a local and unique dependence" on $n$, $\partial_xn$ and $\partial_tj$ [pp. 145–146].
- **No fitted coefficients are given.**

**Finite lifetime** (Sec. 5.5) [pp. 146–152]:
- **Assumptions** [pp. 146–147]:
  - exponential decay at all times;
  - decay toward a Fermi–Dirac ensemble fixed by conservation of energy, momentum and particle number;
  - RTA with a position-independent rate;
  - one extra ground-state electron at the conduction-band bottom, making the system metallic;
  - an RPA homogeneous-gas dielectric function;
  - a softened 1D Coulomb interaction.
- **Equation of motion:** Liouville–von Neumann in the RTA, $\partial_t\rho=-i[\hat H,\rho]+(\rho_{final}-\rho)/\tau$ (Eq. 5.21), starting from $\rho(0)=|\Psi(0)\rangle\langle\Psi(0)|$ [p. 147].
- **Energetics:**
  - Energy zero at the valence-band bottom. The 21-electron ground-state energy is 79.95 eV and the QP energy 21.9 eV, total 101.85 eV.
  - $\langle p_0\rangle=1.16$ a.u., $N=22$ (Eq. 5.23).
  - The final ensemble is $\rho=\sum_kf(\varepsilon_k)|\Psi_k\rangle\langle\Psi_k|$ under the constraints of Eqs 5.25–5.27.
  - Basin hopping plus Powell, minimising $\epsilon=\alpha|E'-E_0|+\beta|p'-p_0|+\gamma|N'-22|$ with $\alpha=0.2$, $\beta=1$, $\gamma=0.1$, gives $\mu_L=12.7$ eV, $\mu_R=12.4$ eV and $k_BT=2.35$ eV. The residual $\epsilon=0.01$ corresponds to roughly 0.3% in energy, 0.8% in momentum or 0.5% in $N$ (Fig. 5.14) [pp. 147–149].
- **Lifetime:**
  - 1D GW-like formula $\mathrm{Im}E_c(p)=\int_{0\le E'\le E}\frac{dk}{2\pi}W(k)\,\mathrm{Im}\frac{1}{\epsilon_{1D}(k,E-E'+i\delta)}$ (Eq. 5.30), with the 1D Lindhard/RPA dielectric function $\epsilon=1-W\chi$ (Eqs 5.31–5.32).
  - **Softened Coulomb** $W(x,x')=1/\sqrt{(x-x')^2+C^2}$ (Eq. 5.33), with $C$ chosen from a waveguide of diameter $d$ such that only the lowest subband is occupied up to $E_{max}\simeq1.5$ Ha. This gives $C\simeq0.866$ a.u. (Eq. 5.35) and $W(k)=\sqrt{2/\pi}K_0(C|k|)$ (Eq. 5.36).
  - Result: $\mathrm{Im}E_c=0.00935$ Ha, so **$\tau\simeq107$ a.u.**, much longer than the simulation time [pp. 149–151].
  - **[obs]** This is *not* iDEA's $1/(|x-x'|+1)$ softening.
- **Decay setups (Fig. 5.15):** (a) decay starts infinitesimally after $t=0$, so the initial current equals the non-decaying one; (b) decay starts exactly at $t=0$, which adds a small current away from the packet [p. 151].
- **Numerical parameters:** **$\Delta x=0.2$ a.u., $\Delta t=\tfrac12\Delta x^2=0.02$ a.u.**, periodic boundary conditions throughout [p. 151].
- **Dynamics (Figs 5.16–5.17).** The packet peak decays slightly. Charge builds up uniformly in the background, ~$10^{-4}$ by 60$\Delta t$ (read off figure) [p. 152].

**KS potentials for decay, case (a)** (Sec. 5.6) [pp. 153–157]:
- The ground-state KS potential is recomputed by VLB for 21 electrons (Fig. 5.18; amplitude about ±2 eV). The 22nd electron is $\phi_{N+1}=e^{i\lambda}\sqrt{n_{QP}}$ (Eq. 5.37).
- **Decay-inducing potentials are huge** (Fig. 5.19): a barrier of ~650 eV at $t=\Delta t$, ~330 eV at $2\Delta t$, then about −320 eV at $3\Delta t$ and decaying (read off figure). They come with $x^2$-shaped wells on either side and a relative step [pp. 154–155].
- **[obs]** The sign alternation between consecutive steps is visible in Fig. 5.19 but not discussed in the text.
- The amplitude grows as $\Delta t$ and $\tau$ shrink, so the exact values are "calculation-specific". The thesis explains: "the size of the corrections to the potential are proportional to the time-derivative of the current density", and the onset of background current is non-adiabatic [p. 155].
- The $x^2$ wells follow from a current that is locally linear in $x$, giving a field linear in $x$ [p. 155].
- **Later times (10–50$\Delta t$)** show four features [pp. 155–157]:
  1. the small dynamic step seen earlier (from the nonlocal hole);
  2. a small long-range $-x^2$ component;
  3. growing, asymmetric, periodic **"buckets"** (up to ~0.16 eV, read off figure) that undo the polarisation the long-range field would cause and "tilt" electrons back to their sites;
  4. **a large step of ~2.2–2.6 eV across the wavepacket** (read off figure). It is "comparable to the band gap in size", varies slowly, must decay to zero, and lets density reflect left and tunnel right.
- The electric field stays localised (±0.025 a.u.; read off figure).
- **Scaling claim:** if $\tau$ is scaled by $\gamma$ and $\Delta t$ by $\gamma$, the decay component of the potential is unchanged at all times ("confirmed computationally") [p. 156].

**Memory test, case (b)** (Sec. 5.7) [pp. 157–158]:
- Now the initial KS state already carries the decaying current. The whole KS system, valence band included, is phase-transformed by $\phi_k\to e^{i\lambda}\phi_k$ with $\partial_z\lambda=(j_{QP}-j'_{KS})/n$ (Eqs 5.38–5.39).
- Result: the huge barriers and wells **do not appear**. Within a few time steps the potential becomes **identical** to case (a) (Fig. 5.20; read off figure: at $t=\Delta t$ the step is ~1.6 eV, at $2\Delta t$ ~2.6 eV, then settling near ~2.1 eV).
- The dependence on the initial *current* distribution therefore has a "remarkably short-ranged memory" [pp. 158–159].

**Chapter conclusions** [pp. 158–160]:
1. The RE algorithms work, both steady-state and nonequilibrium.
2. Steady-state systems with scalar potentials only are "not generally noninteracting-V-representable without the inclusion of an exchange-correlation magnetic field".
3. Exact TD $v_{KS}$ is ultranonlocal, but the KS electric field is amenable to semilocal approximation.
4. Decay can be induced by large but simple, short-lived potentials.
5. "Memory effects due to the differences in the initial charge distribution of the KS system can persist throughout the simulation time, while those that arise due to the differences in the initial current distribution may be short-lived."

The step is claimed to be "a robust feature of all calculations, independent of the precise nature of the self-energy operator or the initial interacting or Kohn–Sham states, so long as the quasiparticle is localised and the self-energy operator nonlocal" [pp. 158–159].

### Chapter 6: Summary and conclusions [pp. 161–168]

The chapter recaps the earlier results (Eqs 6.1–6.3) [pp. 161–164]. Points it adds:
- A claim **not demonstrated in the thesis body**: "two KS systems with the same time-dependent charge and current densities with only slightly different initial states were seen to have starkly different time-dependent KS potentials" [p. 165] (see Sec. 6 below).
- "It is not feasible to construct local or semilocal functionals of the charge and/or current density for the scalar potential that capture the correct physics of nonequilibrium quantum transport" [p. 166].
- **Fields over potentials** [pp. 166–167]:
  - when $\dot j\to0$ (steady state), an electric field alone cannot fix $\mathbf j$, so an XC $\mathbf B$ is needed, and mixed systems need both XC $\mathbf E$ and $\mathbf B$;
  - fields are gauge-invariant, while potentials are space and time integrals of fields and so carry ultranonlocality.
- **Future work** [pp. 167–168]: "many more systems", especially "more rapidly-varying densities"; energy-dependent $\Sigma$; exact 2–3-electron and Hubbard solutions [refs 141, 164].
- The RE algorithms are "a method for studying quantum transport … and not a means of calculating quantum transport systems from first principles" [p. 168].

### Appendix A: Additional proofs [pp. 169–186]

- **A.1 Balanced CDFT with $(n,\mathbf j_d)$** [pp. 169–180]. HK holds at any fixed $\mathbf A$, which gives $(v,\mathbf A)\leftrightarrow(n,\mathbf j_d)$ (Eqs A.1–A.20). Minimisation reduces to ordinary DFT at fixed $\mathbf A$, which the thesis calls "counter-intuitive" because the current is never varied [p. 174]. The KS scheme has the same form as in Sec. 3.5.
- **A.2.1 Spin:** $(v,\mathbf A)\leftrightarrow(n,\mathbf j_g,\mathbf j_d)$ with $\mathbf j_g=\mathbf j_p+\nabla\times\mathbf m$ [pp. 180–183].
- **A.2.2 Degenerate pure states:** uniqueness between $(v,\mathbf A)$ and the *set* of degenerate ground-state densities [pp. 183–186].

---

## 3. Glossary of cross-cutting concepts

| Term | Meaning in the thesis | Ref |
|---|---|---|
| Quasiparticle (QP) | An elementary excitation (added electron) obeying $\varepsilon\psi=\hat H_0\psi+\int\Sigma\psi$; the thesis's "interacting" system | [pp. 39, 109] |
| Model self-energy | $\Sigma=\tfrac12[f(x)+f(x')]g(\lvert x-x'\rvert)$; $f$ periodic amplitude ($F_0$), $g$ normalised Gaussian of range $w$; Hermitian, energy-independent (Chs 4–5.4); non-Hermitian via RTA in 5.5 | [pp. 107–109, 129] |
| Nonlocal current term | Extra contribution to $j$ from a nonlocal $\Sigma$ (Eqs 4.12, 5.8) | [pp. 111, 132] |
| VLB procedure | $v^{(i+1)}=(n^{(i)}/n)v^{(i)}$ with $v^{(0)}>0$ | [p. 112] |
| Additive density update | $v^{(i+1)}=v^{(i)}+\mu(n^{(i)}_{KS}-n)$; used where VLB stalls ($n\to0$) | [p. 116] |
| Steady-state RE | $\mathbf A\leftarrow\mathbf A+(\mathbf j-\mathbf j_{KS})/n$ with $v$ re-tuned to hold $n$ | [pp. 97–100] |
| TD RE / E-field RE | Per-time-step current-matching correction; in 1D converted to a scalar-potential update | [pp. 100–104] |
| Paramagnetic / diamagnetic / magnetisation / physical current | $\mathbf j_p=\langle\hat{\mathbf j}_p\rangle$, $\mathbf j_d=\mathbf An$, $\mathbf j_m=\nabla\times\mathbf m$, $\mathbf j=\mathbf j_p+\mathbf j_d+\mathbf j_m$ | [p. 68] |
| Coulomb gauge vs velocity gauge | $A_{KS}=0$ in propagation and all fields in $v$, vs all time dependence in $A$ | [pp. 102–103, 142] |
| Ultranonlocality | Dependence of $v_{KS}(x,t)$ on densities far away in space (Coulomb gauge) or long ago in time (velocity gauge) | [pp. 139, 142–143] |
| Dynamic XC step / "XC electrochemical potential" | Time-dependent potential difference across a moving localised packet, where the density far away is ground-state-like | [p. 139] |
| KS electric field | $\mathcal E_{KS}=-\partial_xv_{KS}-\partial_tA_{KS}$; localised and semilocal | [pp. 143–145] |
| XC magnetic field | Transverse $\mathbf A_{xc}$ needed for KS to reproduce steady $\mathbf j$ in ≥2D | [pp. 117–120] |
| XC Stern–Gerlach | Coupling of $\mathbf B_{xc}$ to KS magnetisation, requiring a spin-dependent $v_{xc}$ | [pp. 121–124] |
| Derivative discontinuity | Step-like dependence of $E_{xc}$ / $v_{xc}$ on local integer electron number | [pp. 36, 59–60, 133] |
| Initial-state dependence / memory | $v_{KS}$ is a functional of $n(t)$ *and* $\Psi(t_0)$, $\Phi(t_0)$ | [pp. 58, 127, 157–158] |
| Adiabatic approximation / ALDA | $v_{xc}[n](x,t)=v_{xc}^{gs}[n_t](x)$; no memory | [pp. 58–59] |
| Stress-momentum $P$ | Term in the current equation of motion; a functional of $n$ and the initial state | [pp. 56, 144] |
| RTA / Liouville–von Neumann | $\partial_t\rho=-i[H,\rho]+(\rho_f-\rho)/\tau$ | [pp. 34, 147] |
| Buffer regions | Bloch-extrapolated grid extensions that allow non-periodic KS potentials for periodic systems | [pp. 141–142] |
| $V$-representability | Being the ground-state density of some $(v,\mathbf A)$ | [pp. 51, 85] |
| Balanced CDFT | Equal numbers of basic potentials and densities: $(v,\mathbf A)\leftrightarrow(n,\mathbf j_d)$ | [pp. 170–174] |

---

## 4. Relevance to learning density functionals with ML

What the thesis implies a learned map ($n\to v_{xc}$ or $v_s$, possibly time-dependent) must capture, and how it bears on data and representation.

1. **Spatial ultranonlocality is real, even in simple 1D dynamics.**
   - The exact TD $v_{KS}$ has a step across a localised moving packet. In regions far away the density is identical to the ground state but the potential differs ahead of and behind the packet [p. 139].
   - Any local, semilocal or adiabatic map from $n$ (or windowed CNN-style features) predicts the same potential on both sides and must fail [p. 141]. The moving-slab argument says the same [pp. 62–63].
   - Consequence: the model needs a global receptive field. Attention over the whole grid is the natural fit, and a local convolution or a pointwise MLP on local features is structurally insufficient.

2. **The output representation sets how hard the learning problem is.**
   - The same physics is spatially nonlocal in $v$ (Coulomb gauge), temporally nonlocal in $A$ (velocity gauge), and *local and semilocal* in the field $\mathcal E=-\partial_xv-\partial_tA$ [pp. 142–145, 166–167].
   - A strong design option is therefore to predict $\mathcal E_{xc}(x,t)$ (or $\partial_xv_{xc}$) locally and integrate it, $v=-\int^x\mathcal E$. The integration constant / gauge must be fixed explicitly; the thesis anchors $v$ at the left edge ($\int_{-\infty}^x$) [p. 142].
   - **[obs]** This turns a global-output problem into a local one plus a fixed, non-learned integral. It also makes the gauge convention (constant offset, which side is pinned) part of the data specification, and it must be applied consistently to both targets and predictions.

3. **Candidate input features for TD functionals.**
   - $n$, $(\partial_xn)/n$ and $(\partial_tj)/n$, with $j$ obtained in 1D from continuity, $j=-\int^x\partial_tn$ [pp. 136, 144–145].
   - Within one history these give a near-planar, single-valued map to the total KS field [p. 145].
   - **[obs]** Treat this as a feature hint, not a validated functional. It is one trajectory, four snapshots and no fitted coefficients, and transfer to other systems or histories is untested. $(\partial_tj)/n$ is a memory-like (non-adiabatic) quantity: it carries information about $\partial_t^2n$.

4. **Memory and initial-state dependence.**
   - $v_{KS}$ depends on the initial interacting and KS states [pp. 58, 127].
   - In the decay study, the dependence on the initial *current* distribution vanished within a few time steps [p. 158]. The thesis also claims (not shown in the body) that differences in the initial *charge* distribution persist [pp. 159, 165].
   - For learning: a TD model probably needs a history window and/or an encoding of the initial state. The required window length is system-dependent, short in the decay test. Data should use controlled, documented initial states; otherwise targets for the "same" density history are multi-valued.

5. **Derivative-discontinuity-type steps.**
   - Expect sharp steps when local (per-region or per-cell) electron number passes through integers [pp. 36, 59–60, 133].
   - Smooth function approximators will smear these. Consider loss weighting near steps, or predicting fields (where steps become localised peaks).

6. **Periodic or bounded densities do not imply periodic potentials.** Enforcing periodic boundary conditions on the KS potential "would destroy the time-dependent potential step" [p. 141]. Do not build periodicity into model outputs, even for periodic training systems.

7. **Density-matching does not give the right currents or band velocities.**
   - The exact ground-state KS system reproducing $n$ underestimates the QP current by 2.5% (1D) and 5% (3D), because it has the wrong group velocities [pp. 112, 117].
   - A learned $n\to v_s$ inherits exact KS limitations: derived observables (currents, excitation energies, band slopes) are not reproduced even with a perfect $v_s$.
   - In 1D the steady-state current cannot be fixed at all, since $A$ is pure gauge [p. 113]. For 1D ground-state data with net current (e.g. degenerate or ring states), a density-only functional is incomplete.

8. **Huge, $\Delta t$-dependent targets for non-adiabatic onsets.**
   - Exact potentials that induce sudden changes reach hundreds of eV at the first steps, scale with $1/\Delta t$ and $1/\tau$, and are "calculation-specific" [pp. 154–155].
   - Such samples are not converged physical quantities. Exclude them, normalise per sample, or train on fields or on later times. Within a few steps the long-time potential becomes insensitive to how the decay was started [p. 158], which is reassuring for learning the "steady" part.

9. **Reverse-engineering noise ends up in the targets.**
   - TD RE will "correct" accumulated propagation error by adjusting $v$ [pp. 137–138]. $\epsilon_n$ grows roughly linearly with time ($10^{-8}\to10^{-6}$ over 150 steps).
   - VLB stalls where $n\to0$ [p. 117]. The VR functional failed precisely because of $1/n^2$ behaviour at low density [p. 119].
   - For ML: record per-sample RE error metrics ($\epsilon_n$, $\epsilon_j$, Eqs 5.12–5.13), mask or down-weight low-density regions in the loss, and expect noisier targets at late times.

10. **A tunable nonlocality knob gives controlled data.** The model $\Sigma$ with range $w$ interpolates continuously from a local potential ($w=0$, where KS is exact) to strongly nonlocal ($w=2$). The KS error grows monotonically with $w$ (Fig. 4.5) [p. 115]. **[obs]** This suggests a curriculum or benchmark axis: datasets where the true "interaction" is a fixed nonlocal operator of tunable range, giving a cheap single-particle way to generate exact (local) KS targets with known nonlocality.

11. **Gauge and representation hygiene.** Potentials are fixed only up to a constant (time-dependent in TD). Vector potentials in 1D are pure gauge. Uniqueness holds for gauge-variant variables only [pp. 113, 125]. A training set must fix a gauge convention, e.g. $v_{xc}\to0$ at infinity or $\nabla\cdot\mathbf A_{xc}=0$ [p. 96].

12. **The thesis's own call is for more data.** "One would need to calculate the exact KS potentials for many more systems to gather a more complete set of data upon which to base an analysis", especially "more rapidly-varying densities" [p. 167]. A data-driven project answers exactly this call.

---

## 5. Reproducible with iDEA

**Caveat first.** None of the thesis systems is an exact many-body solution. The "interacting" systems are single-particle QP equations with a **fixed nonlocal model self-energy**, plus spectator valence states that are eigenstates of the same nonlocal Hamiltonian. **[obs]** Reproducing them therefore needs a *single-particle solver with a nonlocal potential matrix* (structurally like HF with a frozen exchange operator), not iDEA's many-body solver. Other differences:
- Several systems use periodic supercells. According to `notes/idea_overview.md`, iDEA-latest uses hard walls.
- iDEA's TD RE uses root-finding, not the current-matching update of Sec. 3.7. Its static RE uses $v\leftarrow v+\mu(n^p-n_t^p)$, close to Eq. 4.21 but with a power $p$.
- The thesis does not give the spatial grid for most calculations.

| Thesis calculation | What's needed | Parameters given | Missing / assumptions |
|---|---|---|---|
| **1D steady-state wire, exact DFT (Sec. 4.2)** | Nonlocal single-particle eigen-solver, PBC; ground-state RE (VLB or additive); current via Eq. 4.12 | $\Sigma$: $F_0=4.1$ eV (0.151 Ha), $a=4$, $w=2$ a.u., Gaussian $g$ normalised; $L=40$ a.u. (10 cells); $v_{ext}+v_H=0$; 22 spin-paired electrons (11 per spin, incl. 1 per spin at CB bottom) + 1 QP in the lowest right-moving state; Γ-point sampling [pp. 109–111] | Grid spacing not stated. Reported targets: $j_{QP}=0.0234$ a.u., DFT current −2.5%, density error $<5\times10^{-5}\%$ after 250 VLB iterations [pp. 111–112]. Repeating the $w$ scan (Fig. 4.5) is cheap. |
| **3D cylindrical wire (Sec. 4.3)** | Outside iDEA's scope (3D, vector potentials) | $w=0.5$ a.u., $Hr^6$ with $H=2$ eV/a.u.$^6$ [pp. 114–115] | Not reproducible with a 1D code. |
| **1D TD QP wavepacket (Secs 5.1–5.4)** | Nonlocal $\hat H_{QP}$; CN propagation; 20 spinless valence electrons as static eigenstates; wavepacket from 9 right-moving CB states; KS ground state via RE; TD RE with current matching and the E-field variant; non-periodic KS potential via buffer regions | $L=80$ a.u. (20 cells), $F_0=4.1$ eV, $a=4$, $w=2$; $\Delta t=4\times10^{-3}$ a.u.; 150 steps (main), ~670 steps (Fig. 5.10); KS initial orbital $\sqrt{\Delta n}e^{i\lambda}$ [pp. 129–137] | Grid spacing not given; $c_k$ shown only graphically (Fig. 5.2) and set by Powell minimisation of density in cells 1–2 and 19–20. **[obs]** A reproduction would re-derive the $c_k$ by the stated optimisation, so it would match only qualitatively. Targets: step ≈ −0.07/−0.09 eV at 75/150$\Delta t$; barrier ≈ 0.2 eV; ε metrics [pp. 138–140]. |
| **1D decaying QP (Secs 5.5–5.7)** | Density-matrix propagation with an RTA term (not in iDEA); TD RE | $\Delta x=0.2$ a.u., $\Delta t=0.02$ a.u., PBC; $\tau=107$ a.u.; $\mu_L=12.7$, $\mu_R=12.4$ eV, $k_BT=2.35$ eV; 21 ground-state electrons + 1 QP; energies 79.95 + 21.9 eV [pp. 147–151] | The final-state ensemble construction is specific to this model. The large early potentials are $\Delta t$-dependent [p. 155]. |
| **Hodgson et al. 2013 [ref. 141]** | Exact 2–3-electron TD solutions + this thesis's TD RE | Not in this thesis | The natural iDEA-based follow-on; see that paper for parameters. |

**[obs] Practical suggestions for density_model:**
- The cleanest iDEA-native analogue of the Ch. 5 phenomenon is a few-electron *exact* system in which a localised packet moves through a static background (e.g. a slow electron passing a bound pair, or an excited packet in a long well). Reverse-engineer $v_s(x,t)$, then check for (i) a dynamic step across the packet and (ii) a localised field $\partial_xv_{xc}$.
- The thesis's E-field RE update (accumulate $\Delta j/n$ as an auxiliary $A'$ and integrate $\partial_tA'$ in space) is a current-based alternative to root-finding. It gives an explicit, cheap per-iteration correction, and the thesis reports it converging to within ~0.001% in ~20 iterations [p. 137].

---

## 6. Open questions, stated limitations, and possible inconsistencies

**Limitations and open questions the author states:**
- Whether different KS fields could give the same physical densities but different ground-state wavefunctions is open; it is "immune to a Hohenberg–Kohn-like treatment" [p. 125].
- Physical-current CDFT uniqueness is unproven either way [pp. 82–83].
- Only spatial nonlocality and non-Hermiticity of $\Sigma$ are studied. Energy dependence is not yet investigated [p. 167].
- The attenuation of the long-time step amplitude is of undetermined origin, numerical or physical [p. 139].
- More systems are needed for functional construction, especially more rapidly varying densities. Data on $B_{xc}$ in nanowire cores is limited [p. 167].
- Functionals must ultimately be tested against experiment [pp. 167–168].
- The RE algorithms are analysis tools, not predictive methods [p. 168].
- The VR functional is ill-suited to confined systems [p. 119].
- The exponential-decay and RTA assumptions are acknowledged simplifications (uniform decay rate, decay toward a Fermi–Dirac ensemble) [pp. 146–147].

**My observations** (possible inconsistencies and things to keep in mind; all **[obs]**):

1. **"Interacting" means a QP model.**
   - The densities come from a single-particle equation with a fixed nonlocal $\Sigma$. There is no many-body wavefunction, no self-consistency of $\Sigma$, and no Hartree response of the background to the QP.
   - The "exact KS potentials" are thus exact *local* representations of a *nonlocal one-body* model, conceptually like OEP vs HF.
   - Calling the Fig. 5.8 barrier "the Hartree potential of the quasiparticle … included in the self-energy" [p. 138] is questionable, because $\Sigma$ as defined (Eqs 5.1–5.3) has no density dependence.
2. **Initial-state memory claim not supported by the body.**
   - Sec. 5.2 promises that "an alternative approach will be studied later in this section" [p. 134], and Sec. 5.4's plan mentions "a cursory examination of the initial-state-dependence" [p. 128]. Neither appears in the body.
   - Yet Ch. 6 states that slightly different initial states gave "starkly different" potentials [p. 165], and conclusion 5 states that initial-*charge* memory persists [p. 159]. This may have been in the PRL [ref. 147]. In this document, only the short initial-*current* memory (Sec. 5.7) is demonstrated.
3. **Softening parameter algebra.**
   - Eq. 5.34 gives $1/C=\pi d$, and the text gives $d=\frac{1}{2\pi}\sqrt{2E_{max}}$. Together these imply $1/C=\tfrac12\sqrt{2E_{max}}=\sqrt{E_{max}/2}$, i.e. $C=\sqrt{2/E_{max}}\approx1.155$ a.u. for $E_{max}=1.5$ Ha.
   - The thesis writes $C=\sqrt{E_{max}/2}\simeq0.866$ a.u. (Eq. 5.35), which is $1/C$ under its own definitions. Either the formula or the number is inverted.
   - This affects $W(k)$ and hence $\tau\simeq107$ a.u. [p. 150].
4. **Eq. 5.32** states "where $k_F$ is the Fermi momentum", but $k_F$ does not appear in the formula. As printed, the 1D Lindhard imaginary part is written with $k_F$ apparently set to 1 [p. 150].
5. **Conductance quantum in atomic units.** The thesis says $2e^2/h$ is "4π in atomic units" [p. 21]. With $e=\hbar=1$, $2e^2/h=1/\pi$.
6. **Label and caption slips:**
   - Fig. 4.1 y-axis says "n(z) (a.u.)" for a band energy [p. 110].
   - The VLB-to-additive switch is at $i=133$ in the text but $i=130$ in the Fig. 4.8 caption [pp. 117–118].
   - The Fig. 5.7 caption says $t\le125\Delta t$ but the plot runs to 150$\Delta t$ [p. 138].
   - Fig. 5.11 axis says "t (a.u.)", while the text says 0–125 *time steps* [p. 142].
   - Fig. 5.14 x-axis says "Energy E (a.u.)", while the chemical potentials are quoted in eV [p. 149].
   - Sec. 6.4 says the KS electric fields were studied "in Chapter 4"; they are in Ch. 5 [p. 166].
   - Sec. 6.3 refers to the "time-dependent KS potential" for the steady-state study [p. 164].
   - Electron counts change between sections: 22+1 (Ch. 4, where Sec. 4.2 speaks of a "23-electron charge density" [p. 111]), 20+1 (Secs 5.1–5.4), 21+1 (Secs 5.5–5.7). They are self-consistent, but easy to mix up.
7. **Grid and time step differ between Ch. 5 parts.** The TD wavepacket uses $\Delta t=4\times10^{-3}$ a.u. with no stated $\Delta x$; the decay study uses $\Delta x=0.2$ and $\Delta t=0.02$ a.u. Total simulated times are short (150$\Delta t$ = 0.6 a.u.; 60$\Delta t$ = 1.2 a.u.), much shorter than $\tau=107$ a.u. Fig. 5.6 shows the packet travelling about three cells, but the times of those panels are not given [pp. 132, 136, 151].
8. **Robustness claims are broader than the evidence.**
   - "Independent of the precise nature of the self-energy operator or the initial … states" [pp. 158–159] rests on one $\Sigma$ form, one parameter set for TD, and two initial-state preparations (decay case only).
   - Fig. 5.13's near-planar field map is one history with four snapshots.
9. **Unexplained sign alternation.** The early decay-inducing potentials alternate in sign between consecutive steps (about +650, +330, −320 eV…; read off Fig. 5.19, p. 154). This is not discussed. It could be physical (overshoot correction) or a symptom of RE/CN instability at $\Delta t=0.02$.
10. **Revised document.** The PDF cites a 2014 paper [ref. 163, J. Chem. Phys. 140, 18A515 (2014)] although the title page says May 2013 [pp. 1, 201]. The PDF is probably a post-submission or corrected version.
11. **No convergence analysis.** There is no proof or study of RE convergence (step size or mixing, uniqueness of the fixed point, sensitivity to $\Delta t$ and $\Delta x$). Stated accuracies are empirical, for these runs only.
