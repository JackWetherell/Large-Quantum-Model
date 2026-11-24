# Transformers, attention, and near-orthonormality

A short primer aimed at learning maps such as v_ext → n or n → v_xc on a 1D grid.

## 1. The transformer in one paragraph

The input is a set of **tokens**: here, grid points x_i carrying features such as n(x_i), v_ext(x_i) and an encoding of position. Each token is embedded as a vector h_i ∈ ℝ^d. The collection of h_i is the **residual stream**. A stack of L identical blocks then updates the stream additively:

```
h ← h + Attn(LN(h))     # mixes information *between* tokens (nonlocal)
h ← h + MLP(LN(h))      # transforms each token independently (local, pointwise)
```

Finally a read-out head maps each h_i to the output, for example n(x_i). The residual form h ← h + f(h) matters for two reasons. Each block writes a *small correction* into a shared d-dimensional workspace. And the Jacobian of each block is I + ∂f, so gradients pass through deep stacks without vanishing.

## 2. Attention

Each token emits a query, a key and a value: q_i = W_Q h_i, k_j = W_K h_j, v_j = W_V h_j. Then

  A_ij = softmax_j( q_i · k_j / √d_k ),  out_i = Σ_j A_ij v_j.

Multi-head attention runs H of these in parallel, each in a d_k = d/H subspace, and concatenates the results through W_O. There are two useful readings.

- **Differentiable key–value lookup (associative memory).** Token i asks a question q_i. Every token j advertises what it contains with k_j. The answer is a softmax-weighted blend of the values v_j. It behaves like a soft `dict` lookup, and it is a modern Hopfield network (Ramsauer et al. 2020).
- **A learned, input-dependent integral kernel.** In the continuum limit, out(x) = ∫ K[h](x, x′) v(x′) dx′. Compare the operators already in iDEA: Hartree, v_H(x) = ∫ v_int(x, x′) n(x′) dx′, and Fock exchange, (K̂φ)(x) = −∫ p(x, x′) v_int(x, x′) φ(x′) dx′. Attention generalises these to a kernel that is itself a functional of the input. This is the neural-operator view: Kovachki et al., JMLR 2023, show attention is a Monte-Carlo discretisation of a kernel integral operator.

**Why this matters for density functionals.** The exact maps are **nonlocal**. The exact KS potential has steps and peaks at points far from the density features that cause them, and there is a derivative discontinuity. LDA is exactly the local approximation, and it fails for exactly these effects. A CNN has a finite receptive field. A single attention layer has a global one, and for a few hundred grid points the O(N²) cost is negligible.

One practical point: attention is **permutation-equivariant**, so it knows nothing about geometry unless you supply it. Give it position explicitly, using x itself, Fourier features or RoPE. Otherwise it cannot tell neighbouring points from distant ones.

## 3. Why near-orthonormality matters

**The geometry.** For random unit vectors in ℝ^d, u·v has mean 0 and standard deviation 1/√d. So E\|cos\| = √(2/π)/√d. Measured with 20,000 random vectors (seeded, float64):

| d | 20,000 vectors = ×d | mean \|cos\| | max \|cos\| |
|---|---|---|---|
| 64 | ×312 | 0.100 | 0.65 |
| 256 | ×78 | 0.050 | 0.35 |
| 1024 | ×20 | 0.025 | 0.19 |
| 4096 | ×5 | 0.0125 | 0.09 |

Exact orthonormality caps you at **d** directions. If you tolerate overlaps of size ε, you can fit about exp(c·ε²·d) of them (Johnson–Lindenstrauss). That loosening is what transformers exploit in four places.

1. **Superposition in the residual stream.** The model can represent far more features than it has dimensions by assigning them nearly orthogonal directions. Reading feature m with its direction u_m returns the feature plus crosstalk Σ_{i≠m} aᵢ(uᵢ·u_m). That crosstalk stays small only if the overlaps are about 1/√d **and** the features are sparse (Elhage et al. 2022, *Toy Models of Superposition*).

2. **Clean retrieval in attention.** If the keys are nearly orthonormal, a query aligned with k_m scores about 1 on k_m and about ±1/√d on every other key. With enough sharpness the softmax picks out the right key. The linear-versus-softmax contrast below is the key result. A purely linear memory W = Σ vᵢkᵢᵀ has crosstalk error √((M−1)/d) and fails once M ≳ d. Softmax suppresses the crosstalk exponentially. Measured with d = 512:

   | memory | M | result |
   |---|---|---|
   | linear | 64 / 512 / 4096 | relative error 0.36 / 0.99 / 2.82 (theory 0.35 / 1.00 / 2.83) |
   | softmax, β = 32 | 4096 (8× overcomplete) | weight on correct key 1.000, top-1 accuracy 100% |

3. **The 1/√d_k in the logits.** This follows from the same statistics. If q and k have i.i.d. unit-variance entries, q·k has standard deviation √d_k (measured: 4.0, 8.0 and 16.0 for d_k = 16, 64 and 256). Without the scaling, the logits grow with width, the softmax saturates to one-hot, and gradients vanish. With it, the logits stay O(1) (measured standard deviation 1.00).

4. **Conditioning and signal propagation.** Near-orthogonal weight matrices (WᵀW ≈ I, singular values ≈ 1) preserve norms in the forward pass and gradients in the backward pass. This is dynamical isometry (Saxe et al. 2014; Pennington et al. 2017). It motivates orthogonal and scaled-Gaussian initialisation, and residual blocks, whose Jacobians are close to I. RoPE goes further: it encodes position with exact rotations, which are orthogonal, so q·k depends only on relative position and norms are untouched.

**Physics analogue.** KS orbitals are exactly orthonormal, ⟨φᵢ|φⱼ⟩ = δᵢⱼ, because they are eigenvectors of a Hermitian operator. iDEA gets this from `eigh` and rescales by 1/√dx. Exact orthonormality buys exact Pauli exclusion, but only N_x states fit on N_x grid points. Transformers trade exactness for capacity: approximate orthogonality gives exponentially many nearly independent directions at the price of controlled interference. If a model ever outputs orbitals directly, orthonormality must be imposed explicitly, for example by QR or Löwdin orthogonalisation. The network will not provide it.

## References

- Vaswani et al., *Attention Is All You Need* (2017).
- Elhage et al., *Toy Models of Superposition*, Anthropic (2022).
- Ramsauer et al., *Hopfield Networks Is All You Need* (2020).
- Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces*, JMLR (2023).
- Saxe, McClelland, Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear networks* (2014).
- Su et al., *RoFormer: Rotary Position Embedding* (2021).
