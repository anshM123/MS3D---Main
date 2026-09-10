# MS3D — Mishra–Senthilkumar 3D
## Complete statement of the algorithm, equations, theorems, laws and findings

Ansh Mishra, Aryan Senthilkumar

Everything below is stated at the strength the evidence supports, with the
governing measurement attached. Where a claim was corrected or retracted, that
is marked in place rather than removed.

---

# PART I — SETTING

## I.1 The system

Spectrally discretised incompressible Navier–Stokes on a periodic box:

```
U̇ = F(U),   F(U) = D[ D(FFT(u × ω(u))) − ν K² D(U) ],   D = dealias ∘ Leray
```

`F` is **exactly quadratic** in `U`. Consequences used throughout:

* central finite differencing of `F` has **zero truncation error**, so
  `F(U+εv) − F(U−εv) = 2ε·DF(U)v` exactly for any ε;
* the entire FD residual is floating-point roundoff, amplified by `1/(2ε)`.

Verified: fp32/fp64 residual ratio **5.385e8** against a machine-epsilon ratio
of **5.369e8** — 0.3% agreement across 12 flow/viscosity cases.

## I.2 The observable split

A fixed orthogonal projector `P` selects the observable coordinates;
`Q = I − P`. Any error decomposes exactly:

```
e = Pe + Qe
```

`P` is the `mixed` 24-mode set — 12 conjugate pairs at `|k| ∈ {1, √2, 4.47,
4.58, 5}` — fixed in **absolute** wavenumber, independent of grid size. At
N = 40 it covers 6.25e-5 of the state; at N = 96, 4.5e-6.

**Conjugate-pair structure matters.** The 24 modes are 12 ± pairs. Any
construction that treats them as independent complex directions destroys
Hermitian symmetry, so the reconstructed field is not real. Measured: a single
unit vector gives imag/real = **1.000**; the paired cos/sin construction gives
**0.00e+00**. This caused one retraction (§VII.3).

## I.3 The reduced model (STAR2)

Augment `P` with two hidden closure directions built from the local trajectory
jet:

```
f   = F(U₀)
q₀  = Qf / ‖Qf‖
a   = DF(U₀) f
q₁* = R₁a / ‖R₁a‖,    R₁ = I − P − q₀q₀*
Π   = P + q₀q₀* + q₁*q₁**
U̇_r = Π F(U_r)
```

"Rank 2" is the number of extra **hidden** directions; the resolved state
carries hundreds of coordinates.

**Time convention.** Segment `τ = 0.15`, 20 RK4 substeps, `TAU0 = 40/3`,
`dt = τ·TAU0/20`. Interventions at `τ ∈ {0.90, 1.20, 1.50}`, forecast horizon
2 segments.

---

# PART II — THEOREMS

## Theorem A — dimension-minimal complement jet augmentation

**Statement.** Let `f = F(U₀)`, `a = DF(U₀)f`. For a projection ROM whose first
two trajectory derivatives match the full system at `U₀`, the augmented space
must contain `f` and `a`; with `P` fixed, its hidden component must contain
`Qf` and `Qa`. The minimum hidden dimension is therefore

```
dim span{Qf, Qa}
```

and STAR2 realises exactly this span.

**Proof.** `U'(0) = f`, `U''(0) = a`. For `Π ⊇ P`, `U_r'(0) = Πf = f` iff
`f ∈ range(Π)`; given that, `U_r''(0) = Π·DF(U₀)·U_r'(0) = Πa = a` iff
`a ∈ range(Π)`. Since `P ⊆ Π` is fixed, the free part must contain `Qf, Qa`. ∎

**Status.** Correct, small, low prior-art exposure. STAR2 is not an empirically
lucky second mode — it is the dimension-minimal complement matching velocity
and acceleration.

### Lemma A.1 (exact, originally recorded as numerical)

With `q₁^old ∝ R₁Aq₀`, `q₂* ∝ R₂Af`, `R₂ = R₁ − q₁^old q₁^old*`:

```
R₁ = R₂ + q₁^old q₁^old*
⟹  R₁Af = q₁^old⟨q₁^old, Af⟩ + R₂Af
⟹  q₁* ∈ span{q₁^old, q₂*}    identically
```

**Provenance.** Both operand definitions are traceable to source:
`q1_star = normalized (I − P − q0 q0*) A f` and `r2 = R2 A f`. The relation
was originally recorded as holding "to numerical precision"; it is exact.

**Stated at the strength the algebra supports:** STAR2 isolates in one hidden
coordinate the acceleration-matching direction *contained within* OLD3's
two-dimensional correction subspace. It does **not** follow that OLD3 needs two
dimensions for every purpose.

## Theorem B — observable superconvergence

**Statement.** If a fixed reduced space `S ⊇ P` contains the first `r`
trajectory derivatives, then

```
P[U(t) − U_r(t)] = O(t^(r+2))
```

**Proof.** By induction `U_r^(j)(0) = U^(j)(0)` for `j ≤ r`. The first possible
mismatch is `e^(r+1)(0) = (I − Π)U^(r+1)(0) = R·U^(r+1)(0)`. Since `P ⊆ Π`,
`PR = 0`, so `P e^(r+1)(0) = 0`: the resolved error cannot appear until one
derivative later. ∎

For STAR2, `r = 2`, giving **O(t⁴)** against OLD2's O(t³).

**Explicit leading coefficient.** With `A = DF(U₀)`, `B = D²F(U₀)`,
`g₃ = A²f + B[f,f] = U'''(0)`:

```
P[U(t) − U_r(t)] = (t⁴/24)·P·DF(U₀)·R·[ DF(U₀)²F(U₀) + D²F(U₀)[F(U₀),F(U₀)] ] + O(t⁵)
C₄ = P A R g₃
```

This connects the framework end to end: error is **born** in `Q`, and `DF`
transports it toward `P` — the mechanism later measured directly.

**Numerical confirmation (fp64, analytic Jacobian):**

| flow | p (FD) | p (analytic) | R² | Δp |
|---|---|---|---|---|
| vortex_ring | 3.9778 | 3.9778 | 1.0000 | −0.0000 |
| periodic_shear | 3.9991 | 3.9991 | 1.0000 | +0.0000 |
| skew_tubes | 3.9646 | 3.9646 | 1.0000 | +0.0000 |
| mixed_vortices | 3.9788 | 3.9788 | 1.0000 | −0.0000 |

**The first correction term is measured, not just fitted.** Moving the fit
window 10× earlier shrinks the deviation from 4 by 10×:

| flow | \|p−4\| at window A | \|p−4\| 10× earlier | ratio |
|---|---|---|---|
| vortex_ring | 0.0222 | 0.0023 | 9.7 |
| skew_tubes | 0.0354 | 0.0034 | 10.4 |
| mixed_vortices | 0.0212 | 0.0018 | 11.8 |

exactly as `P-error = Cτ⁴(1 + aτ + …)` predicts.

**Status.** Correct and confirmed to an unusual standard — and the most exposed
on novelty (Boreale & Collodi; derivative-enriched POD). **Do not lead with it.**

## Theorem C — defect-driven second-order error transport

**Statement.** With `Φ_h` the full flow map, `V_k` the ROM state,
`e_k = U_k − V_k`, define the **defect**

```
η_k = Φ_h(V_k) − V_{k+1}
```

which requires no knowledge of the truth. Then exactly

```
e_{k+1} = η_k + Φ_h(V_k + e_k) − Φ_h(V_k)
```

and by Taylor expansion, `J_k = DΦ_h(V_k)`, `H_k = D²Φ_h(V_k)`:

```
e_{k+1} = η_k + J_k e_k + ½H_k[e_k, e_k] + ρ_k ,   ‖ρ_k‖ ≤ (M₃,k/6)‖e_k‖³
```

giving the truth-free recurrence

```
ê_{k+1} = η_k + J_k ê_k + ½H_k[ê_k, ê_k]
```

**Status.** The identity is elementary and **not novel** — see §VIII.1. The
contribution is not the expansion but its use: reconstructing the error
*vector*, extracting `Qê`, and acting on it.

**Measured validity horizon.** Test V put the median second-order residual at
1.005e-4 at percent-level amplitudes. Beyond roughly **15% resolved error** the
`H[e,e]` term dominates and `ê` diverges — one anchor reached
`‖Qê‖/‖V‖ = 25`, producing a state where STAR2 is degenerate. Forced dynamics
reach that bound faster than decaying ones.

## Theorem D — hidden-only correction criterion

**Statement.** For any `c ∈ range(Q)`:

```
P(V + c) = PV        exactly
```

With `b = P[Φ_h(V) − Φ_h(U)]` and `Δ(c) = P[Φ_h(V+c) − Φ_h(V)]`, the
correction improves the forecast **iff**

```
2⟨b, Δ(c)⟩ + ‖Δ(c)‖² < 0
```

**Proof.** `‖b + Δ‖² − ‖b‖² = 2⟨b,Δ⟩ + ‖Δ‖²`. ∎

**Consequence.** There can be no theorem asserting that *any* hidden correction
helps. For small `c`:

```
Δ(+c) = +Tc + ½H[c,c] + O(‖c‖³)
Δ(−c) = −Tc + ½H[c,c] + O(‖c‖³)
```

Reversing the correction flips the leading directional transfer while leaving
the curvature unchanged — which is why the `+c` vs `−c` control isolates
whether the estimate carries causal directional information.

### Corollary D.1 — optimal correction amplitude

**Scope.** Theorem C established `Δ(αq) = αg₁ + (α²/2)g₂ + O(α³)`, so the
response is **not** linear in amplitude. The closed form below assumes linear
response and is therefore the linear-response optimum, `α*_lin`.

**Statement (linear response).** Decompose `b = b_PP + b_QP` into the part
carried by resolved-error persistence and the part leaked from the hidden
block. Under linear response the corrected error is `‖b_PP + (1−α)b_QP‖`,
minimised at

```
α*_lin = 1 + ⟨b_PP, b_QP⟩ / ‖b_QP‖²
```

**Nonlinear refinement.** With `Δ(1)` and `Δ(2)` measured:

```
g₁ = 2Δ(1) − ½Δ(2),   g₂ = Δ(2) − 2Δ(1)
α*_quad = argmin_α ‖b + αg₁ + (α²/2)g₂‖
```

**Recoverable from the pipeline:** with `d = Δ(1)`, `b_QP = −d`, `b_PP = b + d`.

**Interpretation.** `α* > 1` exactly when persistence error and leakage error
are positively aligned — overcorrecting in `Q` then also cancels `P`-error that
the correction is structurally forbidden from touching directly.

**Confirmation.**

| quantity | value |
|---|---|
| median cos(b_PP, b_QP) | +0.8196 |
| cos > 0 | 36/36 windows |
| median ‖b_PP‖/‖b_QP‖ | 1.3387 |
| α\*_lin predicted | 2.0289 |
| α\*_quad predicted | **1.9850** |
| α\* measured (independent sweep) | **2.0000** |

Agreement to **0.75%**. Per-window the prediction lands within 35% of the
measured argmax in 30 of 36 windows. **Caveat:** accurate for `α* ≲ 3`;
predicted 6.02 vs measured 4.0 at the extreme, since `g₁, g₂` are fitted at
α = 1, 2 and extrapolated.

**Prior-art status.** Nothing comparable found. Probably the single most novel
theoretical item in the project, and it made a confirmed quantitative
prediction.

## Theorem E — the estimate→intervention bridge

**Statement.** Let `q = Q(U − V)`, `q̂` its estimate. Assume the projected
finite-time flow is locally Lipschitz, `‖PΦ_h(x) − PΦ_h(y)‖ ≤ L_P‖x − y‖`.
Then

```
‖PΦ_h(V + q̂) − PΦ_h(V + q)‖ ≤ L_P‖q̂ − q‖
E_estimated ≤ E_oracle + L_P‖q̂ − q‖
```

so if `E₀ − E_oracle > L_P‖q̂ − q‖`, then `E_estimated < E₀`. ∎

**Empirical probe.** Degrading `q̂` with random `Q`-noise:

| δ | median gain | win rate |
|---|---|---|
| 0.03 | 1.6421 | 100% |
| 0.10 | 1.6247 | 100% |
| 0.30 | 1.6258 | 100% |
| 1.00 | 1.6133 | 100% |

A 33× range of directional error moves the gain by 2%. In a `Q` of dimension
~10⁴–10⁵ a random perturbation has overlap `≈ √(3/n)` with a rank-3 leakage
subspace, so almost none survives projection through `T`. **Directional
insensitivity and low-rank leakage are the same fact.** Amplitude sensitivity
(D.1) is strong precisely because amplitude is what *does* survive.

## Theorem F — correction reachability

**SCOPE — read first.** Theorem F concerns the **ROM flow map** `Ψ_h^S`, NOT
the full-order flow map `Φ_h`. The proof relies on the state increment lying
inside the fixed reduced subspace, a property of the projection ROM and nothing
else. **Theorem F does not constrain the full-order causal response.**

**Statement.** For an anchored projection ROM with reduced subspace `S`, one
segment from anchor `A` returns `A + s`, `s ∈ S`. For `c ∈ range(Q)`:

```
Ψ_h^S(V + c) − Ψ_h^S(V) = c + s_c ,   s_c ∈ S
Δ(c) = P[Ψ_h^S(V+c) − Ψ_h^S(V)] = P s_c  ∈  P·S
```

for **every** hidden correction `c`. With `ρ = ‖Π_{P·S} b‖/‖b‖`:

```
gain ≤ 1 / √(1 − ρ²)
```

**What does not appear in that bound:** the reservoir ratio, the ROM's
accuracy, or the basis for the rest of `S`.

**Exactness.** The argument needs `S` identical on both branches — exact for
static-basis variants over any number of segments, approximate where the basis
is rebuilt per anchor.

**Confirmation.** 20 unseen trajectories × 5 ROM variants × 3 anchors:

| var | subspace | dim(P·S) | ρ | ceiling | gain |
|---|---|---|---|---|---|
| A | POD only (frozen) | 10/288 | 0.766 | 1.555 | 1.0544 |
| B | POD + all of P | 144/288 | 1.0000 | ∞ | 1.5834 |
| C | POD + P + q₀ | 144/288 | 1.0000 | ∞ | 1.5978 |
| D | POD + P + q₀ + q₁\* | 144/288 | 1.0000 | ∞ | 1.5849 |
| E | P + q₀ + q₁\* (STAR2 subspace) | 144/288 | 1.0000 | ∞ | 1.5639 |

**300 windows (20 unseen initial conditions x 5 variants x 3 intervention
times), zero violations**, constraining the truth-free and oracle corrections
separately in each — 600 comparisons — since the bound covers *every* hidden
correction.

Variant A is **reachability-limited**: only 77% of its observable error is
reachable by any hidden correction, and it achieves 70% of a hard ceiling.
B–E are **dynamics-limited**: ρ = 1 removes the constraint and they land at
1.56–1.60 while their reservoir ratio spans 4.5–25.0 and baseline error spans
0.025%–5.3%. **ρ tracks the gain; R and accuracy do not.**

**Design condition.** `P ⊆ S` removes the reachability constraint. Nothing
else in the ROM does.

**Sharper bound, untested.** `P·S` is the *geometric* reachable set. Under
local linear response with `T = P·DΨ_h^S(V)·Q`, only `Range(T)` is attainable:

```
Range(T) ⊆ P·S ⊆ P
```

giving a tighter ceiling `1/√(1−ρ_T²)`. Finite-amplitude version:
`G* = ‖b‖ / dist(−b, M_V)` with `M_V = {Δ(c) : c ∈ range(Q)}`. This would
unify Theorem F with the rank-3 leakage result and Corollary D.1 as one theory
rather than three findings. **Not yet derived or tested.**

---

# PART III — THE EFFECT-SIZE LAW

## III.1 The identity

The gain is *exactly* determined by two measurable quantities. With `b` the
baseline future observable error and `Δ` the oracle's observable effect:

```
gain = ‖b‖ / ‖b + Δ‖ = 1 / √(1 + 2λ cos θ + λ²)

  λ     = ‖Δ‖/‖b‖                 how large the hidden contribution is
  cos θ = ⟨b,Δ⟩/(‖b‖‖Δ‖)          how well it opposes the existing error
```

**This is algebra, not a model.** Neither the reservoir ratio `R` nor the
reachability fraction `ρ` appears in it — which is why three correlational
hypotheses failed before it.

**Verified across four systems:**

| system | max \|predicted − observed\|/observed |
|---|---|
| Kuramoto–Sivashinsky | 1.97e-15 |
| Navier–Stokes | 2.31e-14 |
| complex Ginzburg–Landau (CUBIC) | 5.38e-12 |
| finite-difference KS + point sensors | 1.12e-15 |

## III.2 Measured values

| system | λ | cos θ | gain | reduction |
|---|---|---|---|---|
| Navier–Stokes | 0.4445 | −0.9412 | 1.6510 | 39.4% |
| KS, unstable band | 0.0891 | −0.4118 | 1.0350 | 3.4% |
| KS, damped band | 1.0298 | −0.9996 | 18.63 | 94.6% |
| FD + point sensors | 0.3909 | −0.9937 | 1.8811 | 46.8% |

**Within Navier–Stokes**, λ varies 0.236–0.813 across the twelve cases and
`Spearman(λ, gain) = +0.986`, with predicted gain matching the oracle to four
decimals in 12/12. The law explains case-to-case variation inside one system,
not only differences between systems.

**Where cos θ turns positive the correction hurts.** In KS's near-neutral band
cos θ = +0.21 and the gain is 0.966 while the wrong-sign arm gives 1.097. The
law predicts the sign flip, not merely the magnitude.

## III.3 What governs λ — partially resolved

**κ, the hidden-driven share.**

**Notation, stated precisely because an earlier draft was ambiguous.** Write
the two *finite-amplitude* block responses, each measured by running the ROM
forecast from the perturbed state:

```
Δ_Q = P[ Φ_ROM(V + Qe) − Φ_ROM(V) ]        exact, not linearised
Δ_P = P[ Φ_ROM(V + Pe) − Φ_ROM(V) ]        exact, not linearised
```

`λ` is defined **once and identically everywhere in this document**:

```
λ = ‖Δ_Q‖ / ‖b‖
```

which is the same `Δ` that appears in the identity of §III.1. The linear-response
operators `A_PQ`, `A_PP` are the *first-order approximations* to `Δ_Q`, `Δ_P`;
they are never used as the measured quantity. Blocking the tangent map,

```
b   ≈ Δ_P + Δ_Q          (with the local defect term neglected)
κ   = ‖Δ_Q‖ / ‖Δ_P‖      the hidden-driven share
λ   ≈ κ / √(1 + 2κc + κ²),   c = cos(Δ_Q, Δ_P)
```

The last line is an **approximation** — it drops the defect's own contribution
to `b` — whereas §III.1's identity is exact. That distinction is the reason the
closed form fails at large κ while the identity does not.

Both block responses are ROM probes, so κ is computable **truth-free**:
`κ̂ = ‖Δ̂(Qê)‖ / ‖Δ̂(Pê)‖`.

**Measured, n = 96:**

| band | κ | λ | λ_pred | reservoir |
|---|---|---|---|---|
| unstable | 0.110 | 0.0752 | 0.1005 | 0.477 |
| neutral | 0.407 | 0.2652 | 0.4336 | 0.440 |
| stable | 611.1 | 1.1961 | 1.0004 | 8.552 |
| damped | 2654.4 | 1.0445 | 1.0000 | 23.673 |

* **`Spearman(κ̂, κ) = +1.0000`** — the truth-free predictor is not an
  approximation of the diagnostic, it *is* the diagnostic.
* κ orders λ correctly across a 24000× range and beats the reservoir ratio
  (0.839 vs 0.721), which was the previous candidate.
* **But the closed form fails as registered:** median `|λ_pred − λ|` = 0.0829
  against a 0.05 threshold, and adding the neglected `η_P` term made it
  *worse* (0.236 vs 0.216 error in reconstructing `‖b‖`). Thresholds were not
  moved.

## III.4 The validity boundary — modal vs pointwise observables

The law's reconstruction accuracy splits sharply by **observation operator**:

| observable | reconstruction error |
|---|---|
| Fourier modes (KS + CGL) | **0.0354** |
| point sensors (FD) | **0.6027** |

**Isolated to the observable, not the discretisation.** One spectral solver,
one equation, one discretisation, matched dimension, only the observable
changed:

| observable | NDIM=6 | NDIM=12 | NDIM=24 |
|---|---|---|---|
| MODES | 0.0052 | **0.0028** | 0.0049 |
| SENSORS | 0.4976 | 0.5552 | 0.4509 |
| MIXED (half and half) | 0.4507 | 0.4884 | 0.5791 |

**200× separation**, with no finite differences anywhere. The confound between
observation operator and discretisation is broken: the exception belongs to the
observable.

### The dose-response curve

An earlier draft of this document concluded from the 50/50 MIXED case that the
failure was "binary, not graded — a single pointwise functional is enough."
**That claim was wrong and is retracted.** A 50/50 comparison cannot separate a
switch from a steeply saturating curve. Sweeping the mixing ratio at fixed
total dimension (24 directions, n = 160) gives the curve:

| n_sensors | fraction of the observable set | reconstruction error | share of saturated damage |
|---|---|---|---|
| 0 | 0.000 | **0.0049** | — |
| 1 | 0.042 | 0.2231 | 49% |
| 2 | 0.083 | 0.3066 | 68% |
| 3 | 0.125 | 0.3777 | 84% |
| 6 | 0.250 | 0.5133 | ~100% |
| 12 | 0.500 | 0.5352 | saturated |
| 18 | 0.750 | 0.4160 | saturated |
| 24 | 1.000 | 0.4509 | saturated |

**The failure is GRADED and saturates early.** It rises monotonically through
four points before plateauing: one pointwise functional among twenty-three
modal ones already costs half the saturated error, three cost 84%, and six —
a quarter of the observable set — cost all of it.

**The statement for the manuscript:**

> The reconstruction error of the effect-size law rises steeply with the
> pointwise fraction of the observable set, reaching half its saturated value
> at a 4% pointwise fraction and saturating by 25%. A predominantly modal
> observable set is not sufficient; the law requires an essentially wholly
> modal one.

**A second effect, not anticipated.** `λ` itself falls monotonically with the
pointwise fraction — 0.9201, 0.6537, 0.6126, 0.4783, 0.3747, 0.3329 — so
adding pointwise functionals does not merely break the *prediction* of the
effect size, it genuinely reduces the share of observable error that is
hidden-driven. The observation operator changes the physics being measured,
not only its representability.

**What this costs elsewhere in the record.** Two candidate mechanisms for the
pointwise exception were rejected (§III.4, below), and part of the stated
reasoning was that both predicted graded behaviour while the data appeared
binary. **That argument is void.** The data are graded. Neither mechanism
automatically revives — each also failed on its own statistics, with
`Spearman(dimension, p) = +0.067` and `Spearman(ℓ, err) = +0.180` against
`+0.175` for horizon alone — but the binary-versus-graded ground for rejecting
them was built on an overstatement and should not be relied on.

### The mechanism — directional misrepresentation

Testing linearity directly, `r(s) = ‖Δ(sq)‖/(s‖Δ(q)‖)`, n = 280:

| scale | MODES r | MODES cos | SENS r | SENS cos |
|---|---|---|---|---|
| 0.1000 | 0.4184 | 0.9802 | 0.9828 | 0.5521 |
| 0.0300 | 0.4050 | 0.9738 | 0.9540 | 0.5718 |
| 0.0100 | 0.4017 | 0.9718 | 0.9461 | 0.5772 |
| 0.0030 | 0.4005 | 0.9711 | 0.9433 | 0.5766 |
| 0.0010 | 0.4911 → conv. | **0.9653** | 1.2308 → conv. | **0.6845** |

`r(s)` **converges for both**, so a linear limit exists in both cases — the
pointwise response *is* differentiable. The difference is **direction**:

```
cos( Δ(s·q), Δ(q) )  as s → 0
   MODES    +0.965
   SENSORS  +0.685
```

**The linearisation exists but points the wrong way.** For modal observables
the infinitesimal response is nearly parallel to the finite-amplitude one, so
the derivative predicts it. For pointwise observables they are far apart: the
derivative is well defined and simply describes a different regime. Shrinking
the perturbation cannot rescue the prediction.

Note also `r → 0.49` for MODES: the finite-amplitude response is *twice* the
linear prediction, strongly nonlinear in **magnitude**, yet the law works.
SENSORS is closer to linear in magnitude (1.23) and the law fails.
**Magnitude nonlinearity is harmless; directional rotation is fatal.**

### Two mechanisms proposed and refuted

Recorded because they constrain what an explanation can look like:

1. **p set by response alignment and observable dimension.** Refuted at
   n = 224: `Spearman(dimension, p) = +0.067`, and |c| vs dimension gave
   r = −0.190 against a predicted −0.5.
2. **Advective locality**, `ℓ = u_rms·T_horizon / sensor spacing`. Refuted at
   n = 160: `Spearman(ℓ, err) = +0.180` against `+0.175` for horizon alone and
   `+0.010` for spacing — no collapse onto the combined group, and the
   lowest-ℓ quartile did not recover the modal regime.

Both predicted **graded** behaviour, and an earlier draft rejected them partly
on the grounds that the observed behaviour was binary. **The observed behaviour
is in fact graded** (see the dose-response curve above), so that ground is
withdrawn. Each rejection stands on its own statistics — the dimension
correlation and the ℓ-collapse test respectively — and on nothing else.

---

# PART IV — THE ALGORITHM

## IV.1 Truth-free error forecast

The W recurrence run forward to the forecast horizon supplies `b̂` using only
the model's own defect history:

```
for each segment k:
    Φ    = Φ_h(V_k)                             (or a COARSE surrogate)
    η    = Φ − V_{k+1}
    ε    = clip( DERIV_REL·‖V_k‖ / ‖ê‖ , 1e-6, 5e-2 )
    ê    = η + [Φ_h(V+εê) − Φ_h(V−εê)]/(2ε)
             + ½[Φ_h(V+εê) + Φ_h(V−εê) − 2Φ]/ε²
b̂ = −P·ê|_{k+h}
```

**Sign convention matters:** `ê` estimates `U_truth − V` while `b` is defined
as `P(forecast − truth)`. Opposite signs. Getting this wrong makes the
algorithm select α = 0 while reporting a perfect-looking `cos = −1.0000`.

**Measured accuracy:** `cos(b̂, b_true) = +1.0000`, magnitude ratio 1.0000–1.0035.

## IV.2 Candidate set and selection

```
candidates:  0                do nothing
             α·Qê             hidden-only        — leaves the observable exact
             α·Pê             observable-only
             α·ê              full state         — see the degeneracy warning
α ∈ {0.5, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0}

Δ̂(c) = P[ Φ_ROM(V + c) − Φ_ROM(V) ]           ROM forecasts only, no truth
pick  = argmin_c ‖ b̂ + Δ̂(c) ‖
Ĝ     = ‖b̂‖ / ‖b̂ + Δ̂(pick)‖                  predicted gain, truth-free
```

**Trust gate.** Act only if `Ĝ > GATE` and `‖Qê‖/‖V‖ < HORIZON` (~0.15), so a
closed loop cannot drive itself past the amplitude horizon of Theorem C.

## IV.3 Performance

**Amplitude selection alone (KS, n = 80 per band):**

| variant | probe cost | captured | p10 |
|---|---|---|---|
| model-free probe | 12 forecasts | **1.000** | 1.000 |
| 2-point quadratic fit | 2 forecasts | 0.605 | 0.036 |
| small-amplitude fit | 2 forecasts | 0.000 | −2.734 |

The probe captures **100% of hindsight-optimal benefit** with α̂\* matching α\*
exactly in 100% of windows. The cheap approximations do not work — the 2-point
fit has 180% model error in the damped band.

**Harm avoidance.** In the KS unstable band, fixed α = 1 lost on 4.2% of
windows (worst 0.9712). Truth-free α̂ eliminated all of them: **win rate
95.8% → 100.0%, worst 1.0000.**

**Full adaptive selection (KS, n = 60 per band):**

| band | Q-only | P-only | full-state | 3DVar *(uses truth)* | **adaptive** | picked |
|---|---|---|---|---|---|---|
| unstable | 1.05 | 3.11 | 3.25 | 3.03 | **14.09** | P=51, full=9 |
| damped | 16.34 | 1.00 | 12.02 | 1.28 | **16.65** | Q=51, full=9 |

**Wins in both regimes and beats ensemble 3DVar in both, using no external
data.** The selector picks P-touching corrections where λ ≈ 0.088 and Q-only
where λ ≈ 1.042 — without ever seeing λ or the truth.

**Decomposition, which must be reported.** With α fixed at 1, adaptive drops to
3.28 in the unstable band (vs 3.25 best fixed) and stays 16.65 in the damped
band. So **subspace selection buys regime-correctness (1–2%); amplitude
selection buys the magnitude (4.3×).** Attributing the unstable-band gain to
subspace choice alone would be wrong.

**Applicability prediction.** `Ĝ` predicts the realised gain:

| system | Spearman(Ĝ, G) | median \|Ĝ/G − 1\| |
|---|---|---|
| KS unstable | +1.0000 | 0.0000 |
| KS damped | +0.9977 | 0.0035 |
| CGL (cubic, unseen) | +0.9997 | 0.0001 |
| FD + point sensors | +1.0000 | 0.0000 |
| 3D Navier–Stokes | +1.0000 | — |

**In 3D Navier–Stokes:** adaptive **3.8106** against best fixed 2.1190 and the
published X-v3 1.6510. `ΔP = 0.000e+00` on all Q-picks. The selector split
P on 26 windows (λ = 0.4201) and Q on 10 (λ = 0.6941).

---

# PART V — THE CAUSAL EXPERIMENT

## V.1 Protocol

Four branches from an **identical observable state**:

```
BASELINE     V
ESTIMATED    V + Qê       truth-free
WRONG SIGN   V − Qê       same magnitude, opposite direction
ORACLE       V + Qe       the true hidden error — the ceiling
```

The design feature: `P(V + Qê) = PV` with mismatch **0.000e+00** — exactly
zero, not approximately. Nothing measurable has changed. Each branch is then
evolved independently and the future observable error compared.

## V.2 Result (Test X-v3, fp64 + analytic Jacobian, 36 interventions)

| quantity | value |
|---|---|
| median EST-Q gain | **1.650998×** |
| 10th percentile | 1.299859× |
| median error reduction | **39.417%** |
| win rate vs baseline | **36/36** |
| beats wrong-sign | **36/36** |
| median wrong-sign gain | 0.6759× |
| oracle benefit captured | 100% |
| max start-P mismatch | **0.000e+00** |
| all six preregistered gates | PASS |

**What this establishes.** Not "a corrected ROM performs better," but: at the
intervention instant nothing observable has changed, yet modifying only the
hidden state changes future observable accuracy — with the sign of the change
determined by the sign of the correction.

## V.3 Truth-free reconstruction (Test W, 132 anchors)

| quantity | value |
|---|---|
| median hidden-Q residual | **1.06e-4** |
| median improvement from the curvature term | 44.8× |
| anchors improved | **132/132** |
| median hidden-direction cosine | 0.99999999 |
| median amplitude ratio | 1.0000108 |

---

# PART VI — STRUCTURE OF THE LEAKAGE

## VI.1 The operator

```
T_k = P · DΦ_h(U_k) · Q
```

Take a hidden perturbation, push it through the flow, read off what lands in
the observables.

## VI.2 Low-rank concentration

Restricted to the **empirically occupied** hidden-error subspace
`span{Qe(t₁),…,Qe(t₉)}` — not arbitrary directions in `Q`:

| config | dim(Q) | ν | empirical rank | top-3 energy |
|---|---|---|---|---|
| N40 | 383976 | 2.5e-3 | 9 | 0.8906 |
| N64 | 1572840 | 2.5e-3 | 9 | 0.9698 |
| N80 | 3071976 | 2.5e-3 | 9 | 0.9544 |
| N64 | 1572840 | 1.0e-3 | 9 | 0.9331 |
| N80 | 3071976 | 4.0e-4 | 9 | 0.8936 |
| N96 | 5308392 | 1.56e-4 | 9 | 0.9190 |

Mean **0.9268**, sd 0.0320, `Spearman(dim Q, top-3) = −0.029` across a 14×
growth in dim(Q) and 16× drop in ν.

**Non-circularity.** Doubling the ambient subspace from 4 to 9 dropped top-3 by
0.052; a uniform spectrum would have fallen 0.417, eight times more.

**Structure.** top-1 is only 0.37–0.44 and `σ₁/σ₃` is 1.29–1.66 — **three
comparably weighted channels**, not one dominant direction.

**Cross-regime, on a matched 9-dimensional ambient:**

| flow | ν | r_leak90 | top-3 |
|---|---|---|---|
| skew_tubes | 2.5e-3 | 3 | 0.9547 |
| skew_tubes | 1.0e-3 | 3 | 0.9285 |
| skew_tubes | 4.0e-4 | 3 | 0.9161 |
| mixed_vortices | 2.5e-3 | 4 | 0.8171 |
| mixed_vortices | 1.0e-3 | 4 | 0.8134 |
| mixed_vortices | 4.0e-4 | 5 | 0.7828 |

Decaying flows gave 3 and ~0.93 on the same ambient. **Leakage rank is 3–5
across trajectory ranks spanning 8 → 35, in both decaying and forced
dynamics.**

## VI.3 Energy ≠ danger — with a credit

The hidden directions that matter are not those carrying the most error. The
gate that "failed" in Test R was testing the wrong thing: the three dominant
leakage directions held >50% of input `Q` energy.

**This is the adjoint-weighting insight and must credit that lineage** —
dual-weighted-residual estimation exists precisely because residual magnitude
does not predict output error.

---

# PART VII — COST

## VII.1 Coarse-grid defect estimation

The defect can be computed on a coarse grid:

```
η_M = Prolong_{M→N}[ Φ_h^M( Restrict_{N→M} V_k ) ] − V_{k+1}
```

Spectral restriction/prolongation verified exact (round-trip 0.0 to 3e-16).

**Required coarse resolution does not grow with N — it falls:**

| case | N=40 | N=64 | N=80 | trend |
|---|---|---|---|---|
| vortex_ring ν=0.25 | 28 | 14 | 10 | decreasing |
| vortex_ring ν=1 | 28 | 14 | 14 | decreasing |
| mixed_vortices ν=0.25 | 28 | 20 | 14 | decreasing |
| mixed_vortices ν=1 | 28 | 20 | 20 | decreasing |
| skew_tubes ν=0.25 | 14 | 10 | 10 | decreasing |
| skew_tubes ν=1 | 14 | 10 | 10 | decreasing |
| periodic_shear ν=0.25 | 14 | 14 | 14 | flat |
| periodic_shear ν=1 | 14 | 14 | 14 | flat |

**Zero cases out of eight where M\* grows.**

**Why:** `k_leak` sits at 4.5–8.2 across every topology and viscosity
(median 5.4, CV 0.19), set by the resolved band (`k_P ≈ 3`) plus the
energy-containing scales (`k_flow ≈ 1–2.3`) — **not** by the dissipation range.
The leakage channel never moves to small scales however the grid is refined.

## VII.2 Wall-clock — the nominal model is wrong

| N | M | nominal `3(M/N)³` | fine solver | **measured** | ratio |
|---|---|---|---|---|---|
| 40 | 28 | 1.029 | 238 ms | **3.061** | 2.98× |
| 40 | 20 | 0.375 | 238 ms | 3.056 | 8.15× |
| 64 | 28 | 0.251 | 1080 ms | **0.711** | 2.83× |
| 80 | 28 | 0.129 | 2187 ms | **0.336** | 2.62× |
| 80 | 20 | 0.047 | 2187 ms | 0.344 | 7.35× |

**Estimator wall time is essentially constant** (728–805 ms) regardless of M.
Per flow-map call: 6.07, 6.06, 6.09 ms at M = 28, 20, 14 — identical. The
coarse work is **latency-bound, not FLOP-bound**. The fine solver is genuinely
FLOP-bound (5.95, 27.0, 54.7 ms per call at N = 40, 64, 80, tracking N³ to
within 15%).

**The correct cost model is a FLOOR, not a ratio.** The estimator costs a
roughly fixed ~750 ms; the saving is whatever the solver costs above that.

| N | measured cost vs one solver segment |
|---|---|
| 40 | 3.06× — **more expensive** |
| 64 | 0.711× |
| 80 | **0.343×** |
| 128 | ~0.084× (extrapolated) |
| 192 | ~0.025× (extrapolated) |

**Retracted:** the "21× cheaper at N=80" figure. Measured is 2.9×. The nominal
model overstates the saving by 2.6–24×. The concern that restrict/prolong would
dominate was also wrong — it is 0.5–0.6% of estimator time.

**Practical result.** At N=80, M=28 and M=20 cost the same wall time (736 vs
753 ms), so coarsening below M=28 buys nothing — and 28 is exactly the
universally safe `M*`. **Use M = 28.**

---

# PART VIII — BASELINES

Six methods, identical trajectories, with an **external information** column:

| method | damped | unstable | observable kept | external info |
|---|---|---|---|---|
| uncorrected ROM | 1.0000 | 1.0000 | n/a | none |
| bigger basis (rank +6) | 4.9718 | 1.0438 | n/a | none |
| full-state defect | 11.5595 | 3.0353 | NO | none |
| **P-only correction** | **1.0000** | 3.0968 | NO | none |
| **Q-only (ours)** | **16.7008** | 1.0416 | **YES** | none |
| ensemble 3DVar | 1.3102 | 2.8056 | NO | 6 obs modes |

**Three findings.**

1. **P-only gives exactly 1.0000 in the damped band.** The hidden component
   does all the work — the cleanest available control for the central claim.
2. **Q-only beats full-state correction there** (16.70 vs 11.56). Correcting
   `P` as well *overshoots*, consistent with Corollary D.1: zeroing the error
   and minimising future error are different objectives.
3. **The regimes reverse.** In the unstable band Q-only is the *worst*
   correction method. λ predicts this: λ ≈ 0.09 means almost none of the
   observable error is hidden-driven. **λ predicts which correction to use.**

**3DVar is a real baseline, not a strawman.** Background covariance from the
model's own error-snapshot ensemble; increment
`dx = B Hᵀ(H B Hᵀ + R)⁻¹(y − H x_b)`, solved exactly in the low-rank ensemble
subspace, so it corrects `Q` *through* the `P` observation. Insensitive to its
one free parameter: R_obs = 1e-8 / 1e-4 / 1e-2 gives 1.26 / 1.20 / 1.12.

**Do not overclaim:** this is single-time, static-B 3DVar, and `B` is built
from truth snapshots. The defensible sentence is "outperforms a fairly-tuned
ensemble 3DVar under this information budget," not "beats data assimilation."

### The full-state degeneracy — a methodological finding

In 3D Navier–Stokes, full-state correction gave a nominal **87.02×** with a
corrected P-error of **5.6e-5**, *below* the estimator's own residual of 1.1e-4.
With an estimator this accurate, `V + ê` reproduces the truth trajectory: that
is **state replacement, not model correction**, and its "gain" measures
estimator precision. It is excluded from the candidate set and retained only as
a diagnostic. In KS the estimator is weak enough that full-state stays in class
(3.04, 11.46); in NS it does not.

---

# PART IX — GENERICITY

## IX.1 Across ROM families

POD-Galerkin under an honest protocol — 12 training trajectories with
randomised ICs, frozen basis, rank chosen on a validation split, 20 unseen test
ICs, statistics over independent **trajectories**:

| gate | result |
|---|---|
| test is LIVE | 20/20, baseline error 1.8–11.3% |
| observable preserved | max P-mismatch 0.000e+00 |
| beats baseline | **20/20** trajectories, 60/60 windows |
| beats wrong-sign | **20/20** |
| median gain | 1.0439×, 95% CI [1.0192, 1.0878] |
| oracle benefit captured | **1.0004** |

**Containment explains the effect-size difference.** Attribution of the gap
between pure POD (1.0544) and the STAR2 subspace (1.5639), log scale:

| contribution | share |
|---|---|
| containment of the observable set | **101.1%** |
| local trajectory jet | 0.1% |
| dropping the global POD modes | −1.2% |

**Gain is a step function in containment.** Across variants B–E it spans only
1.5639–1.5978 (2.2%) while the reservoir ratio spans 13.8–25.0 and baseline
error spans 0.02%–1.62%, an 80× range.

**A separation worth stating alone:** adding the local trajectory jet cuts
baseline ROM error **80-fold** (1.62% → 0.02%) and contributes ~nothing to the
correction. *What makes a reduced model accurate is not what makes it
correctable.*

## IX.2 Across equations and discretisations

| system | nonlinearity | discretisation | observable | causal signature |
|---|---|---|---|---|
| Navier–Stokes | quadratic | spectral | Fourier modes | 36/36 |
| Kuramoto–Sivashinsky | quadratic | spectral | Fourier modes | 12/12 configs |
| complex Ginzburg–Landau | **CUBIC** | spectral | Fourier modes | **120/120** |
| KS finite-difference | quadratic | **finite differences** | **point sensors** | **36/36** |

**CGL prospective test** — four predictions registered before running, all
passed at n = 120: identity 3.763e-12; causal signature 120/120; the
applicability predictor transferring at Spearman +0.9997; and λ's range across
bands staying below 0.30, formalising the earlier negative as a prediction.

**Point sensors in 3D Navier–Stokes**, n = 36: probe readings identical to
**3.119e-11 relative**, causal signature 36/36, identity 5.811e-16,
Spearman(Ĝ, G) = +1.0000, median gain 1.7140.

**CAVEAT, and it is the weakest link in the genericity chain.** The baseline
ROM error in that run is **0.56%**, an order of magnitude below the 1.8–11.3%
band of the POD genericity test and close to the regime that produced the P1
NULL result (1e-7 baseline, nothing to correct). The result is not null — 36/36
with a wrong-sign control and an exact identity is real — but it demonstrates
the observation operator at 3D scale on an *already accurate* ROM. It should be
rerun with the correction band pushed to 2–10% (raise `PB_AMP` or `PB_NSTEP`)
before it carries weight in a submission. The tuning is delicate: at
ν = 2.5e-3 the same ROM tracks the probes to 0.000%.

---

# PART X — NEGATIVE RESULTS

Reported because a framework without its failures is not trustworthy.

## X.1 No fluid-mechanical mechanism (tested twice)

**Test L — leakage-direction identity.** Log-log fits, not rank correlations:

| direction | pooled exponent | R² |
|---|---|---|
| 0 | +0.001 | 0.0000 |
| 1 | +0.195 | 0.3120 |
| 2 | +0.182 | 0.1285 |

against a Kolmogorov prediction of 0.75. Over a 15.6× viscosity change a
dissipation scale would move 7.9×; `k_leak` moves 1.01–1.57×. No
vortex-stretching alignment: cos²(e₂) exceeds the isotropic 1/3 by +0.004,
+0.018, +0.006. Relative helicity +0.024.

**Test TR — the triadic control.** Ordering the *same* hidden-error subspace by
error energy and by leakage:

| ordering | n | median enrichment |
|---|---|---|
| dangerous (top leakage) | 30 | 182.7× |
| energetic (top error energy) | 30 | 151.9× |
| random | 30 | 60.1× |
| dangerous tail (least leakage) | 30 | **0.0×** |

**dangerous/energetic = 1.20× — NOT SEPARATED.** In this testbed the energetic
and dangerous directions largely coincide, so **"energy ≠ danger" is not
demonstrated as a triadic effect.** The 183× enrichment is a property of the
accumulated error subspace, not of the leaking directions.

Two survivors worth reporting: triadic content separates leading from trailing
leakage directions by ~1000× (182.7 vs 0.0), and the error subspace is itself
~60× triadically enriched relative to arbitrary hidden directions.

## X.2 Testbed dimensionality

Unforced decaying random-phase fields stay at trajectory rank 4–11 regardless
of spectral band — **trajectory POD rank measures dynamical exploration, not
spectral breadth**. Lundgren linear forcing with **constant A** (keeping `F`
exactly quadratic) raises rank to 17–35 with `rank ~ ν^-0.273` (R² = 0.99), but
the limit becomes resolution: rank 50 needs `ν ~ 8.7e-5` and `N ~ 800`.

**Rank 50+ is unreachable on the available hardware.**

## X.3 Closed loop at small scale

| M | fired | ROM err | LOOP err | cost | outcome |
|---|---|---|---|---|---|
| 16 | 0/8 | 0.61% | 0.61% | 3.5× | gate declined all; no harm |
| 20 | 2/8 | 0.59% | 7.61% | 10.3× | fired twice, made it worse |
| 24 | 8/8 | 0.59% | 0.00% | 31.9× | degenerate; no coarsening |

At N = 24 no coarse grid is both accurate enough to help and coarse enough to
matter. **The one real finding: the trust gate works** — at M = 16 it declined
every segment and the loop did no harm.

---

# PART XI — WHAT IS NOT CLAIMED

* **Not a general principle of reduced dynamical systems.** The effect size is
  governed by λ and cos θ; λ is orderable by a truth-free quantity but not
  predictable in closed form; and the law itself is valid only for modal
  observables.
* **No fluid-mechanical mechanism.** Tested twice, negative both times.
* **Theorem C is not novel.** See the prior-art positioning.
* **Theorem B is heavily exposed.** Do not lead with it.
* **Effect size is regime-dependent and can be adverse.** Q-only correction is
  the worst of the tested methods when λ is small.
* **Trajectory rank above ~35 is unreachable** on the available hardware.

---

# PART XII — CORRECTIONS AND RETRACTIONS

Eighteen, of which fifteen were found internally.

| # | claim | correction |
|---|---|---|
| 1 | headline gain 1.9147× | float32 artifact; corrected to **1.6510×** |
| 2 | R3 random-probe leakage measurement | wrong operator; empirical subspace used instead |
| 3 | coarse-η validated | over-generalised from one topology |
| 4 | Re^(3/4) Spearman +1.000 | rank correlation on a monotone function proves nothing; log-log fit gives +0.001 |
| 5 | five gate-design errors | non-directional gates; gates keyed off text tags |
| 6 | "21× cheaper at N=80" | measured 2.9×; nominal grid-point model superseded |
| 7 | restrict/prolong would dominate | measured at 0.5–0.6% |
| 8 | P3 containment result | non-Hermitian construction; 16% → 101% |
| 9 | first KS conclusion | sampling error; "cannot reach R≫1" was false |
| 10 | error reduction "65%" | gain 1.65 is **39.4%** reduction, not 65% |
| 11 | Test Q "46%" | printed medians give **44.9%** |
| 12 | Test T `R² = 0.771` | **retracted** — not in any surviving output |
| 13 | Test U "cherry-picked" | vindicated; HARD/CONTROL split is preregistered |
| 14 | p ~ n^0.29 as a law | fitted family absorbing a linearisation failure |
| 15 | two λ mechanisms | alignment/dimension and advective locality, both refuted |
| 18 | identity "holds to one part in 10¹²", then "better than five parts in 10¹²" | **both wrong** — the measured worst case is 5.38e-12, so neither bound holds. Corrected to "at most 5.4 × 10⁻¹²". Caught by external review, twice |
| 17 | reachability confirmed on "540 windows" | **corrected to 300** — the design is 20 trajectories × 5 variants × 3 intervention times = 300 windows (60 per variant, matching the run output); 540 could not be factorised from the design and was wrong. With the oracle arm constrained separately in each, 600 comparisons. Caught by external review |
| 16 | "the pointwise failure is binary, not graded" | **retracted** — n=160 dose-response sweep shows a graded, early-saturating curve (49% of the damage from one sensor in 24, saturating by 25%) |

**Provenance.** Test A–W notebooks recovered and 17 figures verified exactly.
Test X console log recovered — all 36 windows, every summary figure. The
post-X archive recovered and 12 further figures verified exactly, including
X-v3, the α sweep, Z1/Z2 and the wall-clock benchmark. Only the most recent
session's KS/CGL/FD/adaptive work remains unarchived, and all of it is cheap
CPU reruns.
