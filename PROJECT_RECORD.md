# M3D / STAR2 — Complete Project Record

**Hidden-error transport, truth-free reconstruction, and observable-preserving
correction in reduced models of incompressible Navier–Stokes**

Ansh Mishra, Aryan Senthilkumar

Status: pre-manuscript. This document is the consolidated technical record —
every theorem with its proof and its exposure, every test with its result and
its interpretation, and an explicit list of what has been retracted or
corrected along the way.

---

## 0. Executive summary

### 0.1 What the project claims

A projection-based reduced-order model (ROM) accumulates error. Most of that
error lives in the *unresolved* block `Q` of the state, where it is invisible
to the chosen observable `P`. Nonlinear dynamics transport some of it back into
`P`. That transported error can be

1. **estimated without knowing the truth trajectory**, from the model's own
   defect history, and
2. **corrected by modifying only the hidden state**, leaving every observable
   coordinate bit-for-bit unchanged,

and doing so measurably improves the future observable forecast — in a regime
whose boundary is now measured rather than assumed.

### 0.2 The results that carry the paper

| # | Result | Evidence |
|---|---|---|
| 1 | **Causal hidden-state correction.** Modifying only `Q`, with `ΔP = 0.000e+00` exactly, improves future resolved accuracy in 36/36 interventions; the sign-reversed control loses 36/36. | Test X-v3 |
| 2 | **Theorem F: the observable effect of any hidden correction is confined to `P·S`**, giving `gain ≤ 1/√(1−ρ²)`. Confirmed on 300 windows across 5 ROM variants with zero violations, oracle arms included. `P ⊆ S` removes the constraint; nothing else in the ROM does. | Test F |
| 3 | **The optimal correction amplitude is predicted by theory.** Corollary D.1 gives `α* = 1 + ⟨b_PP, b_QP⟩/‖b_QP‖²`; predicted 1.985 against an independently swept 2.000. | Test Z1 |
| 4 | **The mechanism is ROM-generic.** On POD-Galerkin under a frozen-basis, unseen-initial-condition protocol it reproduces at 1.58× against STAR2's 1.65×, provided `P ⊆ S`. Containment explains ~100% of the difference between the two families; ROM accuracy and reservoir ratio explain none. | Tests P1, P4 |
| 5 | **The causal signature survives a second equation.** On Kuramoto–Sivashinsky (independent implementation, trajectory rank to 199), `+q̂` beats `−q̂` in 12/12 configurations. | Test KS |
| 6 | **The effect size is exactly determined by two measurable quantities** — see below. This is the scope condition the project previously lacked. | Test KS, Test LAM |
| 7 | **The estimator becomes cheaper than the solver as resolution grows.** Measured: 3.06× the solver at N=40, 0.71× at N=64, **0.34× at N=80**. | Tests C, CX, CS, benchmark |

### 0.3 The effect-size law

The gain is not a correlational matter. With `b` the baseline future observable
error and `Δ` the oracle's observable effect,

```
gain = ‖b‖ / ‖b + Δ‖ = 1 / √(1 + 2λ cos θ + λ²)

    λ     = ‖Δ‖/‖b‖                    how large the hidden contribution is
    cos θ = ⟨b,Δ⟩/(‖b‖‖Δ‖)             how well it opposes the existing error
```

This is algebra, verified on Kuramoto–Sivashinsky to **2.22e-16** across 12
configurations. Neither the reservoir ratio `R` nor the reachability fraction
`ρ` appears in it, which is why three earlier correlational hypotheses (§5.16,
§6.8) failed.

Measured: KS gives `λ = 0.048–0.134`, `cos θ = −0.64 to +0.12`, gain 1.035.
Navier–Stokes' 1.65 requires `λ ≈ 0.5–0.9` with `cos θ ≈ −0.85`.
**The NS measurement is outstanding** (Test LAM, written and not yet run).

Both quantities are computable *before* any correction is attempted, so this is
an applicability test rather than a post-hoc fit: it says in advance whether the
method will help on a given problem.

### 0.4 What is not claimed

- **Not a general principle of reduced dynamical systems.** The causal
  signature transfers to a second equation; the magnitude does not (39.4%
  error reduction on NS, 3.4% on KS). Until the λ gap is explained on both systems,
  the honest framing is a method with a measured scope condition.
- **No fluid-mechanical mechanism.** Tested twice, negative both times: no
  Kolmogorov scaling in the leakage directions (fitted exponent +0.001 against
  0.75), no vortex-stretching alignment, and triadic enrichment does not
  separate dangerous from energetic directions (ratio 1.20×). §5.14, §5.15.
- **Limited dynamical range on Navier–Stokes.** Trajectory rank reaches 35;
  rank 50 would need N ≈ 800. KS supplies rank to 199 but with a weak effect.
- **Two theorems have close published relatives.** Superconvergence and the
  defect identity both do; do not lead with either (§7).

### 0.5 The open problem

Why is λ an order of magnitude larger on Navier–Stokes than on
Kuramoto–Sivashinsky? A plausible reading is that KS's observable modes are the
linearly unstable ones (`k < 1` grow), so their error self-amplifies and the
hidden contribution is a perturbation on top; whereas NS's observable error is
largely *driven* from unresolved scales. **Untested.** Answering it would
convert the scope condition into a predictive statement about which systems the
method suits, and is the natural opening of the next project.

## 1. Setting and notation

Spectrally discretized incompressible Navier–Stokes on a periodic box,

```
U̇ = F(U),          F(U) = D[ D(FFT(u × ω(u))) − ν K² D(U) ],   D = dealias ∘ Leray
```

`F` is **exactly quadratic** in `U`. This fact is load-bearing throughout
(§2.5, §5.1).

A fixed orthogonal projector `P` selects the observable coordinates;
`Q = I − P`. Any error splits as `e = Pe + Qe`.

- `P` is the `mixed` 24-mode set, spanning `|k| ∈ {1, √2}` (6 modes) and
  `|k| ∈ {4.47, 5}` (6 modes) plus conjugates. `k_P ≈ 3`, `k_P,max = 5`.
- `P` is fixed in **absolute wavenumber**, independent of grid size `N`. At
  N=40 it covers 6.25e-5 of the state; at N=96, 4.5e-6.

**The reduced model (STAR2).** Augment `P` with two hidden directions:

```
q₀  = Qf / ‖Qf‖,                       f = F(U₀)
q₁* = R₁ a / ‖R₁ a‖,                   a = DF(U₀) f,   R₁ = I − P − q₀q₀*
Π   = P + q₀q₀* + q₁*q₁**
U̇_r = Π F(U_r)
```

"Rank 2" means two extra hidden closure directions. The resolved state itself
carries hundreds of coordinates.

**Time convention.** Segment `τ = 0.15`, 20 RK4 substeps, `TAU0 = 40/3`,
`dt = τ·TAU0/20`. Interventions at `τ ∈ {0.90, 1.20, 1.50}`, forecast horizon
2 segments (`τ = 0.30`).

---

## 2. The theorems

This is the section the project is organized around. Each theorem is stated,
proved, and given an explicit honest assessment of novelty.

---

### Theorem A — dimension-minimal complement jet augmentation

**Statement.** Let `f = F(U₀)` and `a = DF(U₀)f`. Suppose `P` is fixed and we
seek a projection ROM whose first two trajectory derivatives match the full
system exactly at `U₀`. Then any admissible augmented space must contain `f`
and `a`, hence its hidden component must contain `Qf` and `Qa`. The minimum
hidden dimension is therefore

```
dim span{Qf, Qa}
```

and STAR2 realizes exactly this span via `q₀ ∝ Qf`, `q₁* ∝ (I − P − q₀q₀*)a`.

**Proof.** `U'(0) = f` and `U''(0) = a`. For a projection ROM with projector
`Π ⊇ P`, `U_r'(0) = ΠF(U₀) = Πf`, which equals `f` iff `f ∈ range(Π)`.
Given that, `U_r''(0) = Π DF(U₀) U_r'(0) = Πa`, equal to `a` iff `a ∈ range(Π)`.
Since `P ⊆ Π` is fixed, the free part of `Π` must contain `Qf` and `Qa`. ∎

**Status: correct, small, and defensible.** STAR2 is not an empirically lucky
second mode; it is the minimal complement augmentation matching velocity and
acceleration.

#### Lemma A.1 (exact, previously reported as numerical)

The project record originally stated that

```
q₁* ∈ span(q₁^old, q₂*)      "to numerical precision"
```

where `q₁^old ∝ R₁ A q₀` is the OLD2 second direction and `q₂* ∝ R₂ A f` is
the STAR3 third direction, `R₂ = R₁ − q₁^old q₁^old*`.

**This is an exact algebraic identity, not a numerical coincidence.** Since
`R₁ = R₂ + q₁^old q₁^old*`,

```
R₁ A f = q₁^old ⟨q₁^old, A f⟩ + R₂ A f
```

and `q₁* ∝ R₁ A f`, so `q₁* ∈ span{q₁^old, q₂*}` identically. ∎

**Consequence.** STAR2 captures in one hidden coordinate the
acceleration-matching direction that OLD3 reaches using two. This should be
stated as a lemma in the manuscript, not as an empirical observation — it
converts "STAR2 got lucky" into "STAR2 is the rank-1 realization of what
OLD3 needs rank 2 for."

---

### Theorem B — observable superconvergence

**Statement.** Let a fixed reduced space `S ⊇ P` contain the first `r`
trajectory derivatives, `U^(j)(0) ∈ S` for `j = 1..r`. Then

```
P[U(t) − U_r(t)] = O(t^(r+2))
```

**Proof.** By induction, `U_r^(j)(0) = Π U^(j)(0) = U^(j)(0)` for `j ≤ r`.
The first possible mismatch is at order `r+1`:

```
e^(r+1)(0) = (I − Π) U^(r+1)(0) = R U^(r+1)(0)
```

Since `P ⊆ Π`, `PR = 0`, so `P e^(r+1)(0) = 0`. The resolved error cannot
appear until one derivative later. ∎

For STAR2, `r = 2`, giving

```
P[U(t) − U_r(t)] = O(t⁴)
```

against OLD2's `O(t³)`.

**Explicit leading coefficient.** With `A = DF(U₀)`, `B = D²F(U₀)`,
`g₃ = A²f + B[f,f] = U'''(0)`, the hidden third-derivative defect is `Rg₃`,
whose first leakage back into `P` gives `C₄ = P A R g₃`. Hence

```
P[U(t) − U_r(t)] = (t⁴/24) · P DF(U₀) R [ DF(U₀)²F(U₀) + D²F(U₀)[F(U₀),F(U₀)] ] + O(t⁵)
```

This connects the beginning and the end of the project: error is *born* in
`Q`, and `DF` transports it toward `P`. That is precisely the mechanism later
measured in Tests Q/R/S/T.

**Numerical confirmation (Test Y-v2, §5.2).** Four topologies in float64,
M-B fit window `τ ∈ [0.015, 0.060]`:

| flow | p (FD Jacobian) | p (analytic Jacobian) | R² | Δp |
|---|---|---|---|---|
| vortex_ring | 3.9778 | 3.9778 | 1.0000 | −0.0000 |
| periodic_shear | 3.9991 | 3.9991 | 1.0000 | +0.0000 |
| skew_tubes | 3.9646 | 3.9646 | 1.0000 | +0.0000 |
| mixed_vortices | 3.9788 | 3.9788 | 1.0000 | −0.0000 |

**The first correction term is measured, not just fitted.** Moving the fit
window 10× earlier shrinks the deviation from 4 by 10×:

| flow | \|p−4\| at M-B window | \|p−4\| at early window | ratio |
|---|---|---|---|
| vortex_ring | 0.0222 | 0.0023 | 9.7 |
| skew_tubes | 0.0354 | 0.0034 | 10.4 |
| mixed_vortices | 0.0212 | 0.0018 | 11.8 |

This is exactly what `P-error = Cτ⁴(1 + aτ + …)` predicts: local slope
`4 + O(τ)`. It is considerably stronger evidence than a single slope fit and
should be a table in the manuscript.

**Status: correct, confirmed to an unusual standard — and the most exposed on
novelty.** See §7.

---

### Theorem C — defect-driven second-order error transport

**Statement.** Let `Φ_h` be the full flow map over one segment, `V_k` the ROM
state, `e_k = U_k − V_k`. Define the **defect**

```
η_k = Φ_h(V_k) − V_{k+1}
```

which requires no knowledge of the truth trajectory. Then exactly

```
e_{k+1} = η_k + Φ_h(V_k + e_k) − Φ_h(V_k)
```

and by Taylor expansion, with `J_k = DΦ_h(V_k)`, `H_k = D²Φ_h(V_k)`,

```
e_{k+1} = η_k + J_k e_k + ½ H_k[e_k, e_k] + ρ_k,     ‖ρ_k‖ ≤ (M₃,k/6)‖e_k‖³
```

giving the truth-free recurrence

```
ê_{k+1} = η_k + J_k ê_k + ½ H_k[ê_k, ê_k]
```

**Proof.** The identity is exact by adding and subtracting `Φ_h(V_k)`. The
expansion is Taylor's theorem with Lagrange remainder. ∎

**Status: correct; the identity is elementary.** Classical defect-based error
estimation is well established. The defensible contribution is not the
expansion but its *use*: reconstructing the full accumulated error vector,
extracting its unresolved component `Q ê`, and acting on it.

---

### Theorem D — hidden-only correction criterion

**Statement.** For any `c ∈ range(Q)`,

```
P(V + c) = PV        exactly
```

Let `b = P[Φ_h(V) − Φ_h(U)]` be the baseline future resolved error and
`Δ(c) = P[Φ_h(V + c) − Φ_h(V)]`. The corrected error is `b_c = b + Δ(c)`, and
the correction improves the forecast **iff**

```
2⟨b, Δ(c)⟩ + ‖Δ(c)‖² < 0
```

**Proof.** `‖b + Δ‖² − ‖b‖² = 2⟨b,Δ⟩ + ‖Δ‖²`. ∎

**Consequence.** There can be no theorem asserting that *any* hidden
correction helps. Direction matters — which is exactly what makes the
sign-reversed control in Test X meaningful. For small `c`,

```
Δ(+c) = +Tc + ½H[c,c] + O(‖c‖³)
Δ(−c) = −Tc + ½H[c,c] + O(‖c‖³)
```

so reversing the correction flips the leading directional transfer while
leaving the quadratic curvature unchanged. The `+c` vs `−c` comparison
therefore isolates whether the estimator carries causal directional
information.

---

### Corollary D.1 — optimal correction amplitude, LINEAR RESPONSE (new)

**This is the newest theoretical result in the project and it was confirmed
by an independent measurement.**

**Scope — read this first.** Theorem C established that the hidden-state
response is *not* linear in amplitude:

```
Δ(αq) = α g₁ + (α²/2) g₂ + O(α³),      so   Δ(αq) ≠ α Δ(q)
```

The closed form below assumes linear response, i.e. that a correction `αq`
cancels exactly `α·b_QP`. It is therefore the **linear-response optimum**,
written `α*_lin`, valid in its approximation regime and not an exact
minimizer of the nonlinear problem. The quadratic refinement is given
immediately after, and is what should be quoted as the theory's prediction.

**Statement (linear response).** Decompose the baseline future resolved error
into the part carried by resolved-error persistence and the part leaked from
the hidden block, `b = b_PP + b_QP`. Under linear response the corrected error
is `‖b_PP + (1−α)b_QP‖`, minimized at

```
α*_lin = 1 + ⟨b_PP, b_QP⟩ / ‖b_QP‖²
```

**Proof.** Differentiate `‖b_PP + (1−α)b_QP‖²` in `α` and set to zero. ∎

**Nonlinear refinement.** With `Δ(1)` and `Δ(2)` measured,

```
g₁ = 2Δ(1) − ½Δ(2),        g₂ = Δ(2) − 2Δ(1)
α*_quad = argmin_α ‖b + α g₁ + (α²/2) g₂‖
```

This is the version that carries the curvature Theorem C requires, and it is
the one that agrees with the measurement.

**Interpretation.** `α* > 1` exactly when the persistence error and the
leakage error are *positively aligned*, in which case overcorrecting in `Q`
also partially cancels `P`-error that the correction is structurally
forbidden from touching directly. This is a stronger statement than Test X
alone supports: the hidden state carries usable information about the
resolved error, not merely about its own leakage.

**Measured quantities are recoverable from the existing pipeline.** With
`d = Δ(1)`:

```
b_QP = −d,        b_PP = b + d
```

**Confirmation (Test Z1, §5.6).** Over 36 windows:

| quantity | value |
|---|---|
| median cos(b_PP, b_QP) | +0.8196 |
| cos > 0 | 36/36 windows |
| median ‖b_PP‖/‖b_QP‖ | 1.3387 |
| **α\*_lin (linear response)** | **2.0289** |
| **α\*_quad (nonlinear refinement)** | **1.9850** |
| **α\* measured (independent sweep)** | **2.0000** |

The nonlinear refinement agrees to 0.75%; the linear-response value to 1.4%.
Both are quoted because the gap between them *is* the curvature term, and
reporting the pair is stronger than reporting either alone. Per-window, the prediction lands within 35% of the
measured argmax in 30 of 36 windows.

**Caveat.** The prediction degrades where `α* > 3`: predicted 6.02 vs measured
4.0, 5.96 vs 4.0, 8.00 vs 6.0. Expected, since `g₁` and `g₂` are fitted at
`α = 1, 2` and then extrapolated. State the result as accurate for `α* ≲ 3`.

---

### Theorem E — the W→X bridge

**Statement.** Let `q = Q(U − V)` be the true hidden error and `q̂` its
estimate. Assume the projected finite-time flow is locally Lipschitz,
`‖PΦ_h(x) − PΦ_h(y)‖ ≤ L_P‖x − y‖`. Then

```
‖PΦ_h(V + q̂) − PΦ_h(V + q)‖ ≤ L_P ‖q̂ − q‖
```

so with `E₀` the baseline future resolved error and `E_oracle` the error after
oracle correction,

```
E_estimated ≤ E_oracle + L_P‖q̂ − q‖
```

Hence if `E₀ − E_oracle > L_P‖q̂ − q‖`, then `E_estimated < E₀`.

**Proof.** Direct from the Lipschitz bound. ∎

**Consequence.** Accurate hidden estimation plus real hidden-state causal
leverage implies a successful truth-free intervention. Tests W and X are two
halves of one mathematical statement.

**Empirical probe of the bound (Test Z, direction family).** Degrading `q̂`
with random `Q`-noise at relative amplitudes `δ ∈ {0.03, 0.10, 0.30, 1.00}`:

| δ | median gain | win rate |
|---|---|---|
| 0.03 | 1.6421 | 100% |
| 0.10 | 1.6247 | 100% |
| 0.30 | 1.6258 | 100% |
| 1.00 | 1.6133 | 100% |

A 33× range of directional error moves the gain by 2%. **Interpretation:** in
a `Q` of dimension ~10⁴–10⁵, a random perturbation has overlap `≈ √(3/n)`
with a rank-3 leakage subspace, so almost none of it survives projection
through `T`. Directional insensitivity and low-rank leakage are the same fact.
Amplitude sensitivity (Corollary D.1) is strong precisely because amplitude is
what *does* survive that projection.
---

## 3. Methodology and provenance discipline

The project operates under rules that should be stated explicitly in the
manuscript, because they are rarer than any individual result:

1. **Preregistered gates.** Thresholds are fixed before a run. When a gate
   fails, the threshold is not moved; the missing physics is diagnosed and a
   new test is run (this happened at Tests R, T, and again at Y-v1 G5).
2. **Artifacts required.** A claim without a backing run artifact is void.
   Two entries have been invalidated under this rule.
3. **Contamination is voided, not salvaged.** Two viscosity-namespace bugs
   were found; affected cases were discarded and rerun rather than reasoned
   around.
4. **Self-identifying builds.** Runners print a `BUILD_ID` and their own
   absolute path before doing work, and assert on their own source content.
   Introduced after three runs silently executed a stale file.
5. **Per-row sentinels.** Every output row records the live viscosity, dtype,
   JVP mode and derivative step, checked against what was requested. Adopted
   after discovering that namespace *self-consistency* does not imply
   *correctness* — the original viscosity bug had every namespace holding the
   same wrong value.
6. **Null-test gates.** A test that cannot discriminate must say so rather
   than report a passing-looking number. Added after Test G1 (§5.9).

---

## 4. The original campaign (Tests A–X, pre-audit)

Condensed; full detail in the project's earlier record.

### 4.1 Construction and speed

The reduced vector field is exactly quadratic, so it admits an exact dense
polynomial representation. Compiled that way:

| N | full CFD RK4 | polynomial ROM RK4 | speedup |
|---|---|---|---|
| 40 | 13.29 ms | 4.13 ms | 3.2× |
| 80 | 32.0 ms | 1.47 ms | 21.8× |

**Test I — adaptive reanchoring.** Teacher-free periodic rebasing of the local
hidden directions at the model's own current state. At `τ = 1`: frozen basis
5.36% resolved error vs adaptive 0.342%, a **15.7×** improvement. The
long-horizon problem is basis aging, not catastrophic truncation.

### 4.2 The local error certificate

`κ₃ = τ₀³‖C₃‖ / (6‖PU₀‖)` with `C₃ = P DF R DF F`.

- **Test G:** all 9 threshold crossings predicted; Spearman ≈ 0.85; median
  multiplicative horizon error 1.19×. Useful as a *local* diagnostic.
- **Test J:** using `κ₃` as a *global* reanchoring scheduler performed
  substantially worse than simple periodic reanchoring. The stronger claim was
  killed.

**Retained statement:** local error certificate ≠ global optimal scheduler.

### 4.3 First contamination event

Functions lived in distinct Python global namespaces. Changing notebook-level
viscosity did not change the viscosity used by every baseline function, so
`0.25ν` and `4ν` cases compared a ROM at one viscosity against truth at
another. Tell: all `ν = 1` results reproduced exactly; only the non-baseline
viscosities moved after repair.

**Root cause, confirmed in this session:** `_pair_rhs_D` reads uppercase
`NU1_D`/`NU2_D`; the bootstrap's sync wrote only lowercase `nu1_D`/`nu2_D`.

Contaminated K1/L2 cross-viscosity conclusions were invalidated. Viscosity
sentinels were added.

### 4.4 Local order

- **Tests M / M-A:** float32 noise produced an apparent order ≈ 2 for periodic
  shear. Not explained away; attacked.
- **Test M-B:** float64/complex128, dtype sync repaired, halved timestep.
  STAR2 gave `p ≈ 3.9985` and `3.9922` on independent windows; mixed vortices
  3.97, 3.93. OLD2 remained cubic.

**Later resolution (this session):** the M-A anomaly is definitively the
float32 floor, reproduced on demand — see §5.2.

- **STAR3:** a third direction `q₂* ∝ R₂Af` cancels the old cubic coefficient
  and preflights perfectly, but barely improves the long horizon. Superseded
  by Lemma A.1.

### 4.5 The local/global gap and its diagnosis

- **Test N:** STAR2 beat OLD2 in resolved error in 10/12 cases, but median
  final improvement at `τ=2` was only **1.07×**, against an enormous local
  order gain. Periodic shear was 9–12× better at early horizons and
  indistinguishable by `τ=2`.
- **Test O:** against an oracle-reanchored STAR2, fresh one-segment error
  stayed ≈ 0.001% while global error reached percent level; **≈ 99.9%** of
  late target error was attributable to accumulated history.
- **Test P:** launching both full NS and STAR2 from the same erroneous state
  gave late median amplification ratio **Γ ≈ 1.06** on target variables — the
  growth is ordinary physical sensitivity, not a ROM pathology. But full-state
  error greatly exceeded resolved error: **error was hiding in Q.**

### 4.6 The hidden reservoir and its leakage

- **Test Q:** injecting `δP` only, `δQ` only, and both. Late in the
  trajectory `‖δQ‖/‖δP‖ ≈ 23×`. Per unit amplitude `Q→P` coupling is much
  weaker than `P→P` persistence, but sheer amplitude made `Q` responsible for
  **≈ 45%** of late one-segment resolved drift (late medians: P-only 0.568%,
  Q-only 0.456%, full 1.017%; 0.456/1.017 = 44.9%). The record previously said
  46%, which could not be reproduced from the printed output — see §6.11. Interaction terms small ⇒
  approximately block-linear local structure, and a new object:

  ```
  T_k = P DΦ_h(U_k) Q
  ```

- **Test R:** empirical `Q`-error snapshot rank 9; 99% of snapshot energy in
  rank 3; first 3 leakage singular directions held **85.8%** of operator
  energy; restricting the actual hidden error to the top 3 empirical
  directions recovered **98.8%** of the realized future `Q→P` response; first-
  order prediction cosine 0.9986.

  The preregistered gate technically failed (the 3 dimensions held >50% of
  input `Q` energy) but was testing the wrong thing. Retained statement:
  **energy ≠ danger.**

- **Test S:** repeated along the trajectory. Median top-3 operator energy
  91.4%, realized 95.1%, first-order prediction cosine **0.999994**. Confound
  acknowledged: `τ`, `P`-error and `Q`-error all grow together, so S is a
  mechanism result, not cross-case proof.

- **Test T:** 4 flows × 3 viscosities × 11 anchors = **132 tests**. Median
  operator top-3 energy **98.5%**, realized **99.95%**. But the first-order
  finite-amplitude gate failed: 90th-percentile residual 0.260 against an
  allowed 0.20. The warning metric also failed its correlation-superiority
  threshold. **RETRACTED:** a leave-one-case-out regression was previously
  reported here at `R² ≈ 0.771`, RMSE ≈ 0.116 against ≈ 0.179. No such
  regression exists in any surviving notebook or console output, and the
  figures cannot be sourced. See §6.11.

  What the output DOES show: the warning metric reached median per-case
  Spearman 0.8636 against 0.8409 for the best simple baseline (`Q_OVER_P`) —
  better, but not by the margin the gate required.

  **Thresholds were not moved.** The question became: what nonlinear term is
  missing?

### 4.7 Finite-amplitude curvature

- **Test U:** `E(α) = P[Φ_h(U + αδQ) − Φ_h(U)]`, comparing
  `E₁ = αg₁` against `E₂ = αg₁ + (α²/2)g₂` on the *hardest* low-viscosity
  cases: vortex ring 0.268 → 0.0130 (**20.6×**), skew tubes 0.505 → 0.0518
  (**9.76×**), mixed vortices 0.309 → 0.0239 (**12.9×**). Scaling laws
  `r₁ ~ α^1.04`, `r₂ ~ α^2.04` — exactly the expected orders. The two
  **CONTROL** cases in the same run gave 58.197× (periodic_shear ν=1) and
  219.465× (vortex_ring ν=4). The three quoted above are exactly the three
  cases the test labels **HARD**; the split is preregistered in the runner, not
  post-hoc. Reporting only the HARD three understates the method — the controls
  show it is safe, not just effective on difficult cases. U remains a
  selected-case test and is not generalization.

- **Test V:** all 132 anchors. Median first-order residual 0.02963; median
  second-order residual **0.0001005** (0.010%); 90th percentile 1.144%; worst
  5.18%; median improvement **262.4×**; 10th-percentile improvement 20.8×;
  **132/132** anchors improved.

  Test T's failure had a clean interpretation: *not wrong hidden structure —
  missing finite-amplitude curvature.* The leakage law became

  ```
  E(q) = Tq + ½H[q,q] + O(‖q‖³)
  ```

### 4.8 Truth-free reconstruction

- **Test W.** Evaluating `q = δQ` required knowing `U_truth − U_ROM`. The
  defect identity (Theorem C) removes that need.

  A second viscosity-sync contamination was found in the bootstrap; the
  affected eight non-`1×` cases were **invalidated and rerun**, not patched.
  The contaminated run is preserved as provenance.

  Corrected 12-case result over 132 mature anchors:

  | quantity | value |
  |---|---|
  | median first-order hidden-Q residual | 0.00476 |
  | median second-order hidden-Q residual | **0.000106** |
  | median improvement | **44.8×** |
  | anchors improved | **132/132** |
  | median hidden-direction cosine | 0.99999999 |
  | median amplitude ratio | 1.0000108 |
  | 90th-percentile Q residual | 1.56% |
  | median P residual | 0.001189 |

  All gates passed. **Limitation stated at the time and still binding in the
  original form: `truth-free ≠ full-order-free`.** W evaluates the full flow
  map. This is what Tests C/CX/CS eventually addressed (§5.11–§5.13).

### 4.9 The causal endpoint

- **Test X.** At `τ ∈ {0.90, 1.20, 1.50}` for all 12 cases, four branches:
  baseline `V`; estimated `V + Q̂e`; wrong-sign `V − Q̂e`; oracle `V + Qe_true`.
  Critical design feature:

  ```
  P(V + Q̂e) = PV
  ```

  Every branch starts from an identical resolved state; only the hidden state
  differs. Original result (36 interventions): EST beat baseline 36/36; median
  gain 1.914705×; 10th percentile 1.353768×; median resolved-error reduction
  47.77%; wrong-sign median gain 0.6759×; `+Q̂e` beat `−Q̂e` 36/36; oracle won
  36/36; median fraction of oracle benefit captured 100%. All six gates passed.

**What X establishes.** Not "a corrected ROM performs better," but: at the
intervention instant `ΔP = 0` exactly, nothing observable has changed, and yet
modifying only the hidden state changes future resolved accuracy — with the
sign of the change determined by the sign of the correction.

---

## 5. The audit and extension campaign

Everything from here is the second campaign. It found a float32 artifact in
the headline, closed the cost objection, and closed the fluid-mechanics
question negatively.

### 5.1 Test Y-v1 — precision and Jacobian audit

**Motivation.** The X log printed `initial RAf ≈ 1e-7` to `1e-8`. If the STAR2
space contains `Af` exactly, that relative residual should be near machine
epsilon. Investigation revealed the bootstrap declared `REAL_DTYPE_D =
torch.float32`, and that `pair_jvp_M` is a **central finite difference** with
`eps = max(1e-5, 1e-3‖U‖)`, not an analytic Jacobian.

**Result.** Two residuals were measured: `rho_self` (against the same FD
acceleration the basis was built from — self-consistent by construction) and
`rho_true` (against an independent reference).

| | rho_true (median) | predicted 10³ × machine eps |
|---|---|---|
| fp32 | 1.388e-04 | 1.192e-04 |
| fp64 | 2.641e-13 | 2.220e-13 |

Across all 12 cases the fp32/fp64 ratio was **5.31e8 ± 4.4%** against a
float-epsilon ratio of **5.37e8**.

**Interpretation.** Because `F` is exactly quadratic, central differencing has
**zero truncation error**:

```
F(U + εv) − F(U − εv) = 2ε·DF(U)v      exactly, for any ε
```

The entire residual is roundoff, amplified by `1/(2ε) ≈ 10³`. A truncation
term would be dtype-independent and could not scale with machine epsilon.

**Consequence.** The referee objection "your superconvergence is limited by
your finite-difference Jacobian" is void. This should be one sentence in the
methods.

fp32 order fits returned `p = −0.12` to `−0.76` — noise, because over
`T ∈ [0.02, 0.15]` the true error straddles float32 epsilon.

**Defect found in Y-v1 itself:** gate G5 was preregistered but never
implemented or printed. Recorded as declared-and-not-evaluated.

### 5.2 Test Y-v2 — analytic-reference audit

Rebuilt with an exact analytic NS Jacobian,

```
DF(U)V = D[ D(FFT(u×ω_v + v×ω_u)) − ν K² V̂ ]
```

G5 implemented (analytic-JVP STAR2 vs FD-JVP STAR2, decided on `Δp` alone),
per-row sentinels, fp64 first, M-B `τ` window. All gates passed.

| gate | result |
|---|---|
| G0 completeness | 24/24 Y2 rows, 16/16 Y3 rows |
| G1 sentinels consistent **and correct** | 40/40 rows |
| G3 precision floor | 12/12 |
| G3b (descriptive) | median ratio 5.385e8 vs eps ratio 5.369e8 → **1.003**, spread 5.1% |
| G4 fp64 order in [3.7, 4.3] | 4/4 |
| G5 `\|Δp\|` on fp64 M-B | 0/4 moved; max 1.2e-3 |

**Acceleration residual, analytic vs FD:**

| | rho / machine eps |
|---|---|
| FD, fp64 | 399 |
| FD, fp32 | 392 |
| analytic, fp64 | **0.31** |
| analytic, fp32 | **0.60** |

The FD path sits at ~400× machine epsilon in both precisions — the `1/ε`
amplification. The analytic path sits *at* machine epsilon, and costs 5
spectral transforms per JVP against the FD path's 6.

**Publishable sentence:** *The quartic local resolved-error scaling persists
under an independently implemented analytic Navier–Stokes Jacobian-vector
product (|Δp| < 1.2e-3 across four topologies and two fit windows) and is
therefore not an artifact of the finite-difference derivative used to
construct STAR2. The finite-difference acceleration residual is confirmed to
be a pure floating-point floor: across twelve flow/viscosity cases the
fp32/fp64 ratio is 5.39e8 against a machine-epsilon ratio of 5.37e8.*

Runtime: 159.9 s on one T4.

### 5.3 Test X-v3 — the headline corrected downward

X was rerun in fp64 with the analytic JVP. The W estimator had *already* been
fp64 in the original X; what changed was the trajectory, STAR2 construction
and forecasts.

| | fp32 + FD (original) | fp64 + analytic (clean) |
|---|---|---|
| median EST-Q gain | 1.914705× | **1.650998×** |
| 10th percentile | 1.353768× | 1.299859× |
| median resolved reduction | 47.773% | **39.417%** |
| win rate | 36/36 | 36/36 |
| beats wrong-sign | 36/36 | 36/36 |
| max start-P mismatch | 0.000e+00 | 0.000e+00 |

**Cause, localized precisely.** Only `periodic_shear` moved:

| case | τ=0.90 | τ=1.20 | τ=1.50 |
|---|---|---|---|
| periodic_shear ν=0.25 | 1.1× | 5.8× | **11.1×** |
| periodic_shear ν=1.0 | 1.1× | 4.0× | **10.4×** |
| periodic_shear ν=4.0 | 1.4× | 4.8× | **10.0×** |
| all nine other cases | 1.0× | 1.0× | 1.0× |

Its fp32 baseline error was up to 11× too large, so the correction was partly
cleaning up numerical garbage; that produced gains of 8.47×, 5.94× and 5.11×
which dragged the median to 1.9147×.

**Read correctly this is a better result for the model, not a worse one:**
STAR2 is ≈10× more accurate on periodic_shear than previously reported —
0.13% at τ=1.50, not 1.36%. The correction gain fell because there was less
error left to correct.

`periodic_shear` was also the M-A "order 2" anomaly and the worst case in
Y-v2's fp32 M-B window (p = 0.0393, R² = 0.0064). Three independent tests,
one topology, one cause.

### 5.4 Attribution: the 2×2 over precision and Jacobian

| | fp32 + FD | fp32 + analytic | fp64 + analytic |
|---|---|---|---|
| median gain | 1.914705× | 1.737095× | 1.650998× |
| median reduction | 47.77% | 42.43% | 39.42% |
| median RAf | 1.01e-7 | 7.12e-8 | 6.80e-17 |

**The `repro` control reproduced the original run exactly** — median, p10,
wrong-sign, oracle, and both reduction figures identical to six decimals, the
twelve `RAf` values the same set, per-window baselines matching to five
decimals. The refactor introduced zero drift, so the entire shift is
attributable to precision and Jacobian.

`periodic_shear ν=1.0` baseline at τ=1.50: 1.361% (fp32+FD) → 0.648%
(fp32+analytic) → 0.131% (fp64+analytic). The analytic Jacobian alone
recovers ≈2× of the 10×; double precision supplies the remaining 5–6×.

**Conclusion: fp64 is load-bearing.** In single precision the analytic path is
already at ≈0.6× machine epsilon, so there is no headroom; only fp64 opens it.

### 5.5 The α sweep

Estimator-tolerance sweep with two perturbation families, both confined to
`range(Q)` so `ΔP = 0` is preserved exactly (verified: P-mismatch 0.000e+00 at
every δ, realized relative estimator error matching δ to four decimals).

| α | median gain | p10 | win rate |
|---|---|---|---|
| 1.00 | 1.6510 | 1.2999 | 100% |
| 1.03 | 1.6709 | 1.3088 | 100% |
| 1.10 | 1.7214 | 1.3317 | 100% |
| 1.30 | 1.8738 | 1.4008 | 100% |
| **2.00** | **2.1445** | **1.6314** | **100%** |
| 3.00 | 1.8362 | 0.9532 | 86.1% |
| 4.00 | 0.9917 | 0.5328 | 47.2% |
| 8.00 | 0.4118 | 0.1524 | 8.3% |

Clean interior maximum at α = 2, which **dominates α = 1 on median and p10
with no loss of the 36/36 win rate.** Degradation begins past α = 3; net harm
between α = 3 and 4.

Per-window argmax spreads: α\* = 1.1 (1 window), 1.3 (6), 2.0 (16), 3.0 (10),
4.0 (2), 6.0 (1). If each window used its own optimum the median gain would be
**2.7050×** with p10 2.0000× — a per-window amplitude rule is worth ≈64% over
the current method.

**Reporting recommendation:** lead with α = 1 (1.6510×, 36/36, truth-free and
parameter-free); present α = 2 as the theory-predicted improvement.

### 5.6 Test Z1 — α\* predicted from error alignment

See Corollary D.1. Predicted 1.9850 against measured 2.0000.

### 5.7 Test Z2 — closed-loop correction

Correction applied at every segment, with the W recurrence fed the corrected
trajectory so the estimator sees its own influence; after each correction
consumes the hidden component, only `Pê` is retained.

| gate | result |
|---|---|
| C1 no divergence | 12/12 cases stable |
| C2 beats open-loop at final time | 12/12 |
| C3 median final improvement | 46.95× |

**The 46.95× must not be reported as a performance result.** At segment 0 the
ROM starts at truth, so `η₀ = Φ_h(V₀) − V₁` is the *exact* error. After
correcting, the state differs from truth only in `P`, so the next `η` recovers
the exact error again, and the recurrence stays near-exact indefinitely. The
closed loop tracks truth because it evaluates the full flow map from the
corrected state every segment.

**Honest contribution:** stability under feedback — a genuine open question,
since corrections feeding back into the defect history that generated them
could have destabilized — and confirmation that continuously removing hidden
error holds resolved error near its per-segment local level instead of letting
it accumulate. That is the project's central claim, directly demonstrated.

At the time Z2 was run this made the cost objection unavoidable: three full
flow-map evaluations per segment, ≈3× the cost of the solver being corrected.
§5.11–§5.13 resolve it.

### 5.8 Test R — resolution and Reynolds scaling

Two sweeps, deliberately separated. Sweep A raises `N` at fixed `ν` with `L`
**pinned at 8.0** (the bootstrap presets scale `L` with `N`, which is domain
extension, not refinement). Sweep B lowers `ν`.

| config | N | ν | dim(Q) | rank(1e-9) | modes@99.9% | STAR2 err | Q/P |
|---|---|---|---|---|---|---|---|
| A_N40_nu1 | 40 | 2.5e-3 | 383976 | 12 | 4 | 3.31% | 16.17 |
| A_N64_nu1 | 64 | 2.5e-3 | 1572840 | 13 | 4 | 8.23% | 12.05 |
| A_N80_nu1 | 80 | 2.5e-3 | 3071976 | 14 | 4 | 9.58% | 11.44 |
| B_N64_nu0.4 | 64 | 1.0e-3 | 1572840 | 16 | 5 | 13.42% | 10.26 |
| B_N80_nu0.16 | 80 | 4.0e-4 | 3071976 | 21 | 6 | 17.66% | 9.10 |
| B_N96_nu0.0625 | 96 | 1.56e-4 | 5308392 | 27 | 8 | 20.00% | 8.70 |

- **R1 (grid):** rank 12 → 14 across N = 40 → 80. **N=40 already resolves
  these flows.** Grid resolution was never the limitation. This permanently
  closes one line of referee attack.
- **R2 (Reynolds):** rank 12 → 27, modes@99.9% 4 → 8. Lowering ν does enrich
  the dynamics — **but weakly.** Even at ν = 1.56e-4 with N = 96, 99.9% of the
  trajectory's energy sits in 8 modes. **These flow families remain
  intrinsically low-dimensional.** This is a real limitation of the testbed
  and must be stated.

**Confounds to disclose.** `n_target` is a fixed 24-mode set at every N, so
dim(Q) grows almost entirely with near-empty high-wavenumber modes. And in
sweep A the STAR2 error triples while rank barely moves, meaning
`topology_pair_L2` resolves more of the analytic initial condition at higher
N — the flows are not identical across the sweep.

**A mild negative:** the Q/P reservoir ratio falls monotonically with rank,
16.17 → 8.70 (Spearman −1.000), against the ≈23× reported in Test Q. The
hidden reservoir is less dominant as dynamics get richer.

**R3 as originally written was invalid** — see §6.2.

### 5.9 Test G1 — POD-Galerkin genericity: NULL

An anchored POD-Galerkin ROM was substituted for STAR2 with the P/Q split, W
estimator and intervention protocol held byte-identical.

**Result: null, not negative.** Baseline P-errors came out at ≈1e-7 — seven
orders below STAR2's. POD rank 10–20 captured 100% of snapshot energy. The
basis was built from snapshots of the *same trajectory the ROM was asked to
reproduce*, and the trajectory is numerically rank ≈13, so the ROM reproduced
it to truncation precision. **There was no accumulated hidden error to
correct**, and the gains scattered around 1.0 as noise-correcting-noise
should.

A `NULL TEST` gate was added so this failure mode announces itself.

**Uncomfortable finding to disclose regardless:** a 13-mode POD-Galerkin ROM
reproduces these flows to 1e-7 while STAR2, with hundreds of resolved modes
plus two hidden directions, sits at percent-level error. A referee will make
that comparison.

**Genericity remains untested.** The corrected design (training window only,
low rank, null gate) has not been run.
### 5.10 Test R3-fix — leakage concentration, correctly measured

The original R3 probed `T = P DΦ_h Q` with **random Gaussian** directions.
That was wrong: random directions in a 3×10⁶-dimensional `Q` have essentially
no overlap with the few directions error actually occupies. The correct
measurement restricts `T` to the span of *actual accumulated Q-errors*, as
Tests R/S/T originally did.

Nine anchor times, Gram-Schmidt to an orthonormal empirical basis, then `T`
applied to each basis vector.

| config | empirical rank | top-1 energy | **top-3 energy** | top-3 realized |
|---|---|---|---|---|
| N40_nu1 | 9 | 0.3833 | 0.8906 | 0.9995 |
| N64_nu1 | 9 | 0.4276 | 0.9698 | 0.9997 |
| N80_nu1 | 9 | 0.4369 | 0.9544 | 0.9979 |
| N64_nu0.4 | 9 | 0.4007 | 0.9331 | 0.9991 |
| N80_nu0.16 | 9 | 0.3696 | 0.8936 | 0.9998 |
| N96_nu0.0625 | 9 | 0.3785 | 0.9190 | 0.9999 |

**Mean top-3 energy 0.9268, sd 0.0320, Spearman against dim(Q) −0.029** over
a 14× growth in dim(Q) and 16× drop in ν. No degradation trend **over the
tested range** — this is evidence, not an asymptotic theorem.

**Scope of the claim.** The object measured is `T` restricted to

```
E_Q = span{ Q e(t₁), …, Q e(t₉) }
```

the empirically occupied hidden-error subspace, NOT `T` on all of `Q`. The
defensible sentence is: *within the empirically occupied hidden-error
subspace, unresolved-to-resolved leakage is strongly concentrated — three
directions carry 92.7% of the restricted operator energy across the tested
resolution and viscosity range.* Arbitrary directions in `Q` were deliberately
not probed, because §6.2 established that probing them measures a different
and uninformative operator.

**Non-circularity check.** An earlier 4-anchor version gave 0.979. Doubling the
ambient subspace from 4 to 9 dropped top-3 by only 0.052; a uniform spectrum
would have dropped from 0.750 to 0.333, a fall of 0.417 — eight times larger.
Three directions out of nine hold 92.7% against a uniform baseline of 33.3%.

**Structure note:** top-1 is only 0.37–0.44 and `σ₁/σ₃` is 1.29–1.66, so this
is **three comparably weighted channels**, not one dominant direction plus
stragglers. Stable across the whole sweep.

**This closes the objection raised at the outset of the audit** — that
low-rank leakage might be an artifact of dim(Q) being small at N=40 — with a
14× span in dim(Q) and 16× in viscosity, and unlike the closed-loop number it
carries no cost caveat.

### 5.11 Test C — coarse-grid defect estimation (pilot)

If the defect is dominated by large scales, a coarse flow map should suffice:

```
η_M = Prolong_{M→N} [ Φ_h^M ( Restrict_{N→M} V_k ) ] − V_{k+1}
```

Spectral restriction/prolongation verified exact (round-trip error 0.0 to
3e-16; coarse physical field matching the fine field on shared grid points).

At N=80 on `skew_tubes`:

| ν | M | cost | gain | vs full |
|---|---|---|---|---|
| 2.5e-3 | 20 | 0.016× | 1.7546 | 1.001× |
| 2.5e-3 | 80 | 1.000× | 1.7524 | — |
| 4.0e-4 | 20 | 0.016× | 1.6555 | **1.035×** |
| 4.0e-4 | 80 | 1.000× | 1.5991 | — |

At low viscosity the coarse estimator *beat* full resolution, with a better
hidden-Q residual too (0.365 vs 0.400).

**This pilot over-generalized** — see §6.3.

### 5.12 Test CX — full protocol, coarse defect, N=40

12 cases × 3 anchors at N=40, matching Test X exactly.

| M | cost | median gain | p10 | win rate | beats −Q | med Qres |
|---|---|---|---|---|---|---|
| 40 (control) | 1.0000 | **1.650998** | 1.2999 | 100% | 100% | 0.0000 |
| 28 | 0.3430 | 1.648814 | 1.2999 | 100% | 100% | 0.1014 |
| 20 | 0.1250 | 1.395640 | 0.5586 | 77.8% | 91.7% | 0.9618 |
| 14 | 0.0429 | 1.128927 | 0.1966 | 55.6% | 83.3% | 1.8809 |
| 10 | 0.0156 | 0.298927 | 0.1070 | 11.1% | 100% | 12.2558 |

The control reproduces the X-v3 headline exactly. **M = 28 works universally**
(0.13% loss, p10 identical, 36/36 preserved). Below that it breaks, and it
breaks **by topology**:

| topology | minimum working M |
|---|---|
| periodic_shear | 14, 14, 14 |
| skew_tubes | 14, 14, 14 |
| vortex_ring | 28, 28, 20 |
| mixed_vortices | 28, 28, 20 |

At N=40, universal M=28 puts the estimator at 3 × 0.343 = **1.03 full-solver
segments — break-even.** The cost objection was *not* answered at N=40.

### 5.13 Test CS — does the required resolution scale with N?

The decisive question: is `M*` absolute (set by the wavenumber content of the
defect) or relative (a fixed fraction of N)? The M ladder was therefore in
**absolute mode counts**, identical at every N.

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

**Zero cases out of eight where `M*` grows.** The requirement is not merely
absolute — it *falls* as the fine grid is refined.

| N | worst-case M\* | **nominal** `3(M*/N)³` vs one solver segment |
|---|---|---|
| 40 | 28 | 1.029× (break-even) |
| 64 | 20 | 0.092× |
| 80 | 20 | **0.047×** |

**These are grid-point-count estimates, not measured times.** Pseudospectral
cost scales as `O(N³ log N)` rather than `N³`, and the pipeline also pays for
restriction and prolongation (which operate at the *fine* resolution and do
not scale down), GPU launch overhead, memory traffic, and the ROM step itself.
The correct claim until benchmarked is: *the three required M=20 coarse-flow
evaluations have a nominal 4.7% grid-point cost relative to one N=80 flow
evaluation.*

Holding `M* = 20` fixed (conservative, since it is still declining): 37× at
N=96, 87× at N=128, 295× at N=192.

**Mechanism.** At N=40 the fine solver is itself marginally resolved, so
"truth" carries grid-scale content the coarse map cannot reproduce. Refine the
grid and the defect becomes cleanly large-scale, so a coarser estimator
suffices.

**Two accuracy notes.** The regularization effect claimed from the pilot does
not survive: coarsening beats full resolution in only 8–9 of 24 windows at
N=80, median ratio 1.0000. Say "no loss," not "better." And the method's own
strength is flat in N (median gain 1.826, 1.866, 1.834 at N = 40, 64, 80) —
the correction is not getting stronger with resolution, only cheaper.

**This is the result that closes the cost objection**, with a measured scaling
argument rather than a single data point.

### 5.14 Test L — physical identity of the leakage directions: NEGATIVE

4 topologies × 4 viscosities = 16 configs at N=64, testing whether the three
leakage directions are a recognizable turbulence structure.

**L1 — no power law in ν.** Log-log fits (not rank correlations):

| direction | pooled exponent | R² |
|---|---|---|
| 0 | +0.001 | 0.0000 |
| 1 | +0.195 | 0.3120 |
| 2 | +0.182 | 0.1285 |

against a Kolmogorov prediction of 0.75. Per-topology dir0 fits span −0.059 to
+0.042. Over a 15.6× viscosity change a dissipation scale would move 7.9×;
`k_leak` moves 1.01× to 1.57×. `periodic_shear` fits with R² = 0.96 and slope
0.003 — "no dependence," fitted well.

**L2 — triadic signature, weak.** Median `k_leak` 5.43 vs median
`k_P + k_flow` 5.14 (ratio 1.057); correlation across 16 configs +0.50.
Suggestive, not compelling.

**L3 — no vortex-stretching alignment.** cos²(e₂) exceeds the isotropic 1/3 by
+0.004 (dir0), +0.018 (dir1), +0.006 (dir2). Vortex stretching would put it
far above. The directions are essentially randomly oriented in the strain
eigenframe.

**L4 — no structural preference.** Correlation with base enstrophy +0.388
against +0.377 with strain — statistically indistinguishable. Relative
helicity +0.024, i.e. none.

**Verdict: the leakage directions are numerically real, low-rank and robust,
but physically unremarkable.** There is no fluid-mechanical mechanism to
report, and JFM is not the venue.

**What L does establish, and it matters.** `k_leak` sits at 4.5–8.2 across
every topology and viscosity (median 5.4, CV 0.19). It is set by the resolved
band (`k_P ≈ 3`) plus the energy-containing scales (`k_flow ≈ 1–2.3`), **not**
by the dissipation range. That is *why* coarse-η works, why `M*` is small, and
why `M*` decreases as N grows: the leakage channel never moves to small scales
no matter how the grid is refined or the viscosity lowered.

This is the physical explanation the cost section needs, now measured on all
four topologies.


### 5.15 Tests F and TR — forced turbulence, and the triadic control

**Motivation.** The largest open risk was testbed dimensionality. Two attempts
to raise it failed instructively before the third worked.

**Failed attempt 1 (BB2/BB3): broadband initial conditions.** Random-phase
divergence-free fields with energy confined to bands up to |k|=18, 122 to
11512 populated modes. Trajectory rank stayed 4-11 for every band. **Reason:
trajectory POD rank measures dynamical exploration, not spectral breadth.** An
unforced decaying random-phase field is dynamically inert -- its snapshots are
the initial condition times a decay envelope. The broadband cases were in fact
*less* dynamically rich than skew_tubes (ROM error 0.05% vs 3-13%), because
random phases carry no coherent structure for the nonlinearity to act on.

A separate methodological error surfaced here: the first version capped POD
snapshots at 11, so measured rank could never exceed 11. **Test R's rank
figures used 33 snapshots and its top value of 27 may likewise be truncated.**

**Failed attempt 2: lowering viscosity further.** With no forcing, energy
cascades to the grid scale and nu must be large enough to dissipate it there.
The resolution guard correctly rejected configurations with up to 62% of
energy above 0.8x the dealiasing cutoff. **The admissible nu is bounded BELOW
by resolution, not above by decay.**

**What worked: Lundgren linear forcing.** `f = A*u` with **A held constant**,
which is essential -- a self-regulating A would make F non-polynomial and
destroy the exactly-quadratic property underpinning Y-v2. A is fixed after a
short self-regulating pilot; the analytic JVP gains a `+A*V` term.

Trajectory rank rose from 4-11 (unforced) to 17-35 (forced), with
`rank ~ nu^-0.273` (R^2 = 0.99 on both flows) and **not saturated**. The limit
became resolution: rank 50 would require nu ~ 8.7e-5 and N ~ 800. **Rank 50+
is unreachable on the available hardware**, and the paper must say so.

#### Test F — full chain on forced turbulence

Six configs, N=64, trajectory rank 17-35.

| flow | nu | r_traj | ambient | r_leak90 | Q/P | triad | median gain |
|---|---|---|---|---|---|---|---|
| mixed_vortices | 2.5e-3 | 17 | 19 | 6 | 8.13 | 167x | 1.57 |
| skew_tubes | 2.5e-3 | 21 | 19 | 5 | 4.97 | 200x | 1.44 |
| mixed_vortices | 1.0e-3 | 22 | 19 | 6 | 7.69 | 120x | 3.48 |
| mixed_vortices | 4.0e-4 | 28 | 19 | 6 | 8.71 | 130x | 3.47 |
| skew_tubes | 1.0e-3 | 29 | 19 | 5 | 9.34 | 206x | 1.92 |
| skew_tubes | 4.0e-4 | 35 | 19 | 5 | 10.55 | 243x | 4.74 |

**F1 — leakage rank is invariant.** `d(log r_leak)/d(log r_traj) = -0.189`
across a 2.1x span in trajectory rank at fixed ambient dimension.

**F2 — the causal correction survives forcing.** Every evaluated anchor won:
**15/15** against baseline and 15/15 against wrong-sign, max start-P mismatch
0.000e+00. Three further anchors were *not evaluated* because the estimator
left its validity regime (below).

**A new limitation, measured rather than assumed.** Theorem C's remainder is
`O(||e||^3)` and Test V validated second-order transport at percent-level
amplitudes. At 35% resolved error the `H[e,e]` term dominates, `e_hat`
diverges (one anchor reached `||Qe_hat||/||V|| = 25`), and `V + Qe_hat`
becomes a state where STAR2 is degenerate. **The truth-free recurrence has an
error-amplitude horizon of roughly 15%, and forced dynamics reach it faster
than decaying ones.** This should be stated as a validity condition, not
discovered by a referee.

#### Test TR — the triadic control, and a negative result

Test F reported triadic enrichment of 120-243x for the leading leakage
direction. That number is meaningless without a control. TR orders the *same*
19-dimensional empirical Q-error subspace two ways -- by error energy (POD of
the Q-error covariance) and by leakage (right singular vectors of the
restricted operator) -- plus a random floor and the leakage tail.

| ordering | n | median enrichment | p10 | p90 |
|---|---|---|---|---|
| dangerous (top leakage) | 30 | 182.7x | 123.6 | 234.6 |
| energetic (top error energy) | 30 | 151.9x | 34.8 | 317.2 |
| random | 30 | 60.1x | 14.2 | 141.0 |
| dangerous tail (least leakage) | 30 | **0.0x** | 0.0 | 0.2 |

**dangerous / energetic = 1.20x. NOT SEPARATED.** Enrichment against the
leakage singular value gives a log-log slope of +0.228 with R^2 = 0.063 -- no
monotone relation.

**Conclusion: "energy != danger" is NOT demonstrated as a triadic effect.** In
this testbed the energetic and dangerous directions largely coincide, so the
comparison cannot separate them. The 183x enrichment reported in Test F is a
property of the accumulated hidden-error subspace, not of the leaking
directions specifically. **The fluid-mechanical thread is closed, negatively,
for the second time (cf. Test L).**

Two findings survive and should be reported:

1. **Triadic content separates leaking from non-leaking directions by ~1000x**
   within the same subspace (182.7x vs 0.0x). Real and large, but confounded
   by the energetic/dangerous overlap.
2. **The accumulated error subspace is itself triadically enriched ~60x
   relative to arbitrary hidden directions** (random ordering vs null). ROM
   error preferentially accumulates in modes triadically connected to the
   resolved band. This is a statement about where error goes, not about which
   error is dangerous.

#### The clean cross-regime result

TR also recomputed leakage rank on a **9-anchor ambient**, matching the
decaying-flow runs exactly and removing the ambient-dimension confound:

| flow | nu | ambient | r_leak90 | top-3 |
|---|---|---|---|---|
| skew_tubes | 2.5e-3 | 9 | 3 | 0.9547 |
| skew_tubes | 1.0e-3 | 9 | 3 | 0.9285 |
| skew_tubes | 4.0e-4 | 9 | 3 | 0.9161 |
| mixed_vortices | 2.5e-3 | 9 | 4 | 0.8171 |
| mixed_vortices | 1.0e-3 | 9 | 4 | 0.8134 |
| mixed_vortices | 4.0e-4 | 9 | 5 | 0.7828 |

Decaying flows gave r_leak90 = 3 with top-3 ~ 0.93 on the same ambient.

**Headline, defensible as stated:** *leakage rank remains 3-5 on a matched
9-dimensional ambient across trajectory ranks spanning 8 to 35, generated by
both decaying and sustained forced dynamics.* This is the strongest
generalization result in the project and it carries no cost caveat.


### 5.16 Tests P1–P4 — genericity, and what governs the effect size

The largest open question in the record was whether observable-preserving
hidden-state correction is a property of STAR2 or of reduced models generally.
Four tests were needed; three of them failed, and the failures are recorded
because two of them were caused by bugs that also invalidate an earlier claim.

#### Test P1 — the kill gate: genericity under an honest protocol

Test G1 (§5.9) had returned NULL because the POD basis was built from the same
trajectory it was asked to reproduce. P1 enforces a real protocol:

* 12 **training** trajectories with randomised initial conditions (Dirichlet
  mixtures of the four canonical fields plus a broadband divergence-free
  perturbation at 12% of field norm)
* POD basis built from training only, then **frozen**
* rank chosen on a separate **validation** split, never on the test set
* 20 **unseen** test initial conditions
* statistics over independent **trajectories**, not anchors — this fixes the
  effective-n problem flagged throughout the record

| gate | result |
|---|---|
| N0 test is LIVE | 20/20 trajectories, baseline error 1.8–11.3% |
| G1 observable preserved | max P-mismatch **0.000e+00** |
| G2 beats baseline | **20/20** trajectories, 60/60 windows |
| G3 beats wrong-sign | **20/20** trajectories |
| G4 median gain | 1.0439×, 95% CI [1.0192, 1.0878], p10 1.0310× |
| oracle benefit captured | **1.0004** |

**The mechanism is not tied to STAR2.** But the effect was 9.5× weaker than
STAR2's — 4.2% error reduction against 39.4% — and the estimator was not the
limitation: it captured 100.04% of the oracle's benefit. There was simply less
exploitable hidden error.

#### Tests P2, P3 — two failed explanations

**P2 (reservoir ratio via rank sweep) — FAILED.** Spearman(R, gain) was +0.782
at rank 6 and **−0.277** at rank 10; the sign flipped and the pooled value
collapsed to +0.334. Median R moved only 3.46 → 4.77 across ranks, a 1.38×
span: rank has almost no leverage on the reservoir. Wrong knob.

**P3 (containment sweep) — RESULT VOID, see §6.8.** P3 augmented the POD basis
with a controlled fraction of the observable modes and reported a monotone
effect in 8/8 trajectories, concluding containment explained 16% of the gap.
The augmentation was constructing non-Hermitian directions. The number is
wrong.

#### Test P4 — the answer

Five variants on identical unseen test trajectories, with the observable
directions built correctly as **conjugate-pair (cos/sin) Hermitian-symmetric
combinations**:

| var | subspace | dim | median R | median base | median gain | reduction |
|---|---|---|---|---|---|---|
| A | POD only (frozen) | 10 | 4.53 | 5.28% | 1.0544 | 5.16% |
| B | POD + all of P | 154 | 13.85 | 1.62% | 1.5834 | 36.84% |
| C | POD + P + q₀ | 155 | 16.41 | 0.29% | 1.5978 | 37.41% |
| D | POD + P + q₀ + q₁\* | 156 | 25.01 | 0.02% | 1.5849 | 36.90% |
| E | P + q₀ + q₁\* (STAR2 subspace) | 146 | 19.87 | 0.03% | 1.5639 | 36.06% |
| — | STAR2, Test X-v3 | — | 16–23 | — | 1.6510 | 39.42% |

**J0 harness control PASSED**: variant E reproduces STAR2 to within 5.3%,
validating the comparison. (Two earlier versions of this harness FAILED J0 and
were debugged rather than reported — see §6.8.)

**Attribution of the A → E gap, log scale:**

| contribution | share |
|---|---|
| containment of the observable set (A→B) | **101.1%** |
| local trajectory jet (B→D) | 0.1% |
| dropping the global POD modes (D→E) | −1.2% |

**Gain behaves as a step function in containment.** Across B, C, D, E the gain
spans only 1.5639–1.5978 (2.2%), while the reservoir ratio spans 13.8–25.0 and
baseline error spans 0.02%–1.62%, an 80× range. **Neither the reservoir ratio
nor the ROM's accuracy moves the gain.**

**A separation worth stating on its own.** Adding the local trajectory jet cuts
baseline ROM error 80-fold (1.62% → 0.02%) and contributes essentially nothing
to the correction. *What makes a reduced model accurate is not what makes it
correctable.*

#### The statement for the manuscript

> Observable-preserving hidden-state correction is not tied to the STAR2
> construction. On a POD-Galerkin ROM under a frozen-basis, unseen-initial-
> condition protocol it reproduces at 1.58× against STAR2's 1.65×, **provided
> the reduced subspace contains the observable set**. Containment accounts for
> essentially the entire difference between the two ROM families; the reduced
> model's accuracy, its hidden-error reservoir ratio, and its local jet
> structure do not.

This is a design principle: build the ROM so that P lies inside the reduced
subspace, and hidden-state correction becomes available at close to full
strength regardless of what else the basis contains.


### 5.17 Test F — Theorem F, the correction-reachability bound

P4 (§5.16) showed the effect size is a step function in containment, with the
reservoir ratio and the ROM's accuracy explaining none of it. Theorem F derives
that from the structure of a projection ROM rather than correlating it.

**SCOPE — read before the statement.** Theorem F is about the **ROM flow map**,
written `Ψ_h^S`, NOT the full-order flow map `Φ_h` used everywhere else in this
document. The proof relies on the state increment lying inside the fixed
reduced subspace `S`, which is a property of the projection ROM and of nothing
else. Theorem F does **not** constrain the full-order causal response, and the
proof below establishes no such thing. An earlier draft used a bare `Φ`, which
invited exactly that misreading.

**Statement.** Let `Ψ_h^S` be the anchored projection-ROM flow map over one
segment with fixed reduced subspace `S`, so that from anchor `A` it returns
`A + s` with `s ∈ S`. For a hidden correction `c ∈ range(Q)`:

```
Ψ_h^S(V + c) − Ψ_h^S(V) = c + s_c ,      s_c ∈ S
```

Applying P and using `Pc = 0`:

```
Δ(c) = P[Ψ_h^S(V+c) − Ψ_h^S(V)] = P s_c  ∈  P·S
```

for EVERY hidden correction `c`. The observable effect of any hidden correction
on the ROM forecast is confined to the image of the reduced subspace under P. With `b` the baseline future observable error and
`ρ = ‖Π_{P·S} b‖ / ‖b‖`:

```
gain ≤ 1 / √(1 − ρ²)
```

**What does not appear in that bound:** the reservoir ratio, the ROM's
accuracy, or the basis used for the rest of `S`. That is exactly P4's result,
now derived rather than observed.

**Exactness.** The argument requires `S` to be identical on both branches. It
is therefore exact for static-basis variants over any number of segments, and
approximate where the basis is rebuilt per anchor.

**Result.** 20 unseen trajectories × 5 ROM variants × 3 anchors:

| var | subspace | dim(P·S) | ρ | ceiling | gain |
|---|---|---|---|---|---|
| A | POD only (frozen) | 10/288 | 0.766 | 1.555 | 1.0544 |
| B | POD + all of P | 144/288 | 1.0000 | ∞ | 1.5834 |
| C | POD + P + q₀ | 144/288 | 1.0000 | ∞ | 1.5978 |
| D | POD + P + q₀ + q₁\* | 144/288 | 1.0000 | ∞ | 1.5849 |
| E | P + q₀ + q₁\* (STAR2 subspace) | 144/288 | 1.0000 | ∞ | 1.5639 |

**300 windows, zero violations**, including every oracle arm (the bound covers
*every* hidden correction, so the oracle must obey it too — and does, 240/240).

Variant A is **reachability-limited**: only 77% of its observable error is
reachable by any hidden correction, and it achieves 70% of a hard ceiling of
1.555. Variants B–E are **dynamics-limited**: ρ = 1 removes the constraint
entirely and they land at 1.56–1.60 while their reservoir ratio spans 4.5–25.0
and baseline error spans 0.025%–5.3%, a 200× range. **ρ tracks the gain; R and
accuracy do not.**

**Design condition.** `P ⊆ S` removes the reachability constraint. Nothing
else in the ROM does.

**A sharper bound is available and untested.** `P·S` is the *geometric*
reachable set. The *dynamical* one is smaller: under local linear response with
`T = P DΨ_h^S(V) Q`, only `Range(T)` is actually attainable by infinitesimal
hidden perturbations, giving

```
Range(T)  ⊆  P·S  ⊆  P
```

and a tighter ceiling `1/√(1 − ρ_T²)` with `ρ_T = ‖Π_{Range(T)} b‖/‖b‖`. This
would connect Theorem F to the rank-3 leakage result (§5.10) — which measured
exactly this operator — and to Corollary D.1, as one theory rather than three
findings. The finite-amplitude version replaces `Range(T)` with the response
manifold `M_V = {Δ(c) : c ∈ range(Q)}`, whose tangent space at zero is
`Range(T)`, giving `G* = ‖b‖ / dist(−b, M_V)`. **Not yet derived or tested.**

**Gate note.** F2 was preregistered at "median gain/ceiling > 0.7" for variant
A and measured 0.6992 — a miss by 0.0008. The threshold was not moved. The
reportable contrast needs no gate: A runs at 70% of its ceiling while B runs
at an unbounded one.

Theorem F also explains three earlier results: P4's step function, P1's
puzzling oracle-capture of 1.0004 at low containment (the oracle was bounded
by the same near-unity ceiling), and why P2 and P3 failed (they correlated
gain against variables the bound does not contain).

---

### 5.18 Test KS — a second equation, and the effect-size law

**Motivation.** The largest remaining caveat was testbed dimensionality: NS
trajectory rank reaches only 8–35. Kuramoto–Sivashinsky was chosen because its
attractor dimension grows with domain length, buying high rank cheaply.

**Independent reimplementation.** New solver, no code shared with the NS
bootstrap: `u_t = −u·u_x − u_xx − u_xxxx`, rfft, 2/3 dealiasing, RK4 at
dx = 0.5, dt = 0.005. The nonlinearity is exactly quadratic as in NS, and the
analytic Jacobian matched central differences to **4.5e-13**, so Y-v2's
FD-exactness result carries over.

**The rank ladder works**, in seconds on CPU:

| L | modes | rank(1e-9) | modes for 99.9% |
|---|---|---|---|
| 22 | 23 | 28 | 11 |
| 50 | 51 | 66 | 23 |
| 100 | 101 | 132 | 30 |
| 200 | 201 | 196 | 36 |
| 400 | 401 | 199+ | 43 |

#### The causal signature transfers

Across L ∈ {100, 200, 400} and P ∈ {2, 3, 4, 6} modes:

| | value |
|---|---|
| `+q̂` beats `−q̂` | **12/12 configurations** |
| median gain | 1.0350 |
| median wrong-sign gain | 0.9880 |
| wrong-sign below 1 | 11/12 |
| `ΔP` | 0.000e+00 throughout |

**The correction has the right sign everywhere.** What does not transfer is the
magnitude: 3.4% error reduction against Navier–Stokes' 39.4%.

#### Two hypotheses tested and rejected

**The reservoir ratio does not explain it.** R reaches **24.84** at L=400 with
P at 2 modes — squarely in NS's 16–23 range, in a system of trajectory rank
199. But `Spearman(R, gain) = −0.580`: R = 24.84 gives gain 0.994 while
R = 1.13 gives 1.044. The correlation runs backwards.

**A CORRECTION TO THE RECORD.** An earlier version of this section concluded
that KS "cannot reach R ≫ 1" and that the framework "does not reproduce."
Both statements were wrong. They were drawn from large-P configurations only,
where R is suppressed by construction — a sampling error, not a property of KS.

**Theorem F does not explain it either.** `P ⊆ S` holds in every KS
configuration, so ρ = 1 and the reachability ceiling is infinite.

#### The effect-size law

The gain is *exactly* determined by two quantities. With `b` the baseline
observable error and `Δ` the oracle's observable effect:

```
gain = ‖b‖/‖b + Δ‖ = 1/√(1 + 2λ cos θ + λ²)
λ = ‖Δ‖/‖b‖ ,   cos θ = ⟨b,Δ⟩/(‖b‖‖Δ‖)
```

Algebra, not a model. **Verified to 2.22e-16 on all 12 KS configurations.**

| L | P modes | R | λ | cos θ | predicted | observed |
|---|---|---|---|---|---|---|
| 100 | 2 | 2.80 | 0.0577 | −0.586 | 1.0371 | 1.0371 |
| 100 | 6 | 0.66 | 0.0814 | −0.637 | 1.0447 | 1.0447 |
| 200 | 2 | 17.19 | 0.1213 | +0.013 | 0.9967 | 0.9967 |
| 200 | 3 | 8.39 | 0.1335 | −0.516 | 1.0584 | 1.0584 |
| 400 | 2 | 24.84 | 0.0482 | +0.062 | 0.9953 | 0.9953 |
| 400 | 4 | 11.44 | 0.0992 | −0.342 | 1.0352 | 1.0352 |

λ spans only 0.048–0.134: the hidden contribution is 5–13% of the observable
error's magnitude, so even perfect anti-alignment caps the gain near 1.14. Where
cos θ turns positive the gain falls below 1 — the two configurations with
gain < 1 are exactly those.

**Prediction for Navier–Stokes.** Gain 1.65 requires `1 + 2λcosθ + λ² = 0.367`,
i.e. λ ≈ 0.5–0.9 with cos θ ≈ −0.80 to −0.88. **Test LAM is written and not
yet run.** If NS lands there, the magnitude gap is a λ gap and both systems are
described by one law.

**Why this matters more than the two rejected hypotheses.** λ and cos θ are
measurable *a priori*, before any correction is attempted. The law is therefore
an applicability test: it says in advance whether the method will help.

**Standing interpretation, untested.** KS's observable modes are the linearly
unstable ones (`k < 1` grow), so their error self-amplifies and the hidden
contribution rides on top — small λ. NS's observable error appears to be driven
from unresolved scales — large λ. If that holds, λ is predictable from the
linear stability structure of the observable set, which would be a genuinely
useful design rule. §0.5.

---

## 6. Corrections and retractions

Recorded explicitly. Several of these were caught only because of the
provenance rules in §3.

### 6.1 The original X headline was inflated by float32

1.914705× → **1.650998×**, and 47.77% → **39.42%**. Cause localized to
`periodic_shear`, whose fp32 baseline error was up to 11× too large (§5.3).
The `repro` control reproduces the original number exactly on the new code, so
this is a precision effect, not a refactor artifact.

### 6.2 R3 measured the wrong operator

Probing `T` with random Gaussian directions instead of the empirical error
subspace. Random directions have overlap `≈√(3/n)` with a rank-3 subspace in
`n ~ 10⁶`, so the resulting 0.45 was uninformative. Corrected in §5.10; the
original Tests R/S/T were **not** contradicted.

### 6.3 The coarse-η pilot over-generalized from one topology

Test C ran `skew_tubes` only — one of the two *tolerant* topologies. The
leakage spectra used for the "60% of energy at |k| ≤ 5" claim were
`skew_tubes`-only as well. Both claims were generalizations from a favorable
case; CX (§5.12) showed `vortex_ring` and `mixed_vortices` need M = 28 where
`skew_tubes` needs 14.

### 6.4 The Re^(3/4) claim was not evidence for the exponent

An earlier "+1.000 Spearman against Re^(3/4)" was reported from six
`skew_tubes` configs. Re^(3/4) is a *monotone function of 1/ν*, so a rank
correlation against it returns ±1 for any increasing relationship and carries
no information about the exponent. The proper log-log fit (§5.14) returns
+0.001 with R² = 0.0000. **Retracted.**

### 6.5 Gate-design errors

- Y-v1 preregistered a G5 that was never implemented or printed.
- X-v2's G1 keyed the expected dtype off a free-text tag string rather than the
  requested precision, so a correct run reported FAIL.
- X-v2's N1 was preregistered for the direction family but coded to fail on
  either family, so the informative amplitude result registered as a failure.
- CS's S1 used `max/min`, which cannot distinguish `M*` growing 2.8× (fatal)
  from `M*` shrinking 2.8× (excellent). It flagged the project's best cost
  result as a failure.
- Several runs silently executed a stale file because a Kaggle dataset version
  was not remounted. Fixed by `BUILD_ID` banners and source self-assertions.

**Lesson worth stating in the methods:** gates must be directional, must own
their own denominators, and must be able to declare a test null.


### 6.11 Three claims audited against the original notebooks

The Colab notebooks covering Tests A–W were recovered and every numerical claim
in §4 was checked against the preserved cell outputs. Seventeen figures matched
exactly (Test V's 0.00010046 and 262.442394×, Test S's 0.99999410, Test T's
98.524% and 99.954%, Test U's 20.599/9.762/12.926 and slopes 1.0398/2.0379,
Test O's 0.999694, Test P's 1.058790, Test Q's 23.299×, Test R's 85.8261% and
0.99855568, and others). Three did not.

**1. Test Q's "46%" — CORRECTED to ~45%.** The printed late medians are
P-only 0.56809%, Q-only 0.45616%, full 1.01682%. The hidden block's share is
0.456/1.017 = **44.9%**, or 44.5% as a fraction of the two injections. 46%
cannot be reproduced from any combination of the printed quantities.

**2. Test T's leave-one-case-out regression — RETRACTED.** `R² ≈ 0.771`,
RMSE ≈ 0.116 and ≈ 0.179 appear nowhere in any of the four recovered notebooks.
A text search across every cell source and output found only coincidental
matches (a bath-RHS percentage, an unrelated gauge value). Either the
regression was run in a session that was never saved, or it was never run. The
qualitative claim it supported — that the warning metric had predictive value
despite failing its gate — is retained on the basis of the Spearman figures
that ARE in the output (0.8636 vs 0.8409 baseline).

**3. Test U's case selection — VINDICATED, and the record was too modest.**
Five cases were run, not three, and the two omitted gave 58.197× and 219.465×.
An earlier draft of this audit flagged that as possible cherry-picking. It is
not: the runner labels the cases explicitly, and the three quoted are exactly
the three marked **HARD**, while the two omitted are marked **CONTROL**:

| case | class | improvement |
|---|---|---|
| vortex_ring ν=0.25 | HARD | 20.599× |
| skew_tubes ν=0.25 | HARD | 9.762× |
| mixed_vortices ν=0.25 | HARD | 12.926× |
| periodic_shear ν=1 | CONTROL | 58.197× |
| vortex_ring ν=4 | CONTROL | 219.465× |

Quoting only the HARD cases understates the result. The controls belong in the
paper as evidence the correction is safe on easy cases, not merely effective on
hard ones.

**Test X IS recovered.** A further notebook was located containing the full
dual-GPU console log for Test X — all 36 intervention windows and the complete
summary block. Every headline figure verifies exactly: median EST-Q gain
1.914705×, p10 1.353768×, wrong-sign 0.675936×, oracle 1.914639×, reduction
47.773%, P mismatch 0.000e+00, all six gates True. The record's earlier note
that X's statistics were "recomputed from the console log" is now backed by the
log itself. The per-window `RAf` values (9.07e-08, 2.99e-08, 1.13e-07) are also
visible, which is the float32 evidence trail that led to the Y-v1 audit and the
1.9147 → 1.6510 correction.

**THE ARCHIVE IS RECOVERED.** A further Kaggle notebook was located containing
the console output for essentially every post-X test: X-v2, X-v3 and the alpha
sweep, Z, G1 (POD genericity), R (resolution), R3-fix, the leakage-direction
analysis, C, CX, CS, L (physical identity), the broadband attempts BB2/BB3,
the forced-turbulence probe, F, TR, the wall-clock benchmark, and P1/P2/P3.

Spot-checked against the record, the three highest-weight results verify
exactly:

| claim | recovered output |
|---|---|
| X-v3 median gain 1.650998× | 1.650998× |
| X-v3 p10 1.299859× | 1.299859× |
| X-v3 reduction 39.417% | 39.417% |
| X-v3 P mismatch 0.000e+00 | 0.000e+00 |
| α sweep argmax α = 2.00, 2.1445× | argmax alpha = 2.00 (gain 2.1445x vs 1.6510x) |
| Z1 cos(b_PP, b_QP) = +0.8196 | +0.8196 |
| Z1 α*_lin 2.0289 / α*_quad 1.9850 | 2.0289 / 1.9850 |
| Z2 12/12 stable, 46.95× | 12/12 PASS, 46.9481× |
| benchmark N=40 measured 3.0613 | 3.0613 |
| benchmark N=64 measured 0.7110 | 0.7110 |
| benchmark N=80 measured 0.3443 | 0.3443 |
| benchmark median measured/nominal 7.35× | 7.35× |

**Remaining provenance gap.** Only the most recent session's work — the KS,
CGL, FD-sensor, adaptive, master, isolate and amplitude experiments — exists
solely as console output in the working record. Those scripts are archived and
all of them are cheap CPU reruns.

**Two provenance questions that cannot now be answered.** Which §4 results
predate each of the two viscosity-namespace fixes is no longer determinable
from surviving artifacts. And Test M-A, whose apparent order-2 result was
attributed to a float32 floor, was never rerun in fp64 — the correction rests
entirely on Test M-B.

### 6.10 An error-reduction percentage was misreported throughout

Gain and percentage reduction were conflated. **A gain of 1.65 is a 39.4%
error reduction, not 65%** (`1 − 1/1.65`). The figure "65%" appeared in the
executive summary and in §5.18, and in several working discussions, always as
the Navier–Stokes comparator against Kuramoto–Sivashinsky.

**Corrected contrast:** NS 39.4% vs KS 3.4%. The qualitative conclusion — an
order-of-magnitude difference in effect size — is unchanged, but every number
quoting 65% was wrong and has been fixed.

Caught by external review, not internally. Worth noting given that every other
entry in this section was found in-house.

### 6.9 The first KS conclusion was wrong and has been rewritten

An earlier §5.18 reported that Kuramoto–Sivashinsky "cannot reach R ≫ 1" and
that "the framework does not reproduce" there, and drew a regime condition from
it. Both claims were wrong.

**Cause:** the gain tests were run only at P = 16–48 modes, where R is
suppressed by construction. A sweep at P = 2–6 modes reaches **R = 24.84** at
L = 400, inside NS's measured range, in a rank-199 system.

**What replaced it:** the causal signature does transfer (12/12), R does not
govern the gain (Spearman −0.580), and the effect size is set exactly by λ and
cos θ. The corrected section stands as §5.18.

**Lesson.** The error was a sampling error — a conclusion drawn from one corner
of a two-parameter space. The eventual fix was to sweep the second parameter,
which took minutes. Three of the four explanatory attempts in this project
(§6.8 and this entry) failed for reasons of measurement design rather than
physics.

### 6.8 P3's containment result is retracted; two harness failures preceded P4

**The bug.** `augment_with_P` inserted each observable Fourier mode as an
independent complex unit vector. The target modes come in **12 conjugate
pairs**; adding them individually destroys Hermitian symmetry, so the
reconstructed field is not real. Verified directly: a single unit vector gives
imag/real = **1.000**, while the paired cos/sin construction gives
**0.00e+00**.

**Consequences.**

* **P3's quantitative result is void.** It reported containment explaining 16%
  of the POD/STAR2 gap. With the correct construction, P4 measures **101%**.
  P3's frac=1.00 arm gave gain 1.11× where P4's variant B gives 1.5834×.
  P3's monotone 8/8 trend survives only as "adding even malformed observable
  directions helped slightly."
* Two earlier P4 harnesses failed the J0 control before the bug was found:
  first with an `IndexError` (empty-basis template), then with variant E at
  gain 1.0001×, reservoir ratio **0.23** and baseline error *worse* than
  pure POD. That last combination — a 74-dimensional subspace containing all
  of P performing worse than a 10-dimensional one containing none of it — was
  the diagnostic that located the bug.

**Lesson for the methods section.** The J0 harness control existed only because
a reviewer-style question ("does your comparison reproduce the known result?")
was preregistered as a gate. It caught two broken harnesses. Without it, P4's
first two runs would have been reported as evidence that containment does not
matter — the exact opposite of the truth.

### 6.7 The cost claim was nominal; it has now been benchmarked, and the
### nominal model is WRONG (though the conclusion survives)

**Measured on a T4, fp64, one segment (20 RK4 steps, 2 fields):**

| N | M | nominal `3(M/N)^3` | fine solver | **measured** | measured/nominal |
|---|---|---|---|---|---|
| 40 | 28 | 1.029 | 238 ms | **3.061** | 2.98x |
| 40 | 20 | 0.375 | 238 ms | 3.056 | 8.15x |
| 40 | 14 | 0.129 | 238 ms | 3.069 | 23.86x |
| 64 | 28 | 0.251 | 1080 ms | **0.711** | 2.83x |
| 64 | 20 | 0.092 | 1080 ms | 0.745 | 8.14x |
| 80 | 28 | 0.129 | 2187 ms | **0.336** | 2.62x |
| 80 | 20 | 0.047 | 2187 ms | 0.344 | 7.35x |

**Why the nominal model fails.** Estimator wall time is essentially constant
(728-805 ms) regardless of M. Per flow-map call it is 6.07, 6.06, 6.09 ms at
M = 28, 20, 14 -- identical. **The coarse work is latency-bound, not
FLOP-bound**: below some size the GPU spends its time on kernel launches and
cost stops scaling as M^3. The fine solver, by contrast, is genuinely
FLOP-bound (5.95, 27.0, 54.7 ms per call at N = 40, 64, 80, tracking N^3 to
within 15%).

**The correct cost model is a FLOOR, not a ratio.** The estimator costs a
roughly fixed ~750 ms on this hardware; the saving is whatever the full solver
costs above that.

| N | measured cost vs one solver segment |
|---|---|
| 40 | 3.06x — **more expensive** |
| 64 | 0.711x — 1.4x cheaper |
| 80 | 0.343x — 2.9x cheaper |
| 128 | ~0.084x — ~12x cheaper (extrapolated) |
| 192 | ~0.025x — ~40x cheaper (extrapolated) |

**Corrections to earlier claims in this document.** The "21x cheaper at N=80"
figure is wrong; the measured value is 2.9x. The nominal model overstates the
saving by 2.6-24x. The concern that restrict/prolong would dominate at large N
was also wrong -- it is 0.5-0.6% of estimator time.

**A clean practical result.** At N=80, M=28 and M=20 cost the same wall time
(736 vs 753 ms), so coarsening below M=28 buys nothing -- and 28 is exactly
the universally safe `M*` from Test CX. **Use M=28.** The coarsening question
answers itself once cost is measured rather than counted.

**Caveat.** The ~750 ms floor is hardware- and implementation-dependent.
Batching the three flow-map evaluations, or CUDA graphs, would lower it and
improve every number above. The paper should report the measured floor and
note that it is not fundamental.

#### (superseded) the original nominal-cost note

`3(M*/N)³` counts grid points. It does not account for the `log N` factor in
FFT cost, for restriction and prolongation (which operate at the fine
resolution and therefore do NOT scale down with M), for GPU launch overhead
and memory traffic, or for the ROM step. Until a wall-clock benchmark exists,

```
C_wall = t[restrict + 3 coarse Phi + prolong + update] / t[one fine FOM segment]
```

the phrase "21× cheaper" must not appear. If `C_wall` lands near 0.05–0.10 the
cost objection is genuinely retired; if the fixed-resolution restrict/prolong
overhead dominates at large N, the favorable scaling could be substantially
eroded. **This is the cheapest outstanding test in the project — minutes of
compute — and it should be run first.**

### 6.6 Provenance status of the original Test X artifacts

T, V and corrected W are artifact-backed. The original X CSVs were lost with
an ephemeral Kaggle session; the full 36-row console log survives and every
summary statistic was independently recomputed from it and matched to six
decimals. The X-v3 reruns supersede this concern entirely.

---

## 7. Prior-art positioning

**Status: literature search completed. Three papers require reading in full
before drafting; they are named below.**

### 7.1 Mori–Zwanzig — the nearest large neighbour

MZ closure for reduced models of Navier–Stokes is an established line:
non-Markovian LES closures applied to Burgers, homogeneous isotropic
turbulence, Taylor–Green and channel flow (Parish & Duraisamy); a priori
estimation of memory effects in nonlinear ROMs; renormalized ROMs with memory
for two-way resolved/unresolved transfer (Price & Stinis); RNN/LSTM memory
closures for POD-Galerkin ROMs; data-driven MZ for boundary-layer transition.
In all of these the unclosed terms are recast as a **memory integral over the
history of the resolved variables**.

**The distinction, which must be made explicitly in the introduction:**

| | Mori–Zwanzig | this work |
|---|---|---|
| object modified | the evolution equation `U̇_r` | the state `U_r` itself |
| mechanism | memory term added to the RHS | vector added in `range(Q)` |
| input | history of resolved variables | the model's own defect `η_k` |
| effect on `P` | changes resolved dynamics | `P(V+c) = PV` exactly |

MZ closes the *dynamics*; this work estimates and corrects the *state error*.
A referee who does not see that distinction drawn will assume it was not
known. Draw it in the first two pages.

### 7.2 Dual-weighted residual / goal-oriented estimation

Also established, and closer than previously acknowledged: DWR and ML error
estimation for projection-based ROMs (Blonigan & Parish, CMAME 2023);
adjoint-based h-refinement for ROMs (Carlberg); goal-oriented adaptive DEIM;
MORe DWR with on-the-fly basis enrichment; goal-oriented adaptive sampling.

**"energy ≠ danger" is the adjoint-weighting insight.** DWR exists precisely
because residual magnitude does not predict output error. The project's
retained slogan must credit that lineage rather than present it as new. What
DWR does *not* do is reconstruct the error vector, split it, and act on the
unresolved component — it estimates a magnitude to drive refinement.

### 7.3 Three papers to read in full before drafting

1. **"Accurate error estimation for model reduction of nonlinear dynamical
   systems via data-enhanced error closure"** (CMAME, 2023). Introduces a
   corrected ROM with a data-driven closure term **related to the local
   truncation error of the time integration scheme**. This is the closest
   published relative of the defect identity (Theorem C). *Read before
   writing Theorem C.*
2. **Huang & Duraisamy, "Predictive reduced order modeling of chaotic
   multi-scale problems using adaptively sampled projections"** (JCP, 2023).
   Uses the same exact orthogonal split `u = V u_r + V_⊥ u_r⊥` into resolved
   and unresolved reduced states, with a full state estimation strategy
   incorporating non-local information. They adapt the **basis**; this work
   corrects the **state** within a fixed projector.
3. **Operator Inference with state constraints** (2025). Departs from
   offline–online decomposition by reintroducing FOM information through
   online **state corrections** — structurally similar to the coarse-defect
   deployment of §5.11–5.13.

### 7.4 Revised novelty assessment

| result | status after the search |
|---|---|
| Theorem A / Lemma A.1 | Small, clean, low exposure. |
| Theorem B (superconvergence) | **High exposure** (Boreale–Collodi; derivative-enriched POD). Do not lead with it. |
| Theorem C (defect transport) | **High exposure** — see 7.3 item 1. The identity is not novel. |
| Theorem D (correction criterion) | Low. Elementary but load-bearing for the wrong-sign control. |
| **Corollary D.1 (optimal amplitude α\*)** | **Nothing comparable found. Probably the single most novel item in the project**, and it made a confirmed quantitative prediction. |
| Q-reservoir → leakage → correction picture | Overlaps MZ in *motivation*, differs in *object*. Defensible with explicit positioning. |
| The intervention protocol (`ΔP = 0` exactly, wrong-sign control, oracle arm) | No published analogue found. This is the methodological contribution. |

**What survives as the paper's claim to novelty:** not the defect identity and
not superconvergence, but the combination — reconstructing the error *vector*
rather than its magnitude, splitting it, intervening on `Q` alone with the
observable held bit-identical, the sign-reversed control isolating
directionality, and the theory-predicted optimal amplitude.

## 8. What is established, what is not

### Established

- Quartic observable superconvergence for STAR2, confirmed under an
  independent analytic Jacobian, with the first correction term measured.
- The acceleration residual is a pure floating-point floor (0.3% agreement
  with the machine-epsilon ratio across 12 cases).
- fp64 is required; the analytic Jacobian is strictly better and slightly
  cheaper.
- Hidden accumulated error is reconstructable from the model's own defect
  history: median hidden-Q residual 1.06e-4, 132/132 anchors improved.
- Hidden-only correction causally improves future resolved accuracy: 36/36,
  `ΔP = 0.000e+00` exactly, sign-reversed control loses 36/36.
- The optimal correction amplitude is predicted by theory and confirmed
  independently (1.985 vs 2.000).
- Leakage is rank-3 dominant, invariant across 14× in dim(Q) and 16× in ν.
- The estimator tolerates ~100% directional error in `q̂` with 2% loss.
- Closed-loop correction is stable across 12/12 cases.
- **The defect can be computed coarsely, and the required resolution decreases
  with N.** Wall-clock measured (§6.7): 0.34× one solver segment at N=80.
- Grid resolution was never the limitation (rank 12→14 across N=40→80).

### Not established

- ~~Wall-clock cost~~ — **now measured** (§6.7). The estimator is 2.9× cheaper
  than the solver at N=80 and more expensive at N=40. The nominal `3(M/N)³`
  model is superseded.
- **The truth-free recurrence has an error-amplitude horizon of ~15%**
  (§5.15). Beyond it the second-order expansion diverges. Forced dynamics
  reach that bound faster than decaying ones.
- ~~Genericity beyond STAR2~~ — **now established** (§5.16). Reproduces on
  POD-Galerkin at 1.58× vs STAR2's 1.65× under a frozen-basis, unseen-IC
  protocol, provided the subspace contains the observable set.
- Any fluid-mechanical identity for the leakage directions (tested, negative).
- Behaviour on a genuinely high-dimensional flow — even at ν=1.56e-4, 99.9% of
  energy sits in 8 modes.
- Behaviour on non-spectral discretizations. An OpenFOAM port faces a specific
  obstacle: the pressure Poisson solve is iterative, so `F` is no longer a
  deterministic smooth function of `U`, the FD Jacobian error becomes `σ/ε`
  and the Hessian error `σ/ε²`. Second-order defect transport does not port to
  a black-box iterative solver by finite differencing.

### Statistical caveats to disclose

- Effective *n* is 12 cases, not 132 anchors or 36 windows; anchors along one
  trajectory are not independent. Report case-level statistics as primary.
- Corollary D.1's `α*` prediction is validated for `α* ≲ 3`.
- The Q/P reservoir ratio declines with dynamical richness (16.17 → 8.70).

---

## 9. Recommended manuscript plan

**Venue: Journal of Computational Physics.** CMAME, SIAM J. Sci. Comput. and
Physical Review Fluids are alternatives. JFM is not the right home — the
result is about reduced models, not about flows, and Test L closed the
fluid-mechanical thread negatively. Nature is not a realistic target for this
subject matter.

**Suggested structure.**

1. **Introduction** — reduced models accumulate error in unobserved
   directions; that error is not inert.
2. **Setting and the P/Q split** — state the framework ROM-independently.
   STAR2 becomes Example 1, not the subject. Theorem A and Lemma A.1 stay but
   move to an application subsection. Do *not* lead with superconvergence
   (novelty exposure, §7).
3. **Where the error lives** — Tests O, P, Q. The hidden reservoir.
4. **How it leaks** — `T = P DΦ_h Q`, rank-3 concentration, R3-fix scaling.
   Position against Mori–Zwanzig and DWR here.
5. **Second-order transport** — Theorems C and D; Tests U, V.
6. **Truth-free reconstruction** — Test W.
7. **The causal intervention** — Test X-v3. This is the centerpiece. Emphasize
   the protocol: `ΔP = 0` exactly, wrong-sign control, oracle arm.
8. **Optimal amplitude** — Corollary D.1 and Test Z1. A theory-predicted,
   independently confirmed result.
9. **Cost** — Tests C/CX/CS. The estimator is cheaper than the solver, with
   `M*` decreasing in N, explained by the `k_leak` measurement from Test L.
10. **Limitations** — testbed dimensionality, genericity untested, effective
    *n*, fp64 requirement.
11. **Methods** — the provenance discipline of §3. State it plainly; it is a
    strength.

**Naming.** §5.16 settles this: the result is about reduced models, not about
STAR2, so the title should not contain `STAR2`. Something like *Observable-
Preserving Hidden-State Correction in Reduced Dynamical Models*. Keep `STAR2`
as the name of the specific construction used as Example 1. Number the
theorems.
Author-surname acronyms read as a claim to significance a first paper has not
earned, and would misdescribe scope if the framework proves ROM-generic.

**Authorship** should be settled explicitly (order, corresponding author)
before submission.

---

## 10. Highest-value work remaining

Ordered by leverage.

1. ~~A genuinely broadband testbed~~ — **DONE, §5.18.** Kuramoto–Sivashinsky
   delivers trajectory rank 28→199 cheaply, and the causal signature DOES
   transfer (12/12 configurations). The magnitude does not (3.4% vs 39.4%),
   and the governing variable is λ, not the reservoir ratio. The central open
   problem is now **what determines λ** (§0.5). Even at
   ν = 1.56e-4, 99.9% of trajectory energy sits in 8 modes and the numerical
   rank is 27. A referee can fairly say the mechanism was discovered on smooth
   decaying flows of near rank-10. The fix is an initial-condition family with
   a broad populated spectrum (target trajectory POD rank 50–200), not a
   bigger grid. Note this requires a new IC generator: `topology_pair_L2`
   produces the current four smooth families only. Then repeat the minimum
   essential chain: Q reservoir → leakage rank → W → X.

2. ~~Rerun genericity properly~~ — **DONE** (§5.16). The paper is about
   reduced models, not about STAR2. The title should reflect that.
3. **Prior-art review against Mori–Zwanzig and DWR.** Reading, not compute.
   Do it before drafting.
4. **Adjoint reformulation of `T`.** Could reduce cost further and situate the
   work in an established literature.
5. **Per-window `α*` rule.** Corollary D.1 gives `α*` in terms of `b_PP` and
   `b_QP`, neither directly observable — but the W recurrence already produces
   `ê`, so `Pê` estimates `b`. A truth-free optimal-amplitude rule would be a
   new capability worth ≈64% over the current fixed-α method.
6. **Closed-loop at coarse resolution.** Z2 established stability; CS
   established affordability. Combining them gives a deployable closure
   correction rather than an estimability result.

---

## Appendix A — reproducibility index

| test | script | key outputs |
|---|---|---|
| Y-v2 | `m3d_test_Y2_runner.py` | `m3d_Y2_jvp_fidelity.csv`, `m3d_Y2_order.csv` |
| X-v3 | `m3d_test_x3_alpha_sweep.py` | `m3d_X3_a1_*`, `m3d_X3_repro_*`, `m3d_X3_fp32_an_*` |
| Z1/Z2 | `m3d_test_z_alignment_closedloop.py` | `m3d_Z_z1_alignment.csv`, `m3d_Z_z1_closedloop.csv` |
| G | `m3d_test_g_pod_genericity.py` | `m3d_G_g1_interventions.csv` (null) |
| R | `m3d_test_r_resolution.py` | `m3d_R_r1_scaling.csv` |
| R3-fix | inline cell | `m3d_R3fix_leakage.csv`, `m3d_R3fix_directions.npz` |
| C | inline cell | `m3d_C_coarse_defect.csv` |
| CX | inline cell | `m3d_CX_coarse_full.csv` |
| CS | inline cell | `m3d_CS_mstar_scaling.csv` |
| L | inline cell | `m3d_L_leakage_identity.csv` |

Every runner prints a `BUILD_ID`, its own absolute path, and per-row
provenance sentinels.

## Appendix B — numerical settings of record

```
precision            float64 / complex128 (fp64 is load-bearing)
STAR2 JVP            analytic NS Jacobian (not finite difference)
projector            'mixed', 24 modes, k_P ~ 3, k_P,max = 5
segment              tau = 0.15, 20 RK4 substeps, TAU0 = 40/3
baseline horizon     12 segments (tau = 1.80)
interventions        tau = 0.90, 1.20, 1.50
forecast horizon     2 segments (tau = 0.30)
map_jet step         DERIV_REL = 1e-5
                     (NOTE: Phi_h is NOT quadratic, so this step has real
                      truncation error; the Hessian second difference
                      amplifies roundoff by 1/eps^2, and eps_machine^(1/4)
                      ~ 1.2e-4 is the textbook optimum. UNTESTED.)
cases                4 topologies x 3 viscosities (nu_mult 0.25, 1.0, 4.0)
base viscosity       2.5e-3
domain               L = 8.0, fixed
```
