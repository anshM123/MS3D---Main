# Results — second round of suggested experiments

Six experiments run since the last exchange, all CPU, all with predictions
registered in the source file before execution. Scripts are in the repository
under `analysis/`; raw outputs accompany each.

Summary: one clear limitation found, one mechanism partially explained, three
hypotheses closed, and two design errors of ours caught and corrected mid-way.
None of it changes the manuscript's central claims; all of it bounds them.

---

## 1. Model-form error — a new limitation, and the most consequential result

`analysis/modelform.py`, n = 100 per perturbed term

The reference evolves under the exact Kuramoto–Sivashinsky equation while the
model — both the ROM and the full map inside the estimator — uses one
coefficient scaled by 1 + δ. Each of the three terms perturbed in turn.

| δ | hyper gain | ratio | adv gain | ratio | diff gain | ratio |
|---|---|---|---|---|---|---|
| 0.000 | 16.72 | 0.99 | 16.72 | 0.99 | 16.72 | 0.99 |
| 0.005 | 12.74 | 0.95 | 11.99 | 0.92 | 11.09 | 1.32 |
| 0.020 | 6.35 | 1.95 | 11.45 | 1.05 | 6.00 | 2.15 |
| 0.050 | 2.98 | 4.12 | 7.41 | 1.41 | 3.16 | 3.90 |
| 0.100 | 1.81 | **6.12** | 4.39 | **2.35** | 1.74 | **6.27** |

ratio = predicted gain / realised gain. Causal signature 20/20 at every δ in
every term.

**What survives:** the causal signature and the qualitative degradation. Hidden
state error remains real and remains transported even when the equations are
wrong, and the gain falls as it must because a growing share of the reported
error is bias rather than state error.

**What fails:** the predictor. At δ = 0.10 the method retains about a tenth of
its benefit while reporting two thirds. The cause is structural: the bias
enters the ROM and the estimator identically, so the method is internally
consistent and cannot detect the inconsistency — it corrects accurately toward
its own attractor, which is no longer the true one.

**Unanticipated detail worth reporting:** advection is the mild case (2.35×
overconfidence, 26% of gain retained) against 6.1–6.3× and 10–11% for the two
dissipative terms. Model error in the dissipative terms is what breaks
calibration.

This is now a Results subsection in the manuscript and it bounds the abstract.

---

## 2. Transport-operator reachability — vacuous, and why

`analysis/rhoT.py`, n = 63 across four bands

ρ_T = **1.0000 in every window**, rank(T) = 12 = full observable dimension,
median oracle/ceiling = 0.024.

A map from roughly four hundred unresolved coordinates into twelve reported
ones is generically surjective, so Range(T) is the whole observable space and
the ceiling 1/√(1−ρ²) is infinite. This is structural, not a sampling artefact.

It also explains why our Theorem F bound is slack: it bound variants B–E at
ρ = 1.000 and bit only on variant A, a POD subspace deliberately built without
P. The geometric constraint binds only when the reduced subspace is
impoverished by construction. That explanation is now in the manuscript.

---

## 3. Correction effort — your correction to us was right

`analysis/effort.py`, n = 63

You pointed out that rank asks whether a direction is reachable with unlimited
intervention, not what it costs. We had conflated the two. Measuring the
minimum-norm correction E* = ‖K⁺b‖:

| band | E* | cond(K) | λ | gain |
|---|---|---|---|---|
| unstable | 11.07 | 10.63 | 0.912 | 8.86 |
| neutral | 3.21 | 3.56 | 0.949 | 12.68 |
| stable | 5.00 | 4.59 | 0.987 | 23.04 |
| damped | 12.23 | 4.03 | 0.983 | 19.43 |

**E1 PASS** — E* spreads 3.81×, so effort is *not* flat even though rank was.
Our earlier conclusion was premature and you were right to say so.

**E4 PASS** — E* from the truth-free forecast matches E* from truth at
Spearman +1.0000, agreeing to four digits. It is exactly computable online.

**E2 FAIL** (−0.192) and **E3 FAIL** (2.99×) — E* varies, is measurable, and
does not predict λ. Effort moves 3.8× while λ moves 8%.

So the geometric route is now closed properly rather than by an incomplete
test: neither rank nor effort predicts the effect size.

---

## 4. Channel occupancy — the first partial mechanism

`analysis/channels.py`, n = 63

Decomposing K = UΣWᵀ, a = Wᵀq, β = Uᵀb, so Δ = UΣa.

| band | participation ratio | cos_chan | cos measured | λ_chan | λ |
|---|---|---|---|---|---|
| unstable | 1.00 / 12 | −0.927 | −0.999 | 0.369 | 0.912 |
| neutral | 1.00 / 12 | −0.971 | −0.999 | 0.339 | 0.949 |
| stable | 1.00 / 12 | −0.968 | −1.000 | 0.288 | 0.987 |
| damped | 1.00 / 12 | −0.963 | −0.999 | 0.116 | 0.983 |

**A1, A2 PASS.** The response is carried by essentially one channel out of
twelve, and the channel decomposition reproduces the measured cos θ to **0.031**
across all four regimes. This is the alignment half of the effect-size law,
explained.

**A3 FAIL** (−0.818). λ_chan decreases monotonically where λ is flat, so the
linear picture gets the magnitude ordering backwards.

**Two designs were wrong before this one, and both produced spurious passes.**
Random probes captured a tenth of the error. Seeding the probe with q̂ captured
all of it but was circular — it makes q̂ the top singular direction by
construction. Only a fixed low-wavenumber basis, motivated by the measured
leakage band at wavenumbers 4.5–8.2, is independent of the error while
capturing 93% of it. A spurious sign in the channel inner product was also
found and fixed.

---

## 5. The amplification factor — magnitude is beyond second order

`analysis/amplif.py`, n = 63

A = ‖Δ(q)‖ / ‖Kq‖, with Kq the true linear response by central difference.

| band | A | λ_lin | λ | cos_lin | cos measured | quad/resid |
|---|---|---|---|---|---|---|
| unstable | 2.386 | 0.391 | 0.912 | −0.914 | −0.999 | 0.307 |
| neutral | 3.163 | 0.367 | 0.949 | −0.984 | −0.999 | 0.241 |
| stable | 5.480 | 0.293 | 0.987 | −0.947 | −1.000 | 0.231 |
| damped | 9.859 | 0.100 | 0.983 | −0.903 | −0.999 | 0.188 |

**F2 PASS** — A spans 4.13×, monotone with damping.

**F4 PASS** — A from the truth-free estimate matches A from truth at Spearman
+1.0000.

**F3 FAIL, and this is the substantive finding.** The quadratic term explains
only 19–31% of the residual, and its share *shrinks* as A grows. A truncation
that were nearly sufficient would do the opposite. λ's magnitude is not
recoverable from any second-order truncation.

Note cos_lin sits between −0.90 and −0.98 in all four bands: the linearisation
gets the direction roughly right everywhere and only the magnitude badly wrong.

---

## 6. The amplitude law — collapses below the operating point, not at it

`analysis/alaw.py`

Testing whether A is a function of relative amplitude x = ‖c‖/‖V‖ rather than
of the band, via the response curve R(x) = ‖Δ(sq)‖/‖Kq‖.

| x bin | spread in R across bands |
|---|---|
| 0.0003–0.0007 | 1.00× |
| 0.0007–0.0021 | 1.01× |
| 0.0021–0.0059 | 1.06× |
| **0.0059–0.0169** | **3.21×** |
| 0.0169–0.0485 | 1.45× |
| 0.0485–0.1389 | 1.15× |

Below x ≈ 0.006 the bands lie on one curve to within 6%. The natural operating
amplitude is x ≈ 0.031, and the collapse fails there.

R also turns over: roughly 0.010 → 0.030 → 0.101 → 0.315 → 2.157 → 1.846 across
the scale grid, so ‖Δ(2q)‖ < ‖Δ(q)‖. The response is non-monotone past the
natural amplitude, which is consistent with the ~15% amplitude horizon
documented for the defect recurrence and was reached here independently.

**A design error of ours, for the record.** An earlier version tested
monotonicity of A = R/s, which carries an explicit 1/s and must fall once the
response saturates. It peaked at s = 1 and dropped at s = 2 for that reason
alone. R is the object with physical content.

---

## Where the λ question now stands

Nine attempts. The question is bounded rather than open.

| attempt | outcome |
|---|---|
| alignment vs observable dimension | refuted, n = 224 |
| advective locality | refuted, n = 160 |
| closed form in κ | failed, n = 96; worse with the neglected term |
| transport range ρ_T | vacuous, ρ_T = 1 identically |
| frozen cross-system calibration | orders at +0.72; fails on pointwise observables |
| minimum-norm effort E* | varies, measurable, does not predict λ |
| channel occupancy | **explains the direction** (cos to 0.031) |
| second-order amplification | magnitude is beyond quadratic |
| amplitude collapse | holds below the operating point, fails at it |

**The statement we can defend:**

> The correction's alignment is explained: the natural hidden error occupies
> essentially one transport channel, and the channel decomposition reproduces
> cos θ to 0.031 across four regimes. Its magnitude is an amplification of
> 2.4–9.9× over the linear response that is not recoverable from any
> second-order truncation and is not a universal function of relative amplitude
> at the amplitudes where corrections are applied. It is, however, exactly
> measurable online: the amplification computed from the truth-free estimate
> matches the true value at Spearman +1.0000.

That is a well-posed opening for a second paper rather than a gap in this one.

---

## A note on where we are stopping

We are stopping the λ investigation here. The last four experiments each
returned a smaller increment than the one before, and two of them contained
design errors we caught only after running — a circular probe basis, and the
1/s in A. The failure modes are now in the code comments so they are not
repeated.

What remains outstanding on the manuscript is your earlier structural points,
which are addressed, plus our acknowledgements. We would rather submit and let
the λ question open the next paper than continue generating experiments against
diminishing returns.
