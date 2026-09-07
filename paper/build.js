const d = require("docx");
const fs = require("fs");
const {Document,Paragraph,TextRun,HeadingLevel,AlignmentType,ImageRun,Packer,
       Table,TableRow,TableCell,WidthType,ShadingType,BorderStyle,PageBreak,
       convertInchesToTwip,LevelFormat,Footer,PageNumber} = d;

const SER="Times New Roman", SANS="Arial";
const P=(t,o={})=>new Paragraph({
  spacing:{after:o.after??120,line:o.line??276},
  alignment:o.align,
  indent:o.indent,
  children:[new TextRun({text:t,font:o.font??SER,size:o.size??20,
    bold:o.bold,italics:o.italics,color:o.color})]});
const RUNS=(runs,o={})=>new Paragraph({
  spacing:{after:o.after??120,line:o.line??276},alignment:o.align,indent:o.indent,
  children:runs.map(r=>new TextRun({text:r.t,font:r.font??SER,size:r.size??20,
    bold:r.b,italics:r.i,color:r.c,superScript:r.sup,subScript:r.sub}))});
const H=(t,lvl)=>new Paragraph({
  heading:lvl,spacing:{before:260,after:130},
  children:[new TextRun({text:t,font:SANS,size:lvl===HeadingLevel.HEADING_1?24:21,
    bold:true,color:"000000"})]});
const EQ=(t,n)=>new Paragraph({
  spacing:{before:130,after:130},alignment:AlignmentType.CENTER,
  children:[new TextRun({text:t,font:"Cambria Math",size:20,italics:true}),
            ...(n?[new TextRun({text:"          ("+n+")",font:SER,size:20})]:[])]});
const FIG=(f,w,h)=>new Paragraph({
  spacing:{before:180,after:60},alignment:AlignmentType.CENTER,
  children:[new ImageRun({type:"png",data:fs.readFileSync(f),
    transformation:{width:w,height:h}})]});
const CAP=(lbl,txt)=>new Paragraph({
  spacing:{after:200},alignment:AlignmentType.JUSTIFIED,
  children:[new TextRun({text:lbl,font:SANS,size:17,bold:true}),
            new TextRun({text:" "+txt,font:SANS,size:17})]});

function TBL(headers,rows,widths){
  const total=widths.reduce((a,b)=>a+b,0);
  const cell=(txt,b,w,shade)=>new TableCell({
    width:{size:w,type:WidthType.DXA},
    shading:shade?{type:ShadingType.CLEAR,fill:shade}:undefined,
    margins:{top:60,bottom:60,left:90,right:90},
    children:[new Paragraph({spacing:{after:0},
      children:[new TextRun({text:txt,font:SANS,size:16,bold:b})]})]});
  return new Table({
    columnWidths:widths,width:{size:total,type:WidthType.DXA},
    rows:[new TableRow({tableHeader:true,
            children:headers.map((h,i)=>cell(h,true,widths[i],"E8EEF4"))}),
          ...rows.map(r=>new TableRow({
            children:r.map((c,i)=>cell(c,false,widths[i]))}))]});
}

const doc = new Document({
  creator:"A. Mishra, A. Senthilkumar",
  title:"Observable-preserving correction of hidden error in reduced models",
  numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,
    text:"\u2022",alignment:AlignmentType.LEFT,
    style:{paragraph:{indent:{left:400,hanging:200}}}}]}]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},
      margin:{top:1440,bottom:1440,left:1440,right:1440}}},
    footers:{default:new Footer({children:[new Paragraph({
      alignment:AlignmentType.CENTER,
      children:[new TextRun({children:[PageNumber.CURRENT],font:SER,size:18})]})]})},
    children:[

// ============================== TITLE ==============================
new Paragraph({spacing:{after:200},children:[new TextRun({
  text:"Observable-preserving correction of hidden error in reduced-order models",
  font:SANS,size:32,bold:true})]}),
RUNS([{t:"Ansh Mishra",b:true},{t:"1"},{t:"* and Aryan Senthilkumar",b:true},{t:"1"}],
     {after:60}),
RUNS([{t:"1",size:16},{t:" Independent researchers.",size:18},
      {t:"  *e-mail: ",size:18},{t:"correspondence to A.M.",size:18,i:true}],
     {after:260}),

// ============================== ABSTRACT ==============================
H("Abstract",HeadingLevel.HEADING_1),
P("Reduced-order models accelerate simulation by evolving a small subspace of a "+
  "high-dimensional state, but they accumulate error, and most of that error "+
  "collects in coordinates the model does not report. We show that this hidden "+
  "error can be reconstructed from the model's own inconsistency with the "+
  "governing equations — using no observations and no reference trajectory — "+
  "and that correcting it improves subsequent forecasts of the reported "+
  "quantities while leaving every reported quantity bit-for-bit unchanged. "+
  "Across 36 controlled interventions in three-dimensional Navier–Stokes the "+
  "correction reduced future observable error by 39.4% in every case, while a "+
  "sign-reversed control degraded it in every case. We derive a reachability "+
  "bound that determines what any hidden correction can achieve, confirmed "+
  "without exception over 600 comparisons, and an exact identity that predicts the "+
  "achievable improvement from two computable quantities. The identity holds to "+
  "better than five parts in 10¹² across four systems spanning quadratic and cubic "+
  "nonlinearity, spectral and finite-difference discretisation, and modal and "+
  "point-sensor observables. Because the improvement can be predicted before "+
  "acting, the method selects its own correction and declines when none would "+
  "help, outperforming a tuned ensemble variational scheme that consumes "+
  "observations the method never sees.",{align:AlignmentType.JUSTIFIED}),

// ============================== MAIN ==============================
H("Main",HeadingLevel.HEADING_1),
P("Simulating a turbulent flow, a combustor, or a climate subsystem at full "+
  "fidelity is often unaffordable, and reduced-order models (ROMs) substitute a "+
  "low-dimensional surrogate for the full state. Every such surrogate is wrong, "+
  "and a large literature addresses how wrong: a posteriori error estimators "+
  "certify accuracy, adjoint-weighted residuals identify where refinement pays, "+
  "and closure models supply the terms projection discards. What unites these "+
  "approaches is that they estimate or bound an error magnitude in order to "+
  "improve the model.",{align:AlignmentType.JUSTIFIED}),
P("We take a different object as the target. Partition the state by a fixed "+
  "projector P selecting the coordinates the model is required to report — "+
  "spectral modes, sensor readings, an engineering functional — and let Q = I − P "+
  "be its complement. Any error splits exactly as e = Pe + Qe. The resolved part "+
  "Pe is visible and measured. The unresolved part Qe is invisible to every "+
  "reported quantity, and it is where most accumulated error lives: in our "+
  "Navier–Stokes testbed the hidden component exceeds the visible one by a "+
  "factor of 23 at late times.",{align:AlignmentType.JUSTIFIED}),
P("Hidden error is not inert. Nonlinear dynamics transport a fraction of it back "+
  "into the reported coordinates, where it accounts for roughly 45% of late "+
  "single-segment drift. This raises a question that error estimation does not "+
  "address: can the hidden component be reconstructed and removed, and does "+
  "doing so improve what the model reports?",{align:AlignmentType.JUSTIFIED}),
P("Both halves are non-obvious. Reconstruction appears to require the reference "+
  "trajectory, which is the object the ROM exists to avoid computing. And a "+
  "correction confined to Q changes nothing observable at the moment it is "+
  "applied, so any subsequent improvement must arrive through the dynamics "+
  "rather than through the correction itself.",{align:AlignmentType.JUSTIFIED}),

H("Reconstructing hidden error without a reference",HeadingLevel.HEADING_2),
P("The route around the reference trajectory is the model's own inconsistency. "+
  "Let Φ be the full-order flow map over one segment and Vₖ the reduced state. "+
  "The defect",{align:AlignmentType.JUSTIFIED}),
EQ("ηₖ  =  Φ(Vₖ) − Vₖ₊₁","1"),
P("asks where the true equations would carry the model's current state relative "+
  "to where the model went. It requires no knowledge of the truth. Writing "+
  "eₖ = Uₖ − Vₖ for the true error, an exact identity follows by adding and "+
  "subtracting Φ(Vₖ), and Taylor expansion gives",{align:AlignmentType.JUSTIFIED}),
EQ("eₖ₊₁  =  ηₖ + Jₖ eₖ + ½ Hₖ[eₖ , eₖ] + O(‖eₖ‖³)","2"),
P("with J and H the first and second derivatives of the flow map. Every term on "+
  "the right is computable from the reduced trajectory alone, so the recurrence "+
  "can be run blind. Defect-based corrections have been used to make error "+
  "estimators independent of the time-integration scheme, where the defect is "+
  "the local truncation error of an imposed scheme measured on full-order "+
  "snapshots; equation (1) is a different object — the mismatch between the full "+
  "flow map and the reduced model's own step — computed online with no training "+
  "data, and used to reconstruct an error vector rather than to bound a "+
  "magnitude.",{align:AlignmentType.JUSTIFIED}),
P("Across 132 test points spanning four flow topologies and three viscosities, "+
  "the reconstructed hidden component matched the true one to a median relative "+
  "residual of 1.06 × 10⁻⁴, with a direction cosine of 0.99999999 and an "+
  "amplitude ratio of 1.0000108. The second-order term in equation (2) is "+
  "essential: retaining only the linear term degrades the reconstruction by a "+
  "median factor of 44.8, and every one of the 132 points improved when it was "+
  "included.",{align:AlignmentType.JUSTIFIED}),

H("Hidden state carries causal information",HeadingLevel.HEADING_2),
P("Because a correction c ∈ range(Q) satisfies P(V + c) = PV identically, the "+
  "intervention can be made rigorous. From a single reduced state we evolved "+
  "four branches with identical reported coordinates: the uncorrected baseline; "+
  "the truth-free estimate V + Qê; a sign-reversed control V − Qê of the same "+
  "magnitude; and an oracle V + Qe using the true hidden error. The measured "+
  "mismatch in the reported coordinates at the moment of intervention was "+
  "0.000 × 10⁰ — exactly zero, not approximately.",{align:AlignmentType.JUSTIFIED}),
FIG("fig/fig1.png",600,262),
CAP("Fig. 1 | The observable-preserving intervention.",
  "a, Four branches are evolved from one reduced state. A correction confined to "+
  "the unresolved block leaves every reported coordinate bit-for-bit unchanged, "+
  "so the branches are indistinguishable to any measurement at the moment of "+
  "intervention. b, Median forecast gain over 36 interventions in "+
  "three-dimensional Navier–Stokes (12 flow configurations × 3 intervention "+
  "times). The truth-free estimate improved the forecast in every window; the "+
  "sign-reversed control, of identical magnitude, degraded it in every window. "+
  "Bar labels give the number of windows improved."),
P("The estimate improved the future reported error in 36 of 36 windows, with a "+
  "median gain of 1.651 (39.4% error reduction) and a tenth percentile of 1.300. "+
  "The sign-reversed control degraded it in 36 of 36, median gain 0.676. The "+
  "estimate captured 100% of the oracle's benefit, so the truth-free "+
  "reconstruction is not the limiting factor.",{align:AlignmentType.JUSTIFIED}),
P("The wrong-sign arm is what makes this causal rather than correlational. For "+
  "small corrections the observable response is Δ(±c) = ±Tc + ½H[c,c] + O(‖c‖³), "+
  "so reversing the sign flips the leading directional transfer while leaving the "+
  "curvature unchanged. A perturbation of the right magnitude but arbitrary "+
  "direction cannot produce this asymmetry. Consistent with this, degrading the "+
  "estimate with random noise in Q at relative amplitudes from 0.03 to 1.00 — a "+
  "33-fold range — moved the median gain by 2%, because a random direction in a "+
  "space of dimension 10⁵ has negligible overlap with the few directions that "+
  "transport error into the reported block.",{align:AlignmentType.JUSTIFIED}),

H("What a hidden correction can reach",HeadingLevel.HEADING_2),
P("The improvement is bounded, and the bound is structural. For an anchored "+
  "projection ROM with reduced subspace S, one segment from anchor A returns "+
  "A + s with s ∈ S. For any c ∈ range(Q), the increment is c + sᶜ with "+
  "sᶜ ∈ S, and applying P annihilates c, so",{align:AlignmentType.JUSTIFIED}),
EQ("Δ(c)  =  P[Ψ(V + c) − Ψ(V)]  =  P sᶜ  ∈  P·S","3"),
P("for every hidden correction. The observable effect of any such correction is "+
  "confined to the image of the reduced subspace under P. Writing ρ for the "+
  "fraction of the baseline observable error lying in that image, the achievable "+
  "gain obeys",{align:AlignmentType.JUSTIFIED}),
EQ("gain  ≤  1 / √(1 − ρ²)","4"),
P("Neither the size of the hidden reservoir nor the accuracy of the reduced "+
  "model appears in this bound. We tested it on 20 unseen initial conditions "+
  "across five reduced-model variants at three intervention times — 300 windows "+
  "— constraining in each the truth-free correction and the oracle correction "+
  "separately, since the bound covers every hidden correction. There were no "+
  "violations in any of the 600 comparisons.",
  {align:AlignmentType.JUSTIFIED}),
FIG("fig/fig3.png",305,247),
CAP("Fig. 2 | The reachability bound.",
  "Achievable gain against the reachable fraction ρ, with the bound of equation "+
  "(4). A proper-orthogonal-decomposition subspace that does not contain the "+
  "reported coordinates (variant A) is reachability-limited: only 77% of its "+
  "observable error is attainable by any hidden correction, and it realises 70% "+
  "of a hard ceiling of 1.555. Variants B–E, whose subspaces contain P, face no "+
  "such constraint and are limited by the dynamics instead. Points are medians "+
  "over 20 unseen initial conditions."),
P("This yields a design condition. Containing the reported coordinates inside "+
  "the reduced subspace removes the reachability constraint; nothing else in the "+
  "model does. Attributing the difference between a plain POD subspace "+
  "(gain 1.054) and one containing P (1.583) on a logarithmic scale assigns "+
  "101.1% to containment, 0.1% to the local trajectory jet, and −1.2% to "+
  "retaining the global POD modes. Notably, adding the trajectory jet reduces "+
  "baseline model error eighty-fold while contributing essentially nothing to "+
  "correctability: what makes a reduced model accurate is not what makes it "+
  "correctable.",{align:AlignmentType.JUSTIFIED}),

H("An exact law for the achievable improvement",HeadingLevel.HEADING_2),
P("Within the reachable set, the realised improvement is fixed exactly by two "+
  "quantities. With b the baseline future observable error and Δ the observable "+
  "effect of the correction,",{align:AlignmentType.JUSTIFIED}),
EQ("gain  =  ‖b‖ / ‖b + Δ‖  =  1 / √(1 + 2λ cos θ + λ²)","5"),
EQ("λ = ‖Δ‖ / ‖b‖          cos θ = ⟨b, Δ⟩ / (‖b‖ ‖Δ‖)","6"),
P("Equation (5) is algebra rather than a model: λ measures how large the hidden "+
  "contribution to the observable error is, and cos θ how well it opposes that "+
  "error. Neither the hidden-reservoir ratio nor the reachable fraction appears, "+
  "which is why three candidate correlational explanations for effect size "+
  "failed before this identity was written down.",{align:AlignmentType.JUSTIFIED}),
FIG("fig/fig2.png",600,253),
CAP("Fig. 3 | The effect-size law.",
  "a, Equation (5) for four values of the alignment cos θ, with measured systems "+
  "overlaid. Gain rises sharply as λ approaches unity and the correction becomes "+
  "well-opposed to the existing error. b, Maximum relative discrepancy between "+
  "predicted and observed gain, per window, across four systems: "+
  "Kuramoto–Sivashinsky and Navier–Stokes (quadratic nonlinearity, spectral, "+
  "modal observables), complex Ginzburg–Landau (cubic nonlinearity), and "+
  "finite-difference Kuramoto–Sivashinsky observed through point sensors. "+
  "Dashed line, the threshold registered before the tests were run."),
P("The identity holds to at most 5.4 × 10⁻¹² across all four systems, and to "+
  "2 × 10⁻¹⁵ on two of them. It also explains variation within a single system: "+
  "across the twelve Navier–Stokes configurations λ ranges from 0.236 to 0.813 "+
  "and the Spearman correlation between λ and realised gain is 0.986, with "+
  "predicted and oracle gains agreeing to four decimals in all twelve. Where "+
  "cos θ turns positive the law predicts that correction should harm, and it "+
  "does: in one regime the gain falls to 0.966 while the sign-reversed arm rises "+
  "to 1.097.",{align:AlignmentType.JUSTIFIED}),

H("Where the law applies",HeadingLevel.HEADING_2),
P("Equation (5) is exact by construction, but predicting λ in advance requires a "+
  "linear-response argument, and that argument has a boundary. Holding the "+
  "solver, the equation, the discretisation and the observable dimension fixed "+
  "and changing only the character of the observable, the reconstruction error "+
  "of the predictive form separates by a factor of 200: 0.003 for modal "+
  "observables against 0.55 for point sensors.",{align:AlignmentType.JUSTIFIED}),
FIG("fig/fig4.png",600,250),
CAP("Fig. 4 | The validity boundary is set by the observation operator.",
  "a, Reconstruction error of the predictive form against the pointwise fraction "+
  "of the observable set, at fixed total dimension (24 directions, n = 160). One "+
  "point sensor among 23 modal directions already produces 49% of the saturated "+
  "error; three produce 84%; the curve saturates by a quarter. b, The "+
  "hidden-driven share λ falls monotonically as the observable set becomes more "+
  "pointwise, so pointwise observation changes the quantity being measured and "+
  "not merely its representability."),
P("The mechanism is directional, not a matter of scale. Testing linearity "+
  "directly by scaling the perturbation over three decades, the response ratio "+
  "converges for both observable types, so a linear limit exists in both cases "+
  "and the pointwise response is differentiable. What differs is where that "+
  "limit points: the cosine between the infinitesimal and finite-amplitude "+
  "responses converges to 0.965 for modal observables and 0.685 for pointwise "+
  "ones. The derivative is well defined and simply does not represent the "+
  "response at the amplitudes that matter, so reducing the perturbation cannot "+
  "recover the prediction. Magnitude nonlinearity is harmless — the modal "+
  "response is twice its linear prediction and the law still holds — whereas "+
  "directional rotation is fatal.",{align:AlignmentType.JUSTIFIED}),

H("Selecting the correction without a reference",HeadingLevel.HEADING_2),
P("Because equation (5) is built from quantities the reduced model can evaluate "+
  "for itself, the improvement can be predicted before acting. Propagating the "+
  "defect recurrence forward supplies an estimate b̂ of the future observable "+
  "error; evaluating the reduced model from perturbed states supplies the "+
  "response Δ̂ to each candidate correction; and the predicted gain is "+
  "Ĝ = ‖b̂‖ / ‖b̂ + Δ̂‖. No reference trajectory is consulted at any point.",
  {align:AlignmentType.JUSTIFIED}),
P("Ĝ predicts the realised gain with a Spearman correlation of 1.0000 on "+
  "Kuramoto–Sivashinsky, 0.9997 on Ginzburg–Landau, 1.0000 on the "+
  "finite-difference sensor system, and 1.0000 in three-dimensional "+
  "Navier–Stokes, with median relative error 0.0000. The method therefore "+
  "chooses among candidate corrections — hidden-only, observable-only, and a "+
  "range of amplitudes — and declines when no candidate is predicted to help.",
  {align:AlignmentType.JUSTIFIED}),
FIG("fig/fig5.png",600,302),
CAP("Fig. 5 | Adaptive selection and cost.",
  "a, Median forecast gain in two dynamical regimes, n = 60 windows each. "+
  "Hidden-only correction is dominant where the hidden-driven share λ is large "+
  "and near-useless where it is small; observable-only correction reverses. "+
  "Selecting between them using only the model's own prediction wins in both, "+
  "and exceeds an ensemble variational scheme that consumes observations the "+
  "selector never sees. b, Measured wall-clock cost of the estimator relative to "+
  "one full-solver segment, against grid resolution, with the grid-point cost "+
  "model for comparison. The estimator becomes cheaper than the solver it "+
  "corrects between N = 40 and N = 64."),
P("The regimes reverse completely. Where λ ≈ 1.04 the hidden-only correction "+
  "gives 16.34 and the observable-only correction gives nothing; where λ ≈ 0.09 "+
  "the ordering inverts and hidden-only correction is the weakest method tested. "+
  "The selector resolves this without seeing λ or any reference, choosing "+
  "observable-touching corrections in 51 of 60 windows in the first regime and "+
  "hidden-only in 51 of 60 in the second. In three-dimensional Navier–Stokes it "+
  "reached 3.811 against 2.119 for the best fixed strategy.",
  {align:AlignmentType.JUSTIFIED}),
P("The decomposition matters for interpretation. Holding the amplitude fixed at "+
  "unity, selection between correction subspaces yields only 1–2% over the best "+
  "fixed strategy; the remaining factor comes from amplitude selection. Subspace "+
  "selection buys correctness of regime, amplitude selection buys magnitude, and "+
  "both are required.",{align:AlignmentType.JUSTIFIED}),
P("Against an ensemble variational baseline the comparison is deliberately "+
  "unfavourable to our method: the baseline is given true observations of the "+
  "reported coordinates at the analysis time, with a background covariance "+
  "estimated from the model's own error ensemble, and it corrects the hidden "+
  "block through the observation via cross-covariance. It attained 1.28 and 3.03 "+
  "in the two regimes against 16.65 and 14.09 for the truth-free selector, and "+
  "was insensitive to its observation-error parameter across four decades. This "+
  "is a single-time, static-covariance scheme; we do not claim to have "+
  "outperformed data assimilation in general.",{align:AlignmentType.JUSTIFIED}),

H("Cost",HeadingLevel.HEADING_2),
P("The estimator evaluates the full flow map, which would be self-defeating if "+
  "it were required at full resolution. It is not: the directions that transport "+
  "hidden error into the reported block sit at wavenumbers 4.5–8.2 across every "+
  "topology and viscosity tested, set by the reported band and the "+
  "energy-containing scales rather than by the dissipation range. The defect can "+
  "therefore be computed on a coarse grid, and the coarse resolution required "+
  "falls as the fine grid is refined — in eight cases spanning three resolutions, "+
  "it decreased or held constant in all eight and increased in none.",
  {align:AlignmentType.JUSTIFIED}),
P("Measured wall-clock cost of the estimator relative to one full-solver segment "+
  "is 3.06 at N = 40, 0.711 at N = 64 and 0.343 at N = 80. A grid-point count "+
  "underestimates this by factors of 2.6 to 24, because the coarse evaluation is "+
  "latency-bound rather than arithmetic-bound: per-call cost is identical at "+
  "coarse grids of 28, 20 and 14 modes, while the fine solver scales as N³ to "+
  "within 15%. The correct model is a fixed floor rather than a ratio, so the "+
  "saving is whatever the solver costs above that floor and grows with "+
  "resolution.",{align:AlignmentType.JUSTIFIED}),

H("Generality",HeadingLevel.HEADING_2),
P("The mechanism is not specific to the reduced model in which it was found. "+
  "Under a frozen-basis protocol — basis built from twelve training "+
  "trajectories, rank selected on a separate validation split, evaluation on "+
  "twenty unseen initial conditions, statistics over independent trajectories — "+
  "a proper-orthogonal-decomposition ROM showed the same behaviour: improvement "+
  "in 20 of 20 trajectories and 60 of 60 windows, the sign-reversed control "+
  "losing in 20 of 20, and 100.04% of the oracle benefit captured.",
  {align:AlignmentType.JUSTIFIED}),
new Paragraph({spacing:{before:60,after:120},children:[new TextRun({
  text:"Table 1 | Transfer across systems.",font:SANS,size:17,bold:true})]}),
TBL(["System","Nonlinearity","Discretisation","Observable","Causal signature"],
 [["Navier–Stokes","quadratic","spectral","modal","36/36"],
  ["Kuramoto–Sivashinsky","quadratic","spectral","modal","12/12 configs"],
  ["Ginzburg–Landau","cubic","spectral","modal","120/120"],
  ["Kuramoto–Sivashinsky","quadratic","finite difference","point sensors","36/36"]],
 [2100,1500,1700,1400,1650]),
P("A prospective test on complex Ginzburg–Landau, with four predictions "+
  "registered before the run, passed all four at n = 120. Its cubic "+
  "nonlinearity matters: the other systems are exactly quadratic, a property "+
  "the estimator exploits, and the identity's survival there shows it is not an "+
  "artefact of that structure.",{align:AlignmentType.JUSTIFIED}),
P("The observation operator generalises to instrument readings. Replacing the "+
  "spectral projector with point velocity probes in a three-dimensional flow, "+
  "the correction left all twenty-four probe readings identical to a relative "+
  "3.1 × 10⁻¹¹ while improving the next reading from those same probes by 46.8%, "+
  "in 36 of 36 windows.",{align:AlignmentType.JUSTIFIED}),
P("Run in closed loop over six segments and gated on its own predicted "+
  "improvement, the method reduced observable error from 6.276% to 0.129% — a "+
  "factor of 48.5 — without divergence in any of twelve configurations, and "+
  "with a median final error fourteen times the estimator's own noise floor, "+
  "confirming that the loop corrects rather than reconstructs.",
  {align:AlignmentType.JUSTIFIED}),

H("Discussion",HeadingLevel.HEADING_2),
P("The results establish that a reduced model's unreported state carries "+
  "recoverable, causally relevant information about the future of what it does "+
  "report, and that the information can be extracted from the model's own "+
  "inconsistency with the governing equations. Two bounds delimit what follows: "+
  "equation (4) fixes what any hidden correction can reach, and equation (5) "+
  "fixes what it will achieve within that reach.",{align:AlignmentType.JUSTIFIED}),
P("The practical consequence is a correction that knows its own worth in "+
  "advance. Because Ĝ predicts the realised gain essentially exactly, the method "+
  "is never a gamble: it selects the correction that suits the regime and "+
  "declines when none would help. Fixed-amplitude correction degraded 4.2% of "+
  "windows in one regime; predicted-amplitude selection eliminated all of them, "+
  "with a worst case of exactly no change. This occupies a gap that data "+
  "assimilation cannot: assimilation requires observations, and this requires "+
  "none.",{align:AlignmentType.JUSTIFIED}),
P("Three limitations bound the claim. The predictive form of the effect-size law "+
  "applies to modal observables and degrades steeply once the observable set "+
  "contains pointwise functionals — the identity remains exact, but λ ceases to "+
  "be predictable in advance. Hidden-state correction is the wrong instrument "+
  "when λ is small, and although the method detects this and switches, the "+
  "detection rather than the correction is what carries that regime. And the "+
  "estimator is cheaper than the solver only above roughly N = 64; below that it "+
  "buys accuracy at a cost premium.",{align:AlignmentType.JUSTIFIED}),
P("The most consequential open question is what sets λ. It is orderable by a "+
  "quantity computable without a reference — the ratio of hidden-block to "+
  "observable-block response, which tracks its own truth-free estimate with "+
  "Spearman correlation 1.0000 and orders λ across a 24,000-fold range — but no "+
  "closed form predicts it, and two candidate mechanisms were tested and "+
  "rejected. Resolving this would convert a characterised domain of validity "+
  "into a predictive theory of which systems the method suits.",
  {align:AlignmentType.JUSTIFIED}),

// ============================== METHODS ==============================
new Paragraph({children:[new PageBreak()]}),
H("Methods",HeadingLevel.HEADING_1),

H("Governing equations and discretisation",HeadingLevel.HEADING_2),
P("Incompressible Navier–Stokes was solved pseudospectrally on a periodic box "+
  "of side 8.0 in rotational form, with Leray projection and two-thirds "+
  "dealiasing applied after every nonlinear evaluation, and advanced by "+
  "classical fourth-order Runge–Kutta. The right-hand side is exactly quadratic "+
  "in the state, so central differencing of the Jacobian carries zero truncation "+
  "error; the entire finite-difference residual is floating-point roundoff "+
  "amplified by the inverse step. This was verified directly: across twelve "+
  "flow-viscosity cases the single- to double-precision residual ratio was "+
  "5.385 × 10⁸ against a machine-epsilon ratio of 5.369 × 10⁸, agreeing to 0.3%. "+
  "All reported results use double precision with an analytically derived "+
  "Jacobian-vector product; single precision was found to inflate the measured "+
  "improvement, and the correction is documented in Supplementary Information.",
  {align:AlignmentType.JUSTIFIED}),
P("Kuramoto–Sivashinsky was solved in one dimension both pseudospectrally and, "+
  "independently, by second-order finite differences with no spectral operators "+
  "anywhere in the solver. Complex Ginzburg–Landau was solved pseudospectrally; "+
  "its cubic nonlinearity makes the Jacobian real-linear rather than "+
  "complex-linear, so the state is treated as a real vector space with "+
  "conjugate-symmetric basis pairs and real reduced coordinates.",
  {align:AlignmentType.JUSTIFIED}),

H("Observable projector",HeadingLevel.HEADING_2),
P("The Navier–Stokes projector selects 24 Fourier modes forming 12 "+
  "conjugate pairs at wavenumber magnitudes 1, √2, 4.47, 4.58 and 5, fixed in "+
  "absolute wavenumber and independent of grid resolution. The conjugate-pair "+
  "structure is load-bearing: constructing observable directions as independent "+
  "complex unit vectors destroys Hermitian symmetry and yields a non-real field "+
  "(measured imaginary-to-real ratio 1.000, against 0.000 for the paired "+
  "cosine–sine construction). For sensor observables the projector is the "+
  "orthonormalised span of the Riesz representers of point-evaluation "+
  "functionals; the hidden block is then the null space of the measurement "+
  "operator, and a hidden correction is one every instrument reads as identical.",
  {align:AlignmentType.JUSTIFIED}),

H("Reduced model",HeadingLevel.HEADING_2),
P("The reduced subspace comprises the observable projector augmented by two "+
  "hidden directions constructed from the local trajectory jet: the component of "+
  "the velocity field outside the observable block, and the component of the "+
  "acceleration outside the observable block and the first hidden direction. "+
  "This is the dimension-minimal augmentation matching the first two trajectory "+
  "derivatives with the observable projector held fixed, and it yields "+
  "fourth-order local observable error against third order for velocity matching "+
  "alone. Measured convergence orders were 3.965 to 3.999 with coefficient of "+
  "determination 1.0000 across four topologies, and moving the fitting window "+
  "tenfold earlier reduced the deviation from four by factors of 9.7 to 11.8, as "+
  "the leading correction term predicts.",{align:AlignmentType.JUSTIFIED}),

H("Intervention protocol",HeadingLevel.HEADING_2),
P("Interventions were performed at three times along each baseline trajectory "+
  "for each of twelve configurations (four topologies × three viscosities), "+
  "giving 36 windows. At each intervention four branches were launched from "+
  "reduced states with identical reported coordinates and evolved independently "+
  "for two segments, after which the relative error in the reported coordinates "+
  "was compared against the reference. The mismatch in reported coordinates at "+
  "the intervention was verified to be identically zero in every window. Gate "+
  "thresholds were fixed before each run and were not adjusted; where a gate "+
  "failed, the missing physics was diagnosed and a new test designed rather than "+
  "the threshold moved.",{align:AlignmentType.JUSTIFIED}),

H("Definition of the effect-size quantities",HeadingLevel.HEADING_2),
P("Throughout, Δ denotes the FINITE-AMPLITUDE observable response to the "+
  "correction, Δ = P[Φ_ROM(V + Qe) − Φ_ROM(V)], evaluated by running the "+
  "reduced model from the perturbed state; it is not a linearisation. The "+
  "baseline observable error is b = P[Φ_ROM(V) − U], with U the reference. "+
  "Then λ = ‖Δ‖/‖b‖ and cos θ = ⟨b, Δ⟩/(‖b‖‖Δ‖), and equation (5) follows by "+
  "expanding ‖b + Δ‖². The identity is therefore exact by construction; what "+
  "requires a linear-response argument, and what carries the validity boundary "+
  "reported in the main text, is predicting λ in advance rather than measuring "+
  "it. The reconstruction error plotted in Fig. 4 is the discrepancy between λ "+
  "measured this way and the value predicted from the ratio of hidden-block to "+
  "observable-block responses under that argument.",
  {align:AlignmentType.JUSTIFIED}),
H("Statistical treatment",HeadingLevel.HEADING_2),
P("For the genericity tests, statistics are reported over independent "+
  "trajectories rather than over intervention windows, since windows along one "+
  "trajectory are not independent; confidence intervals are percentile intervals "+
  "over trajectories. Rank correlations are used where a monotone relationship "+
  "is at issue and log-log fits where an exponent is claimed, following an "+
  "earlier error in which a rank correlation against a monotone function of a "+
  "parameter was mistaken for evidence about a scaling exponent.",
  {align:AlignmentType.JUSTIFIED}),

H("Baselines",HeadingLevel.HEADING_2),
P("The ensemble variational baseline estimates a background covariance from the "+
  "model's own error-snapshot ensemble and applies the analysis increment "+
  "B Hᵀ(H B Hᵀ + R)⁻¹(y − H x_b) with x_b the background state, solved exactly in the low-rank ensemble "+
  "subspace, so that it corrects the hidden block through the observation via "+
  "cross-covariance. It was checked against its observation-error parameter at "+
  "10⁻⁸, 10⁻⁴ and 10⁻², yielding gains of 1.26, 1.20 and 1.12.",
  {align:AlignmentType.JUSTIFIED}),
P("Full-state correction is excluded from the candidate set on grounds "+
  "established by measurement rather than by preference. With an estimator of "+
  "this accuracy, adding the entire estimated error reproduces the reference "+
  "trajectory: in three-dimensional Navier–Stokes it yielded a nominal gain of "+
  "87.0 with a corrected error of 5.6 × 10⁻⁵, below the estimator's own residual "+
  "of 1.1 × 10⁻⁴. That is state replacement rather than model correction, and "+
  "its apparent gain measures estimator precision. A degeneracy guard flags any "+
  "window whose corrected error falls below the estimator residual.",
  {align:AlignmentType.JUSTIFIED}),

H("Data availability",HeadingLevel.HEADING_2),
P("Console outputs, intervention tables and summary statistics supporting the "+
  "findings, including the recovered experimental record with original run "+
  "outputs preserved, are deposited in the project repository and are also "+
  "available from the corresponding author.",{align:AlignmentType.JUSTIFIED}),
H("Code availability",HeadingLevel.HEADING_2),
P("Solver, reduced-model, estimator, baseline and analysis code for all four "+
  "systems is deposited in the project repository. The single-system "+
  "experiments require only NumPy and run on CPU; the three-dimensional "+
  "Navier–Stokes runners additionally require PyTorch and a GPU.",
  {align:AlignmentType.JUSTIFIED}),
H("Author contributions",HeadingLevel.HEADING_2),
P("A.M. developed the computational framework and the physical formulation, "+
  "implemented all numerical experiments and performed the analysis. "+
  "A.S. contributed to model development and the aerodynamic formulation. Both "+
  "authors discussed the results and contributed to the manuscript.",
  {align:AlignmentType.JUSTIFIED}),
H("Competing interests",HeadingLevel.HEADING_2),
P("The authors declare no competing interests.",{align:AlignmentType.JUSTIFIED}),
]}]});

Packer.toBuffer(doc).then(b=>{fs.writeFileSync("MS3D_manuscript.docx",b);
  console.log("written");});
