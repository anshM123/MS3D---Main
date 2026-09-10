# ============================================================================
# M3D TEST P3 — SWEEP CONTAINMENT DIRECTLY
# ============================================================================
# P2 tried to explain why hidden-state correction is 9.5x weaker on
# POD-Galerkin (1.04x) than on STAR2 (1.65x) by sweeping POD RANK and
# correlating gain against the reservoir ratio R = ||Qe||/||Pe||.
#
# THAT DESIGN FAILED, for a reason worth recording:
#   * Spearman(R, gain) was +0.78 at rank 6 but -0.28 at rank 10 -- the sign
#     flipped. Pooled it collapsed to +0.33. R does not predict gain
#     consistently.
#   * Median R moved only 3.46 -> 4.77 across ranks, a 1.38x span. Rank is the
#     wrong knob; it has almost no leverage on the reservoir.
#
# But P2 measured the number that matters:
#
#     CONTAINMENT of the observable set P inside the ROM subspace
#       POD-Galerkin, any rank : 0.0026 - 0.0053   (~0.4%)
#       STAR2                  : 1.0000            (P is in Pi by construction)
#
# A ~250x structural difference alongside a 9.5x gain difference -- but
# essentially BINARY between the two families, so it cannot be tested as a
# continuous variable from a POD rank sweep. Two points are not a curve.
#
# THIS TEST SWEEPS CONTAINMENT DIRECTLY. The POD basis is augmented with a
# controlled fraction of the observable modes:
#
#     frac = 0.00  ->  pure POD-Galerkin      (containment ~ 0.004)
#     frac = 0.25, 0.50, 0.75
#     frac = 1.00  ->  P fully contained      (containment = 1.000, STAR2-like)
#
# Same frozen basis protocol, same 20 unseen test trajectories, same four-arm
# intervention. Only the subspace changes, and it interpolates continuously
# between the two ROM families.
#
# Preregistered gates:
#   C1  containment actually spans the range: min < 0.05 and max > 0.95
#   C2  Spearman(containment, median gain) > 0.6 across the sweep
#   C3  the relationship is MONOTONE: gain at frac=1.00 exceeds gain at
#       frac=0.00 on >= 80% of test trajectories, paired per trajectory
#   C4  DESCRIPTIVE: does the reservoir ratio R rise with containment? If so,
#       containment -> R -> gain is the causal chain and P2's failure was a
#       leverage problem, not a wrong hypothesis.
# ============================================================================
import os, glob, math, json
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT  = SAVE / "m3d_P3_containment.csv"
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT, flush=True)
BUILD_ID = "P3-containment-sweep"
print("BUILD_ID =", BUILD_ID, flush=True)
G = {"__name__": "__m3d__"}
exec(compile(open(BOOT).read(), BOOT, "exec"), G, G)

def spaces():
    out, seen = [G], {id(G)}
    for _, o in list(G.items()):
        if callable(o) and hasattr(o, "__globals__") and id(o.__globals__) not in seen:
            out.append(o.__globals__); seen.add(id(o.__globals__))
    return out
def setup(nu):
    for ns in spaces():
        ns["NU1_D"] = ns["NU2_D"] = ns["nu1_D"] = ns["nu2_D"] = float(nu)
        ns["REAL_DTYPE_D"] = torch.float64
        ns["COMPLEX_DTYPE_D"] = torch.complex128

make_env = G["make_env_D"];  topo  = G["topology_pair_L2"]
full_rk4 = G["full_rk4_D"];  init2 = G["initialize_star2_N"]
packm    = G["pack_modes_D"]; TAU0 = float(G["TAU0_D"])
dealias  = G["dealias_D"];   leray = G["project_divfree_D"]
vrhs     = G["velocity_rhs_D"]

L_BOX, SEG_TAU, SEG_STEPS = 8.0, 0.15, 20
DT   = SEG_TAU * TAU0 / SEG_STEPS
N    = int(os.environ.get("M3D_P3_N", 40))
NU   = float(os.environ.get("M3D_P3_NU", 2.5e-3))
N_TRAIN = int(os.environ.get("M3D_P3_NTRAIN", 12))
N_VAL   = int(os.environ.get("M3D_P3_NVAL", 6))
N_TEST  = int(os.environ.get("M3D_P3_NTEST", 20))
N_SEG   = 12
TAUS    = [0.60, 0.90, 1.20]
FORECAST_SEGS, DERIV_REL = 2, 1e-5
RANKS   = [int(r) for r in os.environ.get("M3D_P3_RANKS", "10").split(",")]
FRACS   = [float(f) for f in os.environ.get(
             "M3D_P3_FRACS", "0.0,0.25,0.5,0.75,1.0").split(",")]
TOPOS   = ["vortex_ring", "periodic_shear", "skew_tubes", "mixed_vortices"]

# ---------------------------------------------------------------- helpers
def redot(a1,a2,b1,b2):
    return float((torch.real(torch.vdot(a1.reshape(-1),b1.reshape(-1)))
                 +torch.real(torch.vdot(a2.reshape(-1),b2.reshape(-1)))).cpu())
def pnorm(a1,a2): return math.sqrt(max(redot(a1,a2,a1,a2),0.0))
def splitPQ(A1,A2,t):
    P1=torch.zeros_like(A1); P2=torch.zeros_like(A2)
    P1.reshape(3,-1)[:,t]=A1.reshape(3,-1)[:,t]; P2.reshape(3,-1)[:,t]=A2.reshape(3,-1)[:,t]
    return P1,P2,A1-P1,A2-P2
def phi(A1,A2,env,nu):
    B1,B2=A1.clone(),A2.clone()
    for _ in range(SEG_STEPS):
        B1=full_rk4(B1,DT,nu,env); B2=full_rk4(B2,DT,nu,env)
    return B1,B2
def terr(A1,A2,T1,T2,t):
    a=packm(A1-T1,A2-T2,t); b=packm(T1,T2,t)
    return float((torch.linalg.vector_norm(a)/torch.linalg.vector_norm(b)).real.cpu())

# ---------------------------------------------------------------- randomised ICs
def random_ic(env, seed, ref):
    """Random weighted mixture of the four canonical fields plus a small
    broadband divergence-free perturbation. Different seed = different flow."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(TOPOS)) * 1.2)
    A1 = None; A2 = None
    for wi, tn in zip(w, TOPOS):
        u1, u2 = topo(tn, env)
        A1 = wi*u1 if A1 is None else A1 + wi*u1
        A2 = wi*u2 if A2 is None else A2 + wi*u2
    # broadband perturbation at 12% of the field norm
    def noise(seed2):
        r = rng.normal(size=(3, N, N, N))
        uh = np.fft.fftn(r, axes=(1,2,3))
        f = np.fft.fftfreq(N)*N
        KX,KY,KZ = np.meshgrid(f,f,f,indexing="ij")
        km = np.sqrt(KX**2+KY**2+KZ**2)
        uh = uh/(np.abs(uh)+1e-300)*((km>=1)&(km<=6))
        return dealias(leray(torch.tensor(uh, dtype=torch.complex128,
                                          device="cuda"), env), env)
    n1 = noise(seed*7+11); n2 = noise(seed*7+22)
    s1 = 0.12*pnorm(A1,A2)/max(pnorm(n1,n2),1e-30)
    A1 = A1 + s1*n1; A2 = A2 + s1*n2
    A1 = dealias(leray(A1, env), env); A2 = dealias(leray(A2, env), env)
    sc = ref/max(pnorm(A1,A2),1e-30)
    return A1*sc, A2*sc

# ---------------------------------------------------------------- anchored POD-Galerkin
def pod_basis(snaps1, snaps2, rank):
    n = len(snaps1)
    m1 = sum(snaps1[1:], snaps1[0].clone())/n
    m2 = sum(snaps2[1:], snaps2[0].clone())/n
    F1 = [s-m1 for s in snaps1]; F2 = [s-m2 for s in snaps2]
    Gm = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i, n):
            v = redot(F1[i],F2[i],F1[j],F2[j]); Gm[i,j]=v; Gm[j,i]=v
    ev, evec = torch.linalg.eigh(Gm)
    o = torch.argsort(ev, descending=True); ev, evec = ev[o], evec[:,o]
    keep = min(rank, int((ev > ev[0]*1e-12).sum()))
    M1, M2 = [], []
    for k in range(keep):
        c = evec[:,k]
        p1 = torch.zeros_like(F1[0]); p2 = torch.zeros_like(F2[0])
        for i in range(n):
            ci = c[i].to(p1.dtype); p1 += ci*F1[i]; p2 += ci*F2[i]
        nn = max(pnorm(p1,p2), 1e-300)
        M1.append(p1/nn); M2.append(p2/nn)
    cap = float(ev[:keep].clamp(min=0).sum()/max(float(ev.clamp(min=0).sum()),1e-300))
    return M1, M2, cap

def augment_with_P(M1, M2, tids, frac, seed=0):
    """Add a controlled fraction of the observable modes to the POD basis, then
    re-orthonormalise. frac=0 leaves POD untouched; frac=1 puts ALL of P in the
    subspace, which is the STAR2 situation. This is the only variable that
    changes across the sweep."""
    idx = tids.detach().cpu().numpy()
    cols = [(c, m) for m in idx for c in range(3)]
    rng = np.random.default_rng(seed)
    rng.shuffle(cols)
    take = int(round(frac * len(cols)))
    B1 = [m.clone() for m in M1]; B2 = [m.clone() for m in M2]
    for c, m in cols[:take]:
        e1 = torch.zeros_like(M1[0]); e2 = torch.zeros_like(M2[0])
        e1.reshape(3, -1)[c, m] = 1.0
        # Gram-Schmidt against everything already in the basis
        for f1, f2 in zip(B1, B2):
            co = redot(f1, f2, e1, e2); e1 = e1 - co*f1; e2 = e2 - co*f2
        nn = pnorm(e1, e2)
        if nn > 1e-10:
            B1.append(e1/nn); B2.append(e2/nn)
    return B1, B2

def pod_rhs(a1,a2,M1,M2,a,env,nu):
    U1 = a1.clone(); U2 = a2.clone()
    for i, ai in enumerate(a):
        U1 = U1 + float(ai)*M1[i]; U2 = U2 + float(ai)*M2[i]
    F1 = vrhs(U1, nu, env); F2 = vrhs(U2, nu, env)
    return [redot(M1[i],M2[i],F1,F2) for i in range(len(a))]
def pod_step(a1,a2,M1,M2,a,dt,env,nu):
    k1=pod_rhs(a1,a2,M1,M2,a,env,nu)
    k2=pod_rhs(a1,a2,M1,M2,[a[i]+0.5*dt*k1[i] for i in range(len(a))],env,nu)
    k3=pod_rhs(a1,a2,M1,M2,[a[i]+0.5*dt*k2[i] for i in range(len(a))],env,nu)
    k4=pod_rhs(a1,a2,M1,M2,[a[i]+dt*k3[i]     for i in range(len(a))],env,nu)
    return [a[i]+dt/6.0*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i in range(len(a))]
def pod_recon(a1,a2,M1,M2,a):
    U1=a1.clone(); U2=a2.clone()
    for i,ai in enumerate(a):
        U1=U1+float(ai)*M1[i]; U2=U2+float(ai)*M2[i]
    return U1,U2
def pod_segment(a1,a2,M1,M2,env,nu):
    """One segment, anchored: coefficients start at zero, anchor carries the
    full state — this is what lets a hidden correction survive."""
    a=[0.0]*len(M1)
    for _ in range(SEG_STEPS):
        a=pod_step(a1,a2,M1,M2,a,DT,env,nu)
    return pod_recon(a1,a2,M1,M2,a)
def pod_forecast(S1,S2,M1,M2,env,nu):
    A1,A2=S1.clone(),S2.clone()
    for _ in range(FORECAST_SEGS):
        A1,A2=pod_segment(A1,A2,M1,M2,env,nu)
    return A1,A2

# ---------------------------------------------------------------- run one trajectory
def trajectory(env, U1, U2, nu, nseg):
    t1,t2=U1.clone(),U2.clone(); out=[(t1.clone(),t2.clone())]
    for _ in range(nseg):
        for _ in range(SEG_STEPS):
            t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
        out.append((t1.clone(),t2.clone()))
    return out

setup(NU); env = make_env(N, L=L_BOX)
ref1, ref2 = topo("skew_tubes", env); REF = pnorm(ref1, ref2); del ref1, ref2
star,_ = init2(*random_ic(env, 0, REF), env); TIDS = star["proj"]["target_ids"]; del star
print(f"N={N} nu={NU:.2e}  P = {int(TIDS.numel())} modes  ref|U|={REF:.4e}", flush=True)

# ============================== TRAINING: build and FREEZE the basis =========
print(f"\n--- building POD basis from {N_TRAIN} TRAINING trajectories ---", flush=True)
S1, S2 = [], []
for k in range(N_TRAIN):
    u1, u2 = random_ic(env, 1000 + k, REF)
    traj = trajectory(env, u1, u2, NU, N_SEG)
    for a, b in traj[::2]:
        S1.append(a); S2.append(b)
    print(f"  train {k+1}/{N_TRAIN}  snapshots={len(S1)}", flush=True)
BASES = {}
for r in RANKS:
    M1, M2, cap = pod_basis(S1, S2, r)
    BASES[r] = (M1, M2, cap)
    print(f"  rank {r:>3}: kept {len(M1):>3} modes, training energy {100*cap:.4f}%",
          flush=True)
del S1, S2; torch.cuda.empty_cache()

# ============================== VALIDATION: choose the rank =================
print(f"\n--- choosing rank on {N_VAL} VALIDATION trajectories (never the test set) ---",
      flush=True)
kmid = int(round(TAUS[1]/SEG_TAU))
val = {}
for r in RANKS:
    M1, M2, _ = BASES[r]; errs = []
    for k in range(N_VAL):
        u1, u2 = random_ic(env, 5000 + k, REF)
        tr = trajectory(env, u1, u2, NU, kmid + FORECAST_SEGS)
        V1, V2 = u1.clone(), u2.clone()
        for _ in range(kmid):
            V1, V2 = pod_segment(V1, V2, M1, M2, env, NU)
        errs.append(terr(V1, V2, tr[kmid][0], tr[kmid][1], TIDS))
    val[r] = float(np.median(errs))
    print(f"  rank {r:>3}: median validation P-error = {100*val[r]:.4f}%", flush=True)
BASE_RANK = RANKS[0]
print(f"  -> base POD rank fixed at {BASE_RANK}; sweeping CONTAINMENT instead",
      flush=True)

# ============================== TEST: sweep rank on unseen ICs ==============
def containment(M1, M2, tids, env):
    """How much of the observable set P the ROM subspace actually contains.
    1.0 = P fully inside the subspace (STAR2 by construction); 0 = disjoint."""
    tot = 0.0; cap = 0.0
    idx = tids.detach().cpu().numpy()
    for j, mi in enumerate(idx):
        for comp in range(3):
            e1 = torch.zeros_like(M1[0]); e2 = torch.zeros_like(M2[0])
            e1.reshape(3, -1)[comp, mi] = 1.0
            n0 = pnorm(e1, e2)
            if n0 < 1e-30:
                continue
            c = sum(redot(M1[i], M2[i], e1, e2)**2 for i in range(len(M1)))
            tot += n0**2; cap += c
    return float(cap/max(tot, 1e-300))

print(f"\n--- TEST: {N_TEST} unseen trajectories x {len(SWEEP)} ranks ---", flush=True)
rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done = {(int(r["traj"]), float(r["frac"])) for r in rows}
BM1, BM2, CAP = BASES[BASE_RANK]
AUG = {}; CONT = {}
for fr in FRACS:
    A1b, A2b = augment_with_P(BM1, BM2, TIDS, fr)
    AUG[fr] = (A1b, A2b)
    CONT[fr] = containment(A1b, A2b, TIDS, env)
    print(f"  frac {fr:.2f}: subspace dim {len(A1b):>3}  "
          f"P-containment = {CONT[fr]:.4f}", flush=True)

for FRAC in FRACS:
    M1, M2 = AUG[FRAC]; RANK = FRAC
    print(f"\n  === frac {FRAC:.2f} (dim {len(M1)}, containment "
          f"{CONT[FRAC]:.4f}) ===", flush=True)
    for ti in range(N_TEST):
        if (ti, FRAC) in done: continue
        try:
            u1, u2 = random_ic(env, 90000 + ti, REF)
            TR = trajectory(env, u1, u2, NU, N_SEG)
            MO = [(u1.clone(), u2.clone())]
            V1, V2 = u1.clone(), u2.clone()
            for _ in range(N_SEG):
                V1, V2 = pod_segment(V1, V2, M1, M2, env, NU)
                MO.append((V1.clone(), V2.clone()))
            E1 = torch.zeros_like(MO[0][0]); E2 = torch.zeros_like(MO[0][1])
            EH = [(E1.clone(), E2.clone())]
            for k in range(int(round(max(TAUS)/SEG_TAU))):
                Vk1, Vk2 = MO[k]; Vn1, Vn2 = MO[k+1]
                B1, B2 = phi(Vk1, Vk2, env, NU)
                e1, e2 = B1-Vn1, B2-Vn2
                if pnorm(E1,E2) < 1e-30:
                    E1, E2 = e1.clone(), e2.clone()
                else:
                    en = max(pnorm(E1,E2), 1e-30)
                    eps = min(max(DERIV_REL*pnorm(Vk1,Vk2)/en, 1e-6), 5e-2)
                    Pp1,Pp2 = phi(Vk1+eps*E1, Vk2+eps*E2, env, NU)
                    Mi1,Mi2 = phi(Vk1-eps*E1, Vk2-eps*E2, env, NU)
                    E1 = e1+(Pp1-Mi1)/(2*eps)+0.5*(Pp1+Mi1-2*B1)/(eps*eps)
                    E2 = e2+(Pp2-Mi2)/(2*eps)+0.5*(Pp2+Mi2-2*B2)/(eps*eps)
                EH.append((E1.clone(), E2.clone()))

            for tau in TAUS:
                k = int(round(tau/SEG_TAU))
                Vi1,Vi2 = MO[k]; Ti1,Ti2 = TR[k]; TF1,TF2 = TR[k+FORECAST_SEGS]
                _,_,EQ1,EQ2 = splitPQ(EH[k][0], EH[k][1], TIDS)
                dP1,dP2,AQ1,AQ2 = splitPQ(Ti1-Vi1, Ti2-Vi2, TIDS)
                # THE MEASUREMENT: how much of the accumulated error is hidden?
                reservoir = pnorm(AQ1,AQ2)/max(pnorm(dP1,dP2), 1e-30)
                rel = pnorm(EQ1,EQ2)/max(pnorm(Vi1,Vi2), 1e-30)
                xb = packm(Vi1,Vi2,TIDS); xe = packm(Vi1+EQ1,Vi2+EQ2,TIDS)
                mism = float((torch.linalg.vector_norm(xe-xb)
                              /torch.linalg.vector_norm(xb)).real.cpu())
                base = dict(traj=ti, tau=tau, frac=FRAC, rank=BASE_RANK,
                            subspace_dim=len(M1), containment=CONT[FRAC],
                            pod_energy=CAP, reservoir=reservoir,
                            start_mismatch=mism, rel_estimate=rel)
                if (not np.isfinite(rel)) or rel > 0.5:
                    rows.append(dict(base, status="diverged",
                        baseline_P_error=float("nan"), est_P_error=float("nan"),
                        neg_P_error=float("nan"), oracle_P_error=float("nan"),
                        gain=float("nan"), gain_oracle=float("nan"),
                        beats=False, beats_neg=False)); continue
                FB = pod_forecast(Vi1, Vi2, M1, M2, env, NU)
                FE = pod_forecast(Vi1+EQ1, Vi2+EQ2, M1, M2, env, NU)
                FN = pod_forecast(Vi1-EQ1, Vi2-EQ2, M1, M2, env, NU)
                FO = pod_forecast(Vi1+AQ1, Vi2+AQ2, M1, M2, env, NU)
                be = terr(*FB, TF1, TF2, TIDS); ee = terr(*FE, TF1, TF2, TIDS)
                ne = terr(*FN, TF1, TF2, TIDS); oe = terr(*FO, TF1, TF2, TIDS)
                rows.append(dict(base, status="ok", baseline_P_error=be,
                    est_P_error=ee, neg_P_error=ne, oracle_P_error=oe,
                    gain=be/max(ee,1e-30), gain_oracle=be/max(oe,1e-30),
                    beats=bool(ee<be), beats_neg=bool(ee<ne)))
            sub=[r for r in rows if r["traj"]==ti and r["frac"]==FRAC and r["status"]=="ok"]
            if sub:
                print(f"    traj {ti:>3}: R={np.median([r['reservoir'] for r in sub]):6.2f}  "
                      f"base={100*np.median([r['baseline_P_error'] for r in sub]):5.2f}%  "
                      f"gain={np.median([r['gain'] for r in sub]):.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        except Exception as exc:
            print(f"    traj {ti}: FAILED {exc!r}", flush=True)
        torch.cuda.empty_cache()

# ============================== REPORT ======================================
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
ok = df[df.status == "ok"]
print("\n" + "="*104)
print("M3D TEST P3 — DOES CONTAINMENT OF THE OBSERVABLE SET GOVERN THE EFFECT?")
print("="*104)
print(f"\n  {'frac':>6}{'dim':>6}{'containment':>13}{'median R':>10}"
      f"{'median base':>13}{'median gain':>13}{'win rate':>10}{'oracle cap':>12}")
tab = {}
for fr in sorted(ok.frac.unique()):
    s_ = ok[ok.frac == fr]; pr = s_.groupby("traj")
    R = float(pr.reservoir.median().median())
    b = float(pr.baseline_P_error.median().median())
    g = float(pr.gain.median().median())
    w = float(s_.beats.mean())
    cap = float(((s_.baseline_P_error-s_.est_P_error)
                 /(s_.baseline_P_error-s_.oracle_P_error).replace(0,np.nan)).median())
    tab[fr] = dict(cont=float(s_.containment.iloc[0]), R=R, base=b, gain=g,
                   win=w, cap=cap, dim=int(s_.subspace_dim.iloc[0]))
    print(f"  {fr:>6.2f}{tab[fr]['dim']:>6}{tab[fr]['cont']:>13.4f}{R:>10.3f}"
          f"{100*b:>12.2f}%{g:>13.4f}{100*w:>9.1f}%{cap:>12.4f}")

conts = [tab[f]["cont"] for f in sorted(tab)]
gains = [tab[f]["gain"] for f in sorted(tab)]
Rs    = [tab[f]["R"] for f in sorted(tab)]
c1 = (min(conts) < 0.05) and (max(conts) > 0.95)
print(f"\nC1 containment spans the range: {min(conts):.4f} to {max(conts):.4f}"
      f"  -> {'PASS' if c1 else 'FAIL'}")

def sp(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
# per (frac, trajectory) so n is not just the number of fracs
pt = ok.groupby(["frac", "traj"]).median(numeric_only=True).reset_index()
sC = sp(pt.containment.values, pt.gain.values)
sB = sp(pt.baseline_P_error.values, pt.gain.values)
sR = sp(pt.reservoir.values, pt.gain.values)
print(f"\nC2 Spearman(containment, gain) = {sC:+.4f}   n={len(pt)}"
      f"  -> {'PASS' if sC > 0.6 else 'FAIL'}")
print(f"     for comparison:  Spearman(reservoir R, gain)   = {sR:+.4f}")
print(f"                      Spearman(baseline err, gain)  = {sB:+.4f}")

lo, hi = min(tab), max(tab)
pl = ok[ok.frac == lo].groupby("traj").gain.median()
ph = ok[ok.frac == hi].groupby("traj").gain.median()
common = pl.index.intersection(ph.index)
n_up = int((ph[common] > pl[common]).sum())
print(f"\nC3 paired per trajectory, frac {hi:.2f} vs {lo:.2f}: "
      f"{n_up}/{len(common)} improved"
      f"  -> {'PASS' if n_up >= 0.8*len(common) else 'FAIL'}")
if len(common):
    print(f"     median gain {pl[common].median():.4f}x -> {ph[common].median():.4f}x"
          f"   ({ph[common].median()/max(pl[common].median(),1e-9):.2f}x larger effect)")

print(f"\nC4 does containment drive the reservoir?  "
      f"Spearman(containment, R) = {sp(conts, Rs):+.4f}")
print(f"     R by frac: " + "  ".join(f"{f:.2f}->{tab[f]['R']:.2f}" for f in sorted(tab)))

print("\n  reference points on the same axes")
print(f"  {'model':<30}{'containment':>13}{'gain':>10}")
for f in sorted(tab):
    print(f"  {'POD + ' + str(int(100*f)) + '% of P':<30}"
          f"{tab[f]['cont']:>13.4f}{tab[f]['gain']:>10.4f}")
print(f"  {'STAR2 (P contained, Test X-v3)':<30}{1.0:>13.4f}{1.6510:>10.4f}")
_v = ("CONFIRMED: the effect size is governed by how much of the observable set "
      "the ROM subspace contains." if (c1 and sC > 0.6 and n_up >= 0.8*len(common))
      else "NOT CONFIRMED: containment does not explain the effect size either. "
           "Report the POD/STAR2 gap as unexplained.")
print(f"\n  VERDICT: {_v}")
print("\nsaved:", OUT)
