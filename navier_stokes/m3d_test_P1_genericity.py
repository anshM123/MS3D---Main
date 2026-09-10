# ============================================================================
# M3D TEST P1 — GENERICITY, DONE PROPERLY
# ============================================================================
# THE KILL GATE. If observable-preserving hidden-state correction does not work
# on a POD-Galerkin ROM under an honest train/test protocol, the phenomenon is
# tied to STAR2 and the paper stays a STAR2 paper.
#
# What was wrong before (Test G1): the POD basis was built from the SAME
# trajectory it was then asked to reproduce, on a numerically rank-13 problem.
# Baseline error came out at 1e-7 -- there was nothing to correct, so the test
# was NULL, not negative.
#
# This version enforces a real protocol:
#   * many TRAINING trajectories with randomised initial conditions
#   * POD basis built from training only, then FROZEN
#   * evaluation on completely UNSEEN initial conditions
#   * rank chosen so the ROM is useful but imperfect (target 1-15% error),
#     selected on a VALIDATION split, never on the test set
#   * statistics reported over independent TRAJECTORIES, not over anchors
#     (the effective-n problem in the existing record)
#
# Everything else -- the P/Q split, the W defect recurrence, the four-arm
# intervention with P(V+c) = PV exactly -- is identical to Test X.
#
# Preregistered gates (fixed BEFORE running):
#   N0  the test is LIVE: median baseline forecast error in [0.005, 0.30] on
#       >= 15 of the test trajectories. Otherwise NULL, and no gate below is
#       evaluated.
#   G1  observable state preserved: max P-mismatch < 1e-9 on every branch
#   G2  truth-free correction beats baseline on >= 80% of test TRAJECTORIES
#   G3  correction beats the wrong-sign control on >= 80% of test trajectories
#   G4  DESCRIPTIVE: median gain, p10, 95% CI over trajectories; fraction of
#       oracle benefit captured
# ============================================================================
import os, glob, math, json
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT  = SAVE / "m3d_P1_pod_genericity.csv"
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT, flush=True)
BUILD_ID = "P1-pod-genericity-trainfrozen"
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
N    = int(os.environ.get("M3D_P1_N", 40))
NU   = float(os.environ.get("M3D_P1_NU", 2.5e-3))
N_TRAIN = int(os.environ.get("M3D_P1_NTRAIN", 12))
N_VAL   = int(os.environ.get("M3D_P1_NVAL", 6))
N_TEST  = int(os.environ.get("M3D_P1_NTEST", 20))
N_SEG   = 12
TAUS    = [0.60, 0.90, 1.20]
FORECAST_SEGS, DERIV_REL = 2, 1e-5
RANKS   = [int(r) for r in os.environ.get("M3D_P1_RANKS", "6,10,16,24").split(",")]
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
usable = [r for r in RANKS if 0.005 <= val[r] <= 0.30]
RANK = usable[len(usable)//2] if usable else min(RANKS, key=lambda r: abs(val[r]-0.05))
print(f"  -> selected rank {RANK} (validation error {100*val[RANK]:.3f}%)", flush=True)
M1, M2, CAP = BASES[RANK]

# ============================== TEST: unseen ICs, frozen basis ==============
print(f"\n--- TEST on {N_TEST} unseen trajectories ---", flush=True)
rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done = {int(r["traj"]) for r in rows}
for ti in range(N_TEST):
    if ti in done: continue
    try:
        u1, u2 = random_ic(env, 90000 + ti, REF)
        TR = trajectory(env, u1, u2, NU, N_SEG)
        MO = [(u1.clone(), u2.clone())]
        V1, V2 = u1.clone(), u2.clone()
        for _ in range(N_SEG):
            V1, V2 = pod_segment(V1, V2, M1, M2, env, NU)
            MO.append((V1.clone(), V2.clone()))

        # W defect recurrence — identical to Test X, no truth used
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
                P1,P2 = phi(Vk1+eps*E1, Vk2+eps*E2, env, NU)
                Mi1,Mi2 = phi(Vk1-eps*E1, Vk2-eps*E2, env, NU)
                E1 = e1+(P1-Mi1)/(2*eps)+0.5*(P1+Mi1-2*B1)/(eps*eps)
                E2 = e2+(P2-Mi2)/(2*eps)+0.5*(P2+Mi2-2*B2)/(eps*eps)
            EH.append((E1.clone(), E2.clone()))

        for tau in TAUS:
            k = int(round(tau/SEG_TAU))
            Vi1,Vi2 = MO[k]; Ti1,Ti2 = TR[k]; TF1,TF2 = TR[k+FORECAST_SEGS]
            _,_,EQ1,EQ2 = splitPQ(EH[k][0], EH[k][1], TIDS)
            _,_,AQ1,AQ2 = splitPQ(Ti1-Vi1, Ti2-Vi2, TIDS)
            rel = pnorm(EQ1,EQ2)/max(pnorm(Vi1,Vi2),1e-30)
            xb = packm(Vi1,Vi2,TIDS); xe = packm(Vi1+EQ1,Vi2+EQ2,TIDS)
            mism = float((torch.linalg.vector_norm(xe-xb)
                          /torch.linalg.vector_norm(xb)).real.cpu())
            if (not np.isfinite(rel)) or rel > 0.5:
                rows.append(dict(traj=ti, tau=tau, rank=RANK, status="diverged",
                    start_mismatch=mism, baseline_P_error=float("nan"),
                    est_P_error=float("nan"), neg_P_error=float("nan"),
                    oracle_P_error=float("nan"), gain=float("nan"),
                    beats=False, beats_neg=False, rel_estimate=rel))
                continue
            FB = pod_forecast(Vi1, Vi2, M1, M2, env, NU)
            FE = pod_forecast(Vi1+EQ1, Vi2+EQ2, M1, M2, env, NU)
            FN = pod_forecast(Vi1-EQ1, Vi2-EQ2, M1, M2, env, NU)
            FO = pod_forecast(Vi1+AQ1, Vi2+AQ2, M1, M2, env, NU)
            be = terr(*FB, TF1, TF2, TIDS); ee = terr(*FE, TF1, TF2, TIDS)
            ne = terr(*FN, TF1, TF2, TIDS); oe = terr(*FO, TF1, TF2, TIDS)
            rows.append(dict(traj=ti, tau=tau, rank=RANK, status="ok",
                start_mismatch=mism, baseline_P_error=be, est_P_error=ee,
                neg_P_error=ne, oracle_P_error=oe, gain=be/max(ee,1e-30),
                gain_neg=be/max(ne,1e-30), gain_oracle=be/max(oe,1e-30),
                beats=bool(ee<be), beats_neg=bool(ee<ne),
                rel_estimate=rel, pod_energy=CAP))
        g = [r["gain"] for r in rows if r["traj"] == ti and r["status"] == "ok"]
        b = [r["baseline_P_error"] for r in rows if r["traj"] == ti and r["status"]=="ok"]
        print(f"  traj {ti:>3}: base err "
              + " ".join(f"{100*x:.2f}%" for x in b)
              + "   gains " + " ".join(f"{x:.3f}" for x in g), flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    except Exception as exc:
        print(f"  traj {ti}: FAILED {exc!r}", flush=True)
    torch.cuda.empty_cache()

# ============================== REPORT ======================================
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
ok = df[df.status == "ok"]
print("\n" + "="*96)
print("M3D TEST P1 — POD-GALERKIN GENERICITY, TRAIN/TEST SEPARATED")
print("="*96)
print(f"  basis: rank {RANK}, frozen, built from {N_TRAIN} training trajectories")
print(f"  test: {df.traj.nunique()} unseen initial conditions, {len(ok)} evaluated windows")

per = ok.groupby("traj")
base_med = per.baseline_P_error.median()
n_live = int(((base_med >= 0.005) & (base_med <= 0.30)).sum())
print(f"\nN0 test is LIVE: {n_live}/{df.traj.nunique()} trajectories with median "
      f"baseline error in [0.5%, 30%]")
if n_live < 15:
    print("  -> NULL TEST. The ROM is either near-perfect or diverged; the gates")
    print("     below are NOT evaluated. Adjust rank via M3D_P1_RANKS and rerun.")
else:
    print("  -> LIVE")
    mm = float(ok.start_mismatch.max())
    print(f"\nG1 observable state preserved: max P-mismatch {mm:.3e}"
          f"  -> {'PASS' if mm < 1e-9 else 'FAIL'}")
    wins = per.beats.mean()
    n2 = int((wins > 0.5).sum())
    print(f"G2 correction beats baseline on {n2}/{len(wins)} trajectories"
          f"  -> {'PASS' if n2 >= 0.8*len(wins) else 'FAIL'}")
    wn = per.beats_neg.mean(); n3 = int((wn > 0.5).sum())
    print(f"G3 correction beats wrong-sign on {n3}/{len(wn)} trajectories"
          f"  -> {'PASS' if n3 >= 0.8*len(wn) else 'FAIL'}")
    gm = per.gain.median()
    lo, hi = np.percentile(gm, [2.5, 97.5])
    print(f"\nG4 per-trajectory statistics (effective n = {len(gm)}, not window count)")
    print(f"     median gain      {gm.median():.4f}x")
    print(f"     95% CI           [{lo:.4f}, {hi:.4f}]")
    print(f"     p10              {np.percentile(gm,10):.4f}x")
    print(f"     median baseline  {100*base_med.median():.3f}%")
    cap = ((ok.baseline_P_error-ok.est_P_error)
           /(ok.baseline_P_error-ok.oracle_P_error).replace(0,np.nan)).median()
    print(f"     oracle benefit captured  {cap:.4f}")
    print("\n  VERDICT: " + ("hidden-state correction is NOT tied to STAR2 — it "
          "reproduces on POD-Galerkin under an honest train/test protocol."
          if n2 >= 0.8*len(wins) and n3 >= 0.8*len(wn) else
          "does NOT reproduce on POD-Galerkin. The phenomenon may be specific "
          "to STAR2; report it as such."))
print("\nsaved:", OUT)
