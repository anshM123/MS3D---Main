# ============================================================================
# M3D TEST F — VERIFY THE CORRECTION-REACHABILITY BOUND
# ============================================================================
# P4 found that gain is a STEP FUNCTION in containment: 1.05x without P in the
# reduced subspace, ~1.58x with it, while the reservoir ratio (13.8-25.0) and
# the ROM baseline error (0.02%-1.62%, an 80x range) moved the gain not at all.
# This test asks WHY, from theory rather than correlation.
#
# THEOREM F (correction reachability).  For an anchored projection ROM with
# reduced subspace S, one segment from anchor A gives A + s with s in S. For a
# hidden correction c in range(Q):
#
#     Delta(c) = P[Phi(V+c) - Phi(V)] = Pc + P(s' - s) = P(s' - s)
#
# since Pc = 0. Both s and s' lie in S, so
#
#     Delta(c)  in  P.S      for EVERY hidden correction c.
#
# The observable effect of any hidden correction is confined to the image of
# the reduced subspace under P. It cannot reach the rest of P at all. With b
# the baseline future observable error and rho = ||Pi_{P.S} b|| / ||b||:
#
#     gain  <=  1 / sqrt(1 - rho^2)                      (the CEILING)
#
# Note what does NOT appear in that bound: the reservoir ratio, the ROM's
# accuracy, the basis used for the rest of S. That is precisely P4's result.
#
# EXACTNESS. The argument needs S to be the same on both branches. Variants A
# and B carry a STATIC basis, so the bound is exact for them over any number of
# segments. C, D and E rebuild the jet directions at each anchor, so S differs
# slightly between branches and the bound is approximate there. The sharp test
# is therefore on A and B; C/D/E are reported as a secondary check.
#
# PREDICTION, from P4's numbers. A's observed gain of 1.0544 corresponds to
# rho ~ 0.31. If A's measured rho is near that, A is REACHABILITY-LIMITED --
# it is doing as well as any hidden correction possibly could. If B's rho is
# near 1 with a ceiling far above its observed 1.58, B is DYNAMICS-LIMITED.
# That would explain the step function from first principles.
#
# Preregistered gates:
#   F1  BOUND HOLDS: observed gain <= ceiling*(1+1e-6) on >= 95% of A and B
#       windows. A single clean violation falsifies Theorem F.
#   F2  A is reachability-limited: median (gain / ceiling) > 0.8 for A
#   F3  B is dynamics-limited:     median (gain / ceiling) < 0.5 for B
#   F4  DESCRIPTIVE: rho, ceiling, gain and dim(P.S) per variant; and the
#       ORACLE gain against the same ceiling, since the bound covers every
#       hidden correction including the true error.
# ============================================================================
import os, glob, math, json
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT  = SAVE / "m3d_F_reachability.csv"
BOOT = os.environ.get("M3D_BOOT",
    "/kaggle/input/datasets/anshmishra0812/broisaboot/m3d_kaggle_bootstrap.py")
if not os.path.exists(BOOT):
    cand = glob.glob("/kaggle/input/*/*/*/m3d_kaggle_bootstrap.py")
    if not cand: raise SystemExit("bootstrap not found; set M3D_BOOT")
    BOOT = cand[0]
print("bootstrap:", BOOT, flush=True)
BUILD_ID = "F-reachability-bound"
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
N    = int(os.environ.get("M3D_F_N", 40))
NU   = float(os.environ.get("M3D_F_NU", 2.5e-3))
N_TRAIN = int(os.environ.get("M3D_F_NTRAIN", 12))
N_VAL   = int(os.environ.get("M3D_F_NVAL", 6))
N_TEST  = int(os.environ.get("M3D_F_NTEST", 20))
N_SEG   = 12
TAUS    = [0.60, 0.90, 1.20]
FORECAST_SEGS, DERIV_REL = 2, 1e-5
RANKS   = [int(r) for r in os.environ.get("M3D_F_RANKS", "10").split(",")]
VARIANTS = os.environ.get("M3D_F_VARIANTS", "A,B,C,D,E").split(",")
VDESC = {"A":"POD only (frozen)", "B":"POD + all of P",
         "C":"POD + P + q0", "D":"POD + P + q0 + q1*",
         "E":"P + q0 + q1*  (STAR2 subspace)"}
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

def _ajvp(Uh, Vh, nu, env):
    curl = G["curl_hat_D"]
    Uc = dealias(leray(Uh, env), env); Vc = dealias(leray(Vh, env), env)
    u  = torch.fft.ifftn(Uc, dim=(-3,-2,-1)).real
    v  = torch.fft.ifftn(Vc, dim=(-3,-2,-1)).real
    wu = torch.fft.ifftn(curl(Uc, env), dim=(-3,-2,-1)).real
    wv = torch.fft.ifftn(curl(Vc, env), dim=(-3,-2,-1)).real
    cr = torch.empty_like(u)
    for i,(a,b) in enumerate(((1,2),(2,0),(0,1))):
        cr[i] = (u[a]*wv[b]-u[b]*wv[a]) + (v[a]*wu[b]-v[b]*wu[a])
    nh = dealias(leray(torch.fft.fftn(cr, dim=(-3,-2,-1)), env), env)
    return dealias(leray(nh - float(nu)*env["K2"].unsqueeze(0)*Vc, env), env)

def gs_append(B1, B2, e1, e2, tol=1e-10):
    for f1, f2 in zip(B1, B2):
        c = redot(f1, f2, e1, e2); e1 = e1 - c*f1; e2 = e2 - c*f2
    n = pnorm(e1, e2)
    if n > tol:
        B1.append(e1/n); B2.append(e2/n); return True
    return False

def local_jet(V1, V2, B1, B2, tids, env, nu, order):
    """q0 and q1* from the LOCAL trajectory jet at this anchor. Uses only the
    ROM state -- no truth. This is exactly what STAR2 rebuilds each anchor."""
    if order < 1: return B1, B2
    f1 = vrhs(V1, nu, env); f2 = vrhs(V2, nu, env)
    _, _, Qf1, Qf2 = splitPQ(f1, f2, tids)
    gs_append(B1, B2, Qf1, Qf2)
    if order >= 2:
        a1 = _ajvp(V1, f1, nu, env); a2 = _ajvp(V2, f2, nu, env)
        _, _, Qa1, Qa2 = splitPQ(a1, a2, tids)
        gs_append(B1, B2, Qa1, Qa2)
    return B1, B2

def build_variant(V1, V2, POD1, POD2, tids, env, nu, variant):
    """Assemble the subspace for one anchor. A/B are static; C/D/E rebuild the
    jet directions here, which is the whole point of the test."""
    if variant == "A":
        return [m.clone() for m in POD1], [m.clone() for m in POD2]
    if variant == "E":
        B1, B2 = [], []
    else:
        B1, B2 = [m.clone() for m in POD1], [m.clone() for m in POD2]
    B1, B2 = augment_with_P(B1, B2, tids, 1.0)
    order = {"B":0, "C":1, "D":2, "E":2}[variant]
    return local_jet(V1, V2, B1, B2, tids, env, nu, order)

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
def PS_projector(M1, M2, tids):
    """Orthonormal basis for P.S = { P s : s in S }, expressed in packed
    observable coordinates. Returns (U, rank) with U columns orthonormal."""
    cols = [packm(m1, m2, tids) for m1, m2 in zip(M1, M2)]
    Bm = torch.stack(cols, dim=1)                       # (dim_P, dim_S)
    U, sv, _ = torch.linalg.svd(Bm, full_matrices=False)
    if sv.numel() == 0:
        return None, 0
    tol = float(sv[0]) * 1e-10
    r = int((sv > tol).sum())
    return U[:, :r], r

def reach_and_ceiling(bvec, U):
    """rho = fraction of the baseline observable error reachable by ANY hidden
    correction; ceiling = the best gain that fraction permits."""
    nb = float(torch.linalg.vector_norm(bvec).real)
    if U is None or nb < 1e-300:
        return 0.0, 1.0
    proj = U @ (U.conj().transpose(0, 1) @ bvec)
    rho = float(torch.linalg.vector_norm(proj).real) / nb
    rho = min(max(rho, 0.0), 1.0 - 1e-15)
    return rho, 1.0 / math.sqrt(max(1.0 - rho*rho, 1e-30))

def seg_variant(a1,a2,POD1,POD2,tids,env,nu,variant):
    """One segment. For C/D/E the basis is REBUILT here from the current
    anchor, which is what STAR2 does and what a frozen POD basis does not."""
    M1,M2 = build_variant(a1,a2,POD1,POD2,tids,env,nu,variant)
    a=[0.0]*len(M1)
    for _ in range(SEG_STEPS):
        a=pod_step(a1,a2,M1,M2,a,DT,env,nu)
    return pod_recon(a1,a2,M1,M2,a)
def fc_variant(S1,S2,POD1,POD2,tids,env,nu,variant):
    A1,A2=S1.clone(),S2.clone()
    for _ in range(FORECAST_SEGS):
        A1,A2=seg_variant(A1,A2,POD1,POD2,tids,env,nu,variant)
    return A1,A2
def pod_segment(a1,a2,M1,M2,env,nu):
    a=[0.0]*len(M1)
    for _ in range(SEG_STEPS):
        a=pod_step(a1,a2,M1,M2,a,DT,env,nu)
    return pod_recon(a1,a2,M1,M2,a)

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
print(f"\n--- TEST: {N_TEST} unseen trajectories x {len(VARIANTS)} variants ---",
      flush=True)
rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done = {(int(r["traj"]), str(r["variant"])) for r in rows}

for VAR in VARIANTS:
    print(f"\n  === variant {VAR}: {VDESC[VAR]} ===", flush=True)
    for ti in range(N_TEST):
        if (ti, VAR) in done: continue
        try:
            u1, u2 = random_ic(env, 90000 + ti, REF)
            TR = trajectory(env, u1, u2, NU, N_SEG)
            MO = [(u1.clone(), u2.clone())]
            V1, V2 = u1.clone(), u2.clone()
            for _ in range(N_SEG):
                V1, V2 = seg_variant(V1, V2, BM1, BM2, TIDS, env, NU, VAR)
                MO.append((V1.clone(), V2.clone()))
            dim = len(build_variant(MO[0][0], MO[0][1], BM1, BM2, TIDS, env, NU, VAR)[0])

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
                res = pnorm(AQ1,AQ2)/max(pnorm(dP1,dP2),1e-30)
                rel = pnorm(EQ1,EQ2)/max(pnorm(Vi1,Vi2),1e-30)
                xb = packm(Vi1,Vi2,TIDS); xe = packm(Vi1+EQ1,Vi2+EQ2,TIDS)
                mism = float((torch.linalg.vector_norm(xe-xb)
                              /torch.linalg.vector_norm(xb)).real.cpu())
                base = dict(traj=ti, tau=tau, variant=VAR, subspace_dim=dim,
                            reservoir=res, start_mismatch=mism, rel_estimate=rel)
                if (not np.isfinite(rel)) or rel > 0.5:
                    rows.append(dict(base, status="diverged",
                        baseline_P_error=float("nan"), est_P_error=float("nan"),
                        neg_P_error=float("nan"), oracle_P_error=float("nan"),
                        gain=float("nan"), beats=False, beats_neg=False)); continue
                Mv1, Mv2 = build_variant(Vi1, Vi2, BM1, BM2, TIDS, env, NU, VAR)
                Umat, ps_rank = PS_projector(Mv1, Mv2, TIDS)
                FB = fc_variant(Vi1, Vi2, BM1, BM2, TIDS, env, NU, VAR)
                FE = fc_variant(Vi1+EQ1, Vi2+EQ2, BM1, BM2, TIDS, env, NU, VAR)
                FN = fc_variant(Vi1-EQ1, Vi2-EQ2, BM1, BM2, TIDS, env, NU, VAR)
                FO = fc_variant(Vi1+AQ1, Vi2+AQ2, BM1, BM2, TIDS, env, NU, VAR)
                be = terr(*FB, TF1, TF2, TIDS); ee = terr(*FE, TF1, TF2, TIDS)
                ne = terr(*FN, TF1, TF2, TIDS); oe = terr(*FO, TF1, TF2, TIDS)
                bvec = packm(FB[0]-TF1, FB[1]-TF2, TIDS)
                rho, ceil = reach_and_ceiling(bvec, Umat)
                base = dict(base, rho=rho, ceiling=ceil, ps_rank=ps_rank,
                            dim_P=int(bvec.numel()),
                            static_basis=bool(VAR in ("A","B")))
                rows.append(dict(base, status="ok", baseline_P_error=be,
                    est_P_error=ee, neg_P_error=ne, oracle_P_error=oe,
                    gain=be/max(ee,1e-30), gain_oracle=be/max(oe,1e-30),
                    beats=bool(ee<be), beats_neg=bool(ee<ne),
                    gain_over_ceiling=(be/max(ee,1e-30))/max(ceil,1e-30),
                    oracle_over_ceiling=(be/max(oe,1e-30))/max(ceil,1e-30),
                    bound_holds=bool(be/max(ee,1e-30) <= ceil*(1+1e-6))))
            sub=[r for r in rows if r["traj"]==ti and r["variant"]==VAR
                 and r["status"]=="ok"]
            if sub:
                print(f"    traj {ti:>3}: dim(P.S)={int(np.median([r['ps_rank'] for r in sub])):>3}"
                      f"/{int(sub[0]['dim_P'])}  rho={np.median([r['rho'] for r in sub]):.4f}"
                      f"  ceiling={np.median([r['ceiling'] for r in sub]):7.3f}"
                      f"  gain={np.median([r['gain'] for r in sub]):.4f}"
                      f"  ratio={np.median([r['gain_over_ceiling'] for r in sub]):.3f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        except Exception as exc:
            print(f"    traj {ti}: FAILED {exc!r}", flush=True)
        torch.cuda.empty_cache()

df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
ok = df[df.status == "ok"]
print("\n" + "="*110)
print("M3D TEST F — CORRECTION-REACHABILITY BOUND")
print("="*110)
print(f"\n  {'var':>4}  {'subspace':<30}{'dim(P.S)':>11}{'rho':>9}{'ceiling':>10}"
      f"{'gain':>9}{'gain/ceil':>11}{'orac/ceil':>11}")
for v in [x for x in VARIANTS if x in set(ok.variant)]:
    q = ok[ok.variant == v]; pr = q.groupby("traj")
    print(f"  {v:>4}  {VDESC[v]:<30}"
          f"{int(pr.ps_rank.median().median()):>5}/{int(q.dim_P.iloc[0]):<5}"
          f"{pr.rho.median().median():>9.4f}{pr.ceiling.median().median():>10.3f}"
          f"{pr.gain.median().median():>9.4f}"
          f"{pr.gain_over_ceiling.median().median():>11.3f}"
          f"{pr.oracle_over_ceiling.median().median():>11.3f}")

st = ok[ok.static_basis]
n_ok, n_tot = int(st.bound_holds.sum()), len(st)
print(f"\nF1 BOUND HOLDS where the theorem is EXACT (static basis, A and B):")
print(f"   {n_ok}/{n_tot} windows"
      f"  -> {'PASS' if n_tot and n_ok >= 0.95*n_tot else 'FAIL'}")
bad = st[~st.bound_holds]
if len(bad):
    print(f"   worst violation gain/ceiling = {bad.gain_over_ceiling.max():.6f}")
    print("   a clean violation falsifies Theorem F; check it is not numerical")

for v, lbl, lo, hi in (("A","F2 A is REACHABILITY-limited",0.8,None),
                       ("B","F3 B is DYNAMICS-limited",None,0.5)):
    if v not in set(ok.variant): continue
    r = float(ok[ok.variant==v].groupby("traj").gain_over_ceiling.median().median())
    good = (r > lo) if lo is not None else (r < hi)
    tgt = f"> {lo}" if lo is not None else f"< {hi}"
    print(f"\n{lbl}: median gain/ceiling = {r:.4f} (need {tgt})"
          f"  -> {'PASS' if good else 'FAIL'}")

dy = ok[~ok.static_basis]
if len(dy):
    print(f"\nF4a rebuilt-basis variants (C,D,E), bound only approximate: "
          f"{int(dy.bound_holds.sum())}/{len(dy)} respect it "
          f"({100*dy.bound_holds.mean():.1f}%)")

print("\nF4b the ORACLE obeys the same bound (it covers EVERY hidden correction):")
for v in [x for x in VARIANTS if x in set(ok.variant)]:
    q = ok[ok.variant == v]
    n3 = int((q.gain_oracle <= q.ceiling*(1+1e-6)).sum())
    print(f"    {v}: {n3}/{len(q)} within ceiling, median oracle/ceiling "
          f"{q.oracle_over_ceiling.median():.3f}")

print("\nF4c what the bound does NOT contain: reservoir ratio, ROM accuracy,")
print("    or the basis for the rest of S:")
for v in [x for x in VARIANTS if x in set(ok.variant)]:
    pr = ok[ok.variant == v].groupby("traj")
    print(f"    {v}: rho={pr.rho.median().median():.4f}  "
          f"R={pr.reservoir.median().median():6.2f}  "
          f"base={100*pr.baseline_P_error.median().median():6.3f}%  "
          f"gain={pr.gain.median().median():.4f}")
print("    If rho tracks gain while R and base do not, Theorem F explains P4.")
print("\nsaved:", OUT)
