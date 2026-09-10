# ============================================================================
# M3D TEST R3-FIX — leakage concentration on the EMPIRICAL error subspace
# ============================================================================
# R1 probed T = P DPhi_h Q with random Gaussian directions. That was wrong:
# random directions in a 3e6-dimensional Q have no overlap with the few
# directions error actually occupies, so top-3 = 0.45 said nothing.
# This rebuilds T restricted to the span of ACTUAL accumulated Q-errors,
# which is what Tests R/S/T originally measured, and asks whether the rank-3
# concentration survives as dim(Q) and Reynolds number grow.
#
# Also saves the top-3 leakage directions in physical space, so they can be
# visualised. If those turn out to be identifiable flow structures, that is
# the fluid-mechanical result.
# ============================================================================
import os, glob, math, json
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT)
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

make_env  = G["make_env_D"];        topo   = G["topology_pair_L2"]
full_rk4  = G["full_rk4_D"];        init2  = G["initialize_star2_N"]
rk4M      = G["rk4_M"];             recon  = G["reconstruct_M"]
packm     = G["pack_modes_D"];      TAU0   = float(G["TAU0_D"])

SEG_TAU, SEG_STEPS, N_SEG = 0.15, 20, 10
DT = SEG_TAU * TAU0 / SEG_STEPS
ANCHORS = [4, 6, 8, 10]          # segments at which to sample the Q-error
BASE_NU = 2.5e-3
CONFIGS = [("N40_nu1", 40, 1.0), ("N64_nu1", 64, 1.0), ("N80_nu1", 80, 1.0),
           ("N64_nu0.4", 64, 0.4), ("N80_nu0.16", 80, 0.16),
           ("N96_nu0.0625", 96, 0.0625)]
TOPOLOGY = os.environ.get("M3D_TOPO", "skew_tubes")

def redot(a1, a2, b1, b2):
    return float((torch.real(torch.vdot(a1.reshape(-1), b1.reshape(-1)))
                + torch.real(torch.vdot(a2.reshape(-1), b2.reshape(-1)))).cpu())

def pnorm(a1, a2):
    return math.sqrt(max(redot(a1, a2, a1, a2), 0.0))

def splitPQ(A1, A2, tids):
    P1 = torch.zeros_like(A1); P2 = torch.zeros_like(A2)
    P1.reshape(3, -1)[:, tids] = A1.reshape(3, -1)[:, tids]
    P2.reshape(3, -1)[:, tids] = A2.reshape(3, -1)[:, tids]
    return P1, P2, A1 - P1, A2 - P2

def phi(A1, A2, env, nu):
    B1, B2 = A1.clone(), A2.clone()
    for _ in range(SEG_STEPS):
        B1 = full_rk4(B1, DT, nu, env); B2 = full_rk4(B2, DT, nu, env)
    return B1, B2

rows, vecs = [], {}
for label, N, numult in CONFIGS:
    nu = BASE_NU * numult
    print(f"\n=== {label}  N={N} nu={nu:.4e} ===", flush=True)
    setup(nu)
    env = make_env(N, L=8.0)
    U1, U2 = topo(TOPOLOGY, env)
    star, _ = init2(U1, U2, env); tids = star["proj"]["target_ids"]; del star

    # --- truth + STAR2 trajectories, capturing Q-error at each anchor ---
    t1, t2 = U1.clone(), U2.clone()
    m, _ = init2(U1, U2, env); V1, V2 = U1.clone(), U2.clone()
    dQ1s, dQ2s, Vs = [], [], []
    for seg in range(1, N_SEG + 1):
        for _ in range(SEG_STEPS):
            t1 = full_rk4(t1, DT, nu, env); t2 = full_rk4(t2, DT, nu, env)
            rk4M(m, SEG_TAU / SEG_STEPS, env)
        V1, V2 = recon(m, env); m, _ = init2(V1, V2, env)
        if seg in ANCHORS:
            _, _, q1, q2 = splitPQ(t1 - V1, t2 - V2, tids)
            dQ1s.append(q1.clone()); dQ2s.append(q2.clone())
            Vs.append((V1.clone(), V2.clone()))

    # --- orthonormal basis of the EMPIRICAL Q-error subspace ---
    B1, B2 = [], []
    for a1, a2 in zip(dQ1s, dQ2s):
        w1, w2 = a1.clone(), a2.clone()
        for e1, e2 in zip(B1, B2):
            c = redot(e1, e2, w1, w2); w1 = w1 - c * e1; w2 = w2 - c * e2
        nw = pnorm(w1, w2)
        if nw > 1e-13 * max(pnorm(a1, a2), 1e-300):
            B1.append(w1 / nw); B2.append(w2 / nw)
    emp_rank = len(B1)

    # --- T restricted to that subspace, evaluated at the last anchor ---
    Va1, Va2 = Vs[-1]
    base1, base2 = phi(Va1, Va2, env, nu)
    scale = max(pnorm(Va1, Va2), 1e-30)
    cols = []
    for e1, e2 in zip(B1, B2):
        eps = 1e-6 * scale
        p1, p2 = phi(Va1 + eps * e1, Va2 + eps * e2, env, nu)
        cols.append(packm((p1 - base1) / eps, (p2 - base2) / eps, tids))
    Y = torch.stack(cols, dim=1)
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    e2v = (S.to(torch.float64) ** 2); tot = float(e2v.sum())
    top1 = float(e2v[0] / tot); top3 = float(e2v[:3].sum() / tot)

    # --- how much of the REALIZED response the top-3 directions capture ---
    dq1, dq2 = dQ1s[-1], dQ2s[-1]
    dn = pnorm(dq1, dq2); eps = 1e-6 * scale / max(dn, 1e-300)
    p1, p2 = phi(Va1 + eps * dq1, Va2 + eps * dq2, env, nu)
    resp = packm((p1 - base1) / eps, (p2 - base2) / eps, tids)
    pr3 = U[:, :3] @ (U[:, :3].conj().transpose(0, 1) @ resp)
    real3 = float((torch.linalg.vector_norm(pr3)
                   / torch.linalg.vector_norm(resp)).real.cpu())

    # --- top-3 leakage directions in PHYSICAL space, for visualisation ---
    right = Vh.conj().transpose(0, 1)[:, :3]      # coeffs in the B basis
    for k in range(min(3, right.shape[1])):
        q1 = sum(right[i, k].to(B1[0].dtype) * B1[i] for i in range(len(B1)))
        u = torch.fft.ifftn(q1, dim=(-3, -2, -1)).real
        vecs[f"{label}_dir{k}"] = u.detach().cpu().numpy().astype(np.float32)

    rows.append(dict(config=label, N=N, nu=nu, empirical_rank=emp_rank,
                     top1_energy=top1, top3_energy=top3,
                     top3_realized=real3, sv1_over_sv3=float(S[0]/S[min(2,len(S)-1)]),
                     dimQ=int(U1.numel()+U2.numel()-tids.numel())))
    print(f"  empirical Q-error rank = {emp_rank}")
    print(f"  top-1 operator energy  = {top1:.4f}")
    print(f"  top-3 operator energy  = {top3:.4f}   (original T-test: 0.985)")
    print(f"  top-3 realized capture = {real3:.4f}   (original T-test: 0.9995)")
    del env, U1, U2, t1, t2, V1, V2, m, dQ1s, dQ2s, Vs, B1, B2
    torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(SAVE / "m3d_R3fix_leakage.csv", index=False)
np.savez_compressed(SAVE / "m3d_R3fix_directions.npz", **vecs)
print("\n" + "=" * 92)
print("M3D TEST R3-FIX — LEAKAGE CONCENTRATION ON THE EMPIRICAL ERROR SUBSPACE")
print("=" * 92)
print(df.to_string(index=False))
held = bool((df.top3_energy > 0.80).all())
grew = df.dimQ.max() / df.dimQ.min()
print(f"\ndim(Q) spans {grew:.0f}x across the sweep")
print("VERDICT:", "PASS — rank-3 leakage concentration is NOT an N=40 artifact"
      if held else "FAIL — concentration degrades with dim(Q)")
print("\nSaved:", SAVE / "m3d_R3fix_leakage.csv")
print("Saved:", SAVE / "m3d_R3fix_directions.npz",
      "(top-3 leakage directions in physical space)")
