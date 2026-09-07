"""Full M3D pipeline on KS, swept over domain size to build a rank ladder."""
import numpy as np, sys, json, time
sys.path.insert(0, '/home/claude/ks')
from ks_core import KS
from ks_frame import *

NPM      = 6          # observable modes
SEG_T    = 0.25       # segment length in time units
NSEG     = 8
FORECAST = 2
ANCHORS  = (2, 3, 4)
BURN     = 40000

def one_case(L, ntraj=8, snaps=400, verbose=True):
    N = int(2*L); ks = KS(L=L, N=N, dt=0.005); ns = int(SEG_T/ks.dt)
    # --- trajectory rank on the attractor ---
    u = ks.phi(ks.ic(1), BURN); S = []
    for _ in range(snaps):
        u = ks.phi(u, 100); S.append(np.fft.irfft(u, n=N).copy())
    S = np.array(S); S -= S.mean(0)
    sv = np.linalg.svd(S, compute_uv=False); e = sv**2; c = np.cumsum(e)/e.sum()
    r_traj = int((sv > sv[0]*1e-9).sum())
    m999 = int(np.searchsorted(c, 0.999))+1
    rows = []
    for ti in range(ntraj):
        u0 = ks.phi(ks.ic(1000+ti), BURN)
        MO = rom_run(ks, u0, NPM, NSEG, ns)
        TR = [u0.copy()]; t = u0.copy()
        for _ in range(NSEG):
            t = ks.phi(t, ns); TR.append(t.copy())
        EH = w_estimate(ks, MO, ns, max(ANCHORS))
        dQs = [split_PQ(TR[k]-MO[k], NPM)[1] for k in range(1, NSEG+1)]
        _, r_emp, r_leak, top3 = leakage(ks, MO[ANCHORS[-1]], dQs, NPM, ns)
        for k in ANCHORS:
            V, T_, TF = MO[k], TR[k], TR[k+FORECAST]
            EQ = split_PQ(EH[k], NPM)[1]
            AQ = split_PQ(T_-V, NPM)[1]
            dP = split_PQ(T_-V, NPM)[0]
            if not np.isfinite(np.linalg.norm(EQ)) or \
               np.linalg.norm(EQ) > 0.5*np.linalg.norm(V):
                continue
            mism = np.linalg.norm(split_PQ(EQ, NPM)[0])/np.linalg.norm(split_PQ(V, NPM)[0])
            def fc(x):
                a = x.copy()
                for _ in range(FORECAST):
                    a = rom_segment(ks, a, star2_basis(ks, a, NPM), ns)
                return a
            FB, FE = fc(V), fc(V+EQ)
            FN, FO = fc(V-EQ), fc(V+AQ)
            be, ee = terr(FB, TF, NPM), terr(FE, TF, NPM)
            ne, oe = terr(FN, TF, NPM), terr(FO, TF, NPM)
            rho, ceil, psr = reach(ks, star2_basis(ks, V, NPM),
                                   split_PQ(FB-TF, NPM)[0], NPM)
            rows.append(dict(L=L, N=N, traj=ti, seg=k, r_traj=r_traj, m999=m999,
                r_emp=r_emp, r_leak=r_leak, top3=top3,
                reservoir=np.linalg.norm(AQ)/max(np.linalg.norm(dP), 1e-30),
                start_mismatch=mism, base=be, est=ee, neg=ne, orac=oe,
                gain=be/max(ee, 1e-30), gain_neg=be/max(ne, 1e-30),
                gain_orac=be/max(oe, 1e-30), beats=bool(ee < be),
                beats_neg=bool(ee < ne), rho=rho, ceiling=ceil, ps_rank=psr,
                bound_ok=bool(be/max(ee, 1e-30) <= ceil*(1+1e-6))))
        if verbose:
            g = [r["gain"] for r in rows if r["traj"] == ti]
            b = [r["base"] for r in rows if r["traj"] == ti]
            print(f"    traj {ti}: base " + " ".join(f"{100*x:.2f}%" for x in b)
                  + "  gain " + " ".join(f"{x:.3f}" for x in g), flush=True)
    return rows

if __name__ == "__main__":
    Ls = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                             else ["22","50","100"])]
    allr = []
    for L in Ls:
        t0 = time.time()
        print(f"\n=== L={L:.0f}  N={int(2*L)} ===", flush=True)
        rs = one_case(L)
        allr += rs
        r = rs[0]
        print(f"  r_traj={r['r_traj']}  99.9%={r['m999']}  r_emp={r['r_emp']}"
              f"  r_leak={r['r_leak']}  top3={r['top3']:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(allr, open("/home/claude/ks/ks_results.json", "w"), default=float)
    print("\nsaved", len(allr), "rows")
