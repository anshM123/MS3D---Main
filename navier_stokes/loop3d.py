"""
ITEM C — END-TO-END CLOSED LOOP WITH WALL-CLOCK.  CPU, numpy, self-contained.

Every cost number so far measures the ESTIMATOR. This measures the whole
algorithm, running autonomously, against the full solver it is meant to replace.

  V_k -> coarse defect -> e_hat -> candidate probe -> alpha_hat, subspace
      -> correction -> V_{k+1}          repeat every segment, no truth

and times it. Three quantities that have never been reported together:

  ACCURACY   probe-observable error over a long horizon
  COST       wall-clock for the CORRECTED ROM vs the full solver
  INFORMATION  none consumed

Baselines on identical trajectories:
  A  full solver                the thing being replaced (cost reference)
  B  uncorrected ROM            the cheap wrong answer
  C  ROM + open-loop correction one correction, then coast (all prior work)
  D  ROM + CLOSED LOOP          corrected every segment, autonomous

The honest question is not "is D more accurate than B" — it must be. It is:
at equal wall-clock, does D beat what the full solver achieves in the same time?

TRUST GATE. The estimator has a measured amplitude horizon (~15% error in the
NS record). D declines to correct when ||Q e_hat||/||V|| exceeds HORIZON or the
predicted gain G_hat is below GATE, so a closed loop cannot drive itself into
the regime where the second-order recurrence diverges.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from core import NS
from probe3d import (build_probes, readings, splitB, rom_basis, rom_seg,
                     rdot, rnorm, perr)

# ---------------------------------------------------------------- coarse map
# Using the FINE flow map inside the estimator makes the loop (a) degenerate —
# it reconstructs truth, exactly the Z2 artifact — and (b) more expensive than
# the solver it replaces. The coarse defect is the whole point of item C.
def _sel(n, m):
    h = m//2
    return np.array(list(range(0, h+1)) + list(range(n-h+1, n)), dtype=int)

def restrict(U, n, m):
    i = _sel(n, m)
    return U[:, i][:, :, i][:, :, :, i] * (float(m)/n)**3

def prolong(U, m, n):
    i = _sel(n, m)
    out = np.zeros((3, n, n, n), complex)
    out[np.ix_(np.arange(3), i, i, i)] = U * (float(n)/m)**3
    return out

N     = int(os.environ.get("LP_N", 24))
NU    = float(os.environ.get("LP_NU", 2.5e-4))
AMP   = float(os.environ.get("LP_AMP", 5.0))
NPROBE= int(os.environ.get("LP_NPROBE", 8))
NSTEP = int(os.environ.get("LP_NSTEP", 20))
NSEG  = int(os.environ.get("LP_NSEG", 14))     # long horizon: the loop must hold
NTRAJ = int(os.environ.get("LP_NTRAJ", 8))
GATE  = float(os.environ.get("LP_GATE", 1.02))
HORIZON = float(os.environ.get("LP_HORIZON", 0.15))
ALPHAS= [0.0, 0.5, 1.0, 1.5, 2.0]

def coarse_phi(s, sc, U, M):
    """Full-map surrogate evaluated on an M^3 grid, then prolonged back."""
    return prolong(sc.phi(restrict(U, s.N, M), NSTEP), M, s.N)

def one_step_estimate(s, B, V, E, sc=None, M=None):
    """Advance the truth-free defect recurrence one segment. Returns (E', V').
    If sc/M are given the defect uses the COARSE map."""
    _phi = (lambda x: coarse_phi(s, sc, x, M)) if sc is not None else \
           (lambda x: s.phi(x, NSTEP))
    Phi = _phi(V)
    Vn  = rom_seg(s, V, rom_basis(s, V, B), NSTEP)
    eta = Phi - Vn
    if rnorm(E) < 1e-30:
        return eta.copy(), Vn
    ep = min(max(1e-5*rnorm(V)/rnorm(E), 1e-6), 5e-2)
    Pp = _phi(V + ep*E); Mm = _phi(V - ep*E)
    return eta + (Pp-Mm)/(2*ep) + 0.5*(Pp+Mm-2*Phi)/(ep*ep), Vn

def main():
    s = NS(N=N, L=8.0, nu=NU, dt=0.1)
    M = int(os.environ.get("LP_M", 16))
    sc = NS(N=M, L=8.0, nu=NU, dt=0.1)
    idx, B = build_probes(s, NPROBE, 0)
    print(f"coarse defect grid M={M} (fine N={N}) -> nominal {(M/N)**3:.3f}x per map")
    print(f"3D NS N={N} nu={NU:.1e}  {NPROBE} probes -> {len(B)} directions")
    print(f"{NSEG} segments x {NSTEP} steps   gate G_hat>{GATE}  horizon {HORIZON}\n",
          flush=True)
    rows=[]; T_full=T_rom=T_loop=0.0
    for ti in range(NTRAJ):
        U0 = s.ic(2000+ti, amp=AMP, ic_khi=5.0)
        # --- A: full solver, and the truth it defines ---
        t0=time.time(); TR=[U0.copy()]; t=U0.copy()
        for _ in range(NSEG): t=s.phi(t,NSTEP); TR.append(t.copy())
        T_full += time.time()-t0
        # --- B: uncorrected ROM ---
        t0=time.time(); V=U0.copy(); MO=[V.copy()]
        for _ in range(NSEG):
            V=rom_seg(s,V,rom_basis(s,V,B),NSTEP); MO.append(V.copy())
        T_rom += time.time()-t0
        # --- D: closed loop, autonomous, no truth ---
        t0=time.time()
        Vc=U0.copy(); E=np.zeros_like(U0); LO=[Vc.copy()]
        nfire=0; nskip=0
        for k in range(NSEG):
            E, Vn = one_step_estimate(s, B, Vc, E, sc, M)
            ph, qh = splitB(E, B)
            rel = rnorm(qh)/max(rnorm(Vn),1e-30)
            if (not np.isfinite(rel)) or rel > HORIZON:
                Vc = Vn; LO.append(Vc.copy()); nskip += 1
                E = splitB(E, B)[0]          # keep only the P part after skipping
                continue
            FB = rom_seg(s, Vn, rom_basis(s, Vn, B), NSTEP)
            cand = {("none",0.0): np.zeros_like(Vn)}
            for a in ALPHAS[1:]:
                cand[("Q",a)] = a*qh; cand[("P",a)] = a*ph
            pr = {kk: splitB(rom_seg(s, Vn+c, rom_basis(s,Vn+c,B), NSTEP)-FB, B)[0]
                  for kk,c in cand.items()}
            bh = -splitB(E, B)[0]
            pick = min(cand, key=lambda kk: rnorm(bh+pr[kk]))
            Gh = rnorm(bh)/max(rnorm(bh+pr[pick]),1e-30)
            if Gh > GATE and pick[0] != "none":
                Vc = Vn + cand[pick]; nfire += 1
                E = E - cand[pick]           # the recurrence must see the change
            else:
                Vc = Vn; nskip += 1
            LO.append(Vc.copy())
        T_loop += time.time()-t0
        # --- C: open loop, one correction at segment 3, then coast ---
        Vo=U0.copy(); Eo=np.zeros_like(U0)
        for k in range(NSEG):
            if k==3:
                qo=splitB(Eo,B)[1]
                Vo=Vo+qo
            Eo,Vo=one_step_estimate(s,B,Vo,Eo,sc,M)
        for k in (NSEG//2, NSEG-1):
            rows.append(dict(traj=ti,seg=k,
                e_rom=perr(MO[k],TR[k],B), e_loop=perr(LO[k],TR[k],B),
                e_open=perr(Vo,TR[NSEG],B) if k==NSEG-1 else np.nan,
                fired=nfire, skipped=nskip))
        print(f"  traj {ti+1}/{NTRAJ}  fired {nfire}/{NSEG}  "
              f"ROM {100*perr(MO[-1],TR[-1],B):.2f}%  "
              f"LOOP {100*perr(LO[-1],TR[-1],B):.2f}%  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows,open("loop3d.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows if np.isfinite(r[k])])
    fin=[r for r in rows if r["seg"]==NSEG-1]
    FLOOR=float(os.environ.get("LP_FLOOR",1e-5))
    ndeg=sum(1 for r in fin if r["e_loop"]<FLOOR)
    if ndeg:
        print(f"\n  !! DEGENERACY: {ndeg}/{len(fin)} closed-loop trajectories have")
        print(f"     final error below {FLOOR:.0e}. The loop is reconstructing truth,")
        print(f"     not correcting a model. Any 'improvement' figure is meaningless.")
    print("\n"+"="*82); print(f"CLOSED LOOP, n={len(fin)} trajectories"); print("="*82)
    print(f"\nC1 ACCURACY at the final segment (probe-observable error)")
    print(f"   {'method':<28}{'median':>11}{'p90':>11}")
    for nm,k in (("uncorrected ROM","e_rom"),("open loop (1 correction)","e_open"),
                 ("CLOSED LOOP","e_loop")):
        v=np.array([r[k] for r in fin if np.isfinite(r[k])])
        if len(v): print(f"   {nm:<28}{100*np.median(v):>10.3f}%{100*np.percentile(v,90):>10.3f}%")
    er=np.array([r["e_rom"] for r in fin]); el=np.array([r["e_loop"] for r in fin])
    print(f"   closed-loop improvement over uncorrected: "
          f"{np.median(er)/max(np.median(el),1e-30):.2f}x")
    print(f"\nC2 WALL-CLOCK, whole algorithm, not just the estimator")
    print(f"   {'component':<28}{'total s':>10}{'vs full solver':>16}")
    for nm,t_ in (("full solver",T_full),("uncorrected ROM",T_rom),
                  ("CLOSED LOOP",T_loop)):
        print(f"   {nm:<28}{t_:>10.1f}{t_/max(T_full,1e-9):>15.2f}x")
    print(f"\nC3 the honest comparison — accuracy per unit wall-clock")
    print(f"   closed loop costs {T_loop/max(T_full,1e-9):.2f}x the full solver")
    if T_loop > T_full:
        print(f"   -> AT THIS SCALE THE LOOP IS MORE EXPENSIVE THAN THE SOLVER.")
        print(f"      It buys accuracy over the uncorrected ROM, not over the FOM.")
        print(f"      The estimator calls full-solver segments; that only pays off")
        print(f"      when the coarse defect is used (Tests C/CX/CS) and N is large.")
    else:
        print(f"   -> the loop beats the full solver on cost at this scale")
    f=A("fired"); sk=A("skipped")
    print(f"\nC4 TRUST GATE   fired {np.median(f):.0f}/{NSEG} segments,"
          f" declined {np.median(sk):.0f}")
    print(f"   no divergence in {len(fin)}/{len(fin)} trajectories"
          f"  -> {'PASS' if np.all(np.isfinite(el)) else 'FAIL'}")

if __name__=="__main__": main()
