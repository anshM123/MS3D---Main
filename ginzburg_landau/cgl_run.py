"""
PROSPECTIVE TEST — Complex Ginzburg-Landau, a system the framework has never
seen, with a CUBIC nonlinearity (NS and KS are both exactly quadratic).

Self-contained: pure numpy, no datasets. Needs cgl_core.py alongside it.

=====================  PREDICTIONS REGISTERED BEFORE RUNNING  =================
From the effect-size law  gain = 1/sqrt(1 + 2*lam*cos + lam^2)  and the KS
stability result (Spearman(growth rate, lambda) = -0.900):

  C1  THE IDENTITY IS ALGEBRA, so it must hold on ANY system:
        per-window |predicted - observed|/observed  <  1e-9
      Failure here means the implementation is wrong, not the theory.

  C2  CAUSAL SIGNATURE TRANSFERS:
        +q_hat beats -q_hat in >= 80% of windows

  C3  STABILITY SETS LAMBDA (the KS mechanism, applied to a new system):
        Spearman(mean growth rate over P, lambda)  <  -0.6
      i.e. lambda RISES as the observable band becomes more damped.

  C4  QUANTITATIVE, the risky one:
        unstable band (growth > 0)     -> lambda < 0.25 and gain < 1.30
        most damped band tested        -> lambda > 0.60 and gain > 2.0

C1 and C2 test whether the framework works at all outside its development
systems. C3 and C4 test whether the KS-derived MECHANISM predicts a system it
was not derived on. C4 is the one that can embarrass us, which is the point.
==============================================================================
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from cgl_core import CGL, splitM, basis, seg, terr, rdot, rnorm

NTRAJ = int(os.environ.get("CGL_NTRAJ", 20))
NS_   = int(os.environ.get("CGL_NS", 250))      # 0.5 time units per segment
NPM   = int(os.environ.get("CGL_NPM", 6))
BURN  = 20000
NSEG, FC = 6, 2
ANCH  = (3, 4)

def band_modes(sys, lo, hi, want):
    gr = sys.growth
    cand = [i for i in range(1, sys.N) if sys.deal[i] and lo <= gr[i] <= hi]
    cand.sort(key=lambda i: -gr[i])
    return cand[:want]

BANDS = [("unstable",  0.0,   1e9),
         ("neutral",  -3.0,   0.0),
         ("mild",    -12.0,  -3.0),
         ("damped",  -40.0, -12.0)]

def main():
    s = CGL(N=256, L=100.0, b=2.0, c=-1.0, dt=0.002)
    print(f"CGL N={s.N} L={s.L:.0f} b={s.b} c={s.c}  segment={NS_*s.dt:.2f} time units")
    print(f"CUBIC nonlinearity — NS and KS are both exactly quadratic.\n")
    rows = []; t0 = time.time()
    for nm, lo, hi in BANDS:
        md = band_modes(s, lo, hi, NPM)
        if len(md) < 3:
            print(f"  {nm}: only {len(md)} modes available, skipped"); continue
        g = float(np.mean(s.growth[md]))
        for ti in range(NTRAJ):
            A = s.phi(s.ic(70000+ti), BURN)
            V = A.copy(); MO = [V.copy()]
            for _ in range(NSEG): V = seg(s, V, basis(s, V, md), NS_); MO.append(V.copy())
            T = [A.copy()]; t = A.copy()
            for _ in range(NSEG): t = s.phi(t, NS_); T.append(t.copy())
            E = np.zeros_like(MO[0]); EH = [E.copy()]
            for k in range(max(ANCH)):
                Phi = s.phi(MO[k], NS_); eta = Phi - MO[k+1]
                if rnorm(E) < 1e-30: E = eta.copy()
                else:
                    ep = min(max(1e-5*rnorm(MO[k])/rnorm(E), 1e-6), 5e-2)
                    Pp = s.phi(MO[k]+ep*E, NS_); Mm = s.phi(MO[k]-ep*E, NS_)
                    E = eta + (Pp-Mm)/(2*ep) + 0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fc(x, n=FC):
                a = x.copy()
                for _ in range(n): a = seg(s, a, basis(s, a, md), NS_)
                return a
            for k in ANCH:
                if k >= len(EH): continue
                Vk, TF = MO[k], T[k+FC]
                qh = splitM(EH[k], md)[1]
                dP, AQ = splitM(T[k]-Vk, md)
                if not np.isfinite(rnorm(qh)) or rnorm(qh) > 0.5*rnorm(Vk): continue
                FB = fc(Vk); FE = fc(Vk+qh); FN = fc(Vk-qh); FO = fc(Vk+AQ)
                b = splitM(FB-TF, md)[0]; D = splitM(FO-FB, md)[0]
                nb, nd = rnorm(b), rnorm(D)
                if nb < 1e-30 or nd < 1e-30: continue
                lam = nd/nb; cos = rdot(b, D)/(nb*nd)
                pred = 1/np.sqrt(max(1+2*lam*cos+lam*lam, 1e-300))
                be = terr(FB, TF, md); oe = terr(FO, TF, md)
                obs = be/max(oe, 1e-30)
                rows.append(dict(band=nm, growth=g, traj=ti, seg=k, base=be,
                    valid=bool(be < 0.50),
                    lam=lam, cos=cos, pred=pred, gain_orac=obs,
                    iden=abs(pred-obs)/max(obs, 1e-30),
                    gain=be/max(terr(FE, TF, md), 1e-30),
                    gain_neg=be/max(terr(FN, TF, md), 1e-30),
                    reservoir=rnorm(AQ)/max(rnorm(dP), 1e-30),
                    mismatch=rnorm(splitM(qh, md)[0])/max(rnorm(splitM(Vk, md)[0]),1e-30)))
        sub = [r for r in rows if r["band"] == nm]
        if sub:
            print(f"  {nm:<9} growth={g:>8.2f}  n={len(sub):>3}  "
                  f"base={100*np.median([x['base'] for x in sub]):6.2f}%  "
                  f"lam={np.median([x['lam'] for x in sub]):.4f}  "
                  f"cos={np.median([x['cos'] for x in sub]):+.4f}  "
                  f"gain={np.median([x['gain'] for x in sub]):8.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open("cgl_prospective.json", "w"), default=float)

    print("\n" + "="*80); print("CGL PROSPECTIVE TEST — REGISTERED PREDICTIONS"); print("="*80)
    nv = [r for r in rows if not r["valid"]]
    if nv:
        bad = sorted(set(r["band"] for r in nv))
        print(f"\nVALIDITY GATE: {len(nv)}/{len(rows)} windows have ROM error > 50%")
        print(f"  affected bands: {bad}  — these are NOT reduced models of their")
        print(f"  observables and are excluded from C2/C3/C4.")
    rows_all = rows; rows = [r for r in rows if r["valid"]]
    if not rows:
        print("\nALL windows failed the validity gate. Retune CGL_NS / bands."); return
    idn = [r["iden"] for r in rows]
    print(f"\nC1 identity holds: max |pred-obs|/obs = {max(idn):.3e}"
          f"  -> {'PASS' if max(idn) < 1e-9 else 'FAIL'}")
    nw = sum(1 for r in rows if r["gain"] > r["gain_neg"])
    print(f"C2 causal signature: +q beats -q in {nw}/{len(rows)}"
          f"  -> {'PASS' if nw >= 0.8*len(rows) else 'FAIL'}")
    bl = sorted(set(r["band"] for r in rows), key=lambda n: -[r for r in rows if r["band"]==n][0]["growth"])
    gr = [[r for r in rows if r["band"]==n][0]["growth"] for n in bl]
    lm = [float(np.median([r["lam"] for r in rows if r["band"]==n])) for n in bl]
    gn = [float(np.median([r["gain"] for r in rows if r["band"]==n])) for n in bl]
    def sp(x, y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0,1])
    if len(bl) >= 3:
        print(f"C3 Spearman(growth, lambda) = {sp(gr,lm):+.3f}"
              f"  -> {'PASS' if sp(gr,lm) < -0.6 else 'FAIL'}")
    print(f"\nC4 quantitative")
    print(f"  {'band':<10}{'growth':>9}{'lambda':>9}{'cos':>9}{'gain':>10}")
    for n, g_, l_, gg in zip(bl, gr, lm, gn):
        c_ = float(np.median([r["cos"] for r in rows if r["band"]==n]))
        print(f"  {n:<10}{g_:>9.2f}{l_:>9.4f}{c_:>+9.4f}{gg:>10.4f}")
    u = [i for i,n in enumerate(bl) if n=="unstable"]
    if u:
        i=u[0]; ok = lm[i] < 0.25 and gn[i] < 1.30
        print(f"  unstable: lambda {lm[i]:.4f} (<0.25) gain {gn[i]:.4f} (<1.30)"
              f"  -> {'PASS' if ok else 'FAIL'}")
    if lm:
        j = int(np.argmin(gr)); ok = lm[j] > 0.60 and gn[j] > 2.0
        print(f"  most damped ({bl[j]}): lambda {lm[j]:.4f} (>0.60) "
              f"gain {gn[j]:.4f} (>2.0)  -> {'PASS' if ok else 'FAIL'}")
    print(f"\n  max start-P mismatch: {max(r['mismatch'] for r in rows):.3e}")

if __name__ == "__main__":
    main()
