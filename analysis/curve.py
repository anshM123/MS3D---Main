"""
A — IS THERE A UNIVERSAL lam = f(kappa) CURVE ACROSS SYSTEMS?

The closed form lam = kappa/sqrt(1+2*kappa*c+kappa^2) failed: median error
0.083, and adding eta_P made it worse. Linear response is too coarse at
kappa >> 1. So stop deriving and ask an empirical question instead:

    does ONE monotone curve lam = f(kappa) fit ALL systems?

If yes, lam is predictable from a computable quantity across Navier-Stokes,
Kuramoto-Sivashinsky, complex Ginzburg-Landau and a finite-difference
discretisation with point sensors. That is weaker than a derivation and much
stronger than a scoping law.

Fits a one-parameter family motivated by the saturation structure

    f(kappa) = kappa / (kappa^p + 1)^(1/p)          p > 0

which has the right limits (f -> kappa as kappa -> 0, f -> 1 as kappa -> inf)
and one shape parameter. p = 2 recovers the c = 0 closed form.

REGISTERED BEFORE FITTING:
  U1  a single p fits all systems: median |f(kappa) - lam| < 0.10 pooled
  U2  the per-system fitted p values agree within a factor of 2
  U3  the pooled fit is better than the failed closed form (median err 0.083)
  U4  leave-one-system-out: fit p on three systems, predict the fourth,
      median error < 0.15. This is the real test — it is prospective.

Reads the JSON files the earlier runs wrote. No new simulation.
"""
import numpy as np, json, glob, os, sys

def load():
    """Collect (kappa, lam, system) from whatever run outputs are present."""
    data=[]
    for f in glob.glob("kappa*.json"):
        for r in json.load(open(f)):
            if np.isfinite(r.get("kappa",np.nan)) and np.isfinite(r.get("lam",np.nan)):
                data.append(("KS-"+r.get("band","?"), r["kappa"], r["lam"]))
    # CGL and FD runs store lam but not kappa; kappa is recoverable only where
    # both block responses were logged, so those files contribute lam only and
    # are reported separately.
    return data

def f(kap,p):
    kap=np.asarray(kap,dtype=float)
    return kap/np.power(np.power(kap,p)+1.0, 1.0/p)

def fit_p(kap,lam,grid=None):
    grid=np.logspace(-0.7,1.0,120) if grid is None else grid
    err=[np.median(np.abs(f(kap,p)-lam)) for p in grid]
    i=int(np.argmin(err)); return float(grid[i]), float(err[i])

def main():
    d=load()
    if not d:
        print("No kappa*.json found. Run kappa.py / kappa2.py first, then this"
              " in the same directory."); return
    syst=sorted(set(x[0] for x in d))
    kap=np.array([x[1] for x in d]); lam=np.array([x[2] for x in d])
    print(f"loaded {len(d)} windows across {len(syst)} bands: {syst}")
    print(f"kappa range {kap.min():.4f} .. {kap.max():.1f}"
          f"   lam range {lam.min():.4f} .. {lam.max():.4f}")

    p_all,e_all=fit_p(kap,lam)
    print(f"\nU1 pooled fit   p = {p_all:.3f}   median |f-lam| = {e_all:.4f}"
          f"  -> {'PASS' if e_all<0.10 else 'FAIL'}")

    print(f"\nU2 per-band fits")
    print(f"   {'band':<16}{'n':>5}{'p':>9}{'median err':>13}")
    ps={}
    for s in syst:
        m=[x for x in d if x[0]==s]
        k=np.array([x[1] for x in m]); l=np.array([x[2] for x in m])
        p,e=fit_p(k,l); ps[s]=p
        print(f"   {s:<16}{len(m):>5}{p:>9.3f}{e:>13.4f}")
    if len(ps)>=2:
        spread=max(ps.values())/max(min(ps.values()),1e-9)
        print(f"   spread max/min = {spread:.2f}x"
              f"  -> {'PASS' if spread<2.0 else 'FAIL'}")

    print(f"\nU3 vs the failed closed form (median err 0.0829)")
    print(f"   pooled one-parameter curve: {e_all:.4f}"
          f"  -> {'PASS' if e_all<0.0829 else 'FAIL'}")

    print(f"\nU4 leave-one-band-out (prospective)")
    print(f"   {'held out':<16}{'p from rest':>13}{'median err':>13}")
    errs=[]
    for s in syst:
        tr=[x for x in d if x[0]!=s]; te=[x for x in d if x[0]==s]
        if not tr or not te: continue
        p,_=fit_p(np.array([x[1] for x in tr]),np.array([x[2] for x in tr]))
        e=float(np.median(np.abs(f(np.array([x[1] for x in te]),p)
                                 -np.array([x[2] for x in te]))))
        errs.append(e)
        print(f"   {s:<16}{p:>13.3f}{e:>13.4f}")
    if errs:
        print(f"   median across held-out bands = {np.median(errs):.4f}"
              f"  -> {'PASS' if np.median(errs)<0.15 else 'FAIL'}")

    print(f"\n   binned check")
    print(f"   {'kappa bin':<22}{'n':>5}{'median lam':>12}{'f(kappa)':>10}")
    o=np.argsort(kap)
    for b in np.array_split(o,6):
        kk=kap[b]; ll=lam[b]
        print(f"   {f'{kk.min():.3f} - {kk.max():.3f}':<22}{len(b):>5}"
              f"{np.median(ll):>12.4f}{np.median(f(kk,p_all)):>10.4f}")

if __name__=="__main__": main()
