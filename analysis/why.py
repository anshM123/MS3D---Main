"""
WHY DOES p GROW WITH SENSOR COUNT?  (p ~ n^0.29, r = +0.974)

HYPOTHESIS. p is not a free shape parameter — it is a proxy for the ALIGNMENT
between the two block responses.

The derived form is

    lam = kappa / sqrt(1 + 2 kappa c + kappa^2),   c = cos(dP, dQ)

At c = 0 that is EXACTLY the p = 2 member of the fitted family
f(kappa) = kappa/(kappa^p + 1)^(1/p). At c -> -1 or +1 it degenerates toward
p = 1. So p measures how orthogonal the observable-block and hidden-block
responses are.

Alignment is dimensional: two vectors in a d-dimensional space have
|cos| ~ 1/sqrt(d). More sensors -> larger observable space -> more orthogonal
responses -> larger p. That is the mechanism, and it predicts the direction of
the measured drift.

TESTS, on data already collected:
  W1  does |c| fall with the observable dimension?   |c| ~ d^(-1/2)
  W2  does the fitted p track the measured |c|?      per configuration
  W3  does using the MEASURED c per window beat the fitted p?
      i.e. is lam = kappa/sqrt(1+2 kappa c + kappa^2) with the true c a
      better predictor than any single p — which would say the p-family was
      only ever a stand-in for c
"""
import numpy as np, json, sys

def f(kap,p):
    kap=np.asarray(kap,float); return kap/np.power(np.power(kap,p)+1.0,1.0/p)
def fit_p(kap,lam):
    grid=np.logspace(-1.2,1.3,300)
    e=[np.median(np.abs(f(kap,p)-lam)) for p in grid]
    i=int(np.argmin(e)); return float(grid[i]),float(e[i])

rows=json.load(open(sys.argv[1] if len(sys.argv)>1 else "master_kappa.json"))
K=np.array([r["kappa"] for r in rows]); L=np.array([r["lam"] for r in rows])
C=np.array([r["c"] for r in rows]); S=np.array([r["system"] for r in rows])
B=np.array([r["band"] for r in rows])

# observable dimension per configuration (REAL dimensions)
def dim_of(s,b):
    if s=="FD":  return int(b.replace("sensors",""))        # 1 real dof/sensor
    return 12                                               # 6 modes x (re,im)

print("="*82); print("W1/W2 — alignment, dimension, and p"); print("="*82)
print(f"  {'config':<18}{'n':>5}{'dim':>6}{'median |c|':>12}{'fitted p':>10}"
      f"{'err':>9}")
cfg=[]
for s in sorted(set(S)):
    for b in sorted(set(B[S==s])):
        m=(S==s)&(B==b)
        if m.sum()<4: continue
        p,e=fit_p(K[m],L[m]); d=dim_of(s,b); ac=float(np.median(np.abs(C[m])))
        cfg.append((f"{s}/{b}",int(m.sum()),d,ac,p,e))
        print(f"  {s+'/'+b:<18}{int(m.sum()):>5}{d:>6}{ac:>12.4f}{p:>10.3f}{e:>9.4f}")

if len(cfg)<3:
    print("\n  too few configurations with enough windows for W1/W2.")
    print("  (this file has n={} rows total)".format(len(rows)))
    import sys as _s; d=ac=pv=None
else:
    d=np.array([c[2] for c in cfg],float); ac=np.array([c[3] for c in cfg])
    pv=np.array([c[4] for c in cfg])
def sp(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])
if d is not None:
  print(f"\nW1 |c| vs dimension:   log-log slope "
      f"{np.polyfit(np.log(d),np.log(ac),1)[0]:+.3f}   "
      f"(prediction -0.5)   r = {np.corrcoef(np.log(d),np.log(ac))[0,1]:+.3f}")
  print(f"W2 p vs |c|:           Spearman {sp(ac,pv):+.3f}"
        f"   (prediction NEGATIVE: more aligned -> smaller p)")
  print(f"   p vs dimension:     Spearman {sp(d,pv):+.3f}")

print("\n"+"="*82); print("W3 — does the MEASURED c beat the fitted p?"); print("="*82)
lam_c = K/np.sqrt(np.maximum(1+2*K*C+K*K,1e-300))
p_all,e_all=fit_p(K,L)
e_c=float(np.median(np.abs(lam_c-L)))
print(f"  best single p over everything: p={p_all:.3f}   median err {e_all:.4f}")
print(f"  derived form with MEASURED c:              median err {e_c:.4f}")
print(f"  -> {'c BEATS the p-family: p was a stand-in for alignment' if e_c<e_all else 'the p-family still wins; c alone does not explain it'}")
print(f"\n  per class")
for nm,msk in (("Fourier (KS+CGL)",(S=="KS")|(S=="CGL")),("sensors (FD)",S=="FD")):
    p_,e_=fit_p(K[msk],L[msk])
    print(f"   {nm:<20} fitted p {p_:.3f} err {e_:.4f}   |"
          f"   measured-c err {np.median(np.abs(lam_c[msk]-L[msk])):.4f}"
          f"   median |c| {np.median(np.abs(C[msk])):.4f}")
