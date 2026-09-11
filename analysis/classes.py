"""
Is p a property of the OBSERVATION OPERATOR rather than the system?

U4 failed at n=224, but not randomly. Per-system fits were excellent
(CGL 0.023, KS 0.025, FD 0.070) — the curve SHAPE is right, the parameter does
not transfer. And the fitted p splits by how the system is OBSERVED:

    CGL  Fourier-mode observable, CUBIC, spectral      p = 1.342
    KS   Fourier-mode observable, quadratic, spectral  p = 0.786
    FD   POINT-SENSOR observable, quadratic, finite diff  p = 0.348

KS and CGL are within 1.7x of each other despite different equations and
different nonlinearity classes. FD sits 2.3-3.9x away, and it is the one with a
different observation operator.

TWO CHECKS, both on data already collected:

  C1  do the two Fourier-observable systems share a curve?
      Fit on KS, predict CGL, and vice versa. If both hold-out errors are small,
      that is a two-system prospective pass within the Fourier class.

  C2  does p drift with SENSOR DENSITY inside FD?
      8, 12 and 20 sensors were run. If p moves systematically with sensor
      count, the observation operator is what sets it, and the two-class
      reading is confirmed rather than assumed.
"""
import numpy as np, json, sys

def f(kap,p):
    kap=np.asarray(kap,float)
    return kap/np.power(np.power(kap,p)+1.0,1.0/p)
def fit_p(kap,lam):
    grid=np.logspace(-1.2,1.3,300)
    err=[np.median(np.abs(f(kap,p)-lam)) for p in grid]
    i=int(np.argmin(err)); return float(grid[i]),float(err[i])

rows=json.load(open(sys.argv[1] if len(sys.argv)>1 else "master_kappa.json"))
K=np.array([r["kappa"] for r in rows]); L=np.array([r["lam"] for r in rows])
S=np.array([r["system"] for r in rows]); B=np.array([r["band"] for r in rows])

print("="*80); print("C1 — do the two FOURIER-OBSERVABLE systems share a curve?")
print("="*80)
fo=(S=="KS")|(S=="CGL")
p_fo,e_fo=fit_p(K[fo],L[fo])
print(f"  pooled KS+CGL:  p = {p_fo:.3f}   median |f-lam| = {e_fo:.4f}   n={int(fo.sum())}")
for held in ("KS","CGL"):
    tr=fo&(S!=held); te=S==held
    p,_=fit_p(K[tr],L[tr])
    e=float(np.median(np.abs(f(K[te],p)-L[te])))
    print(f"  fit on {'CGL' if held=='KS' else 'KS':<4} -> predict {held:<4}"
          f"  p={p:.3f}  median err {e:.4f}"
          f"   {'PASS' if e<0.15 else 'FAIL'}")
pf=(S=="FD")
p_fd,e_fd=fit_p(K[pf],L[pf])
print(f"\n  for contrast, sensor-observable FD alone: p = {p_fd:.3f}"
      f"  median err {e_fd:.4f}   n={int(pf.sum())}")
print(f"  Fourier-class p {p_fo:.3f}  vs  sensor-class p {p_fd:.3f}"
      f"   ratio {p_fo/max(p_fd,1e-9):.2f}x")

print("\n"+"="*80); print("C2 — does p drift with SENSOR DENSITY inside FD?")
print("="*80)
print(f"  {'sensors':<12}{'n':>5}{'p':>9}{'median err':>13}{'median kappa':>14}")
ps=[]; ns=[]
for b in sorted(set(B[pf]), key=lambda x:int(x.replace('sensors',''))):
    m=pf&(B==b)
    if m.sum()<8: continue
    p,e=fit_p(K[m],L[m]); ps.append(p); ns.append(int(b.replace('sensors','')))
    print(f"  {b:<12}{int(m.sum()):>5}{p:>9.3f}{e:>13.4f}{np.median(K[m]):>14.3f}")
if len(ps)>=3:
    r=np.corrcoef(np.log(ns),np.log(ps))[0,1]
    sl=np.polyfit(np.log(ns),np.log(ps),1)[0]
    print(f"\n  log-log slope d(log p)/d(log n_sensors) = {sl:+.3f}   r = {r:+.3f}")
    print(f"  p range {min(ps):.3f} .. {max(ps):.3f}  ({max(ps)/min(ps):.2f}x)")
    if abs(r)>0.9 and max(ps)/min(ps)>1.3:
        print("  -> p DRIFTS with sensor density: the observation operator sets it")
    elif max(ps)/min(ps)<1.3:
        print("  -> p is FLAT in sensor density: it is a property of the")
        print("     operator TYPE (point vs modal), not of how many sensors")
    else:
        print("  -> inconclusive")

print("\n"+"="*80); print("TWO-CLASS MODEL vs ONE-CLASS")
print("="*80)
p_all,e_all=fit_p(K,L)
e_two=np.median(np.concatenate([np.abs(f(K[fo],p_fo)-L[fo]),
                                np.abs(f(K[pf],p_fd)-L[pf])]))
print(f"  one curve for everything:      p={p_all:.3f}   median err {e_all:.4f}")
print(f"  two curves by observable type:               median err {e_two:.4f}")
print(f"  improvement {e_all/max(e_two,1e-9):.2f}x")
