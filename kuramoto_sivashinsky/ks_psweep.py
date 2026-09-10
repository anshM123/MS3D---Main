"""KS P-sweep: hold the system fixed, change only the DYNAMICAL ROLE of P.

KS linear operator is lam_k = k^2 - k^4, so modes with k<1 GROW and k>1 DECAY.
Choosing P from different parts of that spectrum changes whether observable
error self-amplifies or is driven from elsewhere -- with the equation, the
resolution, the ROM construction and the protocol all held fixed.

PREREGISTERED PREDICTION (standing hypothesis, §0.5 of the record):
  if lambda is small because observable error SELF-amplifies, then moving P
  from unstable to stable modes should RAISE lambda and RAISE the gain.
"""
import numpy as np, sys
sys.path.insert(0, '/home/claude/ks')
from ks_core import KS

def splitM(uh, modes):
    P = np.zeros_like(uh); P[modes] = uh[modes]
    return P, uh - P

def basis(ks, uh, modes):
    B = []
    for m in modes:
        for ph in (0, 1):
            e = np.zeros_like(uh); e[m] = 1.0 if ph == 0 else 1.0j
            B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B: v = v - np.vdot(b, v)*b
        n = np.linalg.norm(v); return v/n if n > 1e-12 else None
    f = ks.F(uh)
    q0 = orth(splitM(f, modes)[1])
    if q0 is not None: B.append(q0)
    q1 = orth(splitM(ks.DF(uh, f), modes)[1])
    if q1 is not None: B.append(q1)
    return B

def seg(ks, anchor, B, n):
    a = np.zeros(len(B), dtype=complex)
    def rhs(a):
        u = anchor + sum(a[i]*B[i] for i in range(len(B)))
        F = ks.F(u); return np.array([np.vdot(B[i], F) for i in range(len(B))])
    dt = ks.dt
    for _ in range(n):
        k1 = rhs(a); k2 = rhs(a+0.5*dt*k1)
        k3 = rhs(a+0.5*dt*k2); k4 = rhs(a+dt*k3)
        a = a + dt/6*(k1+2*k2+2*k3+k4)
    return anchor + sum(a[i]*B[i] for i in range(len(B)))

def terrM(A, T, modes):
    return np.linalg.norm(splitM(A-T, modes)[0])/np.linalg.norm(splitM(T, modes)[0])

L, NS_ = 100.0, 200
ks = KS(L=L, N=NS_, dt=0.005); ns = 50
kk = ks.k
BANDS = [("unstable  (k<1)",      list(range(1, 7))),
         ("mixed",                list(range(8, 14))),
         ("near-neutral (k~1)",   list(range(14, 20))),
         ("stable    (k>1)",      list(range(22, 28))),
         ("strongly damped",      list(range(34, 40)))]
print(f"KS L={L:.0f}  N={NS_}   linear growth rate lam_k = k^2 - k^4")
print(f"  {'band':<22}{'modes':>10}{'k range':>14}{'lam_k range':>20}")
for nm, md in BANDS:
    k0, k1 = kk[md[0]], kk[md[-1]]
    l0, l1 = k0**2-k0**4, k1**2-k1**4
    print(f"  {nm:<22}{str(md[0])+'-'+str(md[-1]):>10}"
          f"{f'{k0:.2f}-{k1:.2f}':>14}{f'{l0:+.3f} to {l1:+.3f}':>20}")

print(f"\n  {'band':<22}{'base%':>8}{'R':>8}{'lam':>8}{'cos':>8}{'gain':>8}"
      f"{'neg':>8}{'n':>4}")
res = []
for nm, md in BANDS:
    LA=[];CO=[];GA=[];NG=[];BA=[];RR=[]
    for ti in range(4):
        u = ks.phi(ks.ic(7000+ti), 40000)
        V = u.copy(); MO = [V.copy()]
        for _ in range(6):
            V = seg(ks, V, basis(ks, V, md), ns); MO.append(V.copy())
        t = u.copy(); TR = [u.copy()]
        for _ in range(6): t = ks.phi(t, ns); TR.append(t.copy())
        # W recurrence, truth-free
        E = np.zeros_like(MO[0]); EH = [E.copy()]
        for k in range(4):
            Phi = ks.phi(MO[k], ns); eta = Phi - MO[k+1]
            if np.linalg.norm(E) < 1e-30: E = eta.copy()
            else:
                eps = min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E), 1e-6), 5e-2)
                Pp = ks.phi(MO[k]+eps*E, ns); Mm = ks.phi(MO[k]-eps*E, ns)
                E = eta + (Pp-Mm)/(2*eps) + 0.5*(Pp+Mm-2*Phi)/(eps*eps)
            EH.append(E.copy())
        def fc(x, n=2):
            a = x.copy()
            for _ in range(n): a = seg(ks, a, basis(ks, a, md), ns)
            return a
        for k in (3, 4):
            Vk, T, TF = MO[k], TR[k], TR[k+2]
            EQ = splitM(EH[k], md)[1]; dP, AQ = splitM(T-Vk, md)
            if not np.isfinite(np.linalg.norm(EQ)) or \
               np.linalg.norm(EQ) > 0.5*np.linalg.norm(Vk): continue
            FB, FE, FN, FO = fc(Vk), fc(Vk+EQ), fc(Vk-EQ), fc(Vk+AQ)
            b = splitM(FB-TF, md)[0]; D = splitM(FO-FB, md)[0]
            nb, nd = np.linalg.norm(b), np.linalg.norm(D)
            if nb < 1e-30 or nd < 1e-30: continue
            LA.append(nd/nb); CO.append(float(np.real(np.vdot(b, D))/(nb*nd)))
            be = terrM(FB, TF, md)
            GA.append(be/terrM(FE, TF, md)); NG.append(be/terrM(FN, TF, md))
            BA.append(100*terrM(Vk, T, md))
            RR.append(np.linalg.norm(AQ)/max(np.linalg.norm(dP), 1e-30))
    if not LA: print(f"  {nm:<22}   all skipped"); continue
    res.append((nm, np.median(LA), np.median(CO), np.median(GA)))
    print(f"  {nm:<22}{np.median(BA):>7.2f}%{np.median(RR):>8.2f}"
          f"{np.median(LA):>8.4f}{np.median(CO):>+8.4f}{np.median(GA):>8.4f}"
          f"{np.median(NG):>8.4f}{len(LA):>4}", flush=True)

print("\n  PREDICTION was: unstable P -> small lambda, small gain;")
print("                  stable  P -> large lambda, larger gain")
if len(res) >= 3:
    lam = [r[1] for r in res]; g = [r[3] for r in res]
    print(f"  lambda across bands: " + "  ".join(f"{x:.4f}" for x in lam))
    print(f"  gain   across bands: " + "  ".join(f"{x:.4f}" for x in g))
    print(f"  lambda rises unstable->stable: {lam[-1] > lam[0]}"
          f"  ({lam[0]:.4f} -> {lam[-1]:.4f})")
