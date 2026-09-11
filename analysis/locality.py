"""
WHY DOES LINEAR RESPONSE FAIL FOR POINT-SENSOR OBSERVABLES?

At n=224 the derived form  lam = kappa/sqrt(1 + 2 kappa c + kappa^2)  splits
cleanly by observation operator:

    Fourier observables (KS + CGL)   reconstruction error 0.0354   PASSES
    point sensors       (FD)         reconstruction error 0.6027   FAILS 17x

So lambda IS predictable for modal observables. The exception is pointwise
observation, and it needs a reason.

HYPOTHESIS — ADVECTIVE LOCALITY.
Fourier modes are delocalised: a hidden perturbation couples to them smoothly
and the linearisation holds. A point sensor reads one location, so a hidden
perturbation reaches it by TRANSPORT. Over a finite horizon that is
displacement, not smooth coupling, and displacement is badly represented by a
linear response. The relevant dimensionless group is

    ell = (distance information travels over the horizon) / (sensor spacing)

  ell << 1   information moves less than the sensor gap; each sensor sees a
             locally smooth field; linear response should HOLD
  ell >> 1   information crosses several sensors; the response is dominated by
             advective rearrangement; linear response should BREAK

PREDICTION, registered before running: the reconstruction error collapses onto
ell — a single curve across independently varied sensor spacing AND horizon.
If instead the error depends on spacing and horizon separately, locality is the
wrong explanation and the two-class split should be reported as measured.

Transport distance is measured, not assumed: d = u_rms * T_horizon, with u_rms
taken from the flow itself.

Self-contained: numpy, CPU, finite-difference KS. No dataset.
"""
import numpy as np, sys, os, json, time

NX, LDOM, DT = 200, 100.0, 0.002
NSEG, BURN = 7, 60000
NTRAJ = int(os.environ.get("LOC_NTRAJ", 8))
# independently varied: sensor count (spacing) and forecast horizon
SENSORS  = [int(x) for x in os.environ.get("LOC_SENSORS", "6,10,16,25").split(",")]
HORIZONS = [int(x) for x in os.environ.get("LOC_HORIZONS", "1,2,4").split(",")]
NSTEP = int(os.environ.get("LOC_NSTEP", 125))

class KSFD:
    def __init__(s): s.n, s.L, s.h, s.dt = NX, LDOM, LDOM/NX, DT
    def F(s, u):
        h=s.h; um1=np.roll(u,1); up1=np.roll(u,-1)
        um2=np.roll(u,2); up2=np.roll(u,-2)
        return -u*((up1-um1)/(2*h)) - (up1-2*u+um1)/h**2 \
               - (up2-4*up1+6*u-4*um1+um2)/h**4
    def DF(s, u, v):
        h=s.h; vm1=np.roll(v,1); vp1=np.roll(v,-1)
        vm2=np.roll(v,2); vp2=np.roll(v,-2)
        um1=np.roll(u,1); up1=np.roll(u,-1)
        return -(v*((up1-um1)/(2*h)) + u*((vp1-vm1)/(2*h))) \
               - (vp1-2*v+vm1)/h**2 - (vp2-4*vp1+6*v-4*vm1+vm2)/h**4
    def step(s, u):
        dt=s.dt; k1=s.F(u); k2=s.F(u+.5*dt*k1); k3=s.F(u+.5*dt*k2); k4=s.F(u+dt*k3)
        return u + dt/6*(k1+2*k2+2*k3+k4)
    def phi(s, u, n):
        for _ in range(n): u = s.step(u)
        return u
    def ic(s, seed):
        rng=np.random.default_rng(seed); x=np.arange(s.n)*s.h
        u=0.1*rng.normal(size=s.n)
        for m in range(1,5):
            u+=(0.5/m)*np.cos(2*np.pi*m*x/s.L+rng.uniform(0,2*np.pi))
        return u

def splitS(u,m):
    P=np.zeros_like(u); P[m]=u[m]; return P,u-P
def basisS(sy,u,m):
    B=[]
    for s_ in m:
        e=np.zeros_like(u); e[s_]=1.0; B.append(e/np.linalg.norm(e))
    def orth(v):
        for f in B: v=v-np.dot(f,v)*f
        n=np.linalg.norm(v); return v/n if n>1e-12 else None
    w=sy.F(u)
    for _ in range(2):
        q=orth(splitS(w,m)[1])
        if q is None: break
        B.append(q); w=sy.DF(u,w)
    return B
def segS(sy,anc,B,n):
    a=np.zeros(len(B)); dt=sy.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=sy.F(u)
        return np.array([np.dot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))

def main():
    sy=KSFD(); rows=[]; t0=time.time()
    print(f"FD KS  nx={NX} L={LDOM:.0f}  segment={NSTEP*DT:.2f} time units")
    print(f"sensors {SENSORS}   horizons(segments) {HORIZONS}\n", flush=True)
    for nsen in SENSORS:
        sen=np.linspace(0,NX,nsen,endpoint=False).astype(int)
        spacing=(LDOM/nsen)
        for FC in HORIZONS:
            n0=len(rows)
            for ti in range(NTRAJ):
                u=sy.phi(sy.ic(15000+ti),BURN)
                if not np.all(np.isfinite(u)): continue
                urms=float(np.sqrt(np.mean(u**2)))
                Th=FC*NSTEP*DT
                ell=urms*Th/spacing                 # the dimensionless group
                V=u.copy(); MO=[V.copy()]
                for _ in range(NSEG):
                    V=segS(sy,V,basisS(sy,V,sen),NSTEP); MO.append(V.copy())
                T=[u.copy()]; t=u.copy()
                for _ in range(NSEG): t=sy.phi(t,NSTEP); T.append(t.copy())
                E=np.zeros_like(MO[0]); EH=[E.copy()]
                for k in range(NSEG-FC):
                    Phi=sy.phi(MO[k],NSTEP); eta=Phi-MO[k+1]
                    if np.linalg.norm(E)<1e-30: E=eta.copy()
                    else:
                        ep=min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),
                                   1e-6),5e-2)
                        Pp=sy.phi(MO[k]+ep*E,NSTEP); Mm=sy.phi(MO[k]-ep*E,NSTEP)
                        E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                    EH.append(E.copy())
                def fwd(x,n=FC):
                    a=x.copy()
                    for _ in range(n): a=segS(sy,a,basisS(sy,a,sen),NSTEP)
                    return a
                for k in (3,4):
                    if k>=len(EH) or k+FC>=len(T): continue
                    Vk,TF=MO[k],T[k+FC]
                    pe,qe=splitS(T[k]-Vk,sen)
                    FB=fwd(Vk)
                    b=splitS(FB-TF,sen)[0]; nb=np.linalg.norm(b)
                    base=nb/max(np.linalg.norm(splitS(TF,sen)[0]),1e-30)
                    if not (0.002<base<0.50): continue
                    dQ=splitS(fwd(Vk+qe)-FB,sen)[0]
                    dP=splitS(fwd(Vk+pe)-FB,sen)[0]
                    nQ,nP=np.linalg.norm(dQ),np.linalg.norm(dP)
                    if nQ<1e-30 or nP<1e-30: continue
                    kap=nQ/nP; c=float(np.dot(dQ,dP))/(nQ*nP)
                    lam=nQ/nb
                    lam_pred=kap/np.sqrt(max(1+2*kap*c+kap*kap,1e-300))
                    rows.append(dict(nsen=nsen,spacing=spacing,FC=FC,Th=Th,
                        urms=urms,ell=ell,kappa=kap,c=c,lam=lam,
                        lam_pred=lam_pred,err=abs(lam_pred-lam),base=base))
            g=[r for r in rows[n0:]]
            if g:
                print(f"  {nsen:>3} sensors (dx={spacing:5.2f})  horizon {FC} seg "
                      f"(T={FC*NSTEP*DT:.2f})  ell={np.median([x['ell'] for x in g]):6.2f}"
                      f"  n={len(g):>3}  err={np.median([x['err'] for x in g]):.4f}"
                      f"  [{time.time()-t0:.0f}s]", flush=True)
    if not rows:
        print("\nNo valid windows."); return
    json.dump(rows,open("locality.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows])
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*84); print(f"ADVECTIVE LOCALITY TEST, n={len(rows)}"); print("="*84)
    print(f"\nP1 does the error collapse onto ell?")
    print(f"   Spearman(ell, err)      = {sp(A('ell'),A('err')):+.4f}")
    print(f"   Spearman(spacing, err)  = {sp(A('spacing'),A('err')):+.4f}")
    print(f"   Spearman(horizon, err)  = {sp(A('Th'),A('err')):+.4f}")
    print(f"   -> collapse requires ell to beat BOTH separate variables")
    print(f"\nP2 binned by ell")
    print(f"   {'ell bin':<20}{'n':>5}{'median err':>13}{'median lam':>13}")
    o=np.argsort(A('ell')); e=A('err'); l=A('lam'); el=A('ell')
    for bnd in np.array_split(o,5):
        print(f"   {f'{el[bnd].min():.2f} - {el[bnd].max():.2f}':<20}{len(bnd):>5}"
              f"{np.median(e[bnd]):>13.4f}{np.median(l[bnd]):>13.4f}")
    print(f"\nP3 is there a regime where linear response HOLDS?")
    lo=e[el<np.percentile(el,25)]; hi=e[el>np.percentile(el,75)]
    print(f"   lowest-ell quartile  median err {np.median(lo):.4f}"
          f"  (Fourier reference 0.0354)")
    print(f"   highest-ell quartile median err {np.median(hi):.4f}"
          f"  (FD pooled reference 0.6027)")
    _ok = np.median(lo) < 0.10 and np.median(hi) > 3*np.median(lo)
    _v = ("LOCALITY CONFIRMED: small ell recovers the modal regime" if _ok
          else "NOT CONFIRMED: ell does not separate the regimes")
    print(f"   -> {_v}")

if __name__=="__main__": main()
