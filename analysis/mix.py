"""
HOW MANY POINTWISE FUNCTIONALS DOES IT TAKE TO BREAK THE LAW?

Established: on ONE spectral solver with matched observable dimension,

    MODES    reconstruction error 0.0028
    SENSORS                       0.5552
    MIXED (half and half)         0.4884     <- tracks SENSORS, not the midpoint

so the failure is not graded. But 50/50 is a weak test of that. This sweeps the
MIXING RATIO at fixed total dimension:

    n_sensors = 0, 1, 2, 3, 6, 12, 18, 24   out of 24 total directions

SHARP PREDICTION, registered before running: if ONE pointwise functional among
23 modal ones already gives ~0.5, the statement becomes

    the law holds only when the observable set contains NO pointwise functional

which is a boundary condition with a definite shape, and neither refuted
mechanism (alignment/dimension, advective locality) predicts it — both require
graded behaviour.

If instead the error rises smoothly with n_sensors, the failure IS graded after
all, the 50/50 result was a coincidence of that particular ratio, and the
contamination framing in the record must be softened.

Single file, numpy only, CPU. Paste and run.
"""
import numpy as np, os, json, time

class KS:
    def __init__(self, L, N, dt=0.005):
        self.L, self.N, self.dt = L, N, dt
        self.k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
        self.lin = self.k**2 - self.k**4
        self.deal = np.abs(self.k) < (2/3)*self.k.max()
    def dealias(self, uh): return uh*self.deal
    def F(self, uh):
        u = np.fft.irfft(uh, n=self.N)
        return self.dealias(-0.5j*self.k*np.fft.rfft(u*u) + self.lin*uh)
    def DF(self, uh, vh):
        u = np.fft.irfft(uh, n=self.N); v = np.fft.irfft(vh, n=self.N)
        return self.dealias(-1.0j*self.k*np.fft.rfft(u*v) + self.lin*vh)
    def step(self, uh):
        dt=self.dt
        k1=self.F(uh); k2=self.F(uh+.5*dt*k1); k3=self.F(uh+.5*dt*k2); k4=self.F(uh+dt*k3)
        return self.dealias(uh + dt/6*(k1+2*k2+2*k3+k4))
    def phi(self, uh, n):
        for _ in range(n): uh = self.step(uh)
        return uh
    def ic(self, seed):
        rng=np.random.default_rng(seed); x=np.arange(self.N)*self.L/self.N
        u=0.1*rng.normal(size=self.N)
        for m in range(1,6):
            u += (0.5/m)*np.cos(2*np.pi*m*x/self.L + rng.uniform(0,2*np.pi))
        return self.dealias(np.fft.rfft(u))

L     = float(os.environ.get("MX_L", 100.0))
NDIM  = int(os.environ.get("MX_NDIM", 24))
NTRAJ = int(os.environ.get("MX_NTRAJ", 10))
NSTEP = int(os.environ.get("MX_NSTEP", 50))
NSEG, FC, BURN = 7, 2, 40000
NSENS = [int(x) for x in os.environ.get(
    "MX_NSENS", "0,1,2,3,6,12,18,24").split(",")]

def rdot(a,b): return float(np.real(np.vdot(a,b)))
def rnorm(a):  return float(np.sqrt(max(rdot(a,a),0.0)))

def mixed_basis(ks, ndim, nsen):
    """ndim total REAL directions: nsen point-evaluation functionals, the rest
    Fourier modes. Orthonormalised together, sensors FIRST so the modal set is
    what gets truncated as nsen grows."""
    B=[]
    if nsen > 0:
        pos = np.linspace(0, ks.N, nsen, endpoint=False).astype(int)
        for xj in pos*(ks.L/ks.N):
            v = ks.dealias(np.exp(-1j*ks.k*xj)/ks.N)
            for f in B: v = v - rdot(f,v)*f
            n = rnorm(v)
            if n > 1e-10: B.append(v/n)
            if len(B) >= ndim: return B
    lam = ks.k**2 - ks.k**4
    idx = [i for i in range(1,len(ks.k)) if lam[i] > 0]
    for m in idx:
        for ph in (1.0, 1.0j):
            e = np.zeros(len(ks.k), complex); e[m] = ph
            for f in B: e = e - rdot(f,e)*f
            n = rnorm(e)
            if n > 1e-10: B.append(e/n)
            if len(B) >= ndim: return B
    return B

def splitB(u,B):
    P = np.zeros_like(u)
    for f in B: P = P + rdot(f,u)*f
    return P, u-P
def rom_basis(ks,u,B):
    A=[f.copy() for f in B]
    def orth(v):
        for f in A: v = v - rdot(f,v)*f
        n = rnorm(v); return v/n if n>1e-10 else None
    w = ks.F(u)
    for _ in range(2):
        q = orth(splitB(w,B)[1])
        if q is None: break
        A.append(q); w = ks.DF(u,w)
    return A
def rom_seg(ks,anc,A,n):
    a=np.zeros(len(A)); dt=ks.dt
    def rhs(a):
        u=anc+sum(a[i]*A[i] for i in range(len(A))); F=ks.F(u)
        return np.array([rdot(A[i],F) for i in range(len(A))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*A[i] for i in range(len(A)))
def perr(A_,T_,B):
    pa=splitB(A_-T_,B)[0]; pt=splitB(T_,B)[0]
    return rnorm(pa)/max(rnorm(pt),1e-30)

def main():
    ks=KS(L=L,N=int(2*L),dt=0.005)
    print(f"spectral KS L={L:.0f} N={int(2*L)}  total observable dim = {NDIM}")
    print(f"sweeping n_sensors over {NSENS}\n", flush=True)
    rows=[]; t0=time.time()
    for nsen in NSENS:
        B=mixed_basis(ks,NDIM,nsen)
        if len(B)<NDIM:
            print(f"  n_sensors={nsen}: only {len(B)} directions, skipped"); continue
        n0=len(rows)
        for ti in range(NTRAJ):
            u=ks.phi(ks.ic(31000+ti),BURN)
            V=u.copy(); MO=[V.copy()]
            for _ in range(NSEG): V=rom_seg(ks,V,rom_basis(ks,V,B),NSTEP); MO.append(V.copy())
            T=[u.copy()]; t=u.copy()
            for _ in range(NSEG): t=ks.phi(t,NSTEP); T.append(t.copy())
            def fwd(x,n=FC):
                a=x.copy()
                for _ in range(n): a=rom_seg(ks,a,rom_basis(ks,a,B),NSTEP)
                return a
            for k in (3,4):
                if k+FC>=len(T): continue
                Vk,TF=MO[k],T[k+FC]
                pe,qe=splitB(T[k]-Vk,B)
                FB=fwd(Vk)
                base=perr(FB,TF,B)
                if not (0.002<base<0.50): continue
                b=splitB(FB-TF,B)[0]; nb=rnorm(b)
                dQ=splitB(fwd(Vk+qe)-FB,B)[0]; dP=splitB(fwd(Vk+pe)-FB,B)[0]
                nQ,nP=rnorm(dQ),rnorm(dP)
                if nb<1e-30 or nQ<1e-30 or nP<1e-30: continue
                kap=nQ/nP; c=rdot(dQ,dP)/(nQ*nP); lam=nQ/nb
                pred=kap/np.sqrt(max(1+2*kap*c+kap*kap,1e-300))
                rows.append(dict(nsen=nsen,frac=nsen/NDIM,kappa=kap,c=c,lam=lam,
                    lam_pred=pred,err=abs(pred-lam),base=base))
        g=[r for r in rows[n0:]]
        if g:
            print(f"  n_sensors={nsen:>3} ({100*nsen/NDIM:>5.1f}%)  n={len(g):>3}  "
                  f"err={np.median([x['err'] for x in g]):.4f}  "
                  f"lam={np.median([x['lam'] for x in g]):.4f}  "
                  f"base={100*np.median([x['base'] for x in g]):.2f}%  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("mix.json","w"),default=float)
    print("\n"+"="*72); print(f"CONTAMINATION SWEEP, n={len(rows)}"); print("="*72)
    print(f"\n  {'n_sensors':>10}{'fraction':>10}{'n':>6}{'median err':>13}")
    med={}
    for ns in NSENS:
        g=[r["err"] for r in rows if r["nsen"]==ns]
        if not g: continue
        med[ns]=float(np.median(g))
        print(f"  {ns:>10}{ns/NDIM:>10.3f}{len(g):>6}{med[ns]:>13.4f}")
    if 0 in med and len(med)>=3:
        base=med[0]; full=med[max(med)]
        one=med.get(1)
        print(f"\n  pure modal (0 sensors):        {base:.4f}")
        if one is not None:
            print(f"  ONE sensor among {NDIM-1} modes:      {one:.4f}")
            frac=(one-base)/max(full-base,1e-12)
            print(f"  -> one sensor recovers {100*frac:.0f}% of the full-sensor error")
            if frac > 0.6:
                print(f"\n  CONTAMINATION CONFIRMED. A single pointwise functional")
                print(f"  breaks the law. The boundary is: the observable set must")
                print(f"  contain NO pointwise functional.")
            elif frac < 0.25:
                print(f"\n  GRADED, NOT BINARY. One sensor does little; the failure")
                print(f"  accumulates with sensor count. The contamination framing")
                print(f"  in the record must be softened, and a graded mechanism")
                print(f"  (alignment, locality) is back in play.")
            else:
                print(f"\n  INTERMEDIATE ({100*frac:.0f}%). Report the curve, not a")
                print(f"  binary claim.")
        print(f"  all sensors ({max(med)}):            {full:.4f}")

if __name__=="__main__": main()
