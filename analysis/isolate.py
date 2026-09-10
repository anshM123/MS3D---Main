"""
IS THE EXCEPTION THE OBSERVABLE, OR THE DISCRETISATION?

The law lam = kappa/sqrt(1+2 kappa c + kappa^2) reconstructs to 0.035 for
Fourier observables (KS, CGL) and 0.603 for point sensors (FD). But the FD run
changed TWO things at once: the observable became pointwise AND the solver
became finite-difference. Those are confounded. Two mechanisms have already
been proposed and refuted assuming it was the observable.

This isolates it. ONE spectral KS solver, ONE equation, ONE discretisation.
Only the observable changes:

    MODES      P = span of Fourier modes            (the 0.035 case)
    SENSORS    P = span of point-evaluation functionals on the SAME solver
    MIXED      half modes, half sensors

and the observable DIMENSION is matched across all three, which the earlier
comparison did not do (Fourier P was 12 real dims, FD sensors 8-25).

Point-evaluation functionals in a spectral basis are the Riesz representers
    b_j,k = conj(exp(i k x_j)) / N
orthonormalised. Same construction as the 3D probe test, in 1D.

OUTCOMES
  sensors ~ 0.60 on a spectral solver  -> the OBSERVABLE is the exception.
      Pointwise observation breaks linear response; discretisation is
      irrelevant. The two refuted mechanisms were at least aimed correctly.
  sensors ~ 0.035 on a spectral solver -> the DISCRETISATION is the exception.
      Finite differences break it, not pointwise observation, and the whole
      "modal vs pointwise" framing in the record is WRONG and must be redone.
  in between -> both contribute; report the decomposition.

Either branch is informative. Self-contained: numpy, CPU, needs ks_core.py.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L      = float(os.environ.get("IS_L", 100.0))
NDIM   = int(os.environ.get("IS_NDIM", 12))     # MATCHED observable dimension
NTRAJ  = int(os.environ.get("IS_NTRAJ", 10))
NSTEP  = int(os.environ.get("IS_NSTEP", 50))
NSEG, FC, BURN = 7, 2, 40000

def rdot(a,b): return float(np.real(np.vdot(a,b)))
def rnorm(a):  return float(np.sqrt(max(rdot(a,a),0.0)))

def modal_basis(ks, ndim, band="unstable"):
    """ndim REAL directions from ndim/2 Fourier modes (re + im each)."""
    lam = ks.k**2 - ks.k**4
    idx = ([i for i in range(1,len(ks.k)) if lam[i] > 0] if band=="unstable"
           else [i for i in range(1,len(ks.k)) if -35 <= lam[i] <= -10])
    B = []
    for m in idx:
        for ph in (1.0, 1.0j):
            e = np.zeros_like(ks.ic(0), dtype=complex); e[m] = ph
            n = rnorm(e)
            if n > 1e-12: B.append(e/n)
            if len(B) >= ndim: return B
    return B

def sensor_basis(ks, ndim, seed=0):
    """ndim REAL directions from point-evaluation functionals on the SAME
    spectral solver. Riesz representer of u -> u(x_j)."""
    rng = np.random.default_rng(seed)
    pos = np.linspace(0, ks.N, ndim, endpoint=False).astype(int)
    x = pos*(ks.L/ks.N)
    B = []
    for xj in x:
        v = np.exp(-1j*ks.k*xj)/ks.N
        v = ks.dealias(v)
        for f in B: v = v - rdot(f,v)*f
        n = rnorm(v)
        if n > 1e-10: B.append(v/n)
        if len(B) >= ndim: break
    return B

def splitB(u, B):
    P = np.zeros_like(u)
    for f in B: P = P + rdot(f,u)*f
    return P, u-P

def rom_basis(ks, u, B):
    A = [f.copy() for f in B]
    def orth(v):
        for f in A: v = v - rdot(f,v)*f
        n = rnorm(v); return v/n if n > 1e-10 else None
    w = ks.F(u)
    for _ in range(2):
        q = orth(splitB(w,B)[1])
        if q is None: break
        A.append(q); w = ks.DF(u,w)
    return A

def rom_seg(ks, anc, A, n):
    a = np.zeros(len(A)); dt = ks.dt
    def rhs(a):
        u = anc + sum(a[i]*A[i] for i in range(len(A))); F = ks.F(u)
        return np.array([rdot(A[i],F) for i in range(len(A))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a = a + dt/6*(k1+2*k2+2*k3+k4)
    return anc + sum(a[i]*A[i] for i in range(len(A)))

def perr(A_, T_, B):
    pa = splitB(A_-T_,B)[0]; pt = splitB(T_,B)[0]
    return rnorm(pa)/max(rnorm(pt),1e-30)

def run(ks, B, label, rows):
    ns = NSTEP
    for ti in range(NTRAJ):
        u = ks.phi(ks.ic(31000+ti), BURN)
        V = u.copy(); MO=[V.copy()]
        for _ in range(NSEG): V = rom_seg(ks,V,rom_basis(ks,V,B),ns); MO.append(V.copy())
        T=[u.copy()]; t=u.copy()
        for _ in range(NSEG): t = ks.phi(t,ns); T.append(t.copy())
        E=np.zeros_like(MO[0]); EH=[E.copy()]
        for k in range(NSEG-FC):
            Phi=ks.phi(MO[k],ns); eta=Phi-MO[k+1]
            if rnorm(E)<1e-30: E=eta.copy()
            else:
                ep=min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                Pp=ks.phi(MO[k]+ep*E,ns); Mm=ks.phi(MO[k]-ep*E,ns)
                E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
            EH.append(E.copy())
        def fwd(x,n=FC):
            a=x.copy()
            for _ in range(n): a=rom_seg(ks,a,rom_basis(ks,a,B),ns)
            return a
        for k in (3,4):
            if k>=len(EH) or k+FC>=len(T): continue
            Vk,TF=MO[k],T[k+FC]
            pe,qe=splitB(T[k]-Vk,B)
            base=perr(fwd(Vk),TF,B)
            if not (0.002<base<0.50): continue
            FB=fwd(Vk)
            b=splitB(FB-TF,B)[0]; nb=rnorm(b)
            dQ=splitB(fwd(Vk+qe)-FB,B)[0]; dP=splitB(fwd(Vk+pe)-FB,B)[0]
            nQ,nP=rnorm(dQ),rnorm(dP)
            if nb<1e-30 or nQ<1e-30 or nP<1e-30: continue
            kap=nQ/nP; c=rdot(dQ,dP)/(nQ*nP); lam=nQ/nb
            pred=kap/np.sqrt(max(1+2*kap*c+kap*kap,1e-300))
            rows.append(dict(obs=label,kappa=kap,c=c,lam=lam,lam_pred=pred,
                err=abs(pred-lam),base=base,ndim=len(B)))

def main():
    ks=KS(L=L,N=int(2*L),dt=0.005)
    print(f"spectral KS L={L:.0f} N={int(2*L)}  ONE solver, ONE equation")
    print(f"observable dimension MATCHED at {NDIM} real directions\n",flush=True)
    rows=[]; t0=time.time()
    mb=modal_basis(ks,NDIM); sb=sensor_basis(ks,NDIM)
    mix=(mb[:NDIM//2]+[f for f in sb][:NDIM//2])
    # re-orthonormalise the mixed set
    MB=[]
    for v in mix:
        v=v.copy()
        for f in MB: v=v-rdot(f,v)*f
        n=rnorm(v)
        if n>1e-10: MB.append(v/n)
    for lbl,B in (("MODES",mb),("SENSORS",sb),("MIXED",MB)):
        n0=len(rows); run(ks,B,lbl,rows)
        g=[r for r in rows[n0:]]
        if g: print(f"  {lbl:<9} dim={len(B):>3}  n={len(g):>3}  "
                    f"err={np.median([x['err'] for x in g]):.4f}  "
                    f"lam={np.median([x['lam'] for x in g]):.4f}  "
                    f"base={100*np.median([x['base'] for x in g]):.2f}%  "
                    f"[{time.time()-t0:.0f}s]",flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("isolate.json","w"),default=float)
    print("\n"+"="*78); print(f"OBSERVABLE vs DISCRETISATION, n={len(rows)}"); print("="*78)
    print(f"\n  {'observable':<12}{'n':>5}{'median err':>13}{'p90':>10}")
    md={}
    for lbl in ("MODES","SENSORS","MIXED"):
        g=[r["err"] for r in rows if r["obs"]==lbl]
        if not g: continue
        md[lbl]=float(np.median(g))
        print(f"  {lbl:<12}{len(g):>5}{np.median(g):>13.4f}{np.percentile(g,90):>10.4f}")
    print(f"\n  reference: Fourier on KS/CGL 0.0354   finite-difference sensors 0.6027")
    if "MODES" in md and "SENSORS" in md:
        r=md["SENSORS"]/max(md["MODES"],1e-9)
        print(f"\n  sensors/modes on the SAME spectral solver: {r:.2f}x")
        if md["SENSORS"]>0.3:
            print("  -> THE OBSERVABLE is the exception. Pointwise observation")
            print("     breaks linear response even on a spectral solver.")
        elif md["SENSORS"]<0.10:
            print("  -> THE DISCRETISATION is the exception. Point sensors are")
            print("     FINE on a spectral solver; finite differences are the")
            print("     problem, and the modal-vs-pointwise framing is WRONG.")
        else:
            print("  -> BOTH contribute; report the decomposition.")

if __name__=="__main__": main()
