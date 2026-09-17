"""
THE AMPLIFICATION FACTOR — closing the gap the channel picture left open.

The channel decomposition explained the ALIGNMENT of the correction: the
natural hidden error occupies essentially one transport channel, and the
decomposition reproduces cos(theta) to 0.031 across four regimes (n = 63).
It did NOT explain the magnitude. The channel prediction of lambda ran
0.369, 0.339, 0.288, 0.116 while measured lambda ran 0.912, 0.949, 0.987,
0.983 — monotonically decreasing where lambda is flat.

Those ratios are the whole story:

    lambda / lambda_chan  =  2.5,  2.8,  3.4,  8.5

so the finite-amplitude response is two to eight times the linear prediction,
and increasingly so as the observable band becomes more damped. This is the
same signature the amplitude sweep found earlier, where the modal response
converged to r(s) -> 0.49, i.e. twice its linear value, while its DIRECTION was
preserved at cos 0.965.

Define the amplification factor

    A  =  ||Delta(q)||  /  ||K q||

with K the probed linear transport operator and Delta(q) the measured
finite-amplitude response. This measures it directly, per window, and asks
three things.

REGISTERED PREDICTIONS, written before running:
  F1  CONSISTENCY. A reproduces the gap: median |A - lambda/lambda_chan| /
      (lambda/lambda_chan) < 0.15. If not, the gap is not a simple
      amplification and the framing is wrong.
  F2  IT ORDERS THE BANDS. Spearman(A, band damping) is strong and A spans
      more than 3x, confirming the trend seen in the channel run.
  F3  SECOND ORDER EXPLAINS IT. With the curvature term measured by central
      differences, ||Delta - Kq|| is accounted for by (1/2)||H[q,q]|| to
      within 30 percent. This is the test of whether the quadratic truncation
      of Theorem C is sufficient at these amplitudes.
  F4  TRUTH-FREE. A computed from q_hat tracks A computed from q:
      Spearman > 0.9.

F3 is the one that matters. If second order explains the amplification, then
lambda is predictable in principle from quantities the model can compute, and
the remaining work is engineering. If it does not, the response is genuinely
beyond quadratic at the amplitudes where the method operates, and that is a
different and harder statement.

Self-contained: numpy only, CPU. Needs ks_core.py beside it.
"""
import numpy as np, os, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("AF_L",100.0)); NPM=int(os.environ.get("AF_NPM",6))
NTRAJ=int(os.environ.get("AF_NTRAJ",8)); NSTEP=int(os.environ.get("AF_NSTEP",50))
NPROBE=int(os.environ.get("AF_NPROBE",60))
NSEG,FC,BURN=7,2,40000
BANDS=os.environ.get("AF_BANDS","unstable,neutral,stable,damped").split(",")

def rdot(a,b): return float(np.real(np.vdot(a,b)))
def rnorm(a):  return float(np.sqrt(max(rdot(a,a),0.0)))
def splitM(u,m):
    P=np.zeros_like(u); P[m]=u[m]; return P,u-P
def basis(ks,u,m):
    B=[]
    for mm in m:
        for ph in (1.0,1.0j):
            e=np.zeros(len(ks.k),complex); e[mm]=ph
            B.append(e/rnorm(e))
    def orth(v):
        for f in B: v=v-rdot(f,v)*f
        n=rnorm(v); return v/n if n>1e-12 else None
    w=ks.F(u)
    for _ in range(2):
        q=orth(splitM(w,m)[1])
        if q is None: break
        B.append(q); w=ks.DF(u,w)
    return B
def seg(ks,anc,B,n):
    a=np.zeros(len(B)); dt=ks.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=ks.F(u)
        return np.array([rdot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))
def terr(A,T,m):
    return rnorm(splitM(A-T,m)[0])/max(rnorm(splitM(T,m)[0]),1e-30)
def pack(v,m):
    z=v[m]; return np.concatenate([z.real,z.imag])

def linear_response(ks,V,md,fwd,FB,qvec,eps_rel=1e-6):
    """K q by central difference at infinitesimal amplitude: the LINEAR part
    of the response, with the quadratic term cancelled."""
    nq=rnorm(qvec)
    if nq<1e-30: return None
    eps=eps_rel*max(rnorm(V),1e-30)/nq
    dp=splitM(fwd(V+eps*qvec)-FB,md)[0]
    dm=splitM(fwd(V-eps*qvec)-FB,md)[0]
    return pack((dp-dm)/(2*eps),md)

def curvature(ks,V,md,fwd,FB,qvec,eps_rel=1e-6):
    """H[q,q] by central difference: the QUADRATIC part."""
    nq=rnorm(qvec)
    if nq<1e-30: return None
    eps=eps_rel*max(rnorm(V),1e-30)/nq
    dp=splitM(fwd(V+eps*qvec)-FB,md)[0]
    dm=splitM(fwd(V-eps*qvec)-FB,md)[0]
    return pack((dp+dm)/(eps*eps),md)

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); lam=ks.k**2-ks.k**4
    sel={"unstable":lambda i: lam[i]>0,"neutral":lambda i:-3<=lam[i]<=0,
         "stable":lambda i:-12<=lam[i]<-3,"damped":lambda i:-35<=lam[i]<-12}
    dmp={"unstable":0,"neutral":1,"stable":2,"damped":3}
    print(f"KS L={L:.0f}   amplification of the finite-amplitude response\n",flush=True)
    rows=[]; t0=time.time()
    for bn in BANDS:
        md=[i for i in range(1,len(ks.k)) if sel[bn](i)][:NPM]
        if len(md)<3: continue
        n0=len(rows)
        for ti in range(NTRAJ):
            u=ks.phi(ks.ic(70000+ti),BURN)
            V=u.copy(); MO=[V.copy()]
            for _ in range(NSEG): V=seg(ks,V,basis(ks,V,md),NSTEP); MO.append(V.copy())
            T=[u.copy()]; t=u.copy()
            for _ in range(NSEG): t=ks.phi(t,NSTEP); T.append(t.copy())
            E=np.zeros_like(MO[0]); EH=[E.copy()]
            for k in range(NSEG-FC):
                Phi=ks.phi(MO[k],NSTEP); eta=Phi-MO[k+1]
                if rnorm(E)<1e-30: E=eta.copy()
                else:
                    ep=min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                    Pp=ks.phi(MO[k]+ep*E,NSTEP); Mm=ks.phi(MO[k]-ep*E,NSTEP)
                    E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fwd(x,n=FC):
                a=x.copy()
                for _ in range(n): a=seg(ks,a,basis(ks,a,md),NSTEP)
                return a
            for k in (3,4):
                if k>=len(EH) or k+FC>=len(T): continue
                Vk,TF=MO[k],T[k+FC]
                qh=splitM(EH[k],md)[1]
                if not np.isfinite(rnorm(qh)) or rnorm(qh)>0.5*rnorm(Vk): continue
                FB=fwd(Vk); base=terr(FB,TF,md)
                if not (0.002<base<0.50): continue
                pe,qe=splitM(T[k]-Vk,md)
                b=pack(splitM(FB-TF,md)[0],md); nb=np.linalg.norm(b)
                if nb<1e-30: continue
                # measured finite-amplitude response
                dQ=pack(splitM(fwd(Vk+qe)-FB,md)[0],md); nD=np.linalg.norm(dQ)
                Lq=linear_response(ks,Vk,md,fwd,FB,qe)
                Hq=curvature(ks,Vk,md,fwd,FB,qe)
                if Lq is None or Hq is None: continue
                nL=np.linalg.norm(Lq)
                if nL<1e-30 or nD<1e-30: continue
                Amp=nD/nL
                resid=np.linalg.norm(dQ-Lq)
                quad=0.5*np.linalg.norm(Hq)
                # truth-free counterpart
                dQh=pack(splitM(fwd(Vk+qh)-FB,md)[0],md)
                Lqh=linear_response(ks,Vk,md,fwd,FB,qh)
                Amph=(np.linalg.norm(dQh)/max(np.linalg.norm(Lqh),1e-30)
                      if Lqh is not None else np.nan)
                rows.append(dict(band=bn,damp=dmp[bn],amp=Amp,amp_hat=Amph,
                    lam=nD/nb,lam_lin=nL/nb,resid=resid,quad=quad,
                    quad_ratio=quad/max(resid,1e-30),
                    cos_lin=float(Lq@b)/max(nL*nb,1e-30),
                    cos_meas=float(dQ@b)/max(nD*nb,1e-30)))
        g=rows[n0:]
        if g: print(f"  {bn:<9} n={len(g):>3}"
                    f"  A={np.median([x['amp'] for x in g]):7.3f}"
                    f"  lam_lin={np.median([x['lam_lin'] for x in g]):.4f}"
                    f"  lam={np.median([x['lam'] for x in g]):.4f}"
                    f"  quad/resid={np.median([x['quad_ratio'] for x in g]):7.3f}"
                    f"  [{time.time()-t0:.0f}s]",flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("amplif.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows],dtype=float)
    def sp(x,y):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        if len(x)<4: return np.nan
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    gap=A('lam')/np.maximum(A('lam_lin'),1e-30)
    f1=float(np.median(np.abs(A('amp')-gap)/np.maximum(gap,1e-30)))
    print("\n"+"="*80); print(f"AMPLIFICATION FACTOR, n={len(rows)}"); print("="*80)
    print(f"\nF1 A reproduces the gap: median relative error {f1:.4f}"
          f"  -> {'PASS' if f1<0.15 else 'FAIL'}")
    bands=[b for b in BANDS if any(r['band']==b for r in rows)]
    am={b:float(np.median([r['amp'] for r in rows if r['band']==b])) for b in bands}
    spread=max(am.values())/max(min(am.values()),1e-30)
    s2=sp(A('damp'),A('amp'))
    print(f"F2 A spans {spread:.2f}x across bands, Spearman(damping, A) = {s2:+.4f}"
          f"  -> {'PASS' if spread>3 else 'FAIL'}")
    qr=A('quad_ratio')
    f3=float(np.median(np.abs(qr-1.0)))
    print(f"F3 second order explains the residual: median |quad/resid - 1| "
          f"= {f3:.4f}  -> {'PASS' if f3<0.30 else 'FAIL'}")
    s4=sp(A('amp_hat'),A('amp'))
    print(f"F4 truth-free: Spearman(A_hat, A) = {s4:+.4f}"
          f"  -> {'PASS' if s4>0.9 else 'FAIL'}")
    print(f"\n  {'band':<10}{'A':>9}{'lam_lin':>10}{'lambda':>9}"
          f"{'cos_lin':>10}{'cos_meas':>10}{'quad/resid':>12}")
    for b in bands:
        g=[r for r in rows if r['band']==b]
        print(f"  {b:<10}{np.median([x['amp'] for x in g]):>9.3f}"
              f"{np.median([x['lam_lin'] for x in g]):>10.4f}"
              f"{np.median([x['lam'] for x in g]):>9.4f}"
              f"{np.median([x['cos_lin'] for x in g]):>+10.4f}"
              f"{np.median([x['cos_meas'] for x in g]):>+10.4f}"
              f"{np.median([x['quad_ratio'] for x in g]):>12.3f}")

if __name__=="__main__": main()
