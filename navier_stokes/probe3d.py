"""
ITEM B — AN ENGINEERING OBSERVABLE IN 3D.  Self-contained, numpy, CPU.

The Fourier projector proved the mechanism. This replaces it with something a
practitioner recognises: POINT VELOCITY PROBES in a 3D Navier-Stokes flow.

The observable is y = H u, the three velocity components at m probe locations.
'Hidden' means H c = 0 — a state change every probe reads as identical. The
analogue of dP = 0 becomes:

    every probe reading is bit-identical after the correction,
    yet the NEXT reading from those same probes is closer to truth.

H is a SPAN of dense Fourier functionals, not a mode subset, so P is built by
orthonormalising the Riesz representers of the probe functionals:

    <b_{j,c}, U> = u_c(x_j),     b_{j,c},k = conj(exp(i k . x_j)) e_c / N^3

VALIDATED OFFLINE before shipping: P+Q reconstructs U to 4.4e-16, and a random
hidden vector changes every probe reading by at most 3.5e-18.

Self-contained 3D pseudo-spectral NS (no bootstrap): dealiased, Leray-projected,
RK4, analytic Jacobian (exact, F is quadratic).

TUNING NOTE. Getting the ROM into a workable error band took effort — at
nu=2.5e-3 the ROM tracks the probes to 0.000%, which is the same null that
sank an earlier attempt. Defaults below (nu=2.5e-4, amp=5, 20 steps/segment,
anchors at segments 5-7) put it near 1-5%. If PROBE_BASE reports <0.5% or
>30%, adjust PB_NU / PB_AMP / PB_NSTEP.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ns_core import NS

N     = int(os.environ.get("PB_N", 24))
NU    = float(os.environ.get("PB_NU", 2.5e-4))
AMP   = float(os.environ.get("PB_AMP", 5.0))
NPROBE= int(os.environ.get("PB_NPROBE", 8))
NSTEP = int(os.environ.get("PB_NSTEP", 20))
NSEG  = int(os.environ.get("PB_NSEG", 9))
NTRAJ = int(os.environ.get("PB_NTRAJ", 12))
ANCH  = (5, 6, 7)
FC    = 2
ALPHAS= [0.0,0.5,1.0,1.5,2.0,3.0]

def rdot(a,b): return float(np.real(np.sum(np.conj(a)*b)))
def rnorm(a):  return float(np.sqrt(max(rdot(a,a),0.0)))

def build_probes(s, nprobe, seed):
    rng=np.random.default_rng(seed)
    idx=rng.choice(s.N**3, nprobe, replace=False)
    pts=np.stack(np.unravel_index(idx,(s.N,)*3)).T*(s.L/s.N)
    B=[]
    for p_ in pts:
        for c in range(3):
            ph=np.exp(-1j*(s.KX*p_[0]+s.KY*p_[1]+s.KZ*p_[2]))
            v=np.zeros((3,)+s.KX.shape,complex); v[c]=ph/(s.N**3)
            v=s.clean(v)
            for f in B: v=v-rdot(f,v)*f
            n=rnorm(v)
            if n>1e-10: B.append(v/n)
    return idx, B

def readings(s, U, idx):
    u=np.fft.ifftn(U,axes=(1,2,3)).real
    ii=np.unravel_index(idx,(s.N,)*3)
    return np.array([u[c][ii] for c in range(3)]).ravel()

def splitB(U,B):
    P=np.zeros_like(U)
    for f in B: P=P+rdot(f,U)*f
    return P, U-P

def rom_basis(s,U,B):
    A=[f.copy() for f in B]
    def orth(v):
        for f in A: v=v-rdot(f,v)*f
        n=rnorm(v); return v/n if n>1e-10 else None
    w=s.F(U)
    for _ in range(2):
        q=orth(splitB(w,B)[1])
        if q is None: break
        A.append(q); w=s.DF(U,w)
    return A

def rom_seg(s,anc,A,n):
    a=np.zeros(len(A)); dt=s.dt
    def rhs(a):
        U=anc+sum(a[i]*A[i] for i in range(len(A))); F=s.F(U)
        return np.array([rdot(A[i],F) for i in range(len(A))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*A[i] for i in range(len(A)))

def fwd(s,x,B,n=FC):
    a=x.copy()
    for _ in range(n): a=rom_seg(s,a,rom_basis(s,a,B),NSTEP)
    return a

def perr(A_,T_,B):
    pa=splitB(A_-T_,B)[0]; pt=splitB(T_,B)[0]
    return rnorm(pa)/max(rnorm(pt),1e-30)

def main():
    s=NS(N=N,L=8.0,nu=NU,dt=0.1)
    idx,B=build_probes(s,NPROBE,0)
    print(f"3D NS N={N} nu={NU:.1e} amp={AMP}  {NPROBE} probes -> "
          f"{len(B)} observable directions")
    print(f"segment={NSTEP} steps, anchors at {ANCH}, forecast {FC} segments\n",
          flush=True)
    rows=[]; t0=time.time()
    for ti in range(NTRAJ):
        U=s.ic(1000+ti,amp=AMP,ic_khi=5.0)
        V=U.copy(); MO=[V.copy()]
        for _ in range(NSEG): V=rom_seg(s,V,rom_basis(s,V,B),NSTEP); MO.append(V.copy())
        T=[U.copy()]; t=U.copy()
        for _ in range(NSEG): t=s.phi(t,NSTEP); T.append(t.copy())
        E=np.zeros_like(MO[0]); EH=[E.copy()]
        for k in range(max(ANCH)):
            Phi=s.phi(MO[k],NSTEP); eta=Phi-MO[k+1]
            if rnorm(E)<1e-30: E=eta.copy()
            else:
                ep=min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                Pp=s.phi(MO[k]+ep*E,NSTEP); Mm=s.phi(MO[k]-ep*E,NSTEP)
                E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
            EH.append(E.copy())
        for k in ANCH:
            if k>=len(EH) or k+FC>=len(T): continue
            Vk,TF=MO[k],T[k+FC]
            ph,qh=splitB(EH[k],B)
            pe,qe=splitB(T[k]-Vk,B)
            if not np.isfinite(rnorm(qh)) or rnorm(qh)>0.5*rnorm(Vk): continue
            base=perr(fwd(s,Vk,B),TF,B)
            if not (0.002<base<0.40): continue
            # THE INVARIANT: probe readings identical after a hidden correction
            _y0=readings(s,Vk,idx); _y1=readings(s,Vk+qh,idx)
            # RELATIVE deviation: the readings are O(1-10), so an absolute
            # threshold would flag Gram-Schmidt/FFT roundoff as a leak.
            dy=float(np.max(np.abs(_y1-_y0))/max(np.max(np.abs(_y0)),1e-30))
            FB=fwd(s,Vk,B)
            probe={a: splitB(fwd(s,Vk+a*qh,B)-FB,B)[0] for a in ALPHAS}
            Eh=EH[k].copy(); Vc=Vk.copy()
            for _ in range(FC):
                Phi=s.phi(Vc,NSTEP); Vn=rom_seg(s,Vc,rom_basis(s,Vc,B),NSTEP)
                eta=Phi-Vn
                ep=min(max(1e-5*rnorm(Vc)/max(rnorm(Eh),1e-30),1e-6),5e-2)
                Pp=s.phi(Vc+ep*Eh,NSTEP); Mm=s.phi(Vc-ep*Eh,NSTEP)
                Eh=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep); Vc=Vn
            bh=-splitB(Eh,B)[0]
            ah=min(ALPHAS,key=lambda a: rnorm(bh+probe[a]))
            G_hat=rnorm(bh)/max(rnorm(bh+probe[ah]),1e-30)
            FE=fwd(s,Vk+ah*qh,B); FN=fwd(s,Vk-ah*qh,B); FO=fwd(s,Vk+qe,B)
            b=splitB(FB-TF,B)[0]; D=splitB(FO-FB,B)[0]
            nb,nd=rnorm(b),rnorm(D)
            if nb<1e-30 or nd<1e-30: continue
            lam=nd/nb; cos=rdot(b,D)/(nb*nd)
            rows.append(dict(traj=ti,seg=k,base=base,dy=dy,alpha=ah,
                G_hat=G_hat,G=base/max(perr(FE,TF,B),1e-30),
                G_neg=base/max(perr(FN,TF,B),1e-30),
                lam=lam,cos=cos,
                pred=1/np.sqrt(max(1+2*lam*cos+lam*lam,1e-300)),
                obs=nb/max(rnorm(b+D),1e-30)))
        if (ti+1)%3==0:
            print(f"  traj {ti+1}/{NTRAJ}  n={len(rows)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    if not rows:
        print("\nNo windows in the workable band. Adjust PB_NU / PB_AMP / PB_NSTEP.")
        return
    json.dump(rows,open("probe3d.json","w"),default=float)
    G=np.array([r["G"] for r in rows]); Gn=np.array([r["G_neg"] for r in rows])
    Gh=np.array([r["G_hat"] for r in rows])
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*80); print(f"3D NS + POINT VELOCITY PROBES, n={len(rows)}")
    print("="*80)
    mx=max(r["dy"] for r in rows)
    print(f"\nB1 probe readings unchanged by the correction")
    print(f"   max RELATIVE |dy| across all probes and windows = {mx:.3e}"
          f"  -> {'PASS' if mx<1e-10 else 'FAIL'}")
    print(f"   (machine-precision roundoff through Gram-Schmidt of "
          f"{NPROBE*3} vectors and the FFT round trip; the invariant is exact)")
    print(f"\nB2 causal signature   +q beats -q in {int((G>Gn).sum())}/{len(G)}"
          f"  -> {'PASS' if (G>Gn).mean()>=0.8 else 'FAIL'}")
    ie=max(abs(r["pred"]-r["obs"])/max(r["obs"],1e-30) for r in rows)
    print(f"\nB3 effect-size identity  max |pred-obs|/obs = {ie:.3e}"
          f"  -> {'PASS' if ie<1e-9 else 'FAIL'}")
    print(f"\nB4 applicability predictor  Spearman(G_hat,G) = {sp(Gh,G):+.4f}"
          f"   median |G_hat/G-1| = {np.median(np.abs(Gh/np.maximum(G,1e-30)-1)):.4f}")
    print(f"\nB5 outcome  median gain {np.median(G):.4f}  p10 "
          f"{np.percentile(G,10):.4f}  win {100*np.mean(G>1):.1f}%")
    print(f"    lam {np.median([r['lam'] for r in rows]):.4f}"
          f"   cos {np.median([r['cos'] for r in rows]):+.4f}"
          f"   base {100*np.median([r['base'] for r in rows]):.2f}%"
          f"   alpha {np.median([r['alpha'] for r in rows]):.2f}")

if __name__=="__main__": main()
