"""
CORRECTION EFFORT — what the rank test could not see.

A previous experiment probed the transport operator T = P.DPhi.Q, took its
range, and found rho_T = 1.0000 in every window with rank(T) equal to the full
observable dimension. The conclusion drawn was that geometry cannot limit
correctability because every observable direction is reachable.

That conclusion was too strong, and the test was blind to the thing that
matters. RANK asks whether a direction is reachable with UNLIMITED
intervention. It does not ask how much hidden perturbation is required. Two
configurations can both have full rank and rho_T = 1 while one needs a tiny
correction and the other an enormous one.

The quantity that sees this is the minimum-norm correction

    E*(b) = min { ||c|| : K c = -b }  =  ||K^+ b||

with K the probed transport operator and K^+ its pseudo-inverse, together with
the singular spectrum of K itself:

    K = U Sigma W^T

Small singular values mean an observable direction is reachable only at great
cost. The spectrum, and the orientation of b relative to U, is the mechanism
the range test collapsed away.

REGISTERED PREDICTIONS, written before running:
  E1  E* VARIES where rho_T did not. Spread of E*/||b|| across the four KS
      bands exceeds 3x. If effort is also constant, the geometric route really
      is dead and this closes it properly.
  E2  EFFORT TRACKS THE EFFECT SIZE. Spearman(-log E*_rel, lambda) > 0.6.
      Cheap correction should accompany large lambda.
  E3  THE SPECTRUM CARRIES IT. The conditioning sigma_max/sigma_min of K
      differs across bands by more than 10x, so the bands are not merely
      rescaled copies of one another.
  E4  TRUTH-FREE. E* computed from b_hat tracks E* computed from b_true:
      Spearman > 0.9. Without this the quantity is a diagnostic, not a
      predictor.

E1 is the test of whether the earlier negative was premature. If E* is also
flat, the geometric line is closed for good.

Self-contained: numpy only, CPU. Needs ks_core.py beside it.
"""
import numpy as np, os, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("EF_L",100.0)); NPM=int(os.environ.get("EF_NPM",6))
NTRAJ=int(os.environ.get("EF_NTRAJ",8)); NSTEP=int(os.environ.get("EF_NSTEP",50))
NPROBE=int(os.environ.get("EF_NPROBE",40))
NSEG,FC,BURN=7,2,40000
BANDS=os.environ.get("EF_BANDS","unstable,neutral,stable,damped").split(",")

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

def probe_K(ks,V,md,fwd,FB,nprobe,seed):
    """Build K in the coordinates (orthonormal hidden probe) -> (packed
    observable). Returns K, and the orthonormal hidden directions used."""
    rng=np.random.default_rng(seed)
    dirs=[]; cols=[]
    eps=1e-6*max(rnorm(V),1e-30)
    for _ in range(nprobe):
        r=rng.normal(size=len(ks.k))+1j*rng.normal(size=len(ks.k))
        r=ks.dealias(r); q=splitM(r,md)[1]
        for d in dirs: q=q-rdot(d,q)*d
        nq=rnorm(q)
        if nq<1e-10: continue
        q=q/nq; dirs.append(q)
        cols.append(pack(splitM(fwd(V+eps*q)-FB,md)[0],md)/eps)
    if len(cols)<2: return None,None
    return np.array(cols).T, dirs

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); lam=ks.k**2-ks.k**4
    sel={"unstable":lambda i: lam[i]>0,"neutral":lambda i:-3<=lam[i]<=0,
         "stable":lambda i:-12<=lam[i]<-3,"damped":lambda i:-35<=lam[i]<-12}
    print(f"KS L={L:.0f}   K probed with {NPROBE} orthonormal hidden directions\n",
          flush=True)
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
                K,dirs=probe_K(ks,Vk,md,fwd,FB,NPROBE,1000+ti*13+k)
                if K is None: continue
                S=np.linalg.svd(K,compute_uv=False)
                Kp=np.linalg.pinv(K,rcond=1e-10)
                b_true=pack(splitM(FB-TF,md)[0],md); nb=np.linalg.norm(b_true)
                if nb<1e-30: continue
                # truth-free b_hat at the horizon
                Eh=EH[k].copy(); Vc=Vk.copy()
                for _ in range(FC):
                    Phi=ks.phi(Vc,NSTEP); Vn=seg(ks,Vc,basis(ks,Vc,md),NSTEP)
                    eta=Phi-Vn
                    ep=min(max(1e-5*rnorm(Vc)/max(rnorm(Eh),1e-30),1e-6),5e-2)
                    Pp=ks.phi(Vc+ep*Eh,NSTEP); Mm=ks.phi(Vc-ep*Eh,NSTEP)
                    Eh=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep); Vc=Vn
                b_hat=-pack(splitM(Eh,md)[0],md); nbh=np.linalg.norm(b_hat)
                # MINIMUM-NORM CORRECTION EFFORT, relative to the error it removes
                Es_true=float(np.linalg.norm(Kp@b_true))/nb
                Es_hat=(float(np.linalg.norm(Kp@b_hat))/nbh) if nbh>1e-30 else np.nan
                # realised effect size and gain, oracle correction
                pe,qe=splitM(T[k]-Vk,md)
                FO=fwd(Vk+qe); dQ=splitM(FO-FB,md)[0]
                lam_=rnorm(dQ)/max(rnorm(splitM(FB-TF,md)[0]),1e-30)
                rows.append(dict(band=bn,Estar=Es_true,Estar_hat=Es_hat,
                    lam=lam_,gain=base/max(terr(FO,TF,md),1e-30),
                    smax=float(S[0]),smin=float(S[min(len(S),11)]),
                    cond=float(S[0]/max(S[min(len(S),11)],1e-300)),
                    reservoir=rnorm(qe)/max(rnorm(pe),1e-30)))
        g=rows[n0:]
        if g: print(f"  {bn:<9} n={len(g):>3}"
                    f"  E*={np.median([x['Estar'] for x in g]):10.3e}"
                    f"  cond(K)={np.median([x['cond'] for x in g]):10.3e}"
                    f"  lam={np.median([x['lam'] for x in g]):.4f}"
                    f"  gain={np.median([x['gain'] for x in g]):8.3f}"
                    f"  [{time.time()-t0:.0f}s]",flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("effort.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows],dtype=float)
    def sp(x,y):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        if len(x)<4: return np.nan
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    med={b:np.median([r["Estar"] for r in rows if r["band"]==b])
         for b in BANDS if any(r["band"]==b for r in rows)}
    spread=max(med.values())/max(min(med.values()),1e-300)
    print("\n"+"="*80); print(f"CORRECTION EFFORT, n={len(rows)}"); print("="*80)
    print(f"\nE1 E* varies across bands: spread {spread:.3g}x"
          f"  -> {'PASS' if spread>3 else 'FAIL — effort is flat too'}")
    s2=sp(-np.log(np.maximum(A('Estar'),1e-300)),A('lam'))
    print(f"E2 Spearman(-log E*, lambda) = {s2:+.4f}"
          f"  -> {'PASS' if s2>0.6 else 'FAIL'}")
    cm={b:np.median([r["cond"] for r in rows if r["band"]==b]) for b in med}
    cs=max(cm.values())/max(min(cm.values()),1e-300)
    print(f"E3 conditioning of K differs across bands by {cs:.3g}x"
          f"  -> {'PASS' if cs>10 else 'FAIL'}")
    s4=sp(A('Estar_hat'),A('Estar'))
    print(f"E4 truth-free: Spearman(E*_hat, E*) = {s4:+.4f}"
          f"  -> {'PASS' if s4>0.9 else 'FAIL'}")
    print(f"\n  {'band':<10}{'E*':>12}{'E*_hat':>12}{'cond(K)':>12}"
          f"{'lambda':>9}{'gain':>9}{'reservoir':>11}")
    for b in med:
        g=[r for r in rows if r["band"]==b]
        print(f"  {b:<10}{np.median([x['Estar'] for x in g]):>12.3e}"
              f"{np.median([x['Estar_hat'] for x in g]):>12.3e}"
              f"{np.median([x['cond'] for x in g]):>12.3e}"
              f"{np.median([x['lam'] for x in g]):>9.4f}"
              f"{np.median([x['gain'] for x in g]):>9.3f}"
              f"{np.median([x['reservoir'] for x in g]):>11.3f}")
    print(f"\n  for reference, Spearman(reservoir, lambda) = "
          f"{sp(A('reservoir'),A('lam')):+.4f}")

if __name__=="__main__": main()
