"""
CORRECTABILITY FROM THE TRANSPORT OPERATOR — a prospective ceiling.

Three attempts to predict the effect size from a SCALAR have failed:
alignment/dimension (Spearman +0.067 against dimension), advective locality
(+0.180 against +0.175 for horizon alone), and the closed form in kappa
(median error 0.083, and worse when the neglected term was added).

This is structurally different. Instead of a scalar ratio, characterise the
SUBSPACE that hidden corrections can reach. Define the finite-time transport
operator

    T = P . DPhi_tau . Q            hidden perturbation -> future observable

probe it with r orthonormal directions in Q, take the SVD, and project the
TRUTH-FREE error forecast onto its range:

    rho_T = || Pi_Range(T) b_hat || / || b_hat ||
    G_ceiling = 1 / sqrt(1 - rho_T^2)

Theorem F already bounds the gain by the geometric reachable set P.S; this is
the DYNAMICAL reachable set, which is contained in it and should be sharper.

WHAT MAKES THIS A CEILING AND NOT ANOTHER FIT. G_hat, the existing predictor,
reports what the candidates it tried achieved. rho_T reports what the BEST
POSSIBLE hidden correction could achieve, over all of range(Q), without trying
any of them. Those are different claims.

REGISTERED PREDICTIONS, written before running:
  T1  the bound holds: oracle gain <= G_ceiling in every window
  T2  rho_T is computable TRUTH-FREE and tracks its oracle counterpart:
      Spearman(rho_T from b_hat, rho_T from b_true) > 0.9
  T3  rho_T ORDERS the bands. The damped band, where lambda ~ 1, should have
      rho_T near 1; the unstable band, where lambda ~ 0.09, should be lower.
      Spearman(rho_T, realised gain) > 0.7 across bands.
  T4  THE RISKY ONE: the ceiling is INFORMATIVE, not vacuous. If G_ceiling is
      enormous everywhere it tells us nothing. Require the median ratio
      (oracle gain / G_ceiling) > 0.2 — the achieved gain should be a
      substantial fraction of the predicted ceiling, not a thousandth of it.

T4 is what separates a real theory from a true-but-useless inequality.

Self-contained: numpy only, CPU. Needs ks_core.py beside it.
"""
import numpy as np, os, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("RT_L",100.0)); NPM=int(os.environ.get("RT_NPM",6))
NTRAJ=int(os.environ.get("RT_NTRAJ",8)); NSTEP=int(os.environ.get("RT_NSTEP",50))
NPROBE=int(os.environ.get("RT_NPROBE",24))     # rank of the T probe
NSEG,FC,BURN=7,2,40000
BANDS=os.environ.get("RT_BANDS","unstable,neutral,stable,damped").split(",")

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
    """observable coordinates as a real vector"""
    z=v[m]; return np.concatenate([z.real,z.imag])

def transport_range(ks,V,md,fwd,FB,nprobe,seed):
    """Probe T = P DPhi Q with random hidden directions; return an orthonormal
    basis for its range in the packed observable coordinates."""
    rng=np.random.default_rng(seed)
    cols=[]
    eps_base=1e-6*max(rnorm(V),1e-30)
    for _ in range(nprobe):
        r=rng.normal(size=len(ks.k))+1j*rng.normal(size=len(ks.k))
        r=ks.dealias(r); q=splitM(r,md)[1]
        nq=rnorm(q)
        if nq<1e-30: continue
        q=q/nq*eps_base
        d=splitM(fwd(V+q)-FB,md)[0]
        cols.append(pack(d,md)/eps_base)
    if not cols: return None
    M=np.array(cols).T
    U,S,_=np.linalg.svd(M,full_matrices=False)
    keep=S>1e-10*max(S[0],1e-300)
    return U[:,keep],S[keep]

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); lam=ks.k**2-ks.k**4
    sel={"unstable":lambda i: lam[i]>0,
         "neutral": lambda i: -3<=lam[i]<=0,
         "stable":  lambda i: -12<=lam[i]<-3,
         "damped":  lambda i: -35<=lam[i]<-12}
    print(f"KS L={L:.0f}  T probed with {NPROBE} hidden directions per window\n",flush=True)
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
                out=transport_range(ks,Vk,md,fwd,FB,NPROBE,1000+ti*7+k)
                if out is None: continue
                U,S=out
                b_true=pack(splitM(FB-TF,md)[0],md)
                # forward-propagate the estimate to the horizon for b_hat
                Eh=EH[k].copy(); Vc=Vk.copy()
                for _ in range(FC):
                    Phi=ks.phi(Vc,NSTEP); Vn=seg(ks,Vc,basis(ks,Vc,md),NSTEP)
                    eta=Phi-Vn
                    ep=min(max(1e-5*rnorm(Vc)/max(rnorm(Eh),1e-30),1e-6),5e-2)
                    Pp=ks.phi(Vc+ep*Eh,NSTEP); Mm=ks.phi(Vc-ep*Eh,NSTEP)
                    Eh=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep); Vc=Vn
                b_hat=-pack(splitM(Eh,md)[0],md)
                def rho(v):
                    nv=np.linalg.norm(v)
                    if nv<1e-30: return np.nan
                    return float(np.linalg.norm(U.T@v)/nv)
                rT_hat=rho(b_hat); rT_true=rho(b_true)
                if not (np.isfinite(rT_hat) and np.isfinite(rT_true)): continue
                ceil_hat=1/np.sqrt(max(1-min(rT_hat,0.999999)**2,1e-12))
                # oracle gain: the best a hidden correction actually achieves
                pe,qe=splitM(T[k]-Vk,md)
                FO=fwd(Vk+qe)
                g_or=base/max(terr(FO,TF,md),1e-30)
                rows.append(dict(band=bn,rho_hat=rT_hat,rho_true=rT_true,
                    ceiling=ceil_hat,gain_oracle=g_or,base=base,
                    rank_T=int(len(S)),
                    sv_top3=float(np.sum(S[:3]**2)/max(np.sum(S**2),1e-30))))
        g=[r for r in rows[n0:]]
        if g:
            print(f"  {bn:<9} n={len(g):>3}  rho_hat={np.median([x['rho_hat'] for x in g]):.4f}"
                  f"  rank(T)={int(np.median([x['rank_T'] for x in g])):>3}"
                  f"  top3={np.median([x['sv_top3'] for x in g]):.3f}"
                  f"  ceiling={np.median([x['ceiling'] for x in g]):8.3f}"
                  f"  oracle={np.median([x['gain_oracle'] for x in g]):8.3f}"
                  f"  [{time.time()-t0:.0f}s]",flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("rhoT.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows])
    def sp(x,y):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*80); print(f"TRANSPORT-OPERATOR CORRECTABILITY, n={len(rows)}"); print("="*80)
    viol=int(np.sum(A('gain_oracle')>A('ceiling')*1.001))
    print(f"\nT1 bound holds: {len(rows)-viol}/{len(rows)} windows"
          f"  -> {'PASS' if viol==0 else 'FAIL'}")
    s2=sp(A('rho_hat'),A('rho_true'))
    print(f"T2 rho_T truth-free vs oracle: Spearman {s2:+.4f}"
          f"  -> {'PASS' if s2>0.9 else 'FAIL'}")
    s3=sp(A('rho_hat'),A('gain_oracle'))
    print(f"T3 rho_T orders the gain: Spearman {s3:+.4f}"
          f"  -> {'PASS' if s3>0.7 else 'FAIL'}")
    tight=A('gain_oracle')/np.maximum(A('ceiling'),1e-30)
    print(f"T4 ceiling informative: median oracle/ceiling {np.median(tight):.4f}"
          f"  -> {'PASS' if np.median(tight)>0.2 else 'FAIL — the bound is vacuous'}")
    print(f"\n  {'band':<10}{'rho_hat':>9}{'rho_true':>10}{'rank T':>8}"
          f"{'ceiling':>10}{'oracle':>10}{'ratio':>8}")
    for bn in BANDS:
        g=[r for r in rows if r["band"]==bn]
        if not g: continue
        print(f"  {bn:<10}{np.median([x['rho_hat'] for x in g]):>9.4f}"
              f"{np.median([x['rho_true'] for x in g]):>10.4f}"
              f"{int(np.median([x['rank_T'] for x in g])):>8}"
              f"{np.median([x['ceiling'] for x in g]):>10.3f}"
              f"{np.median([x['gain_oracle'] for x in g]):>10.3f}"
              f"{np.median([x['gain_oracle']/max(x['ceiling'],1e-30) for x in g]):>8.3f}")

if __name__=="__main__": main()
