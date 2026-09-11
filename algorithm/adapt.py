"""
ADAPTIVE CORRECTION-SUBSPACE SELECTION — the deployed algorithm.

The problem this solves: Q-only correction wins hugely where lambda is large
(damped band, 16.70x) and LOSES where lambda is small (unstable band, 1.04x
against P-only's 3.10x). A method that is worst-in-class in half its regimes is
not deployable, however well the theory explains why.

The fix follows from two facts already established:
  * G_hat predicts the realised gain with Spearman +1.0000, TRUTH-FREE
  * P-only and full-state corrections are ALSO truth-free — they use P e_hat
    and e_hat from the same defect recurrence. Only 3DVar needs observations.

So let the algorithm choose the correction SUBSPACE by G_hat. It evaluates each
candidate against its own predicted future error and picks the best, before
seeing any truth.

  candidates:  0            do nothing
               alpha Q e    hidden-only, leaves the observable untouched
               alpha P e    observable-only
               alpha e      full state
  selection:   argmin over candidates of || b_hat + Delta_hat(c) ||

HONEST NOTE ON WHAT IS GIVEN UP. The Delta_P = 0 protocol is what proved that
hidden state carries causal information about future observables; that remains
the scientific instrument and is reported separately. This is the DEPLOYED
method, and when it selects a P-touching candidate it no longer preserves the
observable. The paper should carry both and not conflate them.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("AD_L",100.0)); NPM=int(os.environ.get("AD_NPM",6))
BAND=os.environ.get("AD_BAND","unstable"); NTRAJ=int(os.environ.get("AD_NTRAJ",30))
R_OBS=float(os.environ.get("AD_ROBS",1e-6))
NSEG,FC,BURN=7,2,40000
ALPHAS=[0.5,1.0,1.5,2.0,3.0]

def splitM(u,m):
    P=np.zeros_like(u); P[m]=u[m]; return P,u-P
def basis(ks,u,m):
    B=[]
    for mm in m:
        for ph in (0,1):
            e=np.zeros_like(u); e[mm]=1.0 if ph==0 else 1.0j
            B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B: v=v-np.vdot(b,v)*b
        n=np.linalg.norm(v); return v/n if n>1e-12 else None
    f=ks.F(u)
    q=orth(splitM(f,m)[1])
    if q is not None: B.append(q)
    q=orth(splitM(ks.DF(u,f),m)[1])
    if q is not None: B.append(q)
    return B
def seg(ks,anc,B,n):
    a=np.zeros(len(B),complex); dt=ks.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=ks.F(u)
        return np.array([np.vdot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))
def fc(ks,x,m,ns,n=FC):
    a=x.copy()
    for _ in range(n): a=seg(ks,a,basis(ks,a,m),ns)
    return a
def terr(A,T,m):
    return np.linalg.norm(splitM(A-T,m)[0])/np.linalg.norm(splitM(T,m)[0])

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); ns=50; lam=ks.k**2-ks.k**4
    md=([i for i in range(1,len(ks.k)) if -35<=lam[i]<=-10][:NPM] if BAND=="damped"
        else [i for i in range(1,len(ks.k)) if lam[i]>0][:NPM])
    print(f"KS L={L:.0f} P={BAND} ({len(md)} modes)",flush=True)
    rows=[]; t0=time.time()
    for ti in range(NTRAJ):
        u=ks.phi(ks.ic(40000+ti),BURN)
        V=u.copy(); MO=[V.copy()]
        for _ in range(NSEG): V=seg(ks,V,basis(ks,V,md),ns); MO.append(V.copy())
        T=[u.copy()]; t=u.copy()
        for _ in range(NSEG): t=ks.phi(t,ns); T.append(t.copy())
        E=np.zeros_like(MO[0]); EH=[E.copy()]
        for k in range(NSEG-FC):
            Phi=ks.phi(MO[k],ns); eta=Phi-MO[k+1]
            if np.linalg.norm(E)<1e-30: E=eta.copy()
            else:
                ep=min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                Pp=ks.phi(MO[k]+ep*E,ns); Mm=ks.phi(MO[k]-ep*E,ns)
                E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
            EH.append(E.copy())
        for k in (3,4):
            if k>=len(EH): continue
            Vk,TF=MO[k],T[k+FC]
            eh=EH[k]; ph,qh=splitM(eh,md)
            if not np.isfinite(np.linalg.norm(qh)) or \
               np.linalg.norm(qh)>0.5*np.linalg.norm(Vk): continue
            # ---------------- DECISION TIME: no truth ----------------
            FB=fc(ks,Vk,md,ns)
            cands={("none",0.0): np.zeros_like(Vk)}
            for a in ALPHAS:
                cands[("Q",a)]=a*qh; cands[("P",a)]=a*ph; cands[("full",a)]=a*eh
            probe={key: splitM(fc(ks,Vk+c,md,ns)-FB,md)[0] for key,c in cands.items()}
            Eh=EH[k].copy(); Vc=Vk.copy()
            for _ in range(FC):
                Phi=ks.phi(Vc,ns); Vn=seg(ks,Vc,basis(ks,Vc,md),ns); eta=Phi-Vn
                epp=min(max(1e-5*np.linalg.norm(Vc)/max(np.linalg.norm(Eh),1e-30),
                            1e-6),5e-2)
                Pp=ks.phi(Vc+epp*Eh,ns); Mm=ks.phi(Vc-epp*Eh,ns)
                Eh=eta+(Pp-Mm)/(2*epp)+0.5*(Pp+Mm-2*Phi)/(epp*epp); Vc=Vn
            bh=-splitM(Eh,md)[0]
            pick=min(cands,key=lambda key: np.linalg.norm(bh+probe[key]))
            nbh=np.linalg.norm(bh)
            G_hat=nbh/max(np.linalg.norm(bh+probe[pick]),1e-30)
            # 3DVar baseline: needs TRUTH observations of P
            ens=[splitM(T[j]-MO[j],md)[0]+splitM(T[j]-MO[j],md)[1]
                 for j in range(1,k+1)]
            dx=np.zeros_like(Vk)
            if len(ens)>=2:
                Em=sum(ens)/len(ens); Ec=[e-Em for e in ens]; M=len(Ec)
                HE=np.array([splitM(e,md)[0][md] for e in Ec]).T
                d=(splitM(T[k],md)[0]-splitM(Vk,md)[0])[md]
                S=HE.conj().T@HE+M*R_OBS*np.eye(M)*max(float(np.real(np.vdot(d,d))),1e-30)
                try: dx=sum(np.linalg.solve(S,HE.conj().T@d)[i]*Ec[i] for i in range(M))
                except np.linalg.LinAlgError: pass
            # ---------------- SCORING ----------------
            be=terr(FB,TF,md)
            def g(c): return be/max(terr(fc(ks,Vk+c,md,ns),TF,md),1e-30)
            rows.append(dict(traj=ti,seg=k,base=be,pick=pick[0],alpha=pick[1],
                G_hat=G_hat, G_adapt=g(cands[pick]),
                G_Q=g(qh), G_P=g(ph), G_full=g(eh), G_3dvar=g(dx),
                lam=np.linalg.norm(splitM(fc(ks,Vk+splitM(T[k]-Vk,md)[1],md,ns)
                     -FB,md)[0])/max(np.linalg.norm(splitM(FB-TF,md)[0]),1e-30),
                dP=np.linalg.norm(splitM(cands[pick],md)[0])))
        if (ti+1)%10==0:
            print(f"  traj {ti+1}/{NTRAJ}  n={len(rows)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    json.dump(rows,open(f"adapt_{BAND}.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows])
    print("\n"+"="*88); print(f"ADAPTIVE SUBSPACE SELECTION — KS {BAND}, n={len(rows)}")
    print("="*88)
    print(f"  {'strategy':<30}{'median':>10}{'p10':>10}{'win%':>8}{'worst':>10}{'truth?':>8}")
    for nm,key,tr in (("uncorrected",None,"no"),("Q-only (dP=0)","G_Q","no"),
                      ("P-only","G_P","no"),("full-state","G_full","no"),
                      ("ensemble 3DVar","G_3dvar","YES"),
                      ("ADAPTIVE (ours)","G_adapt","no")):
        v=np.ones(len(rows)) if key is None else A(key)
        print(f"  {nm:<30}{np.median(v):>10.4f}{np.percentile(v,10):>10.4f}"
              f"{100*np.mean(v>=1.0):>7.0f}%{np.min(v):>10.4f}{tr:>8}")
    from collections import Counter
    c=Counter(r["pick"] for r in rows)
    print(f"\n  what it picked: " + "  ".join(f"{k}={v}" for k,v in c.most_common()))
    print(f"  median lambda {np.median(A('lam')):.4f}")
    obs=[r for r in rows if r["pick"]=="Q"]
    print(f"  observable preserved on {len(obs)}/{len(rows)} windows "
          f"({100*len(obs)/len(rows):.0f}%) — those are the dP=0 interventions")
    print(f"\n  vs best fixed strategy: adaptive {np.median(A('G_adapt')):.4f} vs "
          f"{max(np.median(A(k)) for k in ('G_Q','G_P','G_full')):.4f}")
    print(f"  vs 3DVar (uses truth): {np.median(A('G_adapt')):.4f} vs "
          f"{np.median(A('G_3dvar')):.4f}")

if __name__=="__main__": main()
