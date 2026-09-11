"""
ITEM 1, COMPLETED — truth-free APPLICABILITY PREDICTION and a trust gate.

alg.py showed the method can pick the right amplitude without truth. This asks
the harder question: can it predict its own BENEFIT before acting, and decline
when correcting would hurt?

At decision time, using only b_hat and ROM forecasts:

    G_hat   = ||b_hat|| / min_a ||b_hat + Delta_hat(a)||     predicted gain
    lam_hat = ||Delta_hat(a_hat)|| / ||b_hat||
    cos_hat = <b_hat, Delta_hat> / (||b_hat|| ||Delta_hat||)

Then a TRUST GATE: correct only when G_hat > GATE and the estimator is inside
its amplitude horizon (||q_hat||/||V|| < HORIZON). Everything is scored against
truth afterwards, never before.

The demonstration that matters is not median gain. It is:
  * does G_hat track the realized gain G? (diagonal, Spearman)
  * does the gate catch the windows where correction HARMS?
    In alg.py's unstable band, 7.5% of windows lost. If the gate removes those
    while keeping the winners, the method knows when not to fire.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("APP_L",100.0)); NPM=int(os.environ.get("APP_NPM",6))
BAND=os.environ.get("APP_BAND","unstable"); NTRAJ=int(os.environ.get("APP_NTRAJ",60))
GATE=float(os.environ.get("APP_GATE",1.02)); HORIZON=float(os.environ.get("APP_HORIZON",0.15))
NSEG,FC,BURN=7,2,40000
ALPHAS=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0]

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
    print(f"KS L={L:.0f} P={BAND} ({len(md)} modes)  gate G_hat>{GATE}  "
          f"horizon |q|/|V|<{HORIZON}",flush=True)
    rows=[]; t0=time.time()
    for ti in range(NTRAJ):
        u=ks.phi(ks.ic(60000+ti),BURN)
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
            qh=splitM(EH[k],md)[1]
            rel=np.linalg.norm(qh)/np.linalg.norm(Vk)
            if not np.isfinite(rel) or rel>0.5: continue
            # ---------------- DECISION TIME: no truth ----------------
            FB=fc(ks,Vk,md,ns)
            probe={a: splitM(fc(ks,Vk+a*qh,md,ns)-FB,md)[0] for a in ALPHAS}
            Eh=EH[k].copy(); Vc=Vk.copy()
            for _ in range(FC):
                Phi=ks.phi(Vc,ns); Vn=seg(ks,Vc,basis(ks,Vc,md),ns); eta=Phi-Vn
                epp=min(max(1e-5*np.linalg.norm(Vc)/max(np.linalg.norm(Eh),1e-30),
                            1e-6),5e-2)
                Pp=ks.phi(Vc+epp*Eh,ns); Mm=ks.phi(Vc-epp*Eh,ns)
                Eh=eta+(Pp-Mm)/(2*epp)+0.5*(Pp+Mm-2*Phi)/(epp*epp); Vc=Vn
            bh=-splitM(Eh,md)[0]
            ah=min(ALPHAS,key=lambda a: np.linalg.norm(bh+probe[a]))
            nbh=np.linalg.norm(bh); res=np.linalg.norm(bh+probe[ah])
            G_hat=nbh/max(res,1e-30)
            nd=np.linalg.norm(probe[ah])
            lam_h=nd/max(nbh,1e-30)
            cos_h=(float(np.real(np.vdot(bh,probe[ah])))/max(nbh*nd,1e-30)
                   if nd>1e-30 else 0.0)
            fire=bool(G_hat>GATE and rel<HORIZON)
            # ---------------- SCORING ----------------
            be=terr(FB,TF,md)
            G=be/max(terr(fc(ks,Vk+ah*qh,md,ns),TF,md),1e-30)
            G1=be/max(terr(fc(ks,Vk+qh,md,ns),TF,md),1e-30)
            rows.append(dict(traj=ti,seg=k,base=be,alpha=ah,G_hat=G_hat,G=G,
                G_alpha1=G1,lam_hat=lam_h,cos_hat=cos_h,rel=rel,fire=fire,
                G_gated=(G if fire else 1.0)))
        if (ti+1)%10==0:
            print(f"  traj {ti+1}/{NTRAJ}  n={len(rows)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    json.dump(rows,open(f"apply_{BAND}.json","w"),default=float)
    G=np.array([r["G"] for r in rows]); Gh=np.array([r["G_hat"] for r in rows])
    G1=np.array([r["G_alpha1"] for r in rows]); f=np.array([r["fire"] for r in rows])
    Gg=np.array([r["G_gated"] for r in rows])
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*80); print(f"APPLICABILITY PREDICTION — KS {BAND}, n={len(rows)}")
    print("="*80)
    print(f"\nA1 does G_hat predict G?")
    print(f"   Spearman(G_hat, G) = {sp(Gh,G):+.4f}   Pearson(log) "
          f"{np.corrcoef(np.log(Gh),np.log(np.maximum(G,1e-9)))[0,1]:+.4f}")
    print(f"   median G_hat {np.median(Gh):.4f}  vs realized G {np.median(G):.4f}"
          f"   ratio {np.median(Gh/np.maximum(G,1e-30)):.4f}")
    print(f"\nA2 does the gate avoid harm?")
    lose=G<1.0
    print(f"   windows where correcting HURTS: {lose.sum()}/{len(G)}")
    print(f"   of those, gate declined to fire: {int((~f & lose).sum())}/{int(lose.sum())}")
    print(f"   of the winners, gate fired: {int((f & ~lose).sum())}/{int((~lose).sum())}")
    print(f"\nA3 outcome comparison")
    print(f"   {'strategy':<28}{'median':>10}{'p10':>10}{'win rate':>11}{'worst':>10}")
    for nm,v in (("never correct",np.ones_like(G)),("always, alpha=1",G1),
                 ("always, alpha_hat",G),("GATED alpha_hat",Gg)):
        print(f"   {nm:<28}{np.median(v):>10.4f}{np.percentile(v,10):>10.4f}"
              f"{100*np.mean(v>=1.0):>10.1f}%{np.min(v):>10.4f}")
    print(f"\n   gate fired on {100*f.mean():.1f}% of windows")

if __name__=="__main__": main()
