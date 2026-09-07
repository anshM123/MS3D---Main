"""
ITEM 1 — WHAT DETERMINES LAMBDA.  A predictive theory, not a scoping law.

The effect-size identity gain = 1/sqrt(1 + 2*lam*cos + lam^2) is exact but
lam has been an OUTCOME, not a prediction. Stability sets it in KS and NS and
not in CGL, so the mechanism was two-thirds built.

DERIVATION.  Write the finite-time tangent map in P/Q blocks. The future
observable error is

    b  ~  A_PP (P e)  +  A_PQ (Q e)                     (linear response)
    Delta = - A_PQ (Q e)                                (removing Q e)

so lam = ||A_PQ Qe|| / ||b|| is controlled by ONE dimensionless ratio,

    kappa := ||A_PQ (Q e)|| / ||A_PP (P e)||            HIDDEN-DRIVEN SHARE
                                                        (cross-block transport
                                                         over same-block
                                                         persistence)
and

    lam = kappa / sqrt(1 + 2 kappa c + kappa^2),   c = alignment of the two.

PREDICTIONS, registered before running:
  K1  lam is monotone in kappa: Spearman(kappa, lam) > 0.9 pooled across all
      bands and systems
  K2  the closed form holds: median |lam_predicted - lam| < 0.05, where
      lam_predicted uses measured kappa and c
  K3  kappa EXPLAINS the cross-system spread. KS-unstable should give the
      smallest kappa, CGL the largest, NS intermediate — and the ordering of
      kappa should match the ordering of lam.
  K4  kappa is computable TRUTH-FREE from e_hat, and kappa_hat tracks kappa:
      Spearman > 0.9. This is what makes it a predictor rather than a
      diagnosis.

If K1-K4 hold, lam stops being an outcome and becomes a computable property of
the P/Q block structure — which is the missing third of the mechanism.
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

NTRAJ=int(os.environ.get("KP_NTRAJ",12)); NPM=int(os.environ.get("KP_NPM",6))
NSEG,FC,BURN=7,2,40000

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

def run_band(ks,md,ns,label,seed0):
    rows=[]
    for ti in range(NTRAJ):
        u=ks.phi(ks.ic(seed0+ti),BURN)
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
            pe,qe=splitM(T[k]-Vk,md)                     # TRUE split
            ph,qh=splitM(EH[k],md)                       # estimated split
            if not np.isfinite(np.linalg.norm(qh)) or \
               np.linalg.norm(qh)>0.5*np.linalg.norm(Vk): continue
            FB=fc(ks,Vk,md,ns)
            b=splitM(FB-TF,md)[0]; nb=np.linalg.norm(b)
            if nb<1e-30: continue
            # --- eta_P: the estimator's own local defect contribution to b.
            # Dropped in the first derivation. At large kappa, A_PP(Pe) is
            # negligible and eta_P is what remains in the denominator, which is
            # exactly where lam_pred saturated at 1 while lam reached 1.20.
            etaP = splitM(ks.phi(Vk, ns) - seg(ks, Vk, basis(ks, Vk, md), ns),
                          md)[0]
            # --- the two block responses (TRUE) ---
            dQ=splitM(fc(ks,Vk+qe,md,ns)-FB,md)[0]       # A_PQ (Qe)
            dP=splitM(fc(ks,Vk+pe,md,ns)-FB,md)[0]       # A_PP (Pe)
            nQ,nP=np.linalg.norm(dQ),np.linalg.norm(dP)
            if nQ<1e-30 or nP<1e-30: continue
            kap=nQ/nP
            c=float(np.real(np.vdot(dQ,dP)))/(nQ*nP)
            lam=nQ/nb
            lam_pred=kap/np.sqrt(max(1+2*kap*c+kap*kap,1e-300))
            # v2: reconstruct the denominator from all THREE terms rather than
            # assuming ||b|| ~ ||A_PP(Pe) + A_PQ(Qe)||
            b_model = dP + dQ + etaP
            lam_pred2 = nQ / max(np.linalg.norm(b_model), 1e-300)
            # how much of ||b|| does the 2-term model miss vs the 3-term one?
            miss2 = abs(np.linalg.norm(dP+dQ) - nb)/nb
            miss3 = abs(np.linalg.norm(b_model) - nb)/nb
            # --- TRUTH-FREE versions ---
            dQh=splitM(fc(ks,Vk+qh,md,ns)-FB,md)[0]
            dPh=splitM(fc(ks,Vk+ph,md,ns)-FB,md)[0]
            nQh,nPh=np.linalg.norm(dQh),np.linalg.norm(dPh)
            kap_h=nQh/max(nPh,1e-30)
            rows.append(dict(band=label,kappa=kap,kappa_hat=kap_h,c=c,
                lam=lam,lam_pred=lam_pred,lam_pred2=lam_pred2,
                err=abs(lam_pred-lam),err2=abs(lam_pred2-lam),
                miss2=miss2,miss3=miss3,
                etaP_share=np.linalg.norm(etaP)/max(nb,1e-30),
                reservoir=np.linalg.norm(qe)/max(np.linalg.norm(pe),1e-30)))
    return rows

def main():
    ks=KS(L=100.0,N=200,dt=0.005); ns=50; lam_k=ks.k**2-ks.k**4
    bands=[("unstable",[i for i in range(1,len(ks.k)) if lam_k[i]>0][:NPM]),
           ("neutral",[i for i in range(1,len(ks.k)) if -3<=lam_k[i]<=0][:NPM]),
           ("stable",[i for i in range(1,len(ks.k)) if -12<=lam_k[i]<-3][:NPM]),
           ("damped",[i for i in range(1,len(ks.k)) if -35<=lam_k[i]<-12][:NPM])]
    rows=[]; t0=time.time()
    for nm,md in bands:
        if len(md)<3: continue
        r=run_band(ks,md,ns,nm,20000)
        rows+=r
        if r: print(f"  {nm:<9} n={len(r):>3}  kappa={np.median([x['kappa'] for x in r]):8.3f}"
                    f"  lam={np.median([x['lam'] for x in r]):.4f}"
                    f"  lam_pred={np.median([x['lam_pred'] for x in r]):.4f}"
                    f"  [{time.time()-t0:.0f}s]",flush=True)
    json.dump(rows,open("kappa.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows])
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*84); print(f"KAPPA — WHAT DETERMINES LAMBDA, n={len(rows)}"); print("="*84)
    print(f"\nK1 Spearman(kappa, lam) = {sp(A('kappa'),A('lam')):+.4f}"
          f"  -> {'PASS' if sp(A('kappa'),A('lam'))>0.9 else 'FAIL'}")
    print(f"   for reference, Spearman(reservoir, lam) = "
          f"{sp(A('reservoir'),A('lam')):+.4f}")
    e=A('err')
    print(f"\nK2 closed form lam = kappa/sqrt(1+2*kappa*c+kappa^2)")
    print(f"   median |lam_pred - lam| = {np.median(e):.4f}   p90 {np.percentile(e,90):.4f}"
          f"  -> {'PASS' if np.median(e)<0.05 else 'FAIL'}")
    print(f"\nK3 ordering across bands")
    print(f"   {'band':<10}{'kappa':>10}{'lam':>9}{'lam_pred':>10}{'reservoir':>11}")
    for nm in ("unstable","neutral","stable","damped"):
        g=[r for r in rows if r["band"]==nm]
        if not g: continue
        print(f"   {nm:<10}{np.median([x['kappa'] for x in g]):>10.3f}"
              f"{np.median([x['lam'] for x in g]):>9.4f}"
              f"{np.median([x['lam_pred'] for x in g]):>10.4f}"
              f"{np.median([x['reservoir'] for x in g]):>11.3f}")
    e2=A('err2')
    print(f"\nK2b THREE-TERM model  b ~ A_PP(Pe) + A_PQ(Qe) + eta_P")
    print(f"    median |lam_pred2 - lam| = {np.median(e2):.4f}"
          f"   p90 {np.percentile(e2,90):.4f}"
          f"  -> {'PASS' if np.median(e2)<0.05 else 'FAIL'}")
    print(f"    two-term model misses ||b|| by  {np.median(A('miss2')):.4f}")
    print(f"    three-term model misses ||b|| by {np.median(A('miss3')):.4f}")
    print(f"    eta_P share of ||b||:            {np.median(A('etaP_share')):.4f}")
    print(f"    Spearman(kappa, lam) with 3-term denominator = "
          f"{sp(A('kappa'),A('lam')):+.4f}")
    print(f"\n    per band")
    print(f"    {'band':<10}{'lam':>9}{'2-term':>9}{'3-term':>9}{'etaP/b':>9}")
    for nm in ("unstable","neutral","stable","damped"):
        g=[r for r in rows if r["band"]==nm]
        if not g: continue
        print(f"    {nm:<10}{np.median([x['lam'] for x in g]):>9.4f}"
              f"{np.median([x['lam_pred'] for x in g]):>9.4f}"
              f"{np.median([x['lam_pred2'] for x in g]):>9.4f}"
              f"{np.median([x['etaP_share'] for x in g]):>9.4f}")

    print(f"\nK4 truth-free: Spearman(kappa_hat, kappa) = "
          f"{sp(A('kappa_hat'),A('kappa')):+.4f}"
          f"  -> {'PASS' if sp(A('kappa_hat'),A('kappa'))>0.9 else 'FAIL'}")
    print(f"   Spearman(kappa_hat, lam) = {sp(A('kappa_hat'),A('lam')):+.4f}")

if __name__=="__main__": main()
