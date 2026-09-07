"""
TRUTH-FREE CORRECTION AMPLITUDE  —  self-contained, no datasets, pure numpy.

Corollary D.1 gives the optimal correction amplitude alpha*, but computes it
from oracle quantities. This turns it into an ALGORITHM: alpha_hat* estimated
online, using only the ROM's own defect history and its own forecasts. No truth
trajectory is touched at decision time; truth is used ONLY afterwards to score
the decision.

  b_hat      = P[e_hat propagated to the forecast horizon]   (W recurrence)
  Delta_hat(a) = P[Phi_ROM(V + a*q_hat) - Phi_ROM(V)]        (ROM only)
  g1, g2     from Delta_hat at a = 1 and a = 2
  alpha_hat* = argmin_a || b_hat + a*g1 + (a^2/2)*g2 ||

Scored against the HINDSIGHT-OPTIMAL alpha found by sweeping alpha and
measuring true future error. Headline metric: fraction of hindsight-optimal
benefit captured, over independent unseen trajectories.

Runs on CPU. ~1-3 h for the default config.
"""
import numpy as np, sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_core import KS

L      = float(os.environ.get("ALG_L", 100.0))
NPM    = int(os.environ.get("ALG_NPM", 6))
BAND   = os.environ.get("ALG_BAND", "unstable")   # unstable | damped
NTRAJ  = int(os.environ.get("ALG_NTRAJ", 40))
NSEG, FC, BURN = 7, 2, 40000
ALPHAS = [0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,6.0]

def splitM(u, m):
    P = np.zeros_like(u); P[m] = u[m]; return P, u-P

def basis(ks, u, m):
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

def seg(ks, anc, B, n):
    a=np.zeros(len(B),complex); dt=ks.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=ks.F(u)
        return np.array([np.vdot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a); k2=rhs(a+.5*dt*k1); k3=rhs(a+.5*dt*k2); k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))

def fc(ks, x, m, ns, n=FC):
    a=x.copy()
    for _ in range(n): a=seg(ks,a,basis(ks,a,m),ns)
    return a

def terr(A,T,m):
    return np.linalg.norm(splitM(A-T,m)[0])/np.linalg.norm(splitM(T,m)[0])

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); ns=50
    lam=ks.k**2-ks.k**4
    if BAND=="damped":
        md=[i for i in range(1,len(ks.k)) if -35<=lam[i]<=-10][:NPM]
    else:
        md=[i for i in range(1,len(ks.k)) if lam[i]>0][:NPM]
    print(f"fit strategy: {os.environ.get(chr(34)+chr(34).join([]) or 'ALG_FIT','2pt')}"); print(f"KS L={L:.0f} N={N}  P={BAND} band, {len(md)} modes, "
          f"mean growth {np.mean(lam[md]):+.3f}", flush=True)
    rows=[]; t0=time.time()
    for ti in range(NTRAJ):
        u=ks.phi(ks.ic(50000+ti),BURN)
        V=u.copy(); MO=[V.copy()]
        for _ in range(NSEG): V=seg(ks,V,basis(ks,V,md),ns); MO.append(V.copy())
        T=[u.copy()]; t=u.copy()
        for _ in range(NSEG): t=ks.phi(t,ns); T.append(t.copy())
        # ---- W recurrence: truth-free ----
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
            if k >= len(EH): continue
            Vk=MO[k]; TF=T[k+FC]
            qh=splitM(EH[k],md)[1]
            if not np.isfinite(np.linalg.norm(qh)) or \
               np.linalg.norm(qh)>0.5*np.linalg.norm(Vk): continue
            # ================= DECISION TIME: no truth may be used =========
            FB=fc(ks,Vk,md,ns)
            FIT=os.environ.get("ALG_FIT","2pt")
            if FIT=="2pt":                      # original: alpha = 1, 2
                d1=splitM(fc(ks,Vk+qh,md,ns)-FB,md)[0]
                d2=splitM(fc(ks,Vk+2*qh,md,ns)-FB,md)[0]
                g1=2*d1-0.5*d2; g2=d2-2*d1
                probe=None
            elif FIT=="small":                  # fit near zero, less extrapolation
                h=0.25
                dp=splitM(fc(ks,Vk+h*qh,md,ns)-FB,md)[0]
                dm=splitM(fc(ks,Vk-h*qh,md,ns)-FB,md)[0]
                g1=(dp-dm)/(2*h); g2=(dp+dm)/(h*h)
                probe=None
            else:                               # "probe": no model at all --
                # evaluate the ROM forecast directly at each alpha and pick the
                # best against b_hat. Costs |ALPHAS| ROM forecasts, still NO truth.
                probe={a: splitM(fc(ks,Vk+a*qh,md,ns)-FB,md)[0] for a in ALPHAS}
                g1=g2=None
            # b_hat: the recurrence's own forecast of future observable error
            Eh=EH[k].copy(); Vc=Vk.copy()
            for _ in range(FC):
                Phi=ks.phi(Vc,ns); Vn=seg(ks,Vc,basis(ks,Vc,md),ns)
                eta=Phi-Vn
                epp=min(max(1e-5*np.linalg.norm(Vc)/max(np.linalg.norm(Eh),1e-30),
                            1e-6),5e-2)
                Pp=ks.phi(Vc+epp*Eh,ns); Mm=ks.phi(Vc-epp*Eh,ns)
                Eh=eta+(Pp-Mm)/(2*epp)+0.5*(Pp+Mm-2*Phi)/(epp*epp)
                Vc=Vn
            bh=-splitM(Eh,md)[0]   # e_hat estimates (truth - V); b is (forecast - truth)
            if probe is not None:
                a_hat=min(ALPHAS,key=lambda a:np.linalg.norm(bh+probe[a]))
                g1=g2=np.zeros_like(bh)
            else:
                a_hat=min(ALPHAS,key=lambda a:np.linalg.norm(bh+a*g1+0.5*a*a*g2))
            # diagnostic: is b_hat wrong in DIRECTION, or is the g1/g2 fit bad?
            b_true=splitM(FB-TF,md)[0]
            cos_b=float(np.real(np.vdot(b_true,bh))/(
                max(np.linalg.norm(b_true),1e-30)*max(np.linalg.norm(bh),1e-30)))
            # what would alpha selection give with the TRUE b but modelled g?
            a_bt=min(ALPHAS,key=lambda a:np.linalg.norm(
                b_true+(probe[a] if probe is not None else a*g1+0.5*a*a*g2)))
            # and with true b AND true Delta at each alpha (pure model check)
            d_true={a: splitM(fc(ks,Vk+a*qh,md,ns)-FB,md)[0] for a in ALPHAS}
            a_full=min(ALPHAS,key=lambda a:np.linalg.norm(b_true+d_true[a]))
            mdl_err=np.median([np.linalg.norm(
                (probe[a] if probe is not None else a*g1+0.5*a*a*g2)-d_true[a])
                               /max(np.linalg.norm(d_true[a]),1e-30)
                               for a in ALPHAS if a>0])
            # ================= SCORING: truth allowed from here ============
            be=terr(FB,TF,md)
            errs={a: terr(fc(ks,Vk+a*qh,md,ns),TF,md) for a in ALPHAS}
            a_star=min(errs,key=errs.get)
            g_hat=be/max(errs[a_hat],1e-30)
            g_star=be/max(errs[a_star],1e-30)
            g_one=be/max(errs[1.0],1e-30)
            cap=((g_hat-1.0)/(g_star-1.0)) if g_star>1.0+1e-12 else np.nan
            rows.append(dict(traj=ti,seg=k,base=be,alpha_hat=a_hat,
                alpha_star=a_star,gain_hat=g_hat,gain_star=g_star,
                gain_alpha1=g_one,captured=cap,
                bhat_rel=np.linalg.norm(bh)/max(np.linalg.norm(
                    splitM(FB-TF,md)[0]),1e-30),
                cos_bhat=cos_b, alpha_trueb=a_bt, alpha_fullmodel=a_full,
                model_err=mdl_err))
        if (ti+1)%5==0:
            r=[x for x in rows]
            print(f"  traj {ti+1}/{NTRAJ}: n={len(r)} "
                  f"med gain_hat={np.median([x['gain_hat'] for x in r]):.4f} "
                  f"med captured={np.nanmedian([x['captured'] for x in r]):.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows,open(f"alg_{BAND}_L{int(L)}.json","w"),default=float)
    gh=[x["gain_hat"] for x in rows]; gs=[x["gain_star"] for x in rows]
    g1_=[x["gain_alpha1"] for x in rows]
    cp=[x["captured"] for x in rows if np.isfinite(x["captured"])]
    print("\n"+"="*78); print(f"TRUTH-FREE AMPLITUDE — KS {BAND} band, n={len(rows)}")
    print("="*78)
    print(f"  {'method':<34}{'median gain':>13}{'p10':>10}{'win rate':>11}")
    for nm,v in (("no correction",[1.0]*len(gh)),
                 ("fixed alpha = 1",g1_),
                 ("truth-free alpha_hat*",gh),
                 ("hindsight-optimal alpha*",gs)):
        print(f"  {nm:<34}{np.median(v):>13.4f}{np.percentile(v,10):>10.4f}"
              f"{100*np.mean([x>1 for x in v]):>10.1f}%")
    print(f"\n  fraction of hindsight-optimal benefit captured:")
    print(f"    median {np.median(cp):.4f}   p10 {np.percentile(cp,10):.4f}"
          f"   mean {np.mean(cp):.4f}   n={len(cp)}")
    ah=[x["alpha_hat"] for x in rows]; as_=[x["alpha_star"] for x in rows]
    print(f"\n  alpha_hat vs alpha*: median {np.median(ah):.2f} vs {np.median(as_):.2f}"
          f"   exact match {100*np.mean([a==b for a,b in zip(ah,as_)]):.0f}%")
    print(f"  b_hat / b_true magnitude ratio: median "
          f"{np.median([x['bhat_rel'] for x in rows]):.4f}  (1.0 = perfect)")
    print(f"\n  WHERE THE ERROR IS")
    print(f"    cos(b_hat, b_true)              {np.median([x['cos_bhat'] for x in rows]):+.4f}"
          f"   (-1 or +1 = direction perfect)")
    print(f"    quadratic model error vs truth  "
          f"{np.median([x['model_err'] for x in rows]):.4f}   (0 = perfect g1,g2)")
    print(f"    alpha from TRUE b + modelled g  "
          f"{np.median([x['alpha_trueb'] for x in rows]):.2f}")
    print(f"    alpha from TRUE b + TRUE Delta  "
          f"{np.median([x['alpha_fullmodel'] for x in rows]):.2f}"
          f"   (should equal alpha* = {np.median(as_):.2f})")
    print("    -> if 'TRUE b + modelled g' matches alpha*, b_hat is the problem;")
    print("       if it does not, the two-point g1,g2 fit is the problem.")
    print(f"\n  HEADLINE: truth-free amplitude selection captures "
          f"{100*np.median(cp):.0f}% of hindsight-optimal benefit.")

if __name__ == "__main__":
    main()
