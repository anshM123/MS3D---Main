"""
BLIND CORRECTABILITY PREDICTION — freeze on KS, predict CGL and sensors.

Four attempts to predict lambda have failed, and together they narrow the
search sharply:

  1. alignment / observable dimension   refuted   (n=224, Spearman +0.067)
  2. advective locality                 refuted   (n=160, +0.180 vs +0.175)
  3. closed form lam = f(kappa, c)      failed    (n=96,  median err 0.083)
  4. transport-operator range rho_T     VACUOUS   (rho_T = 1 identically,
                                                   rank(T) = full observable
                                                   dimension, n=63)

Attempt 4 is the informative one: a map from ~400 hidden coordinates into 12
observable ones is surjective, so EVERY observable direction is reachable and
the geometry cannot be what limits correctability. Any predictor must be about
the MAGNITUDE of the response, not its direction set.

The only truth-free quantity that survives is kappa_hat, the ratio of hidden-
to observable-block response, which tracks its oracle counterpart at Spearman
+1.0000 and orders lambda across a 24000-fold range. What it lacks is a closed
form. So: stop deriving one, and CALIBRATE one on a single system, freeze it,
and predict others blind.

PROTOCOL, and the freezing is the point:

  STAGE 1  CALIBRATE on Kuramoto-Sivashinsky ONLY. Fit a monotone map
           kappa_hat -> lambda on the KS bands. Write the fitted parameters to
           calibration.json and stop.

  STAGE 2  PREDICT, without refitting. For each held-out configuration compute
           kappa_hat, push it through the frozen map to get lam_pred, and from
           lam_pred and the measured cos_hat produce a predicted gain
           G_pred = 1/sqrt(1 + 2 lam cos + lam^2). Record predictions BEFORE
           the correction is evaluated.

  STAGE 3  SCORE against the realised gain.

Held-out systems: complex Ginzburg-Landau (CUBIC nonlinearity) and
finite-difference KS observed through POINT SENSORS. Neither contributes to
the calibration.

REGISTERED PREDICTIONS, written before running:
  B1  the frozen map transfers: median |lam_pred - lam| < 0.15 on held-out
      systems, versus 0.083 for the failed closed form ON ITS OWN training data
  B2  it orders them: Spearman(lam_pred, lam) > 0.7 on held-out systems
  B3  the predicted GAIN is useful: median |log(G_pred/G)| < 0.35, i.e. within
      about 40 percent
  B4  THE RISKY ONE. The map must beat the trivial baseline of predicting the
      KS median lambda for everything. If a constant does as well, the
      calibration carries no information.

B4 is the test. Everything else can pass on a monotone trend alone.

Self-contained apart from ks_core.py and cgl_core.py. CPU, numpy.
"""
import numpy as np, os, json, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS
from cgl_core import CGL, rdot as crdot, rnorm as crnorm

NTRAJ=int(os.environ.get("BL_NTRAJ",8)); NPM=int(os.environ.get("BL_NPM",6))
NSEG,FC=7,2
ANCH=[int(x) for x in os.environ.get("BL_ANCH","1,2,3,4").split(",")]
BMIN=float(os.environ.get("BL_BMIN",2e-4))
BMAX=float(os.environ.get("BL_BMAX",0.50))
STAGE=os.environ.get("BL_STAGE","all")   # calibrate | predict | all
CAL=os.path.join(os.path.dirname(os.path.abspath(__file__)) or ".","calibration.json")

def rdot(a,b): return float(np.real(np.vdot(a,b)))
def rnorm(a):  return float(np.sqrt(max(rdot(a,a),0.0)))

# ----------------------------------------------------- generic machinery
def make_ops(sysobj, kind):
    """Return split / basis / seg / dot appropriate to the system."""
    if kind=="cgl":
        from cgl_core import splitM as sp, basis as bs, seg as sg
        return sp, (lambda u,m: bs(sysobj,u,m)), (lambda a,B,n: sg(sysobj,a,B,n)), crdot, crnorm
    def spl(u,m):
        P=np.zeros_like(u); P[m]=u[m]; return P,u-P
    def bas(u,m):
        B=[]
        for mm in m:
            for ph in (1.0,1.0j):
                e=np.zeros(len(sysobj.k),complex); e[mm]=ph
                B.append(e/rnorm(e))
        def orth(v):
            for f in B: v=v-rdot(f,v)*f
            n=rnorm(v); return v/n if n>1e-12 else None
        w=sysobj.F(u)
        for _ in range(2):
            q=orth(spl(w,m)[1])
            if q is None: break
            B.append(q); w=sysobj.DF(u,w)
        return B
    def sg(anc,B,n):
        a=np.zeros(len(B)); dt=sysobj.dt
        def rhs(a):
            u=anc+sum(a[i]*B[i] for i in range(len(B))); F=sysobj.F(u)
            return np.array([rdot(B[i],F) for i in range(len(B))])
        for _ in range(n):
            k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
            a=a+dt/6*(k1+2*k2+2*k3+k4)
        return anc+sum(a[i]*B[i] for i in range(len(B)))
    return spl,bas,sg,rdot,rnorm

def harvest(sysobj, kind, md, nstep, burn, seed0, label, rows):
    spl,bas,sg,dot,nrm = make_ops(sysobj,kind)
    for ti in range(NTRAJ):
        u=sysobj.phi(sysobj.ic(seed0+ti),burn)
        V=u.copy(); MO=[V.copy()]
        for _ in range(NSEG): V=sg(V,bas(V,md),nstep); MO.append(V.copy())
        T=[u.copy()]; t=u.copy()
        for _ in range(NSEG): t=sysobj.phi(t,nstep); T.append(t.copy())
        E=np.zeros_like(MO[0]); EH=[E.copy()]
        for k in range(NSEG-FC):
            Phi=sysobj.phi(MO[k],nstep); eta=Phi-MO[k+1]
            if nrm(E)<1e-30: E=eta.copy()
            else:
                ep=min(max(1e-5*nrm(MO[k])/nrm(E),1e-6),5e-2)
                Pp=sysobj.phi(MO[k]+ep*E,nstep); Mm=sysobj.phi(MO[k]-ep*E,nstep)
                E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
            EH.append(E.copy())
        def fwd(x,n=FC):
            a=x.copy()
            for _ in range(n): a=sg(a,bas(a,md),nstep)
            return a
        for k in ANCH:
            if k>=len(EH) or k+FC>=len(T): continue
            Vk,TF=MO[k],T[k+FC]
            ph_,qh=spl(EH[k],md)
            if not np.isfinite(nrm(qh)) or nrm(qh)>0.5*nrm(Vk): continue
            FB=fwd(Vk); b=spl(FB-TF,md)[0]; nb=nrm(b)
            base=nb/max(nrm(spl(TF,md)[0]),1e-30)
            if not (BMIN<base<BMAX) or nb<1e-30: continue
            # ---- TRUTH-FREE: kappa_hat and cos_hat from the ESTIMATE ----
            dQh=spl(fwd(Vk+qh)-FB,md)[0]; dPh=spl(fwd(Vk+ph_)-FB,md)[0]
            nQh,nPh=nrm(dQh),nrm(dPh)
            if nQh<1e-30 or nPh<1e-30: continue
            kap_hat=nQh/nPh
            bh=-spl(EH[k],md)[0]
            nbh=nrm(bh)
            cos_hat=(dot(bh,dQh)/max(nbh*nQh,1e-30)) if nbh>1e-30 else 0.0
            # ---- ORACLE, for scoring only ----
            pe,qe=spl(T[k]-Vk,md)
            dQ=spl(fwd(Vk+qe)-FB,md)[0]; nQ=nrm(dQ)
            if nQ<1e-30: continue
            lam=nQ/nb; cos=dot(b,dQ)/(nb*nQ)
            gain=nb/max(nrm(b+dQ),1e-30)
            rows.append(dict(sys=label,kappa_hat=kap_hat,cos_hat=cos_hat,
                lam=lam,cos=cos,gain=gain,base=base))

# ----------------------------------------------------- the frozen map
def fmap(kap,p,a):
    """lam = (a*kap) / ((a*kap)^p + 1)^(1/p). Two parameters, right limits:
    lam -> a*kap as kap -> 0, lam -> 1 as kap -> inf."""
    x=np.maximum(np.asarray(kap,float)*a,1e-12)
    return x/np.power(np.power(x,p)+1.0,1.0/p)

def fit(kap,lam):
    best=None
    for p in np.logspace(-0.8,1.0,80):
        for a in np.logspace(-1.2,1.2,120):
            e=float(np.median(np.abs(fmap(kap,p,a)-lam)))
            if best is None or e<best[2]: best=(float(p),float(a),e)
    return best

def main():
    rows=[]; t0=time.time()
    # ---------------- STAGE 1: CALIBRATE ON KS ONLY ----------------
    if STAGE in ("calibrate","all"):
        ks=KS(L=100.0,N=200,dt=0.005); lam_k=ks.k**2-ks.k**4
        bands={"unstable":lambda i: lam_k[i]>0,
               "neutral": lambda i: -3<=lam_k[i]<=0,
               "stable":  lambda i: -12<=lam_k[i]<-3,
               "damped":  lambda i: -35<=lam_k[i]<-12}
        print("STAGE 1 — CALIBRATE ON KURAMOTO–SIVASHINSKY ONLY\n",flush=True)
        cal=[]
        for bn,f in bands.items():
            md=[i for i in range(1,len(ks.k)) if f(i)][:NPM]
            if len(md)<3: continue
            n0=len(cal); harvest(ks,"ks",md,50,40000,20000,"KS/"+bn,cal)
            g=cal[n0:]
            if g: print(f"  KS/{bn:<9} n={len(g):>3}"
                        f"  kappa_hat={np.median([x['kappa_hat'] for x in g]):9.3f}"
                        f"  lam={np.median([x['lam'] for x in g]):.4f}"
                        f"  [{time.time()-t0:.0f}s]",flush=True)
        if not cal: print("no calibration data"); return
        K=np.array([r["kappa_hat"] for r in cal]); Lm=np.array([r["lam"] for r in cal])
        p,a,e=fit(K,Lm)
        json.dump({"p":p,"a":a,"fit_median_err":e,"n_calibration":len(cal),
                   "calibrated_on":"Kuramoto-Sivashinsky only",
                   "ks_median_lambda":float(np.median(Lm))},open(CAL,"w"),indent=2)
        print(f"\n  FROZEN MAP  lam = (a k)/((a k)^p + 1)^(1/p)")
        print(f"     p = {p:.4f}   a = {a:.4f}   in-sample median err {e:.4f}"
              f"   n = {len(cal)}")
        print(f"  written to calibration.json — NOT refitted below\n",flush=True)
        rows+=cal
    # ---------------- STAGE 2/3: PREDICT HELD-OUT, THEN SCORE ----------------
    if STAGE in ("predict","all"):
        C=json.load(open(CAL)); p,a=C["p"],C["a"]
        print(f"STAGE 2 — BLIND PREDICTION using p={p:.4f} a={a:.4f}\n",flush=True)
        held=[]
        cg=CGL(N=256,L=100.0,b=2.0,c=-1.0,dt=0.002); gr=cg.growth
        for nm,lo,hi in (("unstable",0.0,1e9),("neutral",-3.0,0.0),("mild",-12.0,-3.0)):
            md=sorted([i for i in range(1,cg.N) if cg.deal[i] and lo<=gr[i]<=hi],
                      key=lambda i:-gr[i])[:NPM]
            if len(md)<3: continue
            n0=len(held); harvest(cg,"cgl",md,100,20000,80000,"CGL/"+nm,held)
            g=held[n0:]
            if g: print(f"  CGL/{nm:<8} n={len(g):>3}"
                        f"  kappa_hat={np.median([x['kappa_hat'] for x in g]):9.3f}"
                        f"  lam_pred={np.median(fmap([x['kappa_hat'] for x in g],p,a)):.4f}"
                        f"  lam={np.median([x['lam'] for x in g]):.4f}"
                        f"  [{time.time()-t0:.0f}s]",flush=True)
        # ---- finite-difference KS observed through POINT SENSORS ----
        NXF,LDF,DTF,NSTF = 200,100.0,0.002,125
        class KSFD:
            def __init__(s_): s_.n,s_.L,s_.h,s_.dt = NXF,LDF,LDF/NXF,DTF
            def F(s_,u):
                h=s_.h; um1=np.roll(u,1); up1=np.roll(u,-1)
                um2=np.roll(u,2); up2=np.roll(u,-2)
                return -u*((up1-um1)/(2*h))-(up1-2*u+um1)/h**2 \
                       -(up2-4*up1+6*u-4*um1+um2)/h**4
            def DF(s_,u,v):
                h=s_.h; vm1=np.roll(v,1); vp1=np.roll(v,-1)
                vm2=np.roll(v,2); vp2=np.roll(v,-2)
                um1=np.roll(u,1); up1=np.roll(u,-1)
                return -(v*((up1-um1)/(2*h))+u*((vp1-vm1)/(2*h))) \
                       -(vp1-2*v+vm1)/h**2-(vp2-4*vp1+6*v-4*vm1+vm2)/h**4
            def step(s_,u):
                dt=s_.dt
                k1=s_.F(u);k2=s_.F(u+.5*dt*k1);k3=s_.F(u+.5*dt*k2);k4=s_.F(u+dt*k3)
                return u+dt/6*(k1+2*k2+2*k3+k4)
            def phi(s_,u,n):
                for _ in range(n): u=s_.step(u)
                return u
            def ic(s_,seed):
                rng=np.random.default_rng(seed); x=np.arange(s_.n)*s_.h
                u=0.1*rng.normal(size=s_.n)
                for m in range(1,5):
                    u+=(0.5/m)*np.cos(2*np.pi*m*x/s_.L+rng.uniform(0,2*np.pi))
                return u
        fd=KSFD()
        def fd_ops(sen):
            def spl(u,m):
                P=np.zeros_like(u); P[m]=u[m]; return P,u-P
            def bas(u,m):
                B=[]
                for s_ in m:
                    e=np.zeros_like(u); e[s_]=1.0; B.append(e/np.linalg.norm(e))
                def orth(v):
                    for f in B: v=v-np.dot(f,v)*f
                    n=np.linalg.norm(v); return v/n if n>1e-12 else None
                w=fd.F(u)
                for _ in range(2):
                    q=orth(spl(w,m)[1])
                    if q is None: break
                    B.append(q); w=fd.DF(u,w)
                return B
            def sg(anc,B,n):
                a=np.zeros(len(B)); dt=fd.dt
                def rhs(a):
                    u=anc+sum(a[i]*B[i] for i in range(len(B))); F=fd.F(u)
                    return np.array([np.dot(B[i],F) for i in range(len(B))])
                for _ in range(n):
                    k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
                    a=a+dt/6*(k1+2*k2+2*k3+k4)
                return anc+sum(a[i]*B[i] for i in range(len(B)))
            return spl,bas,sg
        for nsen in (8,12,20):
            sen=np.linspace(0,NXF,nsen,endpoint=False).astype(int)
            spl,bas,sg=fd_ops(sen); n0=len(held)
            for ti in range(NTRAJ):
                u=fd.phi(fd.ic(15000+ti),60000)
                if not np.all(np.isfinite(u)): continue
                V=u.copy(); MO=[V.copy()]
                for _ in range(NSEG): V=sg(V,bas(V,sen),NSTF); MO.append(V.copy())
                T=[u.copy()]; t=u.copy()
                for _ in range(NSEG): t=fd.phi(t,NSTF); T.append(t.copy())
                E=np.zeros_like(MO[0]); EH=[E.copy()]
                for k in range(NSEG-FC):
                    Phi=fd.phi(MO[k],NSTF); eta=Phi-MO[k+1]
                    if np.linalg.norm(E)<1e-30: E=eta.copy()
                    else:
                        ep=min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                        Pp=fd.phi(MO[k]+ep*E,NSTF); Mm=fd.phi(MO[k]-ep*E,NSTF)
                        E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                    EH.append(E.copy())
                def fwdf(x,n=FC):
                    a=x.copy()
                    for _ in range(n): a=sg(a,bas(a,sen),NSTF)
                    return a
                for k in ANCH:
                    if k>=len(EH) or k+FC>=len(T): continue
                    Vk,TF=MO[k],T[k+FC]
                    ph_,qh=spl(EH[k],sen)
                    if not np.isfinite(np.linalg.norm(qh)) or \
                       np.linalg.norm(qh)>0.5*np.linalg.norm(Vk): continue
                    FB=fwdf(Vk); b=spl(FB-TF,sen)[0]; nb=np.linalg.norm(b)
                    base=nb/max(np.linalg.norm(spl(TF,sen)[0]),1e-30)
                    if not (BMIN<base<BMAX) or nb<1e-30: continue
                    dQh=spl(fwdf(Vk+qh)-FB,sen)[0]; dPh=spl(fwdf(Vk+ph_)-FB,sen)[0]
                    nQh,nPh=np.linalg.norm(dQh),np.linalg.norm(dPh)
                    if nQh<1e-30 or nPh<1e-30: continue
                    bh=-spl(EH[k],sen)[0]; nbh=np.linalg.norm(bh)
                    ch=(float(np.dot(bh,dQh))/max(nbh*nQh,1e-30)) if nbh>1e-30 else 0.0
                    pe,qe=spl(T[k]-Vk,sen)
                    dQ=spl(fwdf(Vk+qe)-FB,sen)[0]; nQ=np.linalg.norm(dQ)
                    if nQ<1e-30: continue
                    held.append(dict(sys=f"FD/{nsen}sensors",kappa_hat=nQh/nPh,
                        cos_hat=ch,lam=nQ/nb,cos=float(np.dot(b,dQ))/(nb*nQ),
                        gain=nb/max(np.linalg.norm(b+dQ),1e-30),base=base))
            g=held[n0:]
            if g: print(f"  FD/{nsen:>2}sensors n={len(g):>3}"
                        f"  kappa_hat={np.median([x['kappa_hat'] for x in g]):9.3f}"
                        f"  lam_pred={np.median(fmap([x['kappa_hat'] for x in g],p,a)):.4f}"
                        f"  lam={np.median([x['lam'] for x in g]):.4f}"
                        f"  [{time.time()-t0:.0f}s]",flush=True)
        json.dump(rows+held,open("blind.json","w"),default=float)
        if not held: print("no held-out data"); return
        K=np.array([r["kappa_hat"] for r in held]); Lm=np.array([r["lam"] for r in held])
        CS=np.array([r["cos"] for r in held]); G=np.array([r["gain"] for r in held])
        LP=fmap(K,p,a)
        GP=1/np.sqrt(np.maximum(1+2*LP*CS+LP**2,1e-12))
        def sp(x,y):
            rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
            return float(np.corrcoef(rx,ry)[0,1])
        e1=float(np.median(np.abs(LP-Lm)))
        s2=sp(LP,Lm)
        e3=float(np.median(np.abs(np.log(np.maximum(GP,1e-9)/np.maximum(G,1e-9)))))
        const=C["ks_median_lambda"]
        e4=float(np.median(np.abs(const-Lm)))
        print("\n"+"="*76); print(f"BLIND PREDICTION SCORED, n={len(held)}"); print("="*76)
        print(f"\nB1 median |lam_pred - lam| = {e1:.4f}"
              f"  -> {'PASS' if e1<0.15 else 'FAIL'}")
        print(f"B2 Spearman(lam_pred, lam) = {s2:+.4f}"
              f"  -> {'PASS' if s2>0.7 else 'FAIL'}")
        print(f"B3 median |log(G_pred/G)| = {e3:.4f}"
              f"  -> {'PASS' if e3<0.35 else 'FAIL'}")
        print(f"\nB4 vs the trivial constant baseline (KS median lambda "
              f"= {const:.4f})")
        print(f"   calibrated map  median |err| = {e1:.4f}")
        print(f"   constant        median |err| = {e4:.4f}")
        print(f"   -> {'PASS — the calibration carries information' if e1<0.8*e4 else 'FAIL — a constant does as well; the map is not informative'}")

if __name__=="__main__": main()
