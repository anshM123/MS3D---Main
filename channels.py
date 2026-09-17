"""
STATE-CONDITIONED CHANNEL OCCUPANCY — why the operator-only theories failed.

Six attempts to predict the effect size have failed, and every one of them
characterised the OPERATOR or the OBSERVABLE and not the STATE:

    alignment vs observable dimension   property of H
    advective locality                  property of H and the flow
    closed form in kappa                ratio of response norms
    transport range rho_T               property of K = P.DPhi.Q
    frozen cross-system calibration     a fit, boundary at H
    minimum-norm effort E*              property of K

The missing ingredient is which channels the NATURALLY OCCURRING hidden error
actually occupies. Write

    K = U Sigma W^T ,   a = W^T q ,   beta = U^T b

so that the observable response decomposes channel by channel,

    Delta = U Sigma a ,   Delta_i = sigma_i a_i

and the gain depends on how well Sigma a opposes beta. An operator with
excellent channels is useless if the error never enters them; a mediocre
operator is enough if the error lands in the few channels that transmit and
oppose.

This computes that decomposition per window and asks whether it explains what
six scalar summaries could not.

REGISTERED PREDICTIONS, written before running:
  A1  OCCUPANCY IS CONCENTRATED. The participation ratio of the normalised
      |sigma_i a_i| is below half the observable dimension, i.e. the response
      is carried by a few channels rather than spread evenly.
  A2  CHANNEL ALIGNMENT PREDICTS cos(theta). The predicted alignment
      cos_chan = -<Sigma a, beta>/(||Sigma a|| ||beta||) matches the measured
      cos theta to within 0.15 median absolute error.
  A3  IT PREDICTS THE EFFECT SIZE where the scalars did not. Spearman between
      the channel prediction of lambda, ||Sigma a||/||b||, and measured lambda
      exceeds 0.8. For reference the best previous operator-only quantity
      reached +0.19 in this same dataset.
  A4  TRUTH-FREE. Everything above computed from q_hat and b_hat rather than
      the true error reproduces A2 and A3 within 0.1 in the correlation.

A3 is the test. A1 and A2 are mechanism; A4 decides whether it is deployable.

If A3 fails, the linear joint state-operator picture is insufficient and the
answer lies in the finite-amplitude response manifold, not in any linear
decomposition.

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

def probe_K(ks,V,md,fwd,FB,nprobe,seed,seed_dirs=None):
    """Build K on a hidden subspace that ACTUALLY CONTAINS the error.

    TWO EARLIER DESIGNS FAILED, and both failures are instructive.

    Random probes: in a ~400-dimensional hidden space, 40 random directions
    capture about a tenth of the error, so decomposing q in that basis measures
    mostly what the probe missed. Participation ratio came out 1.00 and the
    reconstructed response was six times too small with the wrong sign.

    Seeding with the error itself: this captures q fully (A0 = 1.0000) but is
    CIRCULAR. Making q_hat the first probe direction makes it the dominant
    singular direction of K by construction, so "the error occupies one
    channel" describes the basis, not the physics. Participation ratio was
    again exactly 1.00.

    The requirement is a basis that is independent of any particular error yet
    still spans where leakage lives. The project measured that directly: the
    directions transporting hidden error into the observables sit at
    wavenumbers 4.5-8.2 across every topology and viscosity, set by the
    observable band and the energy-containing scales rather than the
    dissipation range. So probe on the LOWEST-|k| unresolved modes — fixed,
    physical, and chosen without reference to q."""
    rng=np.random.default_rng(seed)
    dirs=[]; cols=[]
    eps=1e-6*max(rnorm(V),1e-30)
    def add(q):
        for d in dirs: q=q-rdot(d,q)*d
        nq=rnorm(q)
        if nq<1e-10: return False
        q=q/nq; dirs.append(q)
        cols.append(pack(splitM(fwd(V+eps*q)-FB,md)[0],md)/eps)
        return True
    # fixed physical basis: the lowest-|k| modes outside the observable set
    order=np.argsort(np.abs(ks.k))
    for idx in order:
        if len(dirs)>=nprobe: break
        if idx in md or idx==0 or not ks.deal[idx]: continue
        for ph in (1.0,1.0j):
            if len(dirs)>=nprobe: break
            e=np.zeros(len(ks.k),complex); e[idx]=ph
            add(splitM(ks.dealias(e),md)[1])
    while len(dirs)<nprobe:
        r=rng.normal(size=len(ks.k))+1j*rng.normal(size=len(ks.k))
        r=ks.dealias(r)
        if not add(splitM(r,md)[1]): break
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
                seeds=[splitM(EH[j],md)[1] for j in range(1,k+1)]
                seeds=[x for x in seeds if rnorm(x)>1e-30]
                K,dirs=probe_K(ks,Vk,md,fwd,FB,NPROBE,1000+ti*13+k,seeds)
                if K is None: continue
                U_,S,Wt=np.linalg.svd(K,full_matrices=False)
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
                # ---- channel decomposition, true and truth-free ----
                def chan(qvec,bvec):
                    """coordinates of the hidden error in the probe basis"""
                    aq=np.array([rdot(d,qvec) for d in dirs])
                    a=Wt@aq                      # channel occupancy
                    beta=U_.T@bvec               # future error by channel
                    sa=S*a
                    nsa=np.linalg.norm(sa); nbe=np.linalg.norm(beta)
                    if nsa<1e-30 or nbe<1e-30: return None
                    w=np.abs(sa)/nsa
                    pr=1.0/max(float(np.sum(w**2)),1e-30)   # participation ratio
                    return dict(cos_chan=float(sa@beta)/(nsa*nbe),
                                lam_chan=nsa/max(np.linalg.norm(bvec),1e-30),
                                part_ratio=pr)
                def captured(v):
                    nv=rnorm(v)
                    if nv<1e-30: return np.nan
                    pr=sum(rdot(d,v)**2 for d in dirs)
                    return float(np.sqrt(max(pr,0.0))/nv)
                cap_true=captured(qe); cap_hat=captured(qh)
                ct=chan(qe,b_true)
                ch=chan(qh,b_hat) if nbh>1e-30 else None
                if ct is None: continue
                cos_meas=float(np.dot(pack(dQ,md),b_true))/max(
                    np.linalg.norm(pack(dQ,md))*nb,1e-30)
                rows.append(dict(band=bn,Estar=Es_true,Estar_hat=Es_hat,
                    cos_chan=ct["cos_chan"],lam_chan=ct["lam_chan"],
                    part_ratio=ct["part_ratio"],cos_meas=cos_meas,
                    captured=cap_true,captured_hat=cap_hat,
                    cos_chan_hat=(ch["cos_chan"] if ch else np.nan),
                    lam_chan_hat=(ch["lam_chan"] if ch else np.nan),
                    obs_dim=len(S),
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
    json.dump(rows,open("channels.json","w"),default=float)
    A=lambda k: np.array([r[k] for r in rows],dtype=float)
    med={b:1 for b in BANDS if any(r["band"]==b for r in rows)}
    def sp(x,y):
        m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
        if len(x)<4: return np.nan
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    med={b:np.median([r["Estar"] for r in rows if r["band"]==b])
         for b in BANDS if any(r["band"]==b for r in rows)}
    spread=max(med.values())/max(min(med.values()),1e-300)
    print("\n"+"="*80); print(f"CHANNEL OCCUPANCY, n={len(rows)}"); print("="*80)
    cp=A('captured')
    print(f"\nA0 fraction of the true hidden error inside the probe subspace:"
          f" median {np.median(cp):.4f}")
    print(f"   (a random probe basis captured ~0.1; below ~0.5 the channel"
          f" decomposition is not measuring the error)")
    pr=A('part_ratio'); od=float(np.median(A('obs_dim')))
    print(f"\nA1 occupancy concentrated: median participation ratio "
          f"{np.median(pr):.2f} of {od:.0f} channels"
          f"  -> {'PASS' if np.median(pr)<od/2 else 'FAIL'}")
    e2=float(np.median(np.abs(A('cos_chan')-A('cos_meas'))))
    print(f"A2 channel alignment vs measured cos: median |err| {e2:.4f}"
          f"  -> {'PASS' if e2<0.15 else 'FAIL'}")
    s3=sp(A('lam_chan'),A('lam'))
    print(f"A3 Spearman(channel lambda, measured lambda) = {s3:+.4f}"
          f"  -> {'PASS' if s3>0.8 else 'FAIL'}")
    print(f"   for reference, operator-only E* reached -0.19 on this dataset")
    s4a=sp(A('lam_chan_hat'),A('lam')); s4b=sp(A('cos_chan_hat'),A('cos_meas'))
    print(f"A4 truth-free: Spearman(lambda) {s4a:+.4f}   Spearman(cos) {s4b:+.4f}"
          f"  -> {'PASS' if (np.isfinite(s4a) and abs(s4a-s3)<0.1) else 'FAIL'}")
    print(f"\n  {'band':<10}{'part.ratio':>12}{'cos_chan':>10}{'cos_meas':>10}"
          f"{'lam_chan':>10}{'lambda':>9}{'gain':>9}")
    for b in med:
        g=[r for r in rows if r["band"]==b]
        print(f"  {b:<10}{np.median([x['part_ratio'] for x in g]):>12.2f}"
              f"{np.median([x['cos_chan'] for x in g]):>+10.4f}"
              f"{np.median([x['cos_meas'] for x in g]):>+10.4f}"
              f"{np.median([x['lam_chan'] for x in g]):>10.4f}"
              f"{np.median([x['lam'] for x in g]):>9.4f}"
              f"{np.median([x['gain'] for x in g]):>9.3f}")

if __name__=="__main__": main()
