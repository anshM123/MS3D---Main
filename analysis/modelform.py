"""
DOES THE METHOD SURVIVE MODEL-FORM ERROR?

Every experiment reported so far generates the reference trajectory from the
SAME governing equations the reduced model projects. The only error is
reduction error. Real digital twins are not like that: they have the wrong
viscosity, a missing term, a mis-specified coefficient.

This matters because the defect

    eta = Phi_model(V) - V_next

then contains TWO things: reduction error, which is a state error and is
correctable, and model-form bias, which is a persistent difference between two
different equations and is NOT correctable by any change of state. A state
correction cannot fix a wrong equation.

TEST. Truth evolves under the true KS equation. The model — both the full map
inside the estimator AND the ROM — uses a DELIBERATELY WRONG equation:

    u_t = -(1+a) u u_x - (1+b) u_xx - (1+c) u_xxxx

with a single coefficient perturbed by delta. delta = 0 recovers everything
reported in the manuscript.

REGISTERED PREDICTIONS, written before running:
  M1  at delta = 0 the method reproduces the reported behaviour
  M2  the causal signature (+q beats -q) survives small model error, because
      the hidden state error is still real and still transported
  M3  the GAIN degrades with delta, because a growing fraction of the observable
      error is model-form bias that no state correction can remove
  M4  THE PREDICTOR REMAINS HONEST. G_hat should track the realised gain even
      as the gain falls — the method should KNOW it is helping less. If G_hat
      stays high while G collapses, the trust gate is blind to model-form
      error and that is a serious limitation to report.

M4 is the one that matters. A method that degrades gracefully and knows it is
degrading is deployable; one that degrades while reporting confidence is not.

Self-contained: numpy only, CPU.
"""
import numpy as np, os, json, time

class KS:
    """KS with per-term coefficient perturbations. c=(adv, diff, hyper)."""
    def __init__(self, L, N, dt=0.005, cf=(0.0,0.0,0.0)):
        self.L,self.N,self.dt = L,N,dt
        self.k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
        self.ca,self.cd,self.ch = 1+cf[0], 1+cf[1], 1+cf[2]
        self.lin = self.cd*self.k**2 - self.ch*self.k**4
        self.deal = np.abs(self.k) < (2/3)*self.k.max()
    def dealias(self,uh): return uh*self.deal
    def F(self,uh):
        u=np.fft.irfft(uh,n=self.N)
        return self.dealias(-0.5j*self.ca*self.k*np.fft.rfft(u*u)+self.lin*uh)
    def DF(self,uh,vh):
        u=np.fft.irfft(uh,n=self.N); v=np.fft.irfft(vh,n=self.N)
        return self.dealias(-1.0j*self.ca*self.k*np.fft.rfft(u*v)+self.lin*vh)
    def step(self,uh):
        dt=self.dt
        k1=self.F(uh);k2=self.F(uh+.5*dt*k1);k3=self.F(uh+.5*dt*k2);k4=self.F(uh+dt*k3)
        return self.dealias(uh+dt/6*(k1+2*k2+2*k3+k4))
    def phi(self,uh,n):
        for _ in range(n): uh=self.step(uh)
        return uh
    def ic(self,seed):
        rng=np.random.default_rng(seed); x=np.arange(self.N)*self.L/self.N
        u=0.1*rng.normal(size=self.N)
        for m in range(1,6):
            u+=(0.5/m)*np.cos(2*np.pi*m*x/self.L+rng.uniform(0,2*np.pi))
        return self.dealias(np.fft.rfft(u))

L=float(os.environ.get("MF_L",100.0)); NPM=int(os.environ.get("MF_NPM",6))
NTRAJ=int(os.environ.get("MF_NTRAJ",10)); NSTEP=int(os.environ.get("MF_NSTEP",50))
TERM=os.environ.get("MF_TERM","hyper")   # adv | diff | hyper
DELTAS=[float(x) for x in os.environ.get("MF_DELTAS","0.0,0.005,0.02,0.05,0.10").split(",")]
BAND=os.environ.get("MF_BAND","damped")
NSEG,FC,BURN=7,2,40000
ALPHAS=[0.0,0.5,1.0,1.5,2.0,3.0]

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

def main():
    N=int(2*L)
    truth_sys=KS(L=L,N=N,dt=0.005)                      # the TRUE equation
    lam=truth_sys.k**2-truth_sys.k**4
    md=([i for i in range(1,len(truth_sys.k)) if -35<=lam[i]<=-10][:NPM] if BAND=="damped"
        else [i for i in range(1,len(truth_sys.k)) if lam[i]>0][:NPM])
    idx={"adv":0,"diff":1,"hyper":2}[TERM]
    print(f"KS L={L:.0f}  P={BAND} ({len(md)} modes)  perturbed term: {TERM}")
    print(f"truth uses the exact equation; the MODEL (ROM and estimator) uses")
    print(f"a coefficient scaled by 1+delta\n",flush=True)
    rows=[]; t0=time.time()
    for dl in DELTAS:
        cf=[0.0,0.0,0.0]; cf[idx]=dl
        model=KS(L=L,N=N,dt=0.005,cf=tuple(cf))         # the WRONG equation
        n0=len(rows)
        for ti in range(NTRAJ):
            u=truth_sys.phi(truth_sys.ic(60000+ti),BURN)
            V=u.copy(); MO=[V.copy()]
            for _ in range(NSEG): V=seg(model,V,basis(model,V,md),NSTEP); MO.append(V.copy())
            T=[u.copy()]; t=u.copy()
            for _ in range(NSEG): t=truth_sys.phi(t,NSTEP); T.append(t.copy())
            # estimator sees ONLY the model's own (wrong) flow map
            E=np.zeros_like(MO[0]); EH=[E.copy()]
            for k in range(NSEG-FC):
                Phi=model.phi(MO[k],NSTEP); eta=Phi-MO[k+1]
                if rnorm(E)<1e-30: E=eta.copy()
                else:
                    ep=min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                    Pp=model.phi(MO[k]+ep*E,NSTEP); Mm=model.phi(MO[k]-ep*E,NSTEP)
                    E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fwd(x,n=FC):
                a=x.copy()
                for _ in range(n): a=seg(model,a,basis(model,a,md),NSTEP)
                return a
            for k in (3,4):
                if k>=len(EH) or k+FC>=len(T): continue
                Vk,TF=MO[k],T[k+FC]
                qh=splitM(EH[k],md)[1]
                rel=rnorm(qh)/max(rnorm(Vk),1e-30)
                if not np.isfinite(rel) or rel>0.5: continue
                FB=fwd(Vk); base=terr(FB,TF,md)
                if not (0.001<base<0.90): continue
                probe={a:splitM(fwd(Vk+a*qh)-FB,md)[0] for a in ALPHAS}
                Eh=EH[k].copy(); Vc=Vk.copy()
                for _ in range(FC):
                    Phi=model.phi(Vc,NSTEP); Vn=seg(model,Vc,basis(model,Vc,md),NSTEP)
                    eta=Phi-Vn
                    ep=min(max(1e-5*rnorm(Vc)/max(rnorm(Eh),1e-30),1e-6),5e-2)
                    Pp=model.phi(Vc+ep*Eh,NSTEP); Mm=model.phi(Vc-ep*Eh,NSTEP)
                    Eh=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep); Vc=Vn
                bh=-splitM(Eh,md)[0]
                ah=min(ALPHAS,key=lambda a:rnorm(bh+probe[a]))
                Gh=rnorm(bh)/max(rnorm(bh+probe[ah]),1e-30)
                G   = base/max(terr(fwd(Vk+ah*qh),TF,md),1e-30)
                Gneg= base/max(terr(fwd(Vk-ah*qh),TF,md),1e-30)
                rows.append(dict(delta=dl,base=base,G=G,G_neg=Gneg,G_hat=Gh,alpha=ah))
        g=[r for r in rows[n0:]]
        if g:
            G=np.array([x["G"] for x in g]); Gh=np.array([x["G_hat"] for x in g])
            print(f"  delta={dl:<6.3f} n={len(g):>3}  base={100*np.median([x['base'] for x in g]):6.2f}%"
                  f"  G={np.median(G):7.4f}  G_hat={np.median(Gh):7.4f}"
                  f"  +q>-q {int(sum(1 for x in g if x['G']>x['G_neg']))}/{len(g)}"
                  f"  [{time.time()-t0:.0f}s]",flush=True)
    if not rows: print("\nNo valid windows."); return
    json.dump(rows,open("modelform.json","w"),default=float)
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*78); print(f"MODEL-FORM ERROR, n={len(rows)}"); print("="*78)
    print(f"\n  {'delta':>8}{'n':>5}{'base %':>9}{'gain G':>10}{'G_hat':>10}"
          f"{'G_hat/G':>10}{'causal':>9}")
    med={}
    for dl in DELTAS:
        g=[r for r in rows if r["delta"]==dl]
        if not g: continue
        G=np.array([x["G"] for x in g]); Gh=np.array([x["G_hat"] for x in g])
        cz=sum(1 for x in g if x["G"]>x["G_neg"])
        med[dl]=(float(np.median(G)),float(np.median(Gh)))
        print(f"  {dl:>8.3f}{len(g):>5}{100*np.median([x['base'] for x in g]):>8.2f}%"
              f"{np.median(G):>10.4f}{np.median(Gh):>10.4f}"
              f"{np.median(Gh/np.maximum(G,1e-30)):>10.4f}{cz}/{len(g):<8}")
    G=np.array([r["G"] for r in rows]); Gh=np.array([r["G_hat"] for r in rows])
    print(f"\nM4 DOES THE PREDICTOR STAY HONEST UNDER MODEL ERROR?")
    print(f"   Spearman(G_hat, G) pooled across all delta = {sp(Gh,G):+.4f}")
    print(f"   median G_hat/G = {np.median(Gh/np.maximum(G,1e-30)):.4f}"
          f"   (1.0 = calibrated; >1 = overconfident)")
    if len(med)>=3:
        d0=min(med); dm=max(med)
        print(f"\n   at delta={d0:.3f}: G={med[d0][0]:.4f}  G_hat={med[d0][1]:.4f}")
        print(f"   at delta={dm:.3f}: G={med[dm][0]:.4f}  G_hat={med[dm][1]:.4f}")
        drop=med[dm][0]/max(med[d0][0],1e-9)
        hdrop=med[dm][1]/max(med[d0][1],1e-9)
        print(f"   gain retained {100*drop:.0f}%   predicted gain retained {100*hdrop:.0f}%")
        if abs(np.log(max(drop,1e-9))-np.log(max(hdrop,1e-9)))<0.35:
            print(f"\n   -> THE PREDICTOR DEGRADES WITH THE METHOD. It knows it is")
            print(f"      helping less. That is the deployable behaviour.")
        else:
            print(f"\n   -> WARNING: predicted and realised gain diverge under model")
            print(f"      error. The trust gate would fire on false confidence.")
            print(f"      This is a limitation that must be reported.")

if __name__=="__main__": main()
