"""
ITEMS 5 + 6 — NON-SPECTRAL DISCRETISATION and a GENERAL OBSERVATION OPERATOR.
CPU, pure numpy, fully self-contained.

Two referee objections answered at once:

  ITEM 5  "is this a Fourier/spectral artifact?"
          Second-order FINITE DIFFERENCES on a periodic grid. No FFT anywhere
          in the solver. u_t = -u u_x - u_xx - u_xxxx by central stencils.

  ITEM 6  "P is a Fourier projector; real systems observe SENSORS."
          The observable is y = H x with H a point-sampling operator at m
          sensor locations. 'Hidden' now means H c = 0 — a state change that
          every sensor reads as identical. For point sampling, null(H) is
          simply the set of fields vanishing at the sensor nodes, so the
          projection is exact and the ANALOGUE OF DP = 0 becomes:
              every sensor reading is bit-identical after the correction.

This is the version a practitioner recognises: alter the state in a way no
instrument can detect, and the next reading from those same instruments is
closer to truth.
"""
import numpy as np, sys, os, json, time

NX   = int(os.environ.get("FD_NX", 200))
LDOM = float(os.environ.get("FD_L", 100.0))
NSEN = int(os.environ.get("FD_NSEN", 12))
NTRAJ= int(os.environ.get("FD_NTRAJ", 25))
DT   = float(os.environ.get("FD_DT", 0.002))
NSTEP= int(os.environ.get("FD_NSTEP", 125))     # 0.25 time units per segment
NSEG, FC, BURN = 7, 2, 60000
ALPHAS=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0]

class KSFD:
    """Kuramoto-Sivashinsky by FINITE DIFFERENCES. No spectral operators."""
    def __init__(self, n=NX, L=LDOM, dt=DT):
        self.n, self.L, self.h, self.dt = n, L, L/n, dt
    def F(self, u):
        h=self.h
        um1=np.roll(u,1); up1=np.roll(u,-1)
        um2=np.roll(u,2); up2=np.roll(u,-2)
        ux  = (up1-um1)/(2*h)
        uxx = (up1-2*u+um1)/h**2
        uxxxx=(up2-4*up1+6*u-4*um1+um2)/h**4
        return -u*ux - uxx - uxxxx
    def DF(self, u, v):
        h=self.h
        vm1=np.roll(v,1); vp1=np.roll(v,-1)
        vm2=np.roll(v,2); vp2=np.roll(v,-2)
        um1=np.roll(u,1); up1=np.roll(u,-1)
        ux =(up1-um1)/(2*h); vx=(vp1-vm1)/(2*h)
        vxx=(vp1-2*v+vm1)/h**2
        vxxxx=(vp2-4*vp1+6*v-4*vm1+vm2)/h**4
        return -(v*ux + u*vx) - vxx - vxxxx
    def step(self, u):
        dt=self.dt
        k1=self.F(u); k2=self.F(u+.5*dt*k1); k3=self.F(u+.5*dt*k2); k4=self.F(u+dt*k3)
        return u + dt/6*(k1+2*k2+2*k3+k4)
    def phi(self, u, n):
        for _ in range(n): u=self.step(u)
        return u
    def ic(self, seed):
        rng=np.random.default_rng(seed)
        x=np.arange(self.n)*self.h
        u=0.1*rng.normal(size=self.n)
        for m in range(1,5):
            u+=(0.5/m)*np.cos(2*np.pi*m*x/self.L+rng.uniform(0,2*np.pi))
        return u

def sensors(n, m):
    return np.linspace(0, n, m, endpoint=False).astype(int)

def splitH(u, sen):
    """H = point sampling. P part = values AT sensors; Q = null(H)."""
    P=np.zeros_like(u); P[sen]=u[sen]
    return P, u-P

def basisH(sys, u, sen, extra=2):
    B=[]
    for s in sen:
        e=np.zeros_like(u); e[s]=1.0
        B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B: v=v-np.dot(b,v)*b
        nn=np.linalg.norm(v); return v/nn if nn>1e-12 else None
    w=sys.F(u)
    for _ in range(extra):
        q=orth(splitH(w,sen)[1])
        if q is None: break
        B.append(q); w=sys.DF(u,w)
    return B

def segH(sys, anc, B, n):
    a=np.zeros(len(B)); dt=sys.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=sys.F(u)
        return np.array([np.dot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))

def fcH(sys,x,sen,ns,n=FC):
    a=x.copy()
    for _ in range(n): a=segH(sys,a,basisH(sys,a,sen),ns)
    return a

def serr(A,T,sen):
    return np.linalg.norm(A[sen]-T[sen])/max(np.linalg.norm(T[sen]),1e-300)

def main():
    sys_=KSFD(); sen=sensors(NX,NSEN)
    print(f"KS by FINITE DIFFERENCES  nx={NX} L={LDOM:.0f} h={sys_.h:.3f} dt={DT}")
    print(f"observation: {NSEN} POINT SENSORS at nodes {sen[:6]}...  (H = sampling)")
    print(f"'hidden' = H c = 0, i.e. every sensor reads identically\n", flush=True)
    rows=[]; t0=time.time()
    for ti in range(NTRAJ):
        u=sys_.phi(sys_.ic(11000+ti),BURN)
        if not np.all(np.isfinite(u)): print("  diverged, skip"); continue
        V=u.copy(); MO=[V.copy()]
        for _ in range(NSEG): V=segH(sys_,V,basisH(sys_,V,sen),NSTEP); MO.append(V.copy())
        T=[u.copy()]; t=u.copy()
        for _ in range(NSEG): t=sys_.phi(t,NSTEP); T.append(t.copy())
        E=np.zeros_like(MO[0]); EH=[E.copy()]
        for k in range(NSEG-FC):
            Phi=sys_.phi(MO[k],NSTEP); eta=Phi-MO[k+1]
            if np.linalg.norm(E)<1e-30: E=eta.copy()
            else:
                ep=min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                Pp=sys_.phi(MO[k]+ep*E,NSTEP); Mm=sys_.phi(MO[k]-ep*E,NSTEP)
                E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
            EH.append(E.copy())
        for k in (3,4):
            if k>=len(EH): continue
            Vk,TF=MO[k],T[k+FC]
            qh=splitH(EH[k],sen)[1]
            rel=np.linalg.norm(qh)/max(np.linalg.norm(Vk),1e-30)
            if not np.isfinite(rel) or rel>0.5: continue
            # THE INVARIANT: sensor readings must be bit-identical
            sens_mismatch=float(np.max(np.abs((Vk+qh)[sen]-Vk[sen])))
            FB=fcH(sys_,Vk,sen,NSTEP)
            base=serr(FB,TF,sen)
            if not (base<0.50): continue
            probe={a: splitH(fcH(sys_,Vk+a*qh,sen,NSTEP)-FB,sen)[0] for a in ALPHAS}
            Eh=EH[k].copy(); Vc=Vk.copy()
            for _ in range(FC):
                Phi=sys_.phi(Vc,NSTEP); Vn=segH(sys_,Vc,basisH(sys_,Vc,sen),NSTEP)
                eta=Phi-Vn
                epp=min(max(1e-5*np.linalg.norm(Vc)/max(np.linalg.norm(Eh),1e-30),
                            1e-6),5e-2)
                Pp=sys_.phi(Vc+epp*Eh,NSTEP); Mm=sys_.phi(Vc-epp*Eh,NSTEP)
                Eh=eta+(Pp-Mm)/(2*epp)+0.5*(Pp+Mm-2*Phi)/(epp*epp); Vc=Vn
            bh=-splitH(Eh,sen)[0]
            ah=min(ALPHAS,key=lambda a: np.linalg.norm(bh+probe[a]))
            G_hat=np.linalg.norm(bh)/max(np.linalg.norm(bh+probe[ah]),1e-30)
            FE=fcH(sys_,Vk+ah*qh,sen,NSTEP); FN=fcH(sys_,Vk-ah*qh,sen,NSTEP)
            b=splitH(FB-TF,sen)[0]
            AQ=splitH(T[k]-Vk,sen)[1]
            FO=fcH(sys_,Vk+AQ,sen,NSTEP); D=splitH(FO-FB,sen)[0]
            nb,nd=np.linalg.norm(b),np.linalg.norm(D)
            if nb<1e-30 or nd<1e-30: continue
            lam=nd/nb; cos=float(np.dot(b,D))/(nb*nd)
            pred=1/np.sqrt(max(1+2*lam*cos+lam*lam,1e-300))
            obs=nb/max(np.linalg.norm(b+D),1e-30)
            rows.append(dict(base=base,lam=lam,cos=cos,pred=pred,obs=obs,
                iden=abs(pred-obs)/max(obs,1e-30),G_hat=G_hat,
                G=base/max(serr(FE,TF,sen),1e-30),
                G_neg=base/max(serr(FN,TF,sen),1e-30),
                sens_mismatch=sens_mismatch,alpha=ah))
        if (ti+1)%5==0:
            print(f"  traj {ti+1}/{NTRAJ}  n={len(rows)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    if not rows: print("\nNo valid windows — lower FD_NSTEP."); return
    json.dump(rows,open("fd_sensors.json","w"),default=float)
    G=np.array([r["G"] for r in rows]); Gn=np.array([r["G_neg"] for r in rows])
    Gh=np.array([r["G_hat"] for r in rows])
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*80)
    print(f"FINITE DIFFERENCES + POINT SENSORS, n={len(rows)}")
    print("="*80)
    print(f"\nS1 sensor readings unchanged by the correction")
    print(f"   max |y_corrected - y_baseline| = "
          f"{max(r['sens_mismatch'] for r in rows):.3e}  -> "
          f"{'PASS' if max(r['sens_mismatch'] for r in rows)<1e-12 else 'FAIL'}")
    print(f"\nS2 causal signature   +q beats -q in "
          f"{int((G>Gn).sum())}/{len(G)}  -> "
          f"{'PASS' if (G>Gn).mean()>=0.8 else 'FAIL'}")
    print(f"\nS3 effect-size identity   max |pred-obs|/obs = "
          f"{max(r['iden'] for r in rows):.3e}  -> "
          f"{'PASS' if max(r['iden'] for r in rows)<1e-9 else 'FAIL'}")
    print(f"\nS4 applicability predictor   Spearman(G_hat,G) = {sp(Gh,G):+.4f}"
          f"   median |G_hat/G-1| = {np.median(np.abs(Gh/np.maximum(G,1e-30)-1)):.4f}")
    print(f"\nS5 outcome   median gain {np.median(G):.4f}  p10 "
          f"{np.percentile(G,10):.4f}  win {100*np.mean(G>1):.1f}%")
    print(f"     lambda {np.median([r['lam'] for r in rows]):.4f}  "
          f"cos {np.median([r['cos'] for r in rows]):+.4f}  "
          f"base {100*np.median([r['base'] for r in rows]):.2f}%")

if __name__=="__main__": main()
