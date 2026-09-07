"""
ITEM 3 — BASELINES WITH FAIR INFORMATION BUDGETS.  CPU, pure numpy.

"Uncorrected vs ours" is not a defensible comparison. This runs six methods on
identical trajectories and reports accuracy, wall-clock, AND what external
information each one consumed — that last axis is where a truth-free method
should look good, and it is usually left out.

  A  uncorrected ROM              the do-nothing baseline
  B  bigger basis, no correction  "why not just enlarge the ROM?"  matched by
                                  cost, not by rank
  C  full-state defect correction V += e_hat, corrects P AND Q. NOT
                                  observable-preserving — it moves the very
                                  quantity being scored. An upper bound that
                                  violates the constraint, not a competitor.
  D  P-only correction            V += P e_hat. Isolates whether the HIDDEN
                                  part is doing the work.
  E  Q-only correction (ours)     V += Q e_hat, P(V+c) = PV exactly
  F  ENSEMBLE 3DVar               a proper analysis, not a nudge. Background
                                  covariance B estimated from the model's own
                                  error-snapshot ensemble; the analysis
                                  increment is
                                    dx = B H^T (H B H^T + R)^-1 (y - H x_b)
                                  With B low-rank = (1/M) E E^T this is exact
                                  and cheap. Crucially 3DVar corrects Q THROUGH
                                  the P observation via the B cross-covariance,
                                  so it is a real competitor. It CONSUMES TRUTH
                                  observations of P at the analysis time.

The honest question is not "does E beat C" — C sees more. It is: how close does
E get to C while changing nothing observable and reading no external data?
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS

L=float(os.environ.get("BASE_L",100.0)); NPM=int(os.environ.get("BASE_NPM",6))
BAND=os.environ.get("BASE_BAND","damped"); NTRAJ=int(os.environ.get("BASE_NTRAJ",30))
R_OBS=float(os.environ.get("BASE_ROBS",1e-6))   # obs error variance, relative
NSEG,FC,BURN=7,2,40000

def splitM(u,m):
    P=np.zeros_like(u); P[m]=u[m]; return P,u-P
def basis(ks,u,m,extra=2):
    B=[]
    for mm in m:
        for ph in (0,1):
            e=np.zeros_like(u); e[mm]=1.0 if ph==0 else 1.0j
            B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B: v=v-np.vdot(b,v)*b
        n=np.linalg.norm(v); return v/n if n>1e-12 else None
    v=ks.F(u)
    for _ in range(extra):
        q=orth(splitM(v,m)[1])
        if q is None: break
        B.append(q); v=ks.DF(u,v)
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
def fc(ks,x,m,ns,ex=2,n=FC):
    a=x.copy()
    for _ in range(n): a=seg(ks,a,basis(ks,a,m,ex),ns)
    return a
def terr(A,T,m):
    return np.linalg.norm(splitM(A-T,m)[0])/np.linalg.norm(splitM(T,m)[0])

def main():
    N=int(2*L); ks=KS(L=L,N=N,dt=0.005); ns=50; lam=ks.k**2-ks.k**4
    md=([i for i in range(1,len(ks.k)) if -35<=lam[i]<=-10][:NPM] if BAND=="damped"
        else [i for i in range(1,len(ks.k)) if lam[i]>0][:NPM])
    print(f"KS L={L:.0f} P={BAND} ({len(md)} modes)  3DVar R_obs={R_OBS:.1e}",
          flush=True)
    R={k:[] for k in "ABCDEF"}; TM={k:0.0 for k in "ABCDEF"}; rows=[]
    t0=time.time()
    for ti in range(NTRAJ):
        u=ks.phi(ks.ic(30000+ti),BURN)
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
            obs=splitM(T[k],md)[0]                       # truth, only F may use it
            # ---- ENSEMBLE 3DVar analysis ----
            # B from the model's own accumulated error snapshots (offline-style)
            ens=[splitM(T[j]-MO[j],md)[0]+splitM(T[j]-MO[j],md)[1]
                 for j in range(1,k+1)]
            if len(ens)>=2:
                Em=sum(ens)/len(ens); Ec=[e-Em for e in ens]
                M=len(Ec)
                HE=np.array([splitM(e,md)[0][md] for e in Ec]).T     # (|P|, M)
                d=(obs-splitM(Vk,md)[0])[md]
                S=HE.conj().T@HE + M*R_OBS*np.eye(M)*max(
                    float(np.real(np.vdot(d,d))),1e-30)
                try:
                    w=np.linalg.solve(S, HE.conj().T@d)
                    dx=sum(w[i]*Ec[i] for i in range(M))
                except np.linalg.LinAlgError:
                    dx=np.zeros_like(Vk)
            else:
                dx=np.zeros_like(Vk)
            runs={"A":(Vk,2),"B":(Vk,8),"C":(Vk+eh,2),"D":(Vk+ph,2),
                  "E":(Vk+qh,2),
                  "F":(Vk+dx,2)}
            out={}
            for key,(st,ex) in runs.items():
                tA=time.time(); Fo=fc(ks,st,md,ns,ex); TM[key]+=time.time()-tA
                out[key]=terr(Fo,TF,md)
            base=out["A"]
            for key in "ABCDEF": R[key].append(base/max(out[key],1e-30))
            rows.append({k:float(out[k]) for k in out})
        if (ti+1)%10==0:
            print(f"  traj {ti+1}/{NTRAJ}  n={len(R['A'])}  [{time.time()-t0:.0f}s]",
                  flush=True)
    json.dump(rows,open(f"base_{BAND}.json","w"),default=float)
    NM={"A":"uncorrected ROM","B":"bigger basis (rank+6)",
        "C":"full-state defect  (moves P!)","D":"P-only correction",
        "E":"Q-only, OURS (P unchanged)","F":"ensemble 3DVar (uses truth obs)"}
    INFO={"A":"none","B":"none","C":"none","D":"none","E":"none",
          "F":f"{len(md)} obs modes"}
    PRES={"A":"n/a","B":"n/a","C":"NO","D":"NO","E":"YES","F":"NO"}
    n=len(R["A"]); tot=sum(TM.values())
    print("\n"+"="*94); print(f"BASELINES — KS {BAND}, n={n} windows"); print("="*94)
    print(f"  {'method':<34}{'median gain':>12}{'p10':>9}{'win%':>7}"
          f"{'rel cost':>10}{'P kept':>8}{'ext info':>18}")
    for key in "ABCDEF":
        v=np.array(R[key])
        print(f"  {NM[key]:<34}{np.median(v):>12.4f}{np.percentile(v,10):>9.4f}"
              f"{100*np.mean(v>1.0):>6.0f}%{TM[key]/max(TM['A'],1e-9):>10.2f}x"
              f"{PRES[key]:>8}{INFO[key]:>18}")
    e=np.median(R["E"]); c=np.median(R["C"]); f_=np.median(R["F"])
    print(f"\n  Q-only reaches {100*(e-1)/max(c-1,1e-9):.1f}% of the full-state")
    print(f"  correction's benefit while leaving every observable coordinate")
    print(f"  unchanged and reading no external data.")
    print(f"\n  vs ENSEMBLE 3DVar, which observes P truth at the analysis time:")
    print(f"    3DVar median gain {f_:.4f}   ours {e:.4f}")
    print(f"    ours/3DVar benefit ratio {(e-1)/max(f_-1,1e-9):.2f}x"
          if f_>1.001 else
          f"    3DVar gained <0.1%; quote the raw gains, not a ratio")
    print(f"  vs simply enlarging the basis: {(e-1)/max(np.median(R['B'])-1,1e-9):.2f}x")

if __name__=="__main__": main()
