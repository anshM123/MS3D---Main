# ============================================================================
# M3D TEST BB3 — SELF-CALIBRATING BROADBAND TEST (rank ceiling + nu range fixed)
# ============================================================================
# Stage 1  CALIBRATE.  Cheaply probe (band, nu) pairs -- truth trajectory and
#          POD rank only, no ROM, no W, no leakage -- to find operating points
#          that actually produce high-dimensional dynamics. Also checks the
#          flow stays RESOLVED, so a "high rank" that is really aliasing noise
#          gets rejected instead of celebrated.
# Stage 2  SELECT.  Build a ladder of configs spanning the achieved ranks.
# Stage 3  RUN.  Full chain on each: Q reservoir -> leakage rank -> W -> X,
#          with intervention times chosen adaptively so the ROM error is in a
#          workable band at every anchor.
#
# Headline: d(log r_leak)/d(log r_trajectory).
#   < 0.25  leakage rank nearly invariant while state rank explodes  (dream)
#   < 0.75  sublinear growth                                         (still good)
#   ~ 1.0   leakage rank tracks state rank                           (bad)
#
# Plus the triadic enrichment: are the dangerous hidden modes the effective
# triadic partners of the resolved band rather than the energetic ones?
# ============================================================================
import os, glob, math, json
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT  = SAVE / "m3d_BB3_broadband.csv"
CAL  = SAVE / "m3d_BB3_calibration.csv"
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT, flush=True)
G = {"__name__": "__m3d__"}
exec(compile(open(BOOT).read(), BOOT, "exec"), G, G)

def spaces():
    out, seen = [G], {id(G)}
    for _, o in list(G.items()):
        if callable(o) and hasattr(o,"__globals__") and id(o.__globals__) not in seen:
            out.append(o.__globals__); seen.add(id(o.__globals__))
    return out
def setup(nu):
    for ns in spaces():
        ns["NU1_D"]=ns["NU2_D"]=ns["nu1_D"]=ns["nu2_D"]=float(nu)
        ns["REAL_DTYPE_D"]=torch.float64; ns["COMPLEX_DTYPE_D"]=torch.complex128

make_env=G["make_env_D"]; topo=G["topology_pair_L2"]
full_rk4=G["full_rk4_D"]; init2=G["initialize_star2_N"]
rk4M=G["rk4_M"]; recon=G["reconstruct_M"]; packm=G["pack_modes_D"]
dealias=G["dealias_D"]; leray=G["project_divfree_D"]; TAU0=float(G["TAU0_D"])

L_BOX, SEG_TAU, SEG_STEPS = 8.0, 0.15, 20
DT = SEG_TAU*TAU0/SEG_STEPS
NRES   = int(os.environ.get("M3D_BB_N", 64))
N_SEG  = 10
FORECAST_SEGS, DERIV_REL = 2, 1e-5
TARGET_RANKS = [15, 40, 80, 150]        # ladder we try to hit
K_DEALIAS = NRES/3.0                     # 2/3 rule
CAND_KHI = [4, 8, 12, 16]
CAND_AMP = [1.0, 0.35]   # fraction of the reference norm
# BB2 BUG: all candidate viscosities were too LOW. With no forcing the band
# cascades to the grid scale and nu must be large enough to dissipate it there
# (tail energy hit 0.62 at k_hi=18). The admissible nu is bounded BELOW by
# resolution. Grid now spans two decades upward.
CAND_NU  = [4.0e-3, 1.6e-3, 6.4e-4, 2.5e-4, 1.0e-4]

def redot(a1,a2,b1,b2):
    return float((torch.real(torch.vdot(a1.reshape(-1),b1.reshape(-1)))
                 +torch.real(torch.vdot(a2.reshape(-1),b2.reshape(-1)))).cpu())
def pnorm(a1,a2): return math.sqrt(max(redot(a1,a2,a1,a2),0.0))
def nrm(U): return float(torch.sqrt(torch.real(torch.vdot(U.reshape(-1),U.reshape(-1)))).cpu())
def splitPQ(A1,A2,t):
    P1=torch.zeros_like(A1); P2=torch.zeros_like(A2)
    P1.reshape(3,-1)[:,t]=A1.reshape(3,-1)[:,t]; P2.reshape(3,-1)[:,t]=A2.reshape(3,-1)[:,t]
    return P1,P2,A1-P1,A2-P2
def phi(A1,A2,env,nu):
    B1,B2=A1.clone(),A2.clone()
    for _ in range(SEG_STEPS):
        B1=full_rk4(B1,DT,nu,env); B2=full_rk4(B2,DT,nu,env)
    return B1,B2
def terr(A1,A2,T1,T2,t):
    a=packm(A1-T1,A2-T2,t); b=packm(T1,T2,t)
    return float((torch.linalg.vector_norm(a)/torch.linalg.vector_norm(b)).real.cpu())

def broadband_field(N,k_lo,k_hi,seed,env):
    rng=np.random.default_rng(seed)
    uh=np.fft.fftn(rng.normal(size=(3,N,N,N)),axes=(1,2,3))
    f=np.fft.fftfreq(N)*N
    KX,KY,KZ=np.meshgrid(f,f,f,indexing="ij")
    km=np.sqrt(KX**2+KY**2+KZ**2)
    uh=uh/(np.abs(uh)+1e-300)*((km>=k_lo)&(km<=k_hi)).astype(float)
    U=torch.tensor(uh,dtype=torch.complex128,device="cuda")
    return dealias(leray(U,env),env)

def broadband_pair(N,k_lo,k_hi,env,ref):
    U1=broadband_field(N,k_lo,k_hi,1234,env); U2=broadband_field(N,k_lo,k_hi,5678,env)
    return U1*(ref/max(nrm(U1),1e-30)), U2*(ref/max(nrm(U2),1e-30))

def pod_ev(S1,S2):
    n=len(S1)
    m1=sum(S1[1:],S1[0].clone())/n; m2=sum(S2[1:],S2[0].clone())/n
    F1=[s-m1 for s in S1]; F2=[s-m2 for s in S2]
    Gm=torch.zeros(n,n,dtype=torch.float64)
    for i in range(n):
        for j in range(i,n):
            v=redot(F1[i],F2[i],F1[j],F2[j]); Gm[i,j]=v; Gm[j,i]=v
    return torch.linalg.eigvalsh(Gm).flip(0).clamp(min=0)
def modes_for(ev,frac):
    tot=float(ev.sum()); c=0.0
    for i,e in enumerate(ev):
        c+=float(e)
        if c>=frac*tot: return i+1
    return len(ev)
def tail_fraction(U,N):
    """energy above 0.8 * the dealiasing cutoff -- the under-resolution tell."""
    uh=U.detach().cpu().numpy(); e=(np.abs(uh)**2).sum(axis=0)
    f=np.fft.fftfreq(N)*N
    KX,KY,KZ=np.meshgrid(f,f,f,indexing="ij")
    km=np.sqrt(KX**2+KY**2+KZ**2)
    return float(e[km>0.8*K_DEALIAS].sum()/max(e.sum(),1e-300))

# ---------------------------------------------------------------- reference
setup(1.6e-4); env0=make_env(NRES,L=L_BOX)
R1,_=topo("skew_tubes",env0); REF=nrm(R1); del R1,env0
print(f"reference |U| = {REF:.6e}", flush=True)

# ============================== STAGE 1: CALIBRATE ==========================
# BB2 BUG: POD_SNAPS=11 capped the Gram matrix at 11x11, so measured rank
# could never exceed 11 and every config returned 8-10. The ceiling must sit
# well above any rank we care about. Test R's 33 snapshots may likewise have
# truncated its rank-27 figure.
POD_SNAPS = int(os.environ.get("M3D_BB_SNAPS", 81))
def calibrate(k_hi, nu, amp=1.0):
    setup(nu); env=make_env(NRES,L=L_BOX)
    U1,U2=broadband_pair(NRES,1,k_hi,env,REF*amp)
    t1,t2=U1.clone(),U2.clone(); S1,S2=[t1.clone()],[t2.clone()]
    every=max(1,(N_SEG*SEG_STEPS)//(POD_SNAPS-1))
    for st in range(N_SEG*SEG_STEPS):
        t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
        if (st+1)%every==0 and len(S1)<POD_SNAPS:
            S1.append(t1.clone()); S2.append(t2.clone())
    ev=pod_ev(S1,S2)
    r=int((ev>ev[0]*1e-9).sum()); m999=modes_for(ev,0.999)
    tail=tail_fraction(t1,NRES); ceil=len(S1)
    del env,U1,U2,t1,t2,S1,S2; torch.cuda.empty_cache()
    return r,m999,tail,ceil

cal=[]
if CAL.exists():
    cal=pd.read_csv(CAL).to_dict("records")
    print(f"recovered {len(cal)} calibration rows", flush=True)
else:
    print("\n"+"="*88); print("STAGE 1 — CALIBRATION"); print("="*88, flush=True)
    print(f"  snapshot ceiling on measurable rank: {POD_SNAPS}")
    print(f"  {'k_hi':>5}{'amp':>6}{'nu':>11}{'r_traj':>9}{'99.9%':>8}"
          f"{'tail':>11}{'resolved':>10}{'saturated':>11}")
    for k_hi in CAND_KHI:
        for amp in CAND_AMP:
            for nu in CAND_NU:
                try:
                    r,m9,tail,ceil=calibrate(k_hi,nu,amp)
                except Exception as exc:
                    print(f"  {k_hi:>5}{amp:>6.2f}{nu:>11.2e}  FAILED {exc!r}",
                          flush=True); continue
                okres = tail < 1e-4
                sat  = r >= ceil-2
                cal.append(dict(k_hi=k_hi,amp=amp,nu=nu,r_traj=r,modes_999=m9,
                                tail_frac=tail,resolved=bool(okres),
                                rank_ceiling=ceil,saturated=bool(sat)))
                print(f"  {k_hi:>5}{amp:>6.2f}{nu:>11.2e}{r:>9}{m9:>8}{tail:>11.2e}"
                      f"{'yes' if okres else 'NO':>10}"
                      f"{'SATURATED' if sat else '':>11}", flush=True)
                pd.DataFrame(cal).to_csv(CAL,index=False)

cdf=pd.DataFrame(cal)
if "saturated" in cdf.columns and cdf.saturated.any():
    print(f"\n  WARNING: {int(cdf.saturated.sum())} configs hit the snapshot"
          f" ceiling of {POD_SNAPS}; their ranks are lower bounds.")
    print("  Raise M3D_BB_SNAPS and recalibrate before trusting those rows.")
res=cdf[cdf.resolved] if cdf.resolved.any() else cdf
print(f"\n  achieved trajectory rank: {int(res.r_traj.min())} .. {int(res.r_traj.max())}"
      f"   (existing smooth families: 12-27)")
if int(res.r_traj.max()) < 50:
    print("  WARNING: even the widest band / lowest nu stays below rank 50.")
    print("  The testbed cannot be made high-dimensional at this N. Raise")
    print("  M3D_BB_N, or accept that the flow family is intrinsically simple.")

# ============================== STAGE 2: SELECT =============================
sel=[]
for tgt in TARGET_RANKS:
    c=res.iloc[(res.r_traj-tgt).abs().argsort()].iloc[0]
    key=(int(c.k_hi),float(c.nu),float(getattr(c,"amp",1.0)))
    if key not in [(int(s.k_hi),float(s.nu),float(getattr(s,"amp",1.0))) for s in sel]:
        sel.append(c)
sel=sorted(sel,key=lambda c:c.r_traj)
print("\n"+"="*88); print("STAGE 2 — SELECTED LADDER"); print("="*88)
for c in sel:
    print(f"  k_hi={int(c.k_hi):>3}  amp={float(getattr(c,'amp',1.0)):.2f}  "
          f"nu={c.nu:.2e}  r_traj={int(c.r_traj):>4}  99.9%={int(c.modes_999):>4}  "
          f"resolved={'yes' if c.resolved else 'NO'}")

# ============================== STAGE 3: FULL CHAIN =========================
rows=[]; prev=pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done={(r["k_hi"],r["nu"]) for r in prev}; rows.extend(prev)
print("\n"+"="*88); print("STAGE 3 — FULL CHAIN"); print("="*88, flush=True)

for c in sel:
    k_hi, nu, amp = int(c.k_hi), float(c.nu), float(getattr(c,"amp",1.0))
    if (k_hi,nu) in done:
        print(f"SKIP k_hi={k_hi} nu={nu:.2e}", flush=True); continue
    print(f"\n=== band [1,{k_hi}]  nu={nu:.2e}  (calibrated r_traj={int(c.r_traj)}) ===",
          flush=True)
    setup(nu); env=make_env(NRES,L=L_BOX)
    U1,U2=broadband_pair(NRES,1,k_hi,env,REF*amp)
    star,_=init2(U1,U2,env); tids=star["proj"]["target_ids"]; del star

    t1,t2=U1.clone(),U2.clone(); m,_=init2(U1,U2,env)
    V1,V2=U1.clone(),U2.clone()
    S1,S2=[t1.clone()],[t2.clone()]
    TR,MO=[(t1.clone(),t2.clone())],[(V1.clone(),V2.clone())]
    for seg in range(1,N_SEG+1):
        for _ in range(SEG_STEPS):
            t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
            rk4M(m,SEG_TAU/SEG_STEPS,env)
        V1,V2=recon(m,env); m,_=init2(V1,V2,env)
        S1.append(t1.clone()); S2.append(t2.clone())
        TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))
    ev=pod_ev(S1,S2); r_traj=int((ev>ev[0]*1e-9).sum()); m999=modes_for(ev,0.999)
    tail=tail_fraction(t1,NRES); del S1,S2

    # ---- ADAPTIVE ANCHORS: pick segments where ROM error is workable ----
    errs=[terr(MO[k][0],MO[k][1],TR[k][0],TR[k][1],tids)
          for k in range(len(MO))]
    usable=[k for k in range(2,len(MO)-FORECAST_SEGS) if 0.002<=errs[k]<=0.60]
    if len(usable)>=3:
        anchors=[usable[0], usable[len(usable)//2], usable[-1]]
    else:
        anchors=[k for k in range(2,len(MO)-FORECAST_SEGS)][:3]
        print(f"  NOTE: only {len(usable)} segments in the workable error band;"
              f" falling back to early anchors", flush=True)
    print(f"  r_traj={r_traj} (99.9%: {m999})  tail={tail:.2e}  "
          f"anchors={[round(a*SEG_TAU,2) for a in anchors]}  "
          f"err@anchors={[f'{100*errs[a]:.2f}%' for a in anchors]}", flush=True)

    # ---- empirical Q-error subspace and leakage operator ----
    QA=[k for k in range(2,len(MO))]
    B1,B2=[],[]
    for k in QA:
        _,_,a1,a2=splitPQ(TR[k][0]-MO[k][0],TR[k][1]-MO[k][1],tids)
        w1,w2=a1.clone(),a2.clone()
        for e1,e2 in zip(B1,B2):
            cc=redot(e1,e2,w1,w2); w1=w1-cc*e1; w2=w2-cc*e2
        nw=pnorm(w1,w2)
        if nw>1e-13*max(pnorm(a1,a2),1e-300): B1.append(w1/nw); B2.append(w2/nw)
    r_emp=len(B1)
    Va1,Va2=MO[anchors[-1]]
    base1,base2=phi(Va1,Va2,env,nu); scale=max(pnorm(Va1,Va2),1e-30)
    cols=[]
    for e1,e2 in zip(B1,B2):
        eps=1e-6*scale
        p1,p2=phi(Va1+eps*e1,Va2+eps*e2,env,nu)
        cols.append(packm((p1-base1)/eps,(p2-base2)/eps,tids))
    Y=torch.stack(cols,dim=1); Uu,Sv,Vh=torch.linalg.svd(Y,full_matrices=False)
    e2v=(Sv.to(torch.float64)**2); tot=float(e2v.sum())
    top3=float(e2v[:3].sum()/tot); cum=torch.cumsum(e2v,0)/tot
    r_leak90=int((cum<0.90).sum())+1
    dP1,dP2,dQ1,dQ2=splitPQ(TR[anchors[-1]][0]-Va1,TR[anchors[-1]][1]-Va2,tids)
    reservoir=pnorm(dQ1,dQ2)/max(pnorm(dP1,dP2),1e-30)

    # ---- triadic enrichment of the leading leakage direction ----
    ql=sum(Vh.conj().transpose(0,1)[i,0].to(B1[0].dtype)*B1[i] for i in range(len(B1)))
    qe=(np.abs(ql.detach().cpu().numpy())**2).sum(axis=0).ravel()
    eb=(np.abs(Va1.detach().cpu().numpy())**2).sum(axis=0).ravel()
    order=np.argsort(eb)[::-1]; cse=np.cumsum(eb[order])/max(eb.sum(),1e-300)
    flow=order[:max(1,int((cse<0.90).sum())+1)]
    shp=(NRES,NRES,NRES)
    pv=np.stack(np.unravel_index(tids.detach().cpu().numpy(),shp))
    fv=np.stack(np.unravel_index(flow,shp))
    Sset=set()
    for a in range(pv.shape[1]):
        Sset.update(np.ravel_multi_index((pv[:,a:a+1]-fv)%NRES,shp).tolist())
    Sidx=np.fromiter(Sset,dtype=np.int64)
    frac_S=float(qe[Sidx].sum()/max(qe.sum(),1e-300)); null_S=len(Sidx)/qe.size
    enrich=frac_S/max(null_S,1e-300)

    # ---- W recurrence then X interventions ----
    E1=torch.zeros_like(MO[0][0]); E2=torch.zeros_like(MO[0][1]); EH=[(E1.clone(),E2.clone())]
    for k in range(max(anchors)):
        Vk1,Vk2=MO[k]; Vn1,Vn2=MO[k+1]
        Bp1,Bp2=phi(Vk1,Vk2,env,nu); e1,e2=Bp1-Vn1,Bp2-Vn2
        if pnorm(E1,E2)<1e-30: E1,E2=e1.clone(),e2.clone()
        else:
            en=max(pnorm(E1,E2),1e-30)
            eps=min(max(DERIV_REL*pnorm(Vk1,Vk2)/en,1e-6),5e-2)
            P1,P2=phi(Vk1+eps*E1,Vk2+eps*E2,env,nu)
            Mi1,Mi2=phi(Vk1-eps*E1,Vk2-eps*E2,env,nu)
            E1=e1+(P1-Mi1)/(2*eps)+0.5*(P1+Mi1-2*Bp1)/(eps*eps)
            E2=e2+(P2-Mi2)/(2*eps)+0.5*(P2+Mi2-2*Bp2)/(eps*eps)
        EH.append((E1.clone(),E2.clone()))

    def forecast(A1,A2):
        a1,a2=A1.clone(),A2.clone()
        for _ in range(FORECAST_SEGS):
            mm,_=init2(a1,a2,env)
            for _ in range(SEG_STEPS): rk4M(mm,SEG_TAU/SEG_STEPS,env)
            a1,a2=recon(mm,env)
        return a1,a2

    gains=[]
    for k in anchors:
        Vi1,Vi2=MO[k]; Ti1,Ti2=TR[k]; TF1,TF2=TR[k+FORECAST_SEGS]
        _,_,EQ1,EQ2=splitPQ(EH[k][0],EH[k][1],tids)
        _,_,AQ1,AQ2=splitPQ(Ti1-Vi1,Ti2-Vi2,tids)
        xb=packm(Vi1,Vi2,tids); xe=packm(Vi1+EQ1,Vi2+EQ2,tids)
        mism=float((torch.linalg.vector_norm(xe-xb)/torch.linalg.vector_norm(xb)).real.cpu())
        FB=forecast(Vi1,Vi2); FE=forecast(Vi1+EQ1,Vi2+EQ2); FN=forecast(Vi1-EQ1,Vi2-EQ2)
        be=terr(*FB,TF1,TF2,tids); ee=terr(*FE,TF1,TF2,tids); ne=terr(*FN,TF1,TF2,tids)
        gains.append(be/max(ee,1e-30))
        rows.append(dict(k_hi=k_hi,nu=nu,amp=amp,N=NRES,tau=round(k*SEG_TAU,3),
            r_trajectory=r_traj,modes_999=m999,tail_frac=tail,
            r_empirical_Q=r_emp,top3_energy=top3,r_leak90=r_leak90,
            reservoir=reservoir,triadic_frac=frac_S,triadic_null=null_S,
            triadic_enrichment=enrich,start_mismatch=mism,
            baseline_P_error=be,est_P_error=ee,neg_P_error=ne,
            gain=be/max(ee,1e-30),beats=bool(ee<be),beats_neg=bool(ee<ne),
            hidden_Q_residual=pnorm(EQ1-AQ1,EQ2-AQ2)/max(pnorm(AQ1,AQ2),1e-30)))
    print(f"  r_empQ={r_emp}  top3={top3:.4f}  r_leak90={r_leak90}  "
          f"Q/P={reservoir:.2f}  triad={enrich:.2f}x")
    print(f"  gains={[f'{g:.3f}' for g in gains]}  "
          f"wins {sum(1 for g in gains if g>1)}/{len(gains)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
    del env,U1,U2,t1,t2,V1,V2,m,TR,MO,B1,B2; torch.cuda.empty_cache()

# ============================== SUMMARY =====================================
df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
g=df.groupby(["k_hi","nu"]).first().reset_index().sort_values("r_trajectory")
print("\n"+"="*104); print("M3D TEST BB3 — BROADBAND, SELF-CALIBRATED"); print("="*104)
print(f"\n  {'band':<9}{'nu':>10}{'r_traj':>8}{'99.9%':>7}{'r_empQ':>8}{'top3':>8}"
      f"{'r_leak90':>10}{'Q/P':>7}{'triad':>8}{'med gain':>10}{'wins':>7}")
for _,r in g.iterrows():
    s=df[(df.k_hi==r.k_hi)&(df.nu==r.nu)]
    print(f"  [1,{int(r.k_hi):<5}]{r.nu:>10.1e}{int(r.r_trajectory):>8}"
          f"{int(r.modes_999):>7}{int(r.r_empirical_Q):>8}{r.top3_energy:>8.4f}"
          f"{int(r.r_leak90):>10}{r.reservoir:>7.2f}{r.triadic_enrichment:>8.2f}"
          f"{s.gain.median():>10.4f}{int(s.beats.sum()):>4}/{len(s)}")

print("\n--- B1: is the testbed genuinely high-dimensional? ---")
lo,hi=int(g.r_trajectory.min()),int(g.r_trajectory.max())
print(f"  r_trajectory spans {lo} -> {hi}   (smooth families: 12-27)")
print(f"  max energy above 0.8x dealias cutoff: {g.tail_frac.max():.2e}"
      f"   ({'resolved' if g.tail_frac.max()<1e-4 else 'UNDER-RESOLVED — reject'})")
print(f"  -> {'PASS' if hi>=50 and g.tail_frac.max()<1e-4 else 'MARGINAL/FAIL'}")

print("\n--- B2: does leakage rank grow slower than trajectory rank? ---")
for _,r in g.iterrows():
    print(f"  r_traj={int(r.r_trajectory):>4}  r_leak90={int(r.r_leak90):>3}  "
          f"ratio={r.r_leak90/max(r.r_trajectory,1):.4f}  top3={r.top3_energy:.4f}")
if len(g)>=3:
    p,_=np.polyfit(np.log(g.r_trajectory.values),np.log(g.r_leak90.values),1)
    rr=np.corrcoef(np.log(g.r_trajectory.values),np.log(g.r_leak90.values))[0,1]
    print(f"  log-log slope = {p:+.3f}  (R^2={rr*rr:.3f})")
    v=("DREAM: leakage rank nearly invariant" if p<0.25 else
       "STILL GOOD: sublinear growth" if p<0.75 else
       "BAD: leakage rank tracks state rank")
    print(f"  -> {v}")

print("\n--- B3: does the causal correction survive? ---")
print(f"  win rate {int(df.beats.sum())}/{len(df)}   "
      f"vs wrong-sign {int(df.beats_neg.sum())}/{len(df)}   "
      f"median gain {df.gain.median():.4f}x")
print(f"  max start-P mismatch {df.start_mismatch.max():.3e}   "
      f"median hidden-Q residual {df.hidden_Q_residual.median():.4e}")

print("\n--- B4: triadic test — are dangerous modes the triadic partners of P? ---")
for _,r in g.iterrows():
    print(f"  r_traj={int(r.r_trajectory):>4}  enrichment {r.triadic_enrichment:6.2f}x"
          f"   (frac {r.triadic_frac:.4f} vs null {r.triadic_null:.4f})")
print(f"  median enrichment {g.triadic_enrichment.median():.2f}x")
print("  >>1 means dangerous hidden modes are the effective triadic partners of")
print("  the resolved band, giving 'energy != danger' a Navier-Stokes explanation.")
print("\nsaved:", OUT, "and", CAL)
