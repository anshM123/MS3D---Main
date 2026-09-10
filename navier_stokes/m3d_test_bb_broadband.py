# ============================================================================
# M3D TEST BB — BROADBAND FLOW: DOES THE HIDDEN-ERROR STORY SURVIVE
#               GENUINELY HIGH-DIMENSIONAL DYNAMICS?
# ============================================================================
# The single largest risk to the project: the four existing flow families are
# numerically rank ~10-27, so a referee can say the mechanism was found on
# smooth decaying flows that barely need reducing.
#
# This builds initial conditions with a PRESCRIBED SPECTRAL BAND, sweeping the
# bandwidth to produce a ladder of trajectory ranks, and runs the minimum
# essential chain at each:
#
#   trajectory rank -> Q reservoir -> leakage rank -> W residual -> X gain
#
# The headline is r_leak vs r_trajectory. Three outcomes, per the review:
#   dream      r_leak stays ~3 while r_traj goes 10 -> 200
#   still good r_leak grows but far slower than r_traj  (quantify the ratio)
#   bad        r_leak ~ r_Q and the phenomenon was tied to smooth flows
#
# TRIADIC DIAGNOSTIC (the JFM-relevant one). Navier-Stokes nonlinearity is
# quadratic, so hidden content at wavevector k can reach the resolved band at
# p only through a triad k + q = p with q carried by the flow. If the
# dangerous directions are the effective triadic partners of P rather than
# simply the energetic ones, then
#
#       energy != danger
#
# acquires an actual Navier-Stokes explanation instead of being an MOR
# observation. Measured as an ENRICHMENT FACTOR of leakage energy on the
# triadic partner set S = {p - q} against the null expectation |S|/|Q|.
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT = SAVE / "m3d_BB_broadband.csv"
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
NRES = int(os.environ.get("M3D_BB_N", 64))
NU   = float(os.environ.get("M3D_BB_NU", 1.6e-4))   # low enough that broadband
                                                    # content survives to tau=1.5
N_SEG = 10
ANCHORS = [2,3,4,5,6,7,8,9,10]
TAUS = [0.60, 0.90, 1.20]      # earlier than usual: broadband ROM error grows fast
FORECAST_SEGS = 2
DERIV_REL = 1e-5
# spectral bands, narrow -> broad. The narrow one is a control that should
# behave like the existing smooth families.
BANDS = [(1,3),(1,6),(1,10),(1,14)]

# ---------------------------------------------------------------- broadband IC
def broadband_field(N, k_lo, k_hi, seed, env, slope=0.0):
    """Random-phase divergence-free field with energy confined to [k_lo,k_hi].
    Built from a real random field so Hermitian symmetry is automatic; the
    amplitude is then flattened and reshaped, which preserves that symmetry."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(3,N,N,N))
    uh = np.fft.fftn(u, axes=(1,2,3))
    f = np.fft.fftfreq(N)*N
    KX,KY,KZ = np.meshgrid(f,f,f,indexing="ij")
    kmag = np.sqrt(KX**2+KY**2+KZ**2)
    band = ((kmag>=k_lo)&(kmag<=k_hi)).astype(float)
    shape = np.where(kmag>0, kmag**slope, 0.0)
    uh = uh/(np.abs(uh)+1e-300)*band*shape          # unit amplitude, random phase
    U = torch.tensor(uh, dtype=torch.complex128, device="cuda")
    U = dealias(leray(U, env), env)
    return U

def norm_of(U):
    return float(torch.sqrt(torch.real(torch.vdot(U.reshape(-1),U.reshape(-1)))).cpu())

def broadband_pair(N, k_lo, k_hi, env, ref_norm):
    U1 = broadband_field(N,k_lo,k_hi,1234,env)
    U2 = broadband_field(N,k_lo,k_hi,5678,env)
    U1 = U1*(ref_norm/max(norm_of(U1),1e-30))
    U2 = U2*(ref_norm/max(norm_of(U2),1e-30))
    return U1, U2

# ---------------------------------------------------------------- helpers
def redot(a1,a2,b1,b2):
    return float((torch.real(torch.vdot(a1.reshape(-1),b1.reshape(-1)))
                 +torch.real(torch.vdot(a2.reshape(-1),b2.reshape(-1)))).cpu())
def pnorm(a1,a2): return math.sqrt(max(redot(a1,a2,a1,a2),0.0))
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
def gram_rank(S1,S2):
    B1,B2=[],[]
    for a1,a2 in zip(S1,S2):
        w1,w2=a1.clone(),a2.clone()
        for e1,e2 in zip(B1,B2):
            c=redot(e1,e2,w1,w2); w1=w1-c*e1; w2=w2-c*e2
        nw=pnorm(w1,w2)
        if nw>1e-13*max(pnorm(a1,a2),1e-300): B1.append(w1/nw); B2.append(w2/nw)
    return B1,B2
def pod_spectrum(S1,S2):
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

rows=[]
prev=pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done={(r["k_lo"],r["k_hi"]) for r in prev}; rows.extend(prev)

# reference amplitude: match an existing smooth case so the nonlinear timescale
# is comparable and the comparison is not confounded by forcing amplitude
setup(NU); env0=make_env(NRES,L=L_BOX)
R1,R2=topo("skew_tubes",env0); REF=norm_of(R1); del R1,R2,env0
print(f"reference |U| from skew_tubes = {REF:.6e}", flush=True)

for k_lo,k_hi in BANDS:
    if (k_lo,k_hi) in done:
        print(f"SKIP band [{k_lo},{k_hi}]", flush=True); continue
    print(f"\n=== broadband band [{k_lo},{k_hi}]  N={NRES}  nu={NU:.2e} ===",flush=True)
    setup(NU); env=make_env(NRES,L=L_BOX)
    U1,U2=broadband_pair(NRES,k_lo,k_hi,env,REF)
    star,_=init2(U1,U2,env); tids=star["proj"]["target_ids"]; del star

    # ---- truth + STAR2, collecting snapshots and Q-errors ----
    t1,t2=U1.clone(),U2.clone(); m,_=init2(U1,U2,env)
    V1,V2=U1.clone(),U2.clone()
    S1,S2=[t1.clone()],[t2.clone()]
    dQ1s,dQ2s,Vs,TR,MO=[],[],[],[(t1.clone(),t2.clone())],[(V1.clone(),V2.clone())]
    for seg in range(1,N_SEG+1):
        for _ in range(SEG_STEPS):
            t1=full_rk4(t1,DT,NU,env); t2=full_rk4(t2,DT,NU,env)
            rk4M(m,SEG_TAU/SEG_STEPS,env)
        V1,V2=recon(m,env); m,_=init2(V1,V2,env)
        S1.append(t1.clone()); S2.append(t2.clone())
        TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))
        if seg in ANCHORS:
            _,_,q1,q2=splitPQ(t1-V1,t2-V2,tids)
            dQ1s.append(q1.clone()); dQ2s.append(q2.clone()); Vs.append((V1.clone(),V2.clone()))

    ev=pod_spectrum(S1,S2)
    r_traj=int((ev>ev[0]*1e-9).sum()); m999=modes_for(ev,0.999)
    del S1,S2

    # ---- leakage operator on the empirical Q-error subspace ----
    B1,B2=gram_rank(dQ1s,dQ2s); r_emp=len(B1)
    Va1,Va2=Vs[-1]; base1,base2=phi(Va1,Va2,env,NU)
    scale=max(pnorm(Va1,Va2),1e-30); cols=[]
    for e1,e2 in zip(B1,B2):
        eps=1e-6*scale
        p1,p2=phi(Va1+eps*e1,Va2+eps*e2,env,NU)
        cols.append(packm((p1-base1)/eps,(p2-base2)/eps,tids))
    Y=torch.stack(cols,dim=1)
    Uu,Sv,Vh=torch.linalg.svd(Y,full_matrices=False)
    e2v=(Sv.to(torch.float64)**2); tot=float(e2v.sum())
    top3=float(e2v[:3].sum()/tot)
    cum=torch.cumsum(e2v,0)/tot
    r_leak90=int((cum<0.90).sum())+1

    dP1,dP2,dQ1,dQ2=splitPQ(TR[ANCHORS[-1]][0]-Va1,TR[ANCHORS[-1]][1]-Va2,tids)
    reservoir=pnorm(dQ1,dQ2)/max(pnorm(dP1,dP2),1e-30)

    # ---- TRIADIC ENRICHMENT of the leading leakage direction ----
    q1=sum(Vh.conj().transpose(0,1)[i,0].to(B1[0].dtype)*B1[i] for i in range(len(B1)))
    ql=np.abs(q1.detach().cpu().numpy())**2
    ql=ql.sum(axis=0).ravel()
    ub=Va1.detach().cpu().numpy(); eb=(np.abs(ub)**2).sum(axis=0).ravel()
    idx=np.argsort(eb)[::-1]; cse=np.cumsum(eb[idx])/max(eb.sum(),1e-300)
    flow_modes=idx[:max(1,int((cse<0.90).sum())+1)]
    Pm=tids.detach().cpu().numpy()
    shp=(NRES,NRES,NRES)
    pv=np.stack(np.unravel_index(Pm,shp))
    fv=np.stack(np.unravel_index(flow_modes,shp))
    Sset=set()
    for a in range(pv.shape[1]):
        d=(pv[:,a:a+1]-fv)%NRES
        Sset.update(np.ravel_multi_index(d,shp).tolist())
    Sidx=np.fromiter(Sset,dtype=np.int64)
    frac_S=float(ql[Sidx].sum()/max(ql.sum(),1e-300))
    null_S=len(Sidx)/ql.size
    enrich=frac_S/max(null_S,1e-300)

    # ---- W recurrence, then X interventions ----
    E1=torch.zeros_like(MO[0][0]); E2=torch.zeros_like(MO[0][1]); EH=[(E1.clone(),E2.clone())]
    for k in range(int(round(max(TAUS)/SEG_TAU))):
        Vk1,Vk2=MO[k]; Vn1,Vn2=MO[k+1]
        Bp1,Bp2=phi(Vk1,Vk2,env,NU); e1,e2=Bp1-Vn1,Bp2-Vn2
        if pnorm(E1,E2)<1e-30: E1,E2=e1.clone(),e2.clone()
        else:
            en=max(pnorm(E1,E2),1e-30)
            eps=min(max(DERIV_REL*pnorm(Vk1,Vk2)/en,1e-6),5e-2)
            P1,P2=phi(Vk1+eps*E1,Vk2+eps*E2,env,NU)
            Mi1,Mi2=phi(Vk1-eps*E1,Vk2-eps*E2,env,NU)
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

    gains,wins,dirwins,qres=[],0,0,[]
    for tau in TAUS:
        k=int(round(tau/SEG_TAU))
        Vi1,Vi2=MO[k]; Ti1,Ti2=TR[k]; TF1,TF2=TR[k+FORECAST_SEGS]
        _,_,EQ1,EQ2=splitPQ(EH[k][0],EH[k][1],tids)
        _,_,AQ1,AQ2=splitPQ(Ti1-Vi1,Ti2-Vi2,tids)
        FB=forecast(Vi1,Vi2); FE=forecast(Vi1+EQ1,Vi2+EQ2); FN=forecast(Vi1-EQ1,Vi2-EQ2)
        be=terr(*FB,TF1,TF2,tids); ee=terr(*FE,TF1,TF2,tids); ne=terr(*FN,TF1,TF2,tids)
        gains.append(be/max(ee,1e-30)); wins+=int(ee<be); dirwins+=int(ee<ne)
        qres.append(pnorm(EQ1-AQ1,EQ2-AQ2)/max(pnorm(AQ1,AQ2),1e-30))
        rows.append(dict(k_lo=k_lo,k_hi=k_hi,N=NRES,nu=NU,tau=tau,
            r_trajectory=r_traj,modes_999=m999,r_empirical_Q=r_emp,
            top3_energy=top3,r_leak90=r_leak90,reservoir=reservoir,
            triadic_frac=frac_S,triadic_null=null_S,triadic_enrichment=enrich,
            baseline_P_error=be,est_P_error=ee,neg_P_error=ne,
            gain=be/max(ee,1e-30),beats=bool(ee<be),beats_neg=bool(ee<ne),
            hidden_Q_residual=qres[-1]))
    print(f"  r_traj={r_traj:>4} (99.9%: {m999:>3})  r_empQ={r_emp}  "
          f"top3={top3:.4f}  r_leak90={r_leak90}  Q/P={reservoir:.2f}")
    print(f"  triadic enrichment={enrich:6.2f}x  (frac {frac_S:.4f} vs null {null_S:.4f})")
    _be = "  ".join(f"{100*r['baseline_P_error']:.2f}%" for r in rows[-3:])
    print(f"  base err: {_be}")
    print(f"  gains={[f'{g:.3f}' for g in gains]}  wins {wins}/3  vs -Q {dirwins}/3  "
          f"Qres={np.median(qres):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
    del env,U1,U2,t1,t2,V1,V2,m,TR,MO,dQ1s,dQ2s,Vs,B1,B2
    torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print("\n"+"="*104)
print("M3D TEST BB — BROADBAND FLOW")
print("="*104)
g=df.groupby(["k_lo","k_hi"]).first().reset_index()
print(f"\n  {'band':<10}{'r_traj':>8}{'99.9%':>7}{'r_empQ':>8}{'top3':>8}"
      f"{'r_leak90':>10}{'Q/P':>7}{'triad x':>9}{'med gain':>10}{'wins':>7}")
for _,r in g.iterrows():
    sub=df[(df.k_lo==r.k_lo)&(df.k_hi==r.k_hi)]
    print(f"  [{int(r.k_lo)},{int(r.k_hi)}]{'':<5}{int(r.r_trajectory):>8}"
          f"{int(r.modes_999):>7}{int(r.r_empirical_Q):>8}{r.top3_energy:>8.4f}"
          f"{int(r.r_leak90):>10}{r.reservoir:>7.2f}{r.triadic_enrichment:>9.2f}"
          f"{sub.gain.median():>10.4f}{int(sub.beats.sum()):>4}/{len(sub)}")

print("\n--- B1: does trajectory rank actually increase? ---")
print(f"  r_trajectory: {int(g.r_trajectory.min())} -> {int(g.r_trajectory.max())}"
      f"   (existing smooth families: 12-27)")
ok=int(g.r_trajectory.max())>=50
print(f"  -> {'PASS: genuinely higher-dimensional testbed' if ok else 'MARGINAL: widen the band or lower nu'}")

print("\n--- B2: does leakage rank grow slower than trajectory rank? ---")
for _,r in g.iterrows():
    print(f"  band [{int(r.k_lo)},{int(r.k_hi)}]: r_traj={int(r.r_trajectory):>4}  "
          f"r_leak90={int(r.r_leak90):>3}  ratio r_leak/r_traj = "
          f"{r.r_leak90/max(r.r_trajectory,1):.4f}")
if len(g)>=3:
    p,_=np.polyfit(np.log(g.r_trajectory.values),np.log(g.r_leak90.values),1)
    print(f"  log-log slope d(log r_leak)/d(log r_traj) = {p:+.3f}")
    print(f"  -> {'DREAM: leakage rank nearly invariant' if p<0.25 else ('STILL GOOD: sublinear growth' if p<0.75 else 'BAD: leakage rank tracks state rank')}")

print("\n--- B3: does the causal correction still work? ---")
print(f"  overall win rate {int(df.beats.sum())}/{len(df)}   "
      f"vs wrong-sign {int(df.beats_neg.sum())}/{len(df)}   "
      f"median gain {df.gain.median():.4f}x")
print(f"  median hidden-Q residual {df.hidden_Q_residual.median():.4e}")

print("\n--- B4: TRIADIC test.  are dangerous modes the triadic partners of P? ---")
print("      enrichment = (leakage energy on {p-q}) / (|{p-q}| / |Q|)")
for _,r in g.iterrows():
    print(f"  band [{int(r.k_lo)},{int(r.k_hi)}]: enrichment {r.triadic_enrichment:6.2f}x")
print(f"  median enrichment {g.triadic_enrichment.median():.2f}x")
print("  >> 1 would mean dangerous hidden modes are the effective triadic")
print("  partners of the resolved band, giving 'energy != danger' a genuine")
print("  Navier-Stokes explanation rather than an MOR one.")
print("\nsaved:", OUT)
