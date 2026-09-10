# ============================================================================
# M3D TEST CX — FULL X PROTOCOL WITH COARSE-GRID DEFECT ESTIMATION
# ============================================================================
# Pilot Test C (2 cases, 1 anchor, N=80) showed the defect estimator works at
# 1/64 the flow-map cost, and BEATS full resolution at low viscosity. This is
# the full version: 12 cases x 3 anchors at N=40, matching Test X exactly, so
# M=40 must reproduce the established headline of 1.650998x.
#
# Only the resolution of Phi_h / J / H changes. ROM, P/Q split, intervention
# protocol and forecast are identical to X-v3 (fp64, analytic STAR2 JVP).
#
# Gates:
#   X0  M=40 control reproduces X-v3: median gain 1.6510 +/- 0.002
#   X1  win rate stays 36/36 for every M
#   X2  median gain within 5% of the M=40 control for some M <= N/2
#   X3  DESCRIPTIVE: gain and cost vs M
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT = SAVE / "m3d_CX_coarse_full.csv"
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
print("bootstrap:", BOOT, flush=True)
G = {"__name__": "__m3d__"}
exec(compile(open(BOOT).read(), BOOT, "exec"), G, G)

def spaces():
    out, seen = [G], {id(G)}
    for _, o in list(G.items()):
        if callable(o) and hasattr(o, "__globals__") and id(o.__globals__) not in seen:
            out.append(o.__globals__); seen.add(id(o.__globals__))
    return out

def setup(nu):
    for ns in spaces():
        ns["NU1_D"] = ns["NU2_D"] = ns["nu1_D"] = ns["nu2_D"] = float(nu)
        ns["REAL_DTYPE_D"] = torch.float64
        ns["COMPLEX_DTYPE_D"] = torch.complex128

make_env = G["make_env_D"];   topo  = G["topology_pair_L2"]
full_rk4 = G["full_rk4_D"];   init2 = G["initialize_star2_N"]
rk4M     = G["rk4_M"];        recon = G["reconstruct_M"]
packm    = G["pack_modes_D"]; TAU0  = float(G["TAU0_D"])
dealias  = G["dealias_D"];    leray = G["project_divfree_D"]
curl     = G["curl_hat_D"];   vrhs  = G["velocity_rhs_D"]

# ---- analytic NS JVP, as used by the X-v3 clean run ----
def _ajvp(Uh, Vh, nu, env):
    Uc = dealias(leray(Uh, env), env); Vc = dealias(leray(Vh, env), env)
    u  = torch.fft.ifftn(Uc, dim=(-3,-2,-1)).real
    v  = torch.fft.ifftn(Vc, dim=(-3,-2,-1)).real
    wu = torch.fft.ifftn(curl(Uc, env), dim=(-3,-2,-1)).real
    wv = torch.fft.ifftn(curl(Vc, env), dim=(-3,-2,-1)).real
    cr = torch.empty_like(u)
    for i,(a,b) in enumerate(((1,2),(2,0),(0,1))):
        cr[i] = (u[a]*wv[b]-u[b]*wv[a]) + (v[a]*wu[b]-v[b]*wu[a])
    nh = dealias(leray(torch.fft.fftn(cr, dim=(-3,-2,-1)), env), env)
    return dealias(leray(nh - float(nu)*env["K2"].unsqueeze(0)*Vc, env), env
                   ).to(G["COMPLEX_DTYPE_D"])

def _pair_ajvp(U1,U2,V1,V2,env):
    return (_ajvp(U1,V1,G["NU1_D"],env), _ajvp(U2,V2,G["NU2_D"],env))

for ns in spaces():
    if "pair_jvp_M" in ns: ns["pair_jvp_M"] = _pair_ajvp
G["pair_jvp_M"] = _pair_ajvp
print("STAR2 JVP: analytic", flush=True)

L_BOX, SEG_TAU, SEG_STEPS = 8.0, 0.15, 20
DT = SEG_TAU * TAU0 / SEG_STEPS
N_FINE = 40
COARSE = [40, 28, 20, 14, 10]          # control FIRST so vs_fine is populated
TAUS = [0.90, 1.20, 1.50]
N_BASE_SEG, FORECAST_SEGS, DERIV_REL = 12, 2, 1e-5
BASE_NU = 2.5e-3
CASES = [(t, n) for t in ["vortex_ring","periodic_shear","skew_tubes","mixed_vortices"]
                for n in [0.25, 1.0, 4.0]]

def _idx(n,m):
    h=m//2
    return (list(range(0,h+1))+list(range(n-h+1,n)),
            list(range(0,h+1))+list(range(m-h+1,m)))
def restrict(Uh,N,M):
    if M==N: return Uh.clone()
    s,_=_idx(N,M); si=torch.tensor(s,device=Uh.device,dtype=torch.long)
    return Uh.index_select(1,si).index_select(2,si).index_select(3,si)*(float(M)/N)**3
def prolong(Uh,M,N):
    if M==N: return Uh.clone()
    s,_=_idx(N,M); si=torch.tensor(s,device=Uh.device,dtype=torch.long)
    out=torch.zeros((3,N,N,N),dtype=Uh.dtype,device=Uh.device)
    out[:,si[:,None,None],si[None,:,None],si[None,None,:]]=Uh*(float(N)/M)**3
    return out
def phi_at(A1,A2,envC,nu,N,M):
    if M==N: B1,B2=A1.clone(),A2.clone()
    else:
        B1=dealias(leray(restrict(A1,N,M),envC),envC)
        B2=dealias(leray(restrict(A2,N,M),envC),envC)
    for _ in range(SEG_STEPS):
        B1=full_rk4(B1,DT,nu,envC); B2=full_rk4(B2,DT,nu,envC)
    return (B1,B2) if M==N else (prolong(B1,M,N),prolong(B2,M,N))
def pnorm(a1,a2):
    return float(torch.sqrt(torch.clamp(
        torch.real(torch.vdot(a1.reshape(-1),a1.reshape(-1)))
        +torch.real(torch.vdot(a2.reshape(-1),a2.reshape(-1))),min=0)).cpu())
def splitPQ(A1,A2,t):
    P1=torch.zeros_like(A1); P2=torch.zeros_like(A2)
    P1.reshape(3,-1)[:,t]=A1.reshape(3,-1)[:,t]; P2.reshape(3,-1)[:,t]=A2.reshape(3,-1)[:,t]
    return P1,P2,A1-P1,A2-P2
def terr(A1,A2,T1,T2,t):
    a=packm(A1-T1,A2-T2,t); b=packm(T1,T2,t)
    return float((torch.linalg.vector_norm(a)/torch.linalg.vector_norm(b)).real.cpu())

rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done = {(r["case"], r["M"]) for r in rows}
print(f"recovered {len(rows)} rows", flush=True)

for topo_name, numult in CASES:
    nu = BASE_NU*numult; case = f"{topo_name}__nu{numult:g}"
    if all((case,M) in done for M in COARSE):
        print(f"SKIP {case}", flush=True); continue
    print(f"\n=== {case} ===", flush=True)
    setup(nu)
    envF = make_env(N_FINE, L=L_BOX)
    U1,U2 = topo(topo_name, envF)
    star,_ = init2(U1,U2,envF); tids = star["proj"]["target_ids"]; del star

    t1,t2 = U1.clone(),U2.clone(); m,_ = init2(U1,U2,envF)
    V1,V2 = U1.clone(),U2.clone()
    TR,MO = [(t1.clone(),t2.clone())], [(V1.clone(),V2.clone())]
    for _ in range(N_BASE_SEG):
        for _ in range(SEG_STEPS):
            t1=full_rk4(t1,DT,nu,envF); t2=full_rk4(t2,DT,nu,envF)
            rk4M(m,SEG_TAU/SEG_STEPS,envF)
        V1,V2 = recon(m,envF); m,_ = init2(V1,V2,envF)
        TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))

    def forecast(S1,S2):
        A1,A2=S1.clone(),S2.clone()
        for _ in range(FORECAST_SEGS):
            mm,_=init2(A1,A2,envF)
            for _ in range(SEG_STEPS): rk4M(mm,SEG_TAU/SEG_STEPS,envF)
            A1,A2=recon(mm,envF)
        return A1,A2

    REF = {}
    for M in COARSE:
        if (case,M) in done: continue
        envC = envF if M==N_FINE else make_env(M,L=L_BOX)
        E1=torch.zeros_like(MO[0][0]); E2=torch.zeros_like(MO[0][1])
        EH=[(E1.clone(),E2.clone())]
        for k in range(int(round(max(TAUS)/SEG_TAU))):
            Vk1,Vk2=MO[k]; Vn1,Vn2=MO[k+1]
            B1,B2=phi_at(Vk1,Vk2,envC,nu,N_FINE,M)
            e1,e2=B1-Vn1,B2-Vn2
            if pnorm(E1,E2)<1e-30: E1,E2=e1.clone(),e2.clone()
            else:
                en=max(pnorm(E1,E2),1e-30)
                eps=min(max(DERIV_REL*pnorm(Vk1,Vk2)/en,1e-6),5e-2)
                P1,P2=phi_at(Vk1+eps*E1,Vk2+eps*E2,envC,nu,N_FINE,M)
                Mi1,Mi2=phi_at(Vk1-eps*E1,Vk2-eps*E2,envC,nu,N_FINE,M)
                E1=e1+(P1-Mi1)/(2*eps)+0.5*(P1+Mi1-2*B1)/(eps*eps)
                E2=e2+(P2-Mi2)/(2*eps)+0.5*(P2+Mi2-2*B2)/(eps*eps)
            EH.append((E1.clone(),E2.clone()))
        for tau in TAUS:
            k=int(round(tau/SEG_TAU))
            Vi1,Vi2=MO[k]; Ti1,Ti2=TR[k]; TF1,TF2=TR[k+FORECAST_SEGS]
            _,_,EQ1,EQ2=splitPQ(EH[k][0],EH[k][1],tids)
            _,_,AQ1,AQ2=splitPQ(Ti1-Vi1,Ti2-Vi2,tids)
            xb=packm(Vi1,Vi2,tids); xe=packm(Vi1+EQ1,Vi2+EQ2,tids)
            mism=float((torch.linalg.vector_norm(xe-xb)
                        /torch.linalg.vector_norm(xb)).real.cpu())
            FB=forecast(Vi1,Vi2); FE=forecast(Vi1+EQ1,Vi2+EQ2)
            FN=forecast(Vi1-EQ1,Vi2-EQ2); FO=forecast(Vi1+AQ1,Vi2+AQ2)
            be=terr(*FB,TF1,TF2,tids); ee=terr(*FE,TF1,TF2,tids)
            ne=terr(*FN,TF1,TF2,tids); oe=terr(*FO,TF1,TF2,tids)
            if M==N_FINE: REF[tau]=(EQ1.clone(),EQ2.clone())
            vsf=(pnorm(EQ1-REF[tau][0],EQ2-REF[tau][1])
                 /max(pnorm(*REF[tau]),1e-30)) if tau in REF else float("nan")
            rows.append(dict(case=case,topology=topo_name,nu_mult=numult,nu=nu,
                M=M,cost_ratio=(M/N_FINE)**3,tau=tau,start_mismatch=mism,
                baseline_P_error=be,est_P_error=ee,neg_P_error=ne,oracle_P_error=oe,
                gain=be/max(ee,1e-30),gain_neg=be/max(ne,1e-30),
                gain_oracle=be/max(oe,1e-30),
                est_beats_base=bool(ee<be),est_beats_neg=bool(ee<ne),
                hidden_Q_residual=pnorm(EQ1-AQ1,EQ2-AQ2)/max(pnorm(AQ1,AQ2),1e-30),
                coarse_vs_fine=vsf))
        g=[r["gain"] for r in rows if r["case"]==case and r["M"]==M]
        print(f"  M={M:>3} cost={(M/N_FINE)**3:6.4f}  gains="
              f"{'  '.join(f'{x:.3f}' for x in g)}", flush=True)
        pd.DataFrame(rows).to_csv(OUT,index=False)
        if M!=N_FINE: del envC
    del envF,U1,U2,t1,t2,V1,V2,m,TR,MO
    torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print("\n"+"="*100)
print("M3D TEST CX — FULL X PROTOCOL WITH COARSE DEFECT ESTIMATION")
print("="*100)
print(f"{'M':>4}{'cost':>9}{'n':>5}{'median gain':>13}{'p10':>9}"
      f"{'win rate':>10}{'beats -Q':>10}{'med Qres':>10}")
for M in COARSE:
    s=df[df.M==M]
    if s.empty: continue
    print(f"{M:>4}{(M/N_FINE)**3:>9.4f}{len(s):>5}{s.gain.median():>13.6f}"
          f"{s.gain.quantile(0.1):>9.4f}"
          f"{100*s.est_beats_base.mean():>9.1f}%{100*s.est_beats_neg.mean():>9.1f}%"
          f"{s.hidden_Q_residual.median():>10.4f}")
ctl=df[df.M==N_FINE]
if not ctl.empty:
    med=ctl.gain.median()
    print(f"\nX0 control (M={N_FINE}) median gain = {med:.6f}   "
          f"X-v3 reference = 1.650998   "
          f"{'PASS' if abs(med-1.650998)<0.002 else 'MISMATCH'}")
    print(f"X1 win rate 36/36 at every M: "
          f"{'PASS' if bool((df.groupby('M').est_beats_base.mean()==1.0).all()) else 'FAIL'}")
    ok=df[(df.M<=N_FINE//2)].groupby("M").gain.median()
    good=[m for m,v in ok.items() if abs(v-med)/med<0.05]
    print(f"X2 median gain within 5% of control for M<=N/2: "
          f"{'PASS ' + str(good) if good else 'FAIL'}")
    if good:
        cm=min(good)
        print(f"X3 cheapest working M = {cm} at {(cm/N_FINE)**3:.4f}x flow-map cost;"
              f" W total = 3x{(cm/N_FINE)**3:.4f} = {3*(cm/N_FINE)**3:.4f} of one"
              f" full-solver segment ({1/(3*(cm/N_FINE)**3):.1f}x cheaper)")
print("\nsaved:", OUT)
