# ============================================================================
# M3D TEST CS — DOES THE REQUIRED COARSE RESOLUTION SCALE WITH N?
# ============================================================================
# CX at N=40 found a universal M* = 28, which puts W at 1.03x the full-solver
# cost: break-even, so the cost objection is NOT answered at N=40.
#
# The whole question is whether M* is ABSOLUTE (set by the wavenumber content
# of the defect) or RELATIVE (a fixed fraction of N).
#
#   absolute  -> W cost = 3(M*/N)^3 collapses as N grows; at N=128 it is 0.03x
#                the solver and the method becomes genuinely cheap.
#   relative  -> the ratio is fixed, coarse-eta buys nothing, and the paper
#                stays an estimability result.
#
# The M ladder is therefore in ABSOLUTE mode counts, identical at every N.
# Note P (the 'mixed' 24-mode projector) is itself fixed in absolute
# wavenumber, which is why the absolute hypothesis is plausible.
#
# Gates:
#   S0  M=N control reproduces the known per-case gains at that N
#   S1  M*(N) flat in absolute terms (max/min <= 2) across N for each topology
#   S2  DESCRIPTIVE: implied W cost at each N, and extrapolation to N=128
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT = SAVE / "m3d_CS_mstar_scaling.csv"
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
curl     = G["curl_hat_D"]

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
N_LIST  = [int(x) for x in os.environ.get("M3D_CS_N", "40,64,80").split(",")]
M_LADDER = [10, 14, 20, 28, 40]        # ABSOLUTE mode counts, same at every N
TAUS = [0.90, 1.20, 1.50]
N_BASE_SEG, FORECAST_SEGS, DERIV_REL = 12, 2, 1e-5
BASE_NU = 2.5e-3
# the two harder viscosities; nu=4 was the most tolerant and would flatter M*
CASES = [(t, n) for t in ["vortex_ring","periodic_shear","skew_tubes","mixed_vortices"]
                for n in [0.25, 1.0]]

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
done = {(r["N"], r["case"], r["M"]) for r in rows}
print(f"recovered {len(rows)} rows", flush=True)

for N in N_LIST:
    Ms = sorted(set([m for m in M_LADDER if m < N] + [N]), reverse=True)
    for topo_name, numult in CASES:
        nu = BASE_NU*numult; case = f"{topo_name}__nu{numult:g}"
        if all((N,case,M) in done for M in Ms):
            print(f"SKIP N={N} {case}", flush=True); continue
        print(f"\n=== N={N}  {case} ===", flush=True)
        setup(nu)
        envF = make_env(N, L=L_BOX)
        U1,U2 = topo(topo_name, envF)
        star,_ = init2(U1,U2,envF); tids = star["proj"]["target_ids"]; del star

        t1,t2 = U1.clone(),U2.clone(); m,_ = init2(U1,U2,envF)
        V1,V2 = U1.clone(),U2.clone()
        TR,MO = [(t1.clone(),t2.clone())], [(V1.clone(),V2.clone())]
        for _ in range(N_BASE_SEG):
            for _ in range(SEG_STEPS):
                t1=full_rk4(t1,DT,nu,envF); t2=full_rk4(t2,DT,nu,envF)
                rk4M(m,SEG_TAU/SEG_STEPS,envF)
            V1,V2=recon(m,envF); m,_=init2(V1,V2,envF)
            TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))

        def forecast(S1,S2):
            A1,A2=S1.clone(),S2.clone()
            for _ in range(FORECAST_SEGS):
                mm,_=init2(A1,A2,envF)
                for _ in range(SEG_STEPS): rk4M(mm,SEG_TAU/SEG_STEPS,envF)
                A1,A2=recon(mm,envF)
            return A1,A2

        for M in Ms:
            if (N,case,M) in done: continue
            envC = envF if M==N else make_env(M,L=L_BOX)
            E1=torch.zeros_like(MO[0][0]); E2=torch.zeros_like(MO[0][1])
            EH=[(E1.clone(),E2.clone())]
            for k in range(int(round(max(TAUS)/SEG_TAU))):
                Vk1,Vk2=MO[k]; Vn1,Vn2=MO[k+1]
                B1,B2=phi_at(Vk1,Vk2,envC,nu,N,M)
                e1,e2=B1-Vn1,B2-Vn2
                if pnorm(E1,E2)<1e-30: E1,E2=e1.clone(),e2.clone()
                else:
                    en=max(pnorm(E1,E2),1e-30)
                    eps=min(max(DERIV_REL*pnorm(Vk1,Vk2)/en,1e-6),5e-2)
                    P1,P2=phi_at(Vk1+eps*E1,Vk2+eps*E2,envC,nu,N,M)
                    Mi1,Mi2=phi_at(Vk1-eps*E1,Vk2-eps*E2,envC,nu,N,M)
                    E1=e1+(P1-Mi1)/(2*eps)+0.5*(P1+Mi1-2*B1)/(eps*eps)
                    E2=e2+(P2-Mi2)/(2*eps)+0.5*(P2+Mi2-2*B2)/(eps*eps)
                EH.append((E1.clone(),E2.clone()))
            gains=[]
            for tau in TAUS:
                k=int(round(tau/SEG_TAU))
                Vi1,Vi2=MO[k]; TF1,TF2=TR[k+FORECAST_SEGS]
                _,_,EQ1,EQ2=splitPQ(EH[k][0],EH[k][1],tids)
                FB=forecast(Vi1,Vi2); FE=forecast(Vi1+EQ1,Vi2+EQ2)
                be=terr(*FB,TF1,TF2,tids); ee=terr(*FE,TF1,TF2,tids)
                g=be/max(ee,1e-30); gains.append(g)
                rows.append(dict(N=N,case=case,topology=topo_name,nu_mult=numult,
                    M=M,cost_ratio=(M/N)**3,tau=tau,baseline_P_error=be,
                    est_P_error=ee,gain=g,beats=bool(ee<be)))
            print(f"  M={M:>3} ({(M/N)**3:6.4f}x)  gains="
                  f"{'  '.join(f'{x:.3f}' for x in gains)}"
                  f"   {'ALL PASS' if all(x>1 for x in gains) else ''}", flush=True)
            pd.DataFrame(rows).to_csv(OUT,index=False)
            if M!=N: del envC
        del envF,U1,U2,t1,t2,V1,V2,m,TR,MO
        torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print("\n"+"="*100)
print("M3D TEST CS — DOES M* SCALE WITH N?")
print("="*100)
star={}
for (N,case),s in df.groupby(["N","case"]):
    ok=[M for M,g in s.groupby("M") if bool((g.gain>1.0).all())]
    star[(N,case)] = min(ok) if ok else None
print(f"\nM* = smallest ABSOLUTE mode count with all anchors beating baseline\n")
cases=sorted({c for _,c in star})
print(f"  {'case':<26}" + "".join(f"{'N='+str(n):>9}" for n in N_LIST))
for c in cases:
    print(f"  {c:<26}" + "".join(f"{str(star.get((n,c),'-')):>9}" for n in N_LIST))
print("\nS1 flatness of M* in absolute terms (per case, max/min across N):")
flat=[]
for c in cases:
    v=[star[(n,c)] for n in N_LIST if star.get((n,c))]
    if len(v)>=2:
        r=max(v)/min(v); flat.append(r)
        print(f"  {c:<26} M* = {v}   ratio {r:.2f}")
if flat:
    ok=all(r<=2.0 for r in flat)
    print(f"  -> {'PASS: M* is ABSOLUTE' if ok else 'FAIL: M* scales with N'}"
          f"  (max ratio {max(flat):.2f})")
    Mstar=max(v for c in cases for v in [star[(n,c)] for n in N_LIST if star.get((n,c))])
    print(f"\nS2 universal M* = {Mstar}; implied W cost = 3(M*/N)^3:")
    for N in [40,64,80,96,128,192]:
        cr=3*(Mstar/N)**3
        print(f"  N={N:<4} {cr:8.4f} of one solver segment"
              f"   -> {1/cr:6.1f}x {'cheaper' if cr<1 else 'MORE EXPENSIVE'}")
print("\nsaved:", OUT)
