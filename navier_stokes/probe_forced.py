# ============================================================================
# M3D — FORCED-TURBULENCE RANK PROBE  (calibration only, cheap)
# ============================================================================
# BB3 showed trajectory POD rank stays 4-11 for ANY spectral band, because an
# unforced decaying random-phase field barely explores its state space. Rank
# measures dynamical exploration, not spectral breadth.
#
# This probes Lundgren linear forcing  f = A*u  added to the RHS.
#   * With A CONSTANT, F stays EXACTLY QUADRATIC, so the analytic JVP and the
#     FD-exactness result from Y-v2 both survive unchanged. A self-regulating
#     A would break that and must not be used in the production runs.
#   * Stage A runs a short self-regulating pilot to find the equilibrium A
#     (the value that balances viscous dissipation), then FIXES it.
#   * Stage B runs the truth trajectory at fixed A and measures POD rank with
#     an 81-snapshot ceiling, plus the resolution tail.
#
# Decision rule: if rank clears ~50 with tail < 1e-4, build the full chain on
# forced turbulence. If it does not, the flow family cannot be made
# high-dimensional in this solver and the paper states that as a limitation.
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
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
dealias=G["dealias_D"]; leray=G["project_divfree_D"]; TAU0=float(G["TAU0_D"])
_vrhs = G["velocity_rhs_D"]

L_BOX, SEG_TAU, SEG_STEPS = 8.0, 0.15, 20
DT = SEG_TAU*TAU0/SEG_STEPS
N    = int(os.environ.get("M3D_F_N", 64))
N_SEG= int(os.environ.get("M3D_F_SEG", 20))     # longer horizon: forcing sustains
SNAPS= 81

def rhs_forced(U, nu, env, A):
    return _vrhs(U, nu, env) + float(A)*U      # A constant -> F stays quadratic

def rk4(U, dt, nu, env, A):
    k1=rhs_forced(U,nu,env,A); k2=rhs_forced(U+0.5*dt*k1,nu,env,A)
    k3=rhs_forced(U+0.5*dt*k2,nu,env,A); k4=rhs_forced(U+dt*k3,nu,env,A)
    return U + dt/6.0*(k1+2*k2+2*k3+k4)

def energy(U):  return float(torch.real(torch.vdot(U.reshape(-1),U.reshape(-1))).cpu())
def diss(U,env,nu):
    return float(nu*torch.real(torch.sum(env["K2"].unsqueeze(0)*torch.abs(U)**2)).cpu())
def A_equilibrium(U,env,nu):  return diss(U,env,nu)/max(energy(U),1e-300)

def tail_frac(U):
    e=(np.abs(U.detach().cpu().numpy())**2).sum(axis=0)
    f=np.fft.fftfreq(N)*N
    KX,KY,KZ=np.meshgrid(f,f,f,indexing="ij")
    km=np.sqrt(KX**2+KY**2+KZ**2)
    return float(e[km>0.8*(N/3.0)].sum()/max(e.sum(),1e-300))

def pod_rank(S):
    n=len(S); m=sum(S[1:],S[0].clone())/n; F=[s-m for s in S]
    Gm=torch.zeros(n,n,dtype=torch.float64)
    for i in range(n):
        for j in range(i,n):
            v=float(torch.real(torch.vdot(F[i].reshape(-1),F[j].reshape(-1))).cpu())
            Gm[i,j]=v; Gm[j,i]=v
    ev=torch.linalg.eigvalsh(Gm).flip(0).clamp(min=0)
    r=int((ev>ev[0]*1e-9).sum())
    tot=float(ev.sum()); c=0.0; m999=len(ev)
    for i,e in enumerate(ev):
        c+=float(e)
        if c>=0.999*tot: m999=i+1; break
    return r,m999,n

rows=[]
print(f"\nN={N}  horizon={N_SEG*SEG_TAU:.2f} tau  snapshot ceiling={SNAPS}\n")
print(f"  {'flow':<16}{'nu':>10}{'A_eq':>11}{'rank':>7}{'99.9%':>7}"
      f"{'tail':>11}{'resolved':>10}{'saturated':>11}")
for tname in ["skew_tubes","mixed_vortices"]:
    for nu in [2.5e-3, 1.0e-3, 4.0e-4]:
        setup(nu); env=make_env(N,L=L_BOX)
        U,_=topo(tname,env)
        # --- Stage A: short self-regulating pilot to locate equilibrium A ---
        Up=U.clone(); As=[]
        for _ in range(3*SEG_STEPS):
            a=A_equilibrium(Up,env,nu); As.append(a)
            Up=rk4(Up,DT,nu,env,a)
        A=float(np.median(As))
        # --- Stage B: fixed A, measure rank over a long horizon ---
        Ut=U.clone(); S=[Ut.clone()]
        every=max(1,(N_SEG*SEG_STEPS)//(SNAPS-1))
        for st in range(N_SEG*SEG_STEPS):
            Ut=rk4(Ut,DT,nu,env,A)
            if (st+1)%every==0 and len(S)<SNAPS: S.append(Ut.clone())
        r,m999,ceil=pod_rank(S)
        tf=tail_frac(Ut); ok=tf<1e-4; sat=r>=ceil-2
        rows.append(dict(flow=tname,nu=nu,A=A,rank=r,modes999=m999,
                         tail=tf,resolved=ok,saturated=sat,
                         E_ratio=energy(Ut)/max(energy(U),1e-300)))
        print(f"  {tname:<16}{nu:>10.1e}{A:>11.3e}{r:>7}{m999:>7}{tf:>11.2e}"
              f"{'yes' if ok else 'NO':>10}{'SAT' if sat else '':>11}", flush=True)
        del env,U,Up,Ut,S; torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(SAVE/"m3d_forced_probe.csv",index=False)
res=df[df.resolved]
print("\n" + "="*80)
print("FORCED-TURBULENCE RANK PROBE")
print("="*80)
print(f"  unforced decaying reference (BB3, any band): rank 4-11")
print(f"  forced, resolved configs:  rank {int(res['rank'].min()) if len(res) else 0}"
      f" .. {int(res['rank'].max()) if len(res) else 0}")
print(f"  energy sustained (final/initial): "
      f"{res.E_ratio.min() if len(res) else float('nan'):.3f}"
      f" .. {res.E_ratio.max() if len(res) else float('nan'):.3f}"
      f"   (near 1.0 means forcing is balancing dissipation)")
best = int(res['rank'].max()) if len(res) else 0
if best>=50:
    print("\n  -> PASS. Forcing produces a genuinely high-dimensional testbed.")
    print("     Build the full chain (Q reservoir -> leakage -> W -> X) on it.")
elif best>=25:
    print("\n  -> PARTIAL. Higher rank than decaying flow but short of 50.")
    print("     Lengthen the horizon (M3D_F_SEG) or raise N before deciding.")
else:
    print("\n  -> FAIL. Even forced, this solver/flow family stays low-dimensional.")
    print("     State it as a limitation; do not claim broadband generality.")
print("\nsaved:", SAVE/"m3d_forced_probe.csv")
