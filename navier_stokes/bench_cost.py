# ============================================================================
# M3D — WALL-CLOCK COST BENCHMARK
# ============================================================================
# The project's cost claim is 3(M*/N)^3, a grid-point count. It ignores the
# log N factor in FFT cost, and -- more importantly -- restriction and
# prolongation operate at the FINE resolution and do NOT scale down with M.
# That overhead is fixed in N while the coarse work shrinks as M^3, so it
# could dominate at large N and erode exactly the scaling the claim rests on.
#
# Measures:  C_wall = t[restrict + 3 coarse Phi + prolong] / t[one fine Phi]
# ============================================================================
import os, glob, time, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE=Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True,exist_ok=True)
BOOT=glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py",recursive=True)[0]
G={"__name__":"__m3d__"}; exec(compile(open(BOOT).read(),BOOT,"exec"),G,G)
def spaces():
    out,seen=[G],{id(G)}
    for _,o in list(G.items()):
        if callable(o) and hasattr(o,"__globals__") and id(o.__globals__) not in seen:
            out.append(o.__globals__); seen.add(id(o.__globals__))
    return out
def setup(nu):
    for ns in spaces():
        ns["NU1_D"]=ns["NU2_D"]=ns["nu1_D"]=ns["nu2_D"]=float(nu)
        ns["REAL_DTYPE_D"]=torch.float64; ns["COMPLEX_DTYPE_D"]=torch.complex128
make_env=G["make_env_D"]; topo=G["topology_pair_L2"]; full_rk4=G["full_rk4_D"]
dealias=G["dealias_D"]; leray=G["project_divfree_D"]; TAU0=float(G["TAU0_D"])
SEG_TAU,SEG_STEPS=0.15,20; DT=SEG_TAU*TAU0/SEG_STEPS; NU=2.5e-3
REPS=int(os.environ.get("M3D_B_REPS",5))

def _idx(n,m):
    h=m//2
    return (list(range(0,h+1))+list(range(n-h+1,n)),
            list(range(0,h+1))+list(range(m-h+1,m)))
def restrict(U,N,M):
    s,_=_idx(N,M); si=torch.tensor(s,device=U.device,dtype=torch.long)
    return U.index_select(1,si).index_select(2,si).index_select(3,si)*(float(M)/N)**3
def prolong(U,M,N):
    s,_=_idx(N,M); si=torch.tensor(s,device=U.device,dtype=torch.long)
    o=torch.zeros((3,N,N,N),dtype=U.dtype,device=U.device)
    o[:,si[:,None,None],si[None,:,None],si[None,None,:]]=U*(float(N)/M)**3
    return o
def timeit(fn,reps):
    fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(reps):
        torch.cuda.synchronize(); t0=time.perf_counter()
        fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))

rows=[]
print(f"{'N':>5}{'M':>5}{'nominal':>10}{'fine Phi':>11}{'estimator':>11}"
      f"{'restrict%':>11}{'MEASURED':>10}{'vs nominal':>12}")
for N,Ms in ((40,[28,20,14]),(64,[28,20]),(80,[28,20])):
    setup(NU); envF=make_env(N,L=8.0); U1,U2=topo("skew_tubes",envF)
    def fine():
        a,b=U1.clone(),U2.clone()
        for _ in range(SEG_STEPS):
            a=full_rk4(a,DT,NU,envF); b=full_rk4(b,DT,NU,envF)
        return a,b
    t_fine=timeit(fine,REPS)
    for M in Ms:
        envC=make_env(M,L=8.0)
        def rp_only():
            r1=dealias(leray(restrict(U1,N,M),envC),envC)
            r2=dealias(leray(restrict(U2,N,M),envC),envC)
            return prolong(r1,M,N),prolong(r2,M,N)
        def estimator():           # 3 coarse Phi + the restrict/prolong traffic
            for _ in range(3):
                a=dealias(leray(restrict(U1,N,M),envC),envC)
                b=dealias(leray(restrict(U2,N,M),envC),envC)
                for _ in range(SEG_STEPS):
                    a=full_rk4(a,DT,NU,envC); b=full_rk4(b,DT,NU,envC)
                a=prolong(a,M,N); b=prolong(b,M,N)
            return a,b
        t_est=timeit(estimator,REPS); t_rp=timeit(rp_only,REPS)
        nominal=3*(M/N)**3; meas=t_est/t_fine
        rows.append(dict(N=N,M=M,nominal=nominal,t_fine=t_fine,t_est=t_est,
                         t_restrict_prolong=t_rp,measured=meas,
                         rp_share=3*t_rp/max(t_est,1e-12),ratio=meas/nominal))
        print(f"{N:>5}{M:>5}{nominal:>10.4f}{1000*t_fine:>10.1f}m{1000*t_est:>10.1f}m"
              f"{100*3*t_rp/max(t_est,1e-12):>10.1f}%{meas:>10.4f}{meas/nominal:>11.2f}x",
              flush=True)
        del envC
    del envF,U1,U2; torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(SAVE/"m3d_cost_benchmark.csv",index=False)
print("\n"+"="*88); print("WALL-CLOCK vs NOMINAL GRID-POINT COST"); print("="*88)
print(f"  median measured/nominal = {df.ratio.median():.2f}x")
print(f"  restrict+prolong share of estimator time: "
      f"{100*df.rp_share.min():.1f}% .. {100*df.rp_share.max():.1f}%")
best=df.loc[df.measured.idxmin()]
print(f"\n  cheapest measured: N={int(best.N)} M={int(best.M)} -> "
      f"{best.measured:.4f} of one fine segment ({1/best.measured:.1f}x cheaper)")
for N in sorted(df.N.unique()):
    s=df[df.N==N]; b=s.loc[s.measured.idxmin()]
    verdict=("CHEAPER than the solver" if b.measured<1 else "NOT cheaper")
    print(f"  N={int(N)}: best {b.measured:.4f} ({verdict})")
print("\n  If measured/nominal is near 1, the grid-point model holds and the")
print("  paper may quote 3(M/N)^3. If it is well above 1, quote the MEASURED")
print("  number and report the restrict/prolong overhead explicitly.")
print("\nsaved:",SAVE/"m3d_cost_benchmark.csv")
