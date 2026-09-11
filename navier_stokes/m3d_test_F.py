# ============================================================================
# M3D TEST F — FULL CHAIN ON FORCED TURBULENCE
# ============================================================================
# The forced probe raised trajectory rank from 4-11 (unforced) to 17-33, with
# rank ~ nu^-0.273 (R^2=0.99) and NOT saturated. The limit is now resolution:
# rank 50 would need N ~ 800. So this runs the full chain on the ladder that
# IS reachable, and states the range honestly.
#
# Lundgren linear forcing f = A*u with A CONSTANT, so F stays EXACTLY
# QUADRATIC and the analytic JVP / FD-exactness results survive.
#
# Claim being tested: does r_leak stay ~3 while r_trajectory spans 17 -> 33
# under sustained forced dynamics, as it did for 8 -> 27 under decay?
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE=Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True,exist_ok=True)
OUT=SAVE/"m3d_F_forced_chain.csv"
BOOT=glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py",recursive=True)[0]
print("bootstrap:",BOOT,flush=True)
G={"__name__":"__m3d__"}
exec(compile(open(BOOT).read(),BOOT,"exec"),G,G)

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

make_env=G["make_env_D"]; topo=G["topology_pair_L2"]
full_rk4=G["full_rk4_D"]; init2=G["initialize_star2_N"]
rk4M=G["rk4_M"]; recon=G["reconstruct_M"]; packm=G["pack_modes_D"]
dealias=G["dealias_D"]; leray=G["project_divfree_D"]; curl=G["curl_hat_D"]
TAU0=float(G["TAU0_D"])

L_BOX,SEG_TAU,SEG_STEPS=8.0,0.15,20
DT=SEG_TAU*TAU0/SEG_STEPS
N=int(os.environ.get("M3D_F_N",64))
N_SEG=int(os.environ.get("M3D_F_SEG",20))
SNAPS=51
FORECAST_SEGS,DERIV_REL=2,1e-5
CONFIGS=[(t,nu) for t in ["skew_tubes","mixed_vortices"]
                for nu in [2.5e-3,1.0e-3,4.0e-4]]

# ---- forced RHS: A is a CONSTANT, so F remains exactly quadratic ----
_vrhs=G["velocity_rhs_D"]
BOX={"A":0.0}
def vrhs_forced(Uhat,nu,env):
    return _vrhs(Uhat,nu,env)+BOX["A"]*Uhat
def ajvp_forced(Uh,Vh,nu,env):
    Uc=dealias(leray(Uh,env),env); Vc=dealias(leray(Vh,env),env)
    u=torch.fft.ifftn(Uc,dim=(-3,-2,-1)).real; v=torch.fft.ifftn(Vc,dim=(-3,-2,-1)).real
    wu=torch.fft.ifftn(curl(Uc,env),dim=(-3,-2,-1)).real
    wv=torch.fft.ifftn(curl(Vc,env),dim=(-3,-2,-1)).real
    cr=torch.empty_like(u)
    for i,(a,b) in enumerate(((1,2),(2,0),(0,1))):
        cr[i]=(u[a]*wv[b]-u[b]*wv[a])+(v[a]*wu[b]-v[b]*wu[a])
    nh=dealias(leray(torch.fft.fftn(cr,dim=(-3,-2,-1)),env),env)
    base=dealias(leray(nh-float(nu)*env["K2"].unsqueeze(0)*Vc,env),env)
    return (base+BOX["A"]*Vc).to(G["COMPLEX_DTYPE_D"])   # + A*V from forcing
def pair_ajvp(U1,U2,V1,V2,env):
    return (ajvp_forced(U1,V1,G["NU1_D"],env), ajvp_forced(U2,V2,G["NU2_D"],env))
for ns in spaces():
    if "velocity_rhs_D" in ns: ns["velocity_rhs_D"]=vrhs_forced
    if "pair_jvp_M" in ns:     ns["pair_jvp_M"]=pair_ajvp
G["velocity_rhs_D"]=vrhs_forced; G["pair_jvp_M"]=pair_ajvp
print("patched: forced RHS + analytic JVP with forcing term",flush=True)

def energy(U): return float(torch.real(torch.vdot(U.reshape(-1),U.reshape(-1))).cpu())
def diss(U,env,nu):
    return float(nu*torch.real(torch.sum(env["K2"].unsqueeze(0)*torch.abs(U)**2)).cpu())
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
def tail_frac(U):
    e=(np.abs(U.detach().cpu().numpy())**2).sum(axis=0)
    f=np.fft.fftfreq(N)*N; KX,KY,KZ=np.meshgrid(f,f,f,indexing="ij")
    km=np.sqrt(KX**2+KY**2+KZ**2)
    return float(e[km>0.8*(N/3.0)].sum()/max(e.sum(),1e-300))
def pod_rank(S1,S2):
    n=len(S1); m1=sum(S1[1:],S1[0].clone())/n; m2=sum(S2[1:],S2[0].clone())/n
    F1=[s-m1 for s in S1]; F2=[s-m2 for s in S2]
    Gm=torch.zeros(n,n,dtype=torch.float64)
    for i in range(n):
        for j in range(i,n):
            v=redot(F1[i],F2[i],F1[j],F2[j]); Gm[i,j]=v; Gm[j,i]=v
    ev=torch.linalg.eigvalsh(Gm).flip(0).clamp(min=0)
    r=int((ev>ev[0]*1e-9).sum()); tot=float(ev.sum()); c=0.0; m9=len(ev)
    for i,e in enumerate(ev):
        c+=float(e)
        if c>=0.999*tot: m9=i+1; break
    return r,m9,n

# ---------------- verify the patch actually reached full_rk4_D ----------------
setup(2.5e-3); envv=make_env(N,L=L_BOX); Uv,_=topo("skew_tubes",envv)
BOX["A"]=0.0; a=Uv.clone()
for _ in range(SEG_STEPS): a=full_rk4(a,DT,2.5e-3,envv)
e_unf=energy(a)/energy(Uv)
BOX["A"]=diss(Uv,envv,2.5e-3)/max(energy(Uv),1e-300); b=Uv.clone()
for _ in range(SEG_STEPS): b=full_rk4(b,DT,2.5e-3,envv)
e_for=energy(b)/energy(Uv)
print(f"patch check: energy ratio unforced={e_unf:.4f}  forced={e_for:.4f}",flush=True)
if abs(e_for-e_unf)<1e-6:
    raise SystemExit("ABORT: forcing did not reach full_rk4_D; the patch failed.")
print("  -> forcing IS active inside full_rk4_D\n",flush=True)
BOX["A"]=0.0; del envv,Uv,a,b; torch.cuda.empty_cache()

rows=[]; prev=pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done={(r["flow"],r["nu"]) for r in prev}; rows.extend(prev)

for tname,nu in CONFIGS:
  if (tname,nu) in done:
    print(f"SKIP {tname} nu={nu:.1e}",flush=True); continue
  try:
      print(f"\n=== FORCED {tname}  nu={nu:.2e}  N={N} ===",flush=True)
      setup(nu); env=make_env(N,L=L_BOX)
      U1,U2=topo(tname,env)
      # pilot for the equilibrium A, then FIX it (keeps F quadratic)
      BOX["A"]=0.0; p1,p2=U1.clone(),U2.clone(); As=[]
      for _ in range(3*SEG_STEPS):
          BOX["A"]=diss(p1,env,nu)/max(energy(p1),1e-300); As.append(BOX["A"])
          p1=full_rk4(p1,DT,nu,env); p2=full_rk4(p2,DT,nu,env)
      BOX["A"]=float(np.median(As))
      print(f"  fixed A = {BOX['A']:.4e}",flush=True)
      del p1,p2

      star,_=init2(U1,U2,env); tids=star["proj"]["target_ids"]; del star
      t1,t2=U1.clone(),U2.clone(); m,_=init2(U1,U2,env)
      V1,V2=U1.clone(),U2.clone()
      S1,S2=[t1.clone()],[t2.clone()]
      TR,MO=[(t1.clone(),t2.clone())],[(V1.clone(),V2.clone())]
      every=max(1,(N_SEG*SEG_STEPS)//(SNAPS-1)); step=0
      for seg in range(1,N_SEG+1):
          for _ in range(SEG_STEPS):
              t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
              rk4M(m,SEG_TAU/SEG_STEPS,env); step+=1
              if step%every==0 and len(S1)<SNAPS: S1.append(t1.clone()); S2.append(t2.clone())
          V1,V2=recon(m,env); m,_=init2(V1,V2,env)
          TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))
      r_traj,m999,ceil=pod_rank(S1,S2); tail=tail_frac(t1)
      Eratio=energy(t1)/max(energy(U1),1e-300); del S1,S2

      errs=[terr(MO[k][0],MO[k][1],TR[k][0],TR[k][1],tids) for k in range(len(MO))]
      # Theorem C's remainder is O(||e||^3); Test V validated second-order
      # transport at percent-level amplitudes. At 35% error the H[e,e] term
      # dominates, e_hat diverges, and V+Qe_hat becomes a state where STAR2 is
      # degenerate. Cap the band accordingly.
      ERR_MAX=float(os.environ.get("M3D_F_ERRMAX",0.15))
      usable=[k for k in range(2,len(MO)-FORECAST_SEGS) if 0.002<=errs[k]<=ERR_MAX]
      anchors=[usable[0],usable[len(usable)//2],usable[-1]] if len(usable)>=3 \
              else [k for k in range(2,len(MO)-FORECAST_SEGS)][:3]
      print(f"  r_traj={r_traj} (ceil {ceil}, 99.9%: {m999})  tail={tail:.2e}  "
            f"E_final/E_0={Eratio:.3f}")
      print(f"  anchors tau={[round(a*SEG_TAU,2) for a in anchors]}  "
            f"err={[f'{100*errs[a]:.2f}%' for a in anchors]}",flush=True)

      B1,B2=[],[]
      for k in range(2,len(MO)):
          _,_,a1,a2=splitPQ(TR[k][0]-MO[k][0],TR[k][1]-MO[k][1],tids)
          w1,w2=a1.clone(),a2.clone()
          for e1,e2 in zip(B1,B2):
              cc=redot(e1,e2,w1,w2); w1=w1-cc*e1; w2=w2-cc*e2
          nw=pnorm(w1,w2)
          if nw>1e-13*max(pnorm(a1,a2),1e-300): B1.append(w1/nw); B2.append(w2/nw)
      r_emp=len(B1)
      Va1,Va2=MO[anchors[-1]]; base1,base2=phi(Va1,Va2,env,nu)
      scale=max(pnorm(Va1,Va2),1e-30); cols=[]
      for e1,e2 in zip(B1,B2):
          eps=1e-6*scale; q1,q2=phi(Va1+eps*e1,Va2+eps*e2,env,nu)
          cols.append(packm((q1-base1)/eps,(q2-base2)/eps,tids))
      Y=torch.stack(cols,dim=1); Uu,Sv,Vh=torch.linalg.svd(Y,full_matrices=False)
      e2v=(Sv.to(torch.float64)**2); tot=float(e2v.sum())
      top3=float(e2v[:3].sum()/tot); cum=torch.cumsum(e2v,0)/tot
      r_leak90=int((cum<0.90).sum())+1
      dP1,dP2,dQ1,dQ2=splitPQ(TR[anchors[-1]][0]-Va1,TR[anchors[-1]][1]-Va2,tids)
      reservoir=pnorm(dQ1,dQ2)/max(pnorm(dP1,dP2),1e-30)

      ql=sum(Vh.conj().transpose(0,1)[i,0].to(B1[0].dtype)*B1[i] for i in range(len(B1)))
      qe=(np.abs(ql.detach().cpu().numpy())**2).sum(axis=0).ravel()
      eb=(np.abs(Va1.detach().cpu().numpy())**2).sum(axis=0).ravel()
      order=np.argsort(eb)[::-1]; cse=np.cumsum(eb[order])/max(eb.sum(),1e-300)
      flow=order[:max(1,int((cse<0.90).sum())+1)]
      shp=(N,N,N); pv=np.stack(np.unravel_index(tids.detach().cpu().numpy(),shp))
      fv=np.stack(np.unravel_index(flow,shp)); Sset=set()
      for a in range(pv.shape[1]):
          Sset.update(np.ravel_multi_index((pv[:,a:a+1]-fv)%N,shp).tolist())
      Sidx=np.fromiter(Sset,dtype=np.int64)
      frac_S=float(qe[Sidx].sum()/max(qe.sum(),1e-300)); null_S=len(Sidx)/qe.size
      enrich=frac_S/max(null_S,1e-300)

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
          # guard: if the estimate has diverged, the expansion is out of regime
          rel_eq=pnorm(EQ1,EQ2)/max(pnorm(Vi1,Vi2),1e-30)
          if not np.isfinite(rel_eq) or rel_eq>0.5:
              print(f"    tau={k*SEG_TAU:.2f}: SKIP, |Qe_hat|/|V|={rel_eq:.3f} "
                    f"(estimator out of regime)",flush=True)
              rows.append(dict(flow=tname,nu=nu,A=BOX["A"],N=N,tau=round(k*SEG_TAU,3),
                  r_trajectory=r_traj,rank_ceiling=ceil,modes_999=m999,tail_frac=tail,
                  E_ratio=Eratio,r_empirical_Q=r_emp,top3_energy=top3,r_leak90=r_leak90,
                  reservoir=reservoir,triadic_frac=frac_S,triadic_null=null_S,
                  triadic_enrichment=enrich,start_mismatch=float("nan"),
                  baseline_P_error=errs[k],est_P_error=float("nan"),
                  neg_P_error=float("nan"),gain=float("nan"),beats=False,
                  beats_neg=False,hidden_Q_residual=float("nan"),
                  rel_estimate=rel_eq,status="estimator_diverged"))
              continue
          xb=packm(Vi1,Vi2,tids); xe=packm(Vi1+EQ1,Vi2+EQ2,tids)
          mism=float((torch.linalg.vector_norm(xe-xb)/torch.linalg.vector_norm(xb)).real.cpu())
          try:
              FB=forecast(Vi1,Vi2); FE=forecast(Vi1+EQ1,Vi2+EQ2); FN=forecast(Vi1-EQ1,Vi2-EQ2)
          except RuntimeError as exc:
              print(f"    tau={k*SEG_TAU:.2f}: SKIP, forecast failed ({exc})",flush=True)
              rows.append(dict(flow=tname,nu=nu,A=BOX["A"],N=N,tau=round(k*SEG_TAU,3),
                  r_trajectory=r_traj,rank_ceiling=ceil,modes_999=m999,tail_frac=tail,
                  E_ratio=Eratio,r_empirical_Q=r_emp,top3_energy=top3,r_leak90=r_leak90,
                  reservoir=reservoir,triadic_frac=frac_S,triadic_null=null_S,
                  triadic_enrichment=enrich,start_mismatch=mism,
                  baseline_P_error=errs[k],est_P_error=float("nan"),
                  neg_P_error=float("nan"),gain=float("nan"),beats=False,
                  beats_neg=False,hidden_Q_residual=float("nan"),
                  rel_estimate=rel_eq,status="star2_degenerate"))
              continue
          be=terr(*FB,TF1,TF2,tids); ee=terr(*FE,TF1,TF2,tids); ne=terr(*FN,TF1,TF2,tids)
          gains.append(be/max(ee,1e-30))
          rows.append(dict(flow=tname,nu=nu,A=BOX["A"],N=N,tau=round(k*SEG_TAU,3),
              r_trajectory=r_traj,rank_ceiling=ceil,modes_999=m999,tail_frac=tail,
              E_ratio=Eratio,r_empirical_Q=r_emp,top3_energy=top3,r_leak90=r_leak90,
              reservoir=reservoir,triadic_frac=frac_S,triadic_null=null_S,
              triadic_enrichment=enrich,start_mismatch=mism,baseline_P_error=be,
              est_P_error=ee,neg_P_error=ne,gain=be/max(ee,1e-30),
              beats=bool(ee<be),beats_neg=bool(ee<ne),
              hidden_Q_residual=pnorm(EQ1-AQ1,EQ2-AQ2)/max(pnorm(AQ1,AQ2),1e-30),
              rel_estimate=rel_eq,status="ok"))
      print(f"  r_empQ={r_emp}  top3={top3:.4f}  r_leak90={r_leak90}  "
            f"Q/P={reservoir:.2f}  triad={enrich:.1f}x")
      print(f"  gains={[f'{g:.3f}' for g in gains]}  "
            f"wins {sum(1 for g in gains if g>1)}/{len(gains)}",flush=True)
      pd.DataFrame(rows).to_csv(OUT,index=False)
      BOX["A"]=0.0
      del env,U1,U2,t1,t2,V1,V2,m,TR,MO,B1,B2; torch.cuda.empty_cache()
  except Exception as exc:
    print(f"  CONFIG FAILED: {exc!r}",flush=True)
    BOX["A"]=0.0; torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
g=df.groupby(["flow","nu"]).first().reset_index().sort_values("r_trajectory")
print("\n"+"="*112); print("M3D TEST F — FULL CHAIN ON FORCED TURBULENCE"); print("="*112)
print(f"\n  {'flow':<16}{'nu':>9}{'r_traj':>8}{'99.9%':>7}{'tail':>10}{'E_r':>7}"
      f"{'r_empQ':>8}{'top3':>8}{'r_leak':>8}{'Q/P':>7}{'triad':>8}{'gain':>9}{'wins':>7}")
for _,r in g.iterrows():
    s=df[(df.flow==r.flow)&(df.nu==r.nu)]
    print(f"  {r.flow:<16}{r.nu:>9.1e}{int(r.r_trajectory):>8}{int(r.modes_999):>7}"
          f"{r.tail_frac:>10.1e}{r.E_ratio:>7.2f}{int(r.r_empirical_Q):>8}"
          f"{r.top3_energy:>8.4f}{int(r.r_leak90):>8}{r.reservoir:>7.2f}"
          f"{r.triadic_enrichment:>8.1f}{s.gain.median():>9.4f}"
          f"{int(s.beats.sum()):>4}/{len(s)}")
strict=g[g.tail_frac<1e-4]; loose=g[g.tail_frac<1e-3]
print(f"\n  strictly resolved (tail<1e-4): {len(strict)}/{len(g)} configs")
print(f"  acceptably resolved (tail<1e-3): {len(loose)}/{len(g)} configs")
print("\n--- F1: leakage rank vs trajectory rank under FORCING ---")
for _,r in g.iterrows():
    unif=3.0/max(r.r_empirical_Q,1)
    print(f"  r_traj={int(r.r_trajectory):>3}  ambient={int(r.r_empirical_Q):>3}  "
          f"r_leak90={int(r.r_leak90):>3}  top3={r.top3_energy:.4f}  "
          f"vs uniform {unif:.3f} = {r.top3_energy/unif:.2f}x  "
          f"r_leak/ambient={r.r_leak90/max(r.r_empirical_Q,1):.3f}")
if len(g)>=3:
    p,_=np.polyfit(np.log(g.r_trajectory.values),np.log(g.r_leak90.values),1)
    print(f"  log-log slope d(log r_leak)/d(log r_traj) = {p:+.3f}")
    print(f"  -> {'leakage rank essentially INVARIANT' if p<0.25 else ('sublinear growth' if p<0.75 else 'tracks state rank')}")
print("\n--- F2: does the causal correction survive forcing? ---")
print(f"  win rate {int(df.beats.sum())}/{len(df)}  vs wrong-sign "
      f"{int(df.beats_neg.sum())}/{len(df)}  median gain {df.gain.median():.4f}x")
print(f"  max start-P mismatch {df.start_mismatch.max():.3e}  "
      f"median hidden-Q residual {df.hidden_Q_residual.median():.3e}")
print("\n--- F3: triadic enrichment on forced flows ---")
for _,r in g.iterrows():
    print(f"  r_traj={int(r.r_trajectory):>3}  enrichment {r.triadic_enrichment:7.1f}x"
          f"  (frac {r.triadic_frac:.4f} vs null {r.triadic_null:.5f})")
print(f"  median {g.triadic_enrichment.median():.1f}x   "
      f"(decaying flows gave 91x and 147x)")
print("\nsaved:",OUT)
