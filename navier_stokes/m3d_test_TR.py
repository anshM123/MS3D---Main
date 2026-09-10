# ============================================================================
# M3D TEST TR — IS "ENERGY != DANGER" A TRIADIC EFFECT?
# ============================================================================
# Test F found triadic enrichment of 120-243x for the LEADING leakage
# direction. That number is meaningless without a control: maybe ANY vector in
# the hidden-error subspace looks triadic.
#
# The decisive comparison orders the SAME 19-dimensional empirical Q-error
# subspace two different ways:
#
#   ENERGETIC ordering   POD eigenvectors of the Q-error covariance
#                        -> the directions carrying the most hidden error
#   DANGEROUS ordering   right singular vectors of T restricted to that
#                        subspace -> the directions that actually leak into P
#
# plus a RANDOM ordering as a floor. If triadic enrichment is high for the
# dangerous directions and low for the energetic ones, then
#
#       energy != danger
#
# has a Navier-Stokes mechanism: the dangerous hidden modes are the effective
# triadic partners of the resolved band, not the energetic ones. That is a
# fluid-dynamical claim, not an MOR observation.
#
# Also measures enrichment DIRECTION BY DIRECTION against the leakage singular
# value, so a monotone relationship (more dangerous -> more triadic) can be
# reported rather than a single number.
#
# Secondary: r_leak recomputed on a 9-anchor ambient subspace, matching the
# decaying-flow runs, so forced-vs-decaying leakage rank is apples-to-apples.
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE=Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True,exist_ok=True)
OUT=SAVE/"m3d_TR_triadic_control.csv"
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
N=int(os.environ.get("M3D_TR_N",64))
N_SEG=int(os.environ.get("M3D_TR_SEG",20))
NDIR=int(os.environ.get("M3D_TR_NDIR",5))       # directions per category
CONFIGS=[(t,nu) for t in ["skew_tubes","mixed_vortices"]
                for nu in [2.5e-3,1.0e-3,4.0e-4]]

_vrhs=G["velocity_rhs_D"]; BOX={"A":0.0}
def vrhs_forced(Uhat,nu,env): return _vrhs(Uhat,nu,env)+BOX["A"]*Uhat
def ajvp(Uh,Vh,nu,env):
    Uc=dealias(leray(Uh,env),env); Vc=dealias(leray(Vh,env),env)
    u=torch.fft.ifftn(Uc,dim=(-3,-2,-1)).real; v=torch.fft.ifftn(Vc,dim=(-3,-2,-1)).real
    wu=torch.fft.ifftn(curl(Uc,env),dim=(-3,-2,-1)).real
    wv=torch.fft.ifftn(curl(Vc,env),dim=(-3,-2,-1)).real
    cr=torch.empty_like(u)
    for i,(a,b) in enumerate(((1,2),(2,0),(0,1))):
        cr[i]=(u[a]*wv[b]-u[b]*wv[a])+(v[a]*wu[b]-v[b]*wu[a])
    nh=dealias(leray(torch.fft.fftn(cr,dim=(-3,-2,-1)),env),env)
    base=dealias(leray(nh-float(nu)*env["K2"].unsqueeze(0)*Vc,env),env)
    return (base+BOX["A"]*Vc).to(G["COMPLEX_DTYPE_D"])
def pair_ajvp(U1,U2,V1,V2,env):
    return (ajvp(U1,V1,G["NU1_D"],env), ajvp(U2,V2,G["NU2_D"],env))
for ns in spaces():
    if "velocity_rhs_D" in ns: ns["velocity_rhs_D"]=vrhs_forced
    if "pair_jvp_M" in ns:     ns["pair_jvp_M"]=pair_ajvp
G["velocity_rhs_D"]=vrhs_forced; G["pair_jvp_M"]=pair_ajvp

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

# --- verify the forcing patch reached full_rk4_D ---
setup(2.5e-3); ev=make_env(N,L=L_BOX); Uv,_=topo("skew_tubes",ev)
BOX["A"]=0.0; a=Uv.clone()
for _ in range(SEG_STEPS): a=full_rk4(a,DT,2.5e-3,ev)
e0=energy(a)/energy(Uv)
BOX["A"]=diss(Uv,ev,2.5e-3)/max(energy(Uv),1e-300); b=Uv.clone()
for _ in range(SEG_STEPS): b=full_rk4(b,DT,2.5e-3,ev)
e1=energy(b)/energy(Uv)
print(f"patch check: unforced={e0:.4f} forced={e1:.4f}",flush=True)
if abs(e1-e0)<1e-6: raise SystemExit("ABORT: forcing patch did not take.")
BOX["A"]=0.0; del ev,Uv,a,b; torch.cuda.empty_cache()

rows=[]; prev=pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done={(r["flow"],r["nu"]) for r in prev}; rows.extend(prev)

for tname,nu in CONFIGS:
  if (tname,nu) in done: print(f"SKIP {tname} nu={nu:.1e}",flush=True); continue
  try:
    print(f"\n=== {tname}  nu={nu:.2e} ===",flush=True)
    setup(nu); env=make_env(N,L=L_BOX)
    U1,U2=topo(tname,env)
    BOX["A"]=0.0; p1=U1.clone(); As=[]
    for _ in range(3*SEG_STEPS):
        BOX["A"]=diss(p1,env,nu)/max(energy(p1),1e-300); As.append(BOX["A"])
        p1=full_rk4(p1,DT,nu,env)
    BOX["A"]=float(np.median(As)); del p1
    print(f"  fixed A={BOX['A']:.4e}",flush=True)

    star,_=init2(U1,U2,env); tids=star["proj"]["target_ids"]; del star
    t1,t2=U1.clone(),U2.clone(); m,_=init2(U1,U2,env)
    V1,V2=U1.clone(),U2.clone()
    TR,MO=[(t1.clone(),t2.clone())],[(V1.clone(),V2.clone())]
    for seg in range(N_SEG):
        for _ in range(SEG_STEPS):
            t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
            rk4M(m,SEG_TAU/SEG_STEPS,env)
        V1,V2=recon(m,env); m,_=init2(V1,V2,env)
        TR.append((t1.clone(),t2.clone())); MO.append((V1.clone(),V2.clone()))

    # --- empirical Q-error subspace, two anchor counts ---
    for label,anch in (("full",list(range(2,len(MO)))),
                       ("matched9",list(range(2,11)))):
        B1,B2,raw=[],[],[]
        for k in anch:
            _,_,a1,a2=splitPQ(TR[k][0]-MO[k][0],TR[k][1]-MO[k][1],tids)
            raw.append((a1.clone(),a2.clone()))
            w1,w2=a1.clone(),a2.clone()
            for e1,e2 in zip(B1,B2):
                c=redot(e1,e2,w1,w2); w1=w1-c*e1; w2=w2-c*e2
            nw=pnorm(w1,w2)
            if nw>1e-13*max(pnorm(a1,a2),1e-300): B1.append(w1/nw); B2.append(w2/nw)
        nb=len(B1)
        ka=anch[-1]; Va1,Va2=MO[ka]
        base1,base2=phi(Va1,Va2,env,nu); scale=max(pnorm(Va1,Va2),1e-30)
        cols=[]
        for e1,e2 in zip(B1,B2):
            eps=1e-6*scale; q1,q2=phi(Va1+eps*e1,Va2+eps*e2,env,nu)
            cols.append(packm((q1-base1)/eps,(q2-base2)/eps,tids))
        Y=torch.stack(cols,dim=1)
        Uu,Sv,Vh=torch.linalg.svd(Y,full_matrices=False)
        e2v=(Sv.to(torch.float64)**2); tot=float(e2v.sum())
        cum=torch.cumsum(e2v,0)/tot
        r_leak90=int((cum<0.90).sum())+1
        top3=float(e2v[:3].sum()/tot)

        if label=="matched9":
            rows.append(dict(flow=tname,nu=nu,ordering="__rank_only__",idx=-1,
                ambient=nb,r_leak90=r_leak90,top3=top3,sigma=float("nan"),
                enrichment=float("nan"),frac=float("nan"),null=float("nan")))
            print(f"  [matched9] ambient={nb} r_leak90={r_leak90} top3={top3:.4f}",
                  flush=True)
            continue

        # --- triadic partner set S = {p - q} ---
        eb=(np.abs(Va1.detach().cpu().numpy())**2).sum(axis=0).ravel()
        order=np.argsort(eb)[::-1]; cse=np.cumsum(eb[order])/max(eb.sum(),1e-300)
        flow=order[:max(1,int((cse<0.90).sum())+1)]
        shp=(N,N,N); pv=np.stack(np.unravel_index(tids.detach().cpu().numpy(),shp))
        fv=np.stack(np.unravel_index(flow,shp)); Sset=set()
        for a in range(pv.shape[1]):
            Sset.update(np.ravel_multi_index((pv[:,a:a+1]-fv)%N,shp).tolist())
        Sidx=np.fromiter(Sset,dtype=np.int64)
        def enrich_of(vec1):
            qe=(np.abs(vec1.detach().cpu().numpy())**2).sum(axis=0).ravel()
            fr=float(qe[Sidx].sum()/max(qe.sum(),1e-300))
            nl=len(Sidx)/qe.size
            return fr, nl, fr/max(nl,1e-300)

        right=Vh.conj().transpose(0,1)
        def combo(coef):
            return sum(coef[i].to(B1[0].dtype)*B1[i] for i in range(nb))

        # ENERGETIC ordering: POD of the Q-error snapshots in this basis
        Cm=torch.zeros(nb,nb,dtype=torch.float64)
        for a1,a2 in raw:
            c=torch.tensor([redot(B1[i],B2[i],a1,a2) for i in range(nb)],
                           dtype=torch.float64)
            Cm+=torch.outer(c,c)
        evals,evecs=torch.linalg.eigh(Cm)
        eorder=torch.argsort(evals,descending=True)
        gen=torch.Generator().manual_seed(7)

        for i in range(min(NDIR,nb)):
            for ordering,coef,sig in (
                ("dangerous", right[:,i],            float(Sv[i])),
                ("dangerous_tail", right[:,nb-1-i],  float(Sv[nb-1-i])),
                ("energetic", evecs[:,eorder[i]].to(torch.float64), float("nan")),
                ("random",  torch.randn(nb,generator=gen,dtype=torch.float64),
                            float("nan"))):
                c=coef/torch.linalg.vector_norm(coef)
                fr,nl,en=enrich_of(combo(c.to(torch.complex128)))
                rows.append(dict(flow=tname,nu=nu,ordering=ordering,idx=i,
                    ambient=nb,r_leak90=r_leak90,top3=top3,sigma=sig,
                    enrichment=en,frac=fr,null=nl))
        for o in ("dangerous","energetic","random","dangerous_tail"):
            v=[r["enrichment"] for r in rows if r["flow"]==tname and r["nu"]==nu
               and r["ordering"]==o]
            if v: print(f"    {o:<16} median enrichment {np.median(v):8.1f}x",flush=True)
        del B1,B2,raw,cols,Y
    pd.DataFrame(rows).to_csv(OUT,index=False)
    BOX["A"]=0.0
    del env,U1,U2,t1,t2,V1,V2,m,TR,MO; torch.cuda.empty_cache()
  except Exception as exc:
    print(f"  CONFIG FAILED: {exc!r}",flush=True); BOX["A"]=0.0
    torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
d=df[df.ordering!="__rank_only__"]
print("\n"+"="*100); print("M3D TEST TR — IS 'ENERGY != DANGER' TRIADIC?"); print("="*100)
print(f"\n  {'ordering':<18}{'n':>5}{'median enrich':>16}{'p10':>10}{'p90':>10}")
for o in ("dangerous","energetic","random","dangerous_tail"):
    s=d[d.ordering==o]
    if s.empty: continue
    print(f"  {o:<18}{len(s):>5}{s.enrichment.median():>16.1f}"
          f"{s.enrichment.quantile(0.1):>10.1f}{s.enrichment.quantile(0.9):>10.1f}")
dd=d[d.ordering=="dangerous"].enrichment.median()
ee=d[d.ordering=="energetic"].enrichment.median()
rr=d[d.ordering=="random"].enrichment.median()
print(f"\n  dangerous / energetic = {dd/max(ee,1e-30):.2f}x")
print(f"  dangerous / random    = {dd/max(rr,1e-30):.2f}x")
sep = dd > 2*ee
line1 = ("ENERGY != DANGER IS TRIADIC: the leaking directions are the"
         if sep else "NOT SEPARATED: enrichment is a property of the subspace,")
line2 = ("triadic partners of P, the energetic ones are not."
         if sep else "not of the dangerous directions specifically.")
print(f"\n  -> {line1}")
print(f"     {line2}")
print("\n--- enrichment vs leakage singular value (per direction) ---")
s=d[(d.ordering=="dangerous")&np.isfinite(d.sigma)]
if len(s)>=4:
    x=np.log(s.sigma.values); y=np.log(s.enrichment.values)
    p=np.polyfit(x,y,1)[0]; r=np.corrcoef(x,y)[0,1]
    print(f"  log-log slope {p:+.3f}   R^2={r*r:.3f}   n={len(s)}")
    verdict=("monotone: more dangerous = more triadic" if r>0.4
             else "no clear monotone relation")
    print(f"  -> {verdict}")
print("\n--- leakage rank on a MATCHED 9-anchor ambient (vs decaying flows) ---")
mm=df[df.ordering=="__rank_only__"]
for _,r in mm.iterrows():
    print(f"  {r.flow:<16} nu={r.nu:.1e}  ambient={int(r.ambient)}  "
          f"r_leak90={int(r.r_leak90)}  top3={r.top3:.4f}   (decaying: 9, 3, ~0.93)")
print("\nsaved:",OUT)
