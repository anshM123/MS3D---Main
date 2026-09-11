# ============================================================================
# M3D TEST L — WHAT ARE THE LEAKAGE DIRECTIONS, PHYSICALLY?
# ============================================================================
# The JFM thread. R3-fix showed 3 directions carry ~93% of unresolved ->
# resolved feedback, stable across resolution and Reynolds number. This asks
# what they ARE, on ALL FOUR topologies and FOUR viscosities (16 configs),
# instead of the single topology the earlier spectra used.
#
# IMPORTANT correction to an earlier claim: a Spearman correlation against
# Re^(3/4) is NOT evidence for the 3/4 exponent -- Re^(3/4) is a monotone
# function of 1/nu, so ANY increasing relationship gives rank correlation 1.
# Testing the exponent requires a LOG-LOG FIT, which is what this does.
#
# Diagnostics per direction:
#   spectral    centroid, k90, and the fitted exponent in k_leak ~ nu^(-p)
#   triadic     is k_leak ~ k_P + k_flow, the signature of one triadic step
#               from the resolved band?
#   vortical    relative helicity; alignment of the direction's vorticity with
#               the base flow's strain-rate eigenvectors (the classic
#               vortex-stretching diagnostic: alignment with e2)
#   spatial     does the direction live where the base flow has high
#               enstrophy, or high strain?
# ============================================================================
import os, glob, math
import torch, numpy as np, pandas as pd
from pathlib import Path

SAVE = Path("/kaggle/working/M3D"); SAVE.mkdir(parents=True, exist_ok=True)
OUT = SAVE / "m3d_L_leakage_identity.csv"
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

make_env = G["make_env_D"];  topo  = G["topology_pair_L2"]
full_rk4 = G["full_rk4_D"];  init2 = G["initialize_star2_N"]
rk4M     = G["rk4_M"];       recon = G["reconstruct_M"]
packm    = G["pack_modes_D"]; TAU0 = float(G["TAU0_D"])

L_BOX, SEG_TAU, SEG_STEPS, N_SEG = 8.0, 0.15, 20, 10
DT = SEG_TAU * TAU0 / SEG_STEPS
NRES = int(os.environ.get("M3D_L_N", 64))
ANCHORS = [2,3,4,5,6,7,8,9,10]
TOPOS = ["vortex_ring","periodic_shear","skew_tubes","mixed_vortices"]
NUS = [2.5e-3, 1.0e-3, 4.0e-4, 1.6e-4]

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

def spectrum_np(u):
    N=u.shape[-1]; uh=np.fft.fftn(u,axes=(1,2,3))/N**3
    e=0.5*np.sum(np.abs(uh)**2,axis=0); f=np.fft.fftfreq(N)*N
    KX,KY,KZ=np.meshgrid(f,f,f,indexing="ij")
    kb=np.rint(np.sqrt(KX**2+KY**2+KZ**2)).astype(int); nb=N//2+1
    E=np.bincount(kb.ravel(),weights=e.ravel(),minlength=nb)[:nb]
    return np.arange(nb), E

def grad_tensor(u,N):
    """du_i/dx_j at every point, spectrally."""
    uh=np.fft.fftn(u,axes=(1,2,3))
    f=np.fft.fftfreq(N)*N*(2*math.pi/L_BOX)
    K=np.stack(np.meshgrid(f,f,f,indexing="ij"))
    Ghat=1j*uh[:,None,...]*K[None,...]
    return np.real(np.fft.ifftn(Ghat,axes=(2,3,4)))   # (3,3,N,N,N)

def vorticity(u,N):
    Gr=grad_tensor(u,N)
    return np.stack([Gr[2,1]-Gr[1,2], Gr[0,2]-Gr[2,0], Gr[1,0]-Gr[0,1]])

rows=[]
prev=pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
done={(r["topology"],r["nu"],r["direction"]) for r in prev}
rows.extend(prev)

for tname in TOPOS:
    for nu in NUS:
        if all((tname,nu,d) in done for d in range(3)):
            print(f"SKIP {tname} nu={nu:.1e}", flush=True); continue
        print(f"\n=== {tname}  nu={nu:.3e}  N={NRES} ===", flush=True)
        setup(nu)
        env=make_env(NRES,L=L_BOX)
        U1,U2=topo(tname,env)
        star,_=init2(U1,U2,env); tids=star["proj"]["target_ids"]; del star

        t1,t2=U1.clone(),U2.clone(); m,_=init2(U1,U2,env)
        V1,V2=U1.clone(),U2.clone(); dQ1s,dQ2s,Vs=[],[],[]
        for seg in range(1,N_SEG+1):
            for _ in range(SEG_STEPS):
                t1=full_rk4(t1,DT,nu,env); t2=full_rk4(t2,DT,nu,env)
                rk4M(m,SEG_TAU/SEG_STEPS,env)
            V1,V2=recon(m,env); m,_=init2(V1,V2,env)
            if seg in ANCHORS:
                _,_,q1,q2=splitPQ(t1-V1,t2-V2,tids)
                dQ1s.append(q1.clone()); dQ2s.append(q2.clone())
                Vs.append((V1.clone(),V2.clone()))

        B1,B2=[],[]
        for a1,a2 in zip(dQ1s,dQ2s):
            w1,w2=a1.clone(),a2.clone()
            for e1,e2 in zip(B1,B2):
                c=redot(e1,e2,w1,w2); w1=w1-c*e1; w2=w2-c*e2
            nw=pnorm(w1,w2)
            if nw>1e-13*max(pnorm(a1,a2),1e-300):
                B1.append(w1/nw); B2.append(w2/nw)

        Va1,Va2=Vs[-1]
        base1,base2=phi(Va1,Va2,env,nu)
        scale=max(pnorm(Va1,Va2),1e-30); cols=[]
        for e1,e2 in zip(B1,B2):
            eps=1e-6*scale
            p1,p2=phi(Va1+eps*e1,Va2+eps*e2,env,nu)
            cols.append(packm((p1-base1)/eps,(p2-base2)/eps,tids))
        Y=torch.stack(cols,dim=1)
        Uu,S,Vh=torch.linalg.svd(Y,full_matrices=False)
        e2v=(S.to(torch.float64)**2); top3=float(e2v[:3].sum()/e2v.sum())
        right=Vh.conj().transpose(0,1)

        # base-flow fields for the alignment diagnostics
        ub=np.real(np.fft.ifftn(Va1.detach().cpu().numpy(),axes=(1,2,3)))
        Gb=grad_tensor(ub,NRES)
        Sb=0.5*(Gb+np.transpose(Gb,(1,0,2,3,4)))
        Sflat=np.transpose(Sb.reshape(3,3,-1),(2,0,1))
        evals,evecs=np.linalg.eigh(Sflat)          # ascending: e1<e2<e3
        wb=vorticity(ub,NRES); ensb=np.sum(wb**2,axis=0)
        strb=np.sum(Sb**2,axis=(0,1))
        kf,Ef=spectrum_np(ub); k_flow=float((kf*Ef).sum()/max(Ef.sum(),1e-300))

        for d in range(min(3,right.shape[1])):
            q1=sum(right[i,d].to(B1[0].dtype)*B1[i] for i in range(len(B1)))
            ud=np.real(np.fft.ifftn(q1.detach().cpu().numpy(),axes=(1,2,3)))
            kk,Ek=spectrum_np(ud); tot=max(Ek.sum(),1e-300)
            cum=np.cumsum(Ek)/tot
            k_leak=float((kk*Ek).sum()/tot)
            wd=vorticity(ud,NRES)
            nw=np.sqrt(np.sum(wd**2,axis=0))+1e-300
            wn=(wd/nw).reshape(3,-1).T
            cos2=[float(np.mean(np.einsum('ij,ij->i',wn,evecs[:,:,j])**2)) for j in range(3)]
            nu_=np.sqrt(np.mean(np.sum(ud**2,axis=0)))
            nwm=np.sqrt(np.mean(np.sum(wd**2,axis=0)))
            relhel=float(np.mean(np.sum(ud*wd,axis=0))/max(nu_*nwm,1e-30))
            amp=np.sum(ud**2,axis=0).ravel()
            c_ens=float(np.corrcoef(amp,ensb.ravel())[0,1])
            c_str=float(np.corrcoef(amp,strb.ravel())[0,1])
            rows.append(dict(topology=tname,nu=nu,N=NRES,direction=d,
                top3_energy=top3,empirical_rank=len(B1),
                k_leak=k_leak,k90=int(np.searchsorted(cum,0.9)),k_flow=k_flow,
                cos2_e1=cos2[0],cos2_e2=cos2[1],cos2_e3=cos2[2],
                relative_helicity=relhel,corr_enstrophy=c_ens,corr_strain=c_str))
            print(f"  dir{d}: k_leak={k_leak:6.2f} (k_flow={k_flow:5.2f})  "
                  f"cos2(e1,e2,e3)={cos2[0]:.3f},{cos2[1]:.3f},{cos2[2]:.3f}  "
                  f"relhel={relhel:+.3f}  r(enstr)={c_ens:+.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT,index=False)
        del env,U1,U2,t1,t2,V1,V2,m,dQ1s,dQ2s,Vs,B1,B2,ub,Gb,Sb,Sflat,evecs
        torch.cuda.empty_cache()

df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
print("\n"+"="*104)
print("M3D TEST L — PHYSICAL IDENTITY OF THE LEAKAGE DIRECTIONS")
print("="*104)
print(f"\nconfigs: {df.groupby(['topology','nu']).ngroups}  "
      f"(4 topologies x 4 viscosities)")
print(f"median top-3 operator energy: {df.top3_energy.median():.4f}")

print("\n--- L1: does k_leak follow a power law in nu?  (LOG-LOG FIT, not a rank test) ---")
print("     Kolmogorov dissipation scaling would give exponent p = 0.75 in k ~ nu^-p")
for d in (0,1,2):
    s=df[df.direction==d]
    print(f"  direction {d}:")
    for t in TOPOS:
        q=s[s.topology==t].sort_values("nu")
        if len(q)<3: continue
        p,b=np.polyfit(np.log(1/q.nu.values),np.log(q.k_leak.values),1)
        r=np.corrcoef(np.log(1/q.nu.values),np.log(q.k_leak.values))[0,1]
        print(f"    {t:<16} p = {p:+.3f}   R^2 = {r*r:.4f}   "
              f"k_leak {q.k_leak.min():.2f}->{q.k_leak.max():.2f}")
    p,b=np.polyfit(np.log(1/s.nu.values),np.log(s.k_leak.values),1)
    r=np.corrcoef(np.log(1/s.nu.values),np.log(s.k_leak.values))[0,1]
    print(f"    {'POOLED':<16} p = {p:+.3f}   R^2 = {r*r:.4f}")

print("\n--- L2: triadic signature.  one step from the resolved band predicts")
print("        k_leak ~ k_P + k_flow, with k_P ~ 3 for the 'mixed' projector ---")
d0=df[df.direction==0]
pred=3.0+d0.k_flow.values
print(f"  median k_leak = {d0.k_leak.median():.2f}   "
      f"median (k_P + k_flow) = {np.median(pred):.2f}   "
      f"ratio {d0.k_leak.median()/np.median(pred):.3f}")
print(f"  correlation across the 16 configs: "
      f"{np.corrcoef(d0.k_leak.values,pred)[0,1]:+.4f}")

print("\n--- L3: vortex-stretching alignment.  Isotropic turbulence puts vorticity")
print("        preferentially along the INTERMEDIATE strain eigenvector e2 ---")
print(f"  {'direction':<11}{'cos2(e1)':>10}{'cos2(e2)':>10}{'cos2(e3)':>10}{'random=0.333':>14}")
for d in (0,1,2):
    s=df[df.direction==d]
    print(f"  dir{d:<8}{s.cos2_e1.median():>10.4f}{s.cos2_e2.median():>10.4f}"
          f"{s.cos2_e3.median():>10.4f}{'':>14}")

print("\n--- L4: where do the directions live? ---")
print(f"  median corr(|u_dir|^2, base enstrophy) = {df.corr_enstrophy.median():+.4f}")
print(f"  median corr(|u_dir|^2, base strain)    = {df.corr_strain.median():+.4f}")
print(f"  median relative helicity               = {df.relative_helicity.median():+.4f}")
print("\nsaved:", OUT)
