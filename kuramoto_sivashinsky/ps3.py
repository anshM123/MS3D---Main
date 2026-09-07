import numpy as np, sys, json
sys.path.insert(0,'/home/claude/ks')
from ks_core import KS
def splitM(uh,m):
    P=np.zeros_like(uh); P[m]=uh[m]; return P,uh-P
def basis(ks,uh,m):
    B=[]
    for mm in m:
        for ph in (0,1):
            e=np.zeros_like(uh); e[mm]=1.0 if ph==0 else 1.0j
            B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B: v=v-np.vdot(b,v)*b
        n=np.linalg.norm(v); return v/n if n>1e-12 else None
    f=ks.F(uh)
    q=orth(splitM(f,m)[1])
    if q is not None: B.append(q)
    q=orth(splitM(ks.DF(uh,f),m)[1])
    if q is not None: B.append(q)
    return B
def seg(ks,anc,B,n):
    a=np.zeros(len(B),dtype=complex); dt=ks.dt
    def rhs(a):
        u=anc+sum(a[i]*B[i] for i in range(len(B))); F=ks.F(u)
        return np.array([np.vdot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a); k2=rhs(a+.5*dt*k1); k3=rhs(a+.5*dt*k2); k4=rhs(a+dt*k3)
        a=a+dt/6*(k1+2*k2+2*k3+k4)
    return anc+sum(a[i]*B[i] for i in range(len(B)))
def terrM(A,T,m):
    return np.linalg.norm(splitM(A-T,m)[0])/np.linalg.norm(splitM(T,m)[0])

NT=int(sys.argv[1]); rows=[]
for L,NN in ((100.,200),(200.,400)):
    ks=KS(L=L,N=NN,dt=0.005); ns=50; lam=ks.k**2-ks.k**4
    BND=[("unstable",[i for i in range(1,len(ks.k)) if lam[i]>0][:6]),
         ("damped",[i for i in range(1,len(ks.k)) if -35<=lam[i]<=-10][:6])]
    for nm,md in BND:
        lk=float(np.mean(lam[md]))
        for ti in range(NT):
            u=ks.phi(ks.ic(9000+ti),40000)
            V=u.copy(); MO=[V.copy()]
            for _ in range(6): V=seg(ks,V,basis(ks,V,md),ns); MO.append(V.copy())
            t=u.copy(); TR=[u.copy()]
            for _ in range(6): t=ks.phi(t,ns); TR.append(t.copy())
            E=np.zeros_like(MO[0]); EH=[E.copy()]
            for k in range(4):
                Phi=ks.phi(MO[k],ns); eta=Phi-MO[k+1]
                if np.linalg.norm(E)<1e-30: E=eta.copy()
                else:
                    ep=min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                    Pp=ks.phi(MO[k]+ep*E,ns); Mm=ks.phi(MO[k]-ep*E,ns)
                    E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fc(x,n=2):
                a=x.copy()
                for _ in range(n): a=seg(ks,a,basis(ks,a,md),ns)
                return a
            for k in (3,4):
                Vk,T,TF=MO[k],TR[k],TR[k+2]
                EQ=splitM(EH[k],md)[1]; dP,AQ=splitM(T-Vk,md)
                if not np.isfinite(np.linalg.norm(EQ)) or \
                   np.linalg.norm(EQ)>0.5*np.linalg.norm(Vk): continue
                FB,FE,FO=fc(Vk),fc(Vk+EQ),fc(Vk+AQ)
                bv=splitM(FB-TF,md)[0]; D=splitM(FO-FB,md)[0]
                nb,nd=np.linalg.norm(bv),np.linalg.norm(D)
                if nb<1e-30 or nd<1e-30: continue
                l=nd/nb; c=float(np.real(np.vdot(bv,D))/(nb*nd))
                pr=1/np.sqrt(max(1+2*l*c+l*l,1e-300))
                be=terrM(FB,TF,md); oo=be/max(terrM(FO,TF,md),1e-30)
                rows.append(dict(L=L,band=nm,lam_k=lk,lam=l,cos=c,pred=pr,
                    gain=be/max(terrM(FE,TF,md),1e-30),orac=oo,
                    iden=abs(pr-oo)/max(oo,1e-30)))
        g=[r for r in rows if r["L"]==L and r["band"]==nm]
        print(f"  L={L:.0f} {nm:<9} lam_k={lk:>+8.2f} n={len(g):>3} "
              f"lam={np.median([x['lam'] for x in g]):.4f} "
              f"cos={np.median([x['cos'] for x in g]):+.4f} "
              f"gain={np.median([x['gain'] for x in g]):9.4f} "
              f"p10={np.percentile([x['gain'] for x in g],10):7.4f}",flush=True)
json.dump(rows,open('/home/claude/ks/ps3.json','w'),default=float)
i=[r["iden"] for r in rows]
print(f"\n  PER-WINDOW identity |pred-obs|/obs: median {np.median(i):.2e} max {np.max(i):.2e}")
print(f"  n={len(rows)} windows total")
