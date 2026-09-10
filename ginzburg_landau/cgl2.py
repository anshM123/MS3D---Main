"""
ITEM 2, COMPLETED — prospective test on CGL with the ALGORITHM, not just the
phenomenon. Predictions registered before running.

The first CGL run confirmed the identity (5.4e-12) and the causal signature
(63/63) but showed the KS stability mechanism does NOT transfer: lambda ~ 1 in
every band. It also had a validity problem — 94/157 windows had ROM error above
50%. This run shortens the segment so the bands stay valid, and tests the thing
that actually matters for deployment.

=====================  PREDICTIONS REGISTERED BEFORE RUNNING  =================
  D1  identity holds: max |pred-obs|/obs < 1e-9
      (algebra; confirmed on KS 2e-15 and CGL 5e-12, expected pass)

  D2  causal signature: +q beats -q in >= 80% of valid windows
      (confirmed 63/63 on CGL already, expected pass)

  D3  THE REAL TEST — the applicability predictor, developed on KS, applied to
      a system it has never seen:
          Spearman(G_hat, G) > 0.90   AND   median |G_hat/G - 1| < 0.05
      On KS this was Spearman +1.0000, ratio 1.0000. If it holds on a CUBIC
      system, the algorithm's self-assessment is not system-specific.

  D4  STABILITY DOES NOT SET LAMBDA (opposite of the KS-derived claim):
          lambda range across bands < 0.30 while the growth rate spans > 10x
      The first CGL run gave lambda in [0.963, 1.011] over a 4x growth span.
      Registering the NEGATIVE prediction, since KS said the opposite.

D3 is the one worth running. D4 formalises yesterday's failure as a prediction.
==============================================================================
"""
import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from cgl_core import CGL, splitM, basis, seg, terr, rdot, rnorm

NTRAJ=int(os.environ.get("CGL2_NTRAJ",20)); NS_=int(os.environ.get("CGL2_NS",100))
NPM=int(os.environ.get("CGL2_NPM",6)); BURN=20000
NSEG,FC,ANCH=6,2,(3,4)
ALPHAS=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0]
BANDS=[("unstable",0.0,1e9),("neutral",-3.0,0.0),("mild",-12.0,-3.0)]

def main():
    s=CGL(N=256,L=100.0,b=2.0,c=-1.0,dt=0.002)
    print(f"CGL segment={NS_*s.dt:.2f} time units  CUBIC nonlinearity")
    rows=[]; t0=time.time()
    for nm,lo,hi in BANDS:
        gr=s.growth
        md=sorted([i for i in range(1,s.N) if s.deal[i] and lo<=gr[i]<=hi],
                  key=lambda i:-gr[i])[:NPM]
        if len(md)<3: continue
        g=float(np.mean(gr[md]))
        for ti in range(NTRAJ):
            A=s.phi(s.ic(80000+ti),BURN)
            V=A.copy(); MO=[V.copy()]
            for _ in range(NSEG): V=seg(s,V,basis(s,V,md),NS_); MO.append(V.copy())
            T=[A.copy()]; t=A.copy()
            for _ in range(NSEG): t=s.phi(t,NS_); T.append(t.copy())
            E=np.zeros_like(MO[0]); EH=[E.copy()]
            for k in range(max(ANCH)):
                Phi=s.phi(MO[k],NS_); eta=Phi-MO[k+1]
                if rnorm(E)<1e-30: E=eta.copy()
                else:
                    ep=min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                    Pp=s.phi(MO[k]+ep*E,NS_); Mm=s.phi(MO[k]-ep*E,NS_)
                    E=eta+(Pp-Mm)/(2*ep)+0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fc(x,n=FC):
                a=x.copy()
                for _ in range(n): a=seg(s,a,basis(s,a,md),NS_)
                return a
            for k in ANCH:
                if k>=len(EH): continue
                Vk,TF=MO[k],T[k+FC]
                qh=splitM(EH[k],md)[1]; dP,AQ=splitM(T[k]-Vk,md)
                rel=rnorm(qh)/max(rnorm(Vk),1e-30)
                if not np.isfinite(rel) or rel>0.5: continue
                FB=fc(Vk)
                base=terr(FB,TF,md)
                if not (base<0.50): continue          # validity gate, up front
                probe={a: splitM(fc(Vk+a*qh)-FB,md)[0] for a in ALPHAS}
                Eh=EH[k].copy(); Vc=Vk.copy()
                for _ in range(FC):
                    Phi=s.phi(Vc,NS_); Vn=seg(s,Vc,basis(s,Vc,md),NS_); eta=Phi-Vn
                    epp=min(max(1e-5*rnorm(Vc)/max(rnorm(Eh),1e-30),1e-6),5e-2)
                    Pp=s.phi(Vc+epp*Eh,NS_); Mm=s.phi(Vc-epp*Eh,NS_)
                    Eh=eta+(Pp-Mm)/(2*epp)+0.5*(Pp+Mm-2*Phi)/(epp*epp); Vc=Vn
                bh=-splitM(Eh,md)[0]
                ah=min(ALPHAS,key=lambda a: rnorm(bh+probe[a]))
                nbh=rnorm(bh); G_hat=nbh/max(rnorm(bh+probe[ah]),1e-30)
                FO=fc(Vk+AQ); FE=fc(Vk+ah*qh); FN=fc(Vk-ah*qh)
                b=splitM(FB-TF,md)[0]; D=splitM(FO-FB,md)[0]
                nb,nd=rnorm(b),rnorm(D)
                if nb<1e-30 or nd<1e-30: continue
                lam=nd/nb; cos=rdot(b,D)/(nb*nd)
                pred=1/np.sqrt(max(1+2*lam*cos+lam*lam,1e-300))
                obs=nb/max(rnorm(b+D),1e-30)
                rows.append(dict(band=nm,growth=g,base=base,lam=lam,cos=cos,
                    pred=pred,obs=obs,iden=abs(pred-obs)/max(obs,1e-30),
                    G_hat=G_hat,G=base/max(terr(FE,TF,md),1e-30),
                    G_neg=base/max(terr(FN,TF,md),1e-30),alpha=ah,
                    mismatch=rnorm(splitM(qh,md)[0])/max(rnorm(splitM(Vk,md)[0]),1e-30)))
        sub=[r for r in rows if r["band"]==nm]
        if sub: print(f"  {nm:<9} growth={g:>8.2f} n={len(sub):>3} "
                      f"base={100*np.median([x['base'] for x in sub]):5.1f}% "
                      f"lam={np.median([x['lam'] for x in sub]):.4f} "
                      f"G={np.median([x['G'] for x in sub]):8.4f} "
                      f"[{time.time()-t0:.0f}s]",flush=True)
    json.dump(rows,open("cgl2.json","w"),default=float)
    if not rows: print("\nNo valid windows. Lower CGL2_NS."); return
    def sp(x,y):
        rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
        return float(np.corrcoef(rx,ry)[0,1])
    print("\n"+"="*80); print(f"CGL PROSPECTIVE — REGISTERED PREDICTIONS, n={len(rows)}")
    print("="*80)
    idn=max(r["iden"] for r in rows)
    print(f"\nD1 identity        max {idn:.3e}  -> {'PASS' if idn<1e-9 else 'FAIL'}")
    nw=sum(1 for r in rows if r["G"]>r["G_neg"])
    print(f"D2 causal sign     {nw}/{len(rows)}  -> "
          f"{'PASS' if nw>=0.8*len(rows) else 'FAIL'}")
    Gh=np.array([r["G_hat"] for r in rows]); G=np.array([r["G"] for r in rows])
    s1=sp(Gh,G); r1=float(np.median(np.abs(Gh/np.maximum(G,1e-30)-1)))
    print(f"\nD3 APPLICABILITY PREDICTOR on an unseen cubic system")
    print(f"   Spearman(G_hat, G) = {s1:+.4f}  (need > 0.90)")
    print(f"   median |G_hat/G - 1| = {r1:.4f}  (need < 0.05)")
    print(f"   -> {'PASS' if s1>0.90 and r1<0.05 else 'FAIL'}")
    print(f"   KS reference: Spearman +1.0000, ratio 1.0000")
    bl=sorted(set(r["band"] for r in rows),
              key=lambda n:-[r for r in rows if r["band"]==n][0]["growth"])
    lm=[float(np.median([r["lam"] for r in rows if r["band"]==n])) for n in bl]
    gr_=[[r for r in rows if r["band"]==n][0]["growth"] for n in bl]
    rng=max(lm)-min(lm); span=max(gr_)-min(gr_)
    print(f"\nD4 stability does NOT set lambda")
    print(f"   {'band':<10}{'growth':>9}{'lambda':>9}{'cos':>9}{'gain':>10}")
    for n,g_,l_ in zip(bl,gr_,lm):
        c_=float(np.median([r["cos"] for r in rows if r["band"]==n]))
        gg=float(np.median([r["G"] for r in rows if r["band"]==n]))
        print(f"   {n:<10}{g_:>9.2f}{l_:>9.4f}{c_:>+9.4f}{gg:>10.4f}")
    print(f"   lambda range {rng:.4f} (need < 0.30) over growth span {span:.2f}")
    print(f"   -> {'PASS' if rng<0.30 else 'FAIL'}  "
          f"(KS spanned lambda 0.09-1.04 = 0.95)")
    print(f"\n   max start-P mismatch {max(r['mismatch'] for r in rows):.3e}")

if __name__=="__main__": main()
