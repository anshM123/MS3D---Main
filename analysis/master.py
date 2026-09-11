"""
MASTER — kappa and lambda across every system, in one run, then the curve test.

Answers one question: does a single curve lam = f(kappa) hold across
fundamentally different systems?

  KS   Kuramoto-Sivashinsky, spectral, Fourier-mode observable, quadratic
  CGL  complex Ginzburg-Landau, spectral, Fourier-mode observable, CUBIC
  FD   Kuramoto-Sivashinsky by FINITE DIFFERENCES, POINT-SENSOR observable

Three discretisations, two nonlinearity classes, two kinds of observation
operator. For every window it logs

    kappa = ||A_PQ (Qe)|| / ||A_PP (Pe)||        hidden-driven share
    lam   = ||A_PQ (Qe)|| / ||b||                effect-size parameter

then fits f(kappa) = kappa/(kappa^p+1)^(1/p) and — the test that matters —
holds out one SYSTEM at a time and predicts it from the other two.

Self-contained apart from ks_core.py and cgl_core.py beside it. CPU, numpy.
"""
import numpy as np, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from ks_core import KS
from cgl_core import CGL, rdot, rnorm

NT_KS  = int(os.environ.get("M_KS",  10))
NT_CGL = int(os.environ.get("M_CGL", 12))
NT_FD  = int(os.environ.get("M_FD",  12))
NSEG, FC = 7, 2

# ----------------------------------------------------------------- generic
def make_split(dot):
    def splitM(u, m):
        P = np.zeros_like(u); P[m] = u[m]; return P, u - P
    return splitM

def block_kappa(fwd, V, pe, qe, b, nrm, dotn):
    """kappa and lam from the two block responses. fwd(x) = ROM forecast."""
    FB = fwd(V)
    dQ = b["split"](fwd(V + qe) - FB, b["m"])[0]
    dP = b["split"](fwd(V + pe) - FB, b["m"])[0]
    nQ, nP = nrm(dQ), nrm(dP)
    nb = nrm(b["vec"])
    if nQ < 1e-30 or nP < 1e-30 or nb < 1e-30: return None
    kap = nQ / nP
    c = dotn(dQ, dP) / (nQ * nP)
    return dict(kappa=kap, lam=nQ / nb, c=c,
                lam_cf=kap / np.sqrt(max(1 + 2*kap*c + kap*kap, 1e-300)))

# ----------------------------------------------------------------- KS
def run_ks(rows):
    ks = KS(L=100.0, N=200, dt=0.005); ns = 50
    lam_k = ks.k**2 - ks.k**4
    splitM = make_split(None)
    bands = [("unstable", [i for i in range(1,len(ks.k)) if lam_k[i] > 0][:6]),
             ("neutral",  [i for i in range(1,len(ks.k)) if -3 <= lam_k[i] <= 0][:6]),
             ("stable",   [i for i in range(1,len(ks.k)) if -12 <= lam_k[i] < -3][:6]),
             ("damped",   [i for i in range(1,len(ks.k)) if -35 <= lam_k[i] < -12][:6])]
    def basis(u, m):
        B = []
        for mm in m:
            for ph in (0,1):
                e = np.zeros_like(u); e[mm] = 1.0 if ph==0 else 1.0j
                B.append(e/np.linalg.norm(e))
        def orth(v):
            for f_ in B: v = v - np.vdot(f_, v)*f_
            n = np.linalg.norm(v); return v/n if n > 1e-12 else None
        w = ks.F(u)
        for _ in range(2):
            q = orth(splitM(w, m)[1])
            if q is None: break
            B.append(q); w = ks.DF(u, w)
        return B
    def seg(anc, B, n):
        a = np.zeros(len(B), complex); dt = ks.dt
        def rhs(a):
            u = anc + sum(a[i]*B[i] for i in range(len(B))); F = ks.F(u)
            return np.array([np.vdot(B[i], F) for i in range(len(B))])
        for _ in range(n):
            k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
            a = a + dt/6*(k1+2*k2+2*k3+k4)
        return anc + sum(a[i]*B[i] for i in range(len(B)))
    for nm, md in bands:
        if len(md) < 3: continue
        t0 = time.time(); n0 = len(rows)
        for ti in range(NT_KS):
            u = ks.phi(ks.ic(20000+ti), 40000)
            V = u.copy(); MO = [V.copy()]
            for _ in range(NSEG): V = seg(V, basis(V, md), ns); MO.append(V.copy())
            T = [u.copy()]; t = u.copy()
            for _ in range(NSEG): t = ks.phi(t, ns); T.append(t.copy())
            E = np.zeros_like(MO[0]); EH = [E.copy()]
            for k in range(NSEG-FC):
                Phi = ks.phi(MO[k], ns); eta = Phi - MO[k+1]
                if np.linalg.norm(E) < 1e-30: E = eta.copy()
                else:
                    ep = min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                    Pp = ks.phi(MO[k]+ep*E, ns); Mm = ks.phi(MO[k]-ep*E, ns)
                    E = eta + (Pp-Mm)/(2*ep) + 0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fwd(x, n=FC):
                a = x.copy()
                for _ in range(n): a = seg(a, basis(a, md), ns)
                return a
            for k in (3,4):
                if k >= len(EH): continue
                Vk, TF = MO[k], T[k+FC]
                pe, qe = splitM(T[k]-Vk, md)
                qh = splitM(EH[k], md)[1]
                if not np.isfinite(np.linalg.norm(qh)) or \
                   np.linalg.norm(qh) > 0.5*np.linalg.norm(Vk): continue
                bvec = splitM(fwd(Vk)-TF, md)[0]
                r = block_kappa(fwd, Vk, pe, qe,
                                dict(split=splitM, m=md, vec=bvec),
                                np.linalg.norm,
                                lambda a_,b_: float(np.real(np.vdot(a_,b_))))
                if r: rows.append(dict(system="KS", band=nm, **r))
        print(f"  KS/{nm:<9} +{len(rows)-n0:>3} windows  [{time.time()-t0:.0f}s]",
              flush=True)

# ----------------------------------------------------------------- CGL (cubic)
def run_cgl(rows):
    s = CGL(N=256, L=100.0, b=2.0, c=-1.0, dt=0.002); ns = 100
    from cgl_core import splitM as csplit, basis as cbasis, seg as cseg
    gr = s.growth
    bands = [("unstable", 0.0, 1e9), ("neutral", -3.0, 0.0), ("mild", -12.0, -3.0)]
    for nm, lo, hi in bands:
        md = sorted([i for i in range(1,s.N) if s.deal[i] and lo <= gr[i] <= hi],
                    key=lambda i: -gr[i])[:6]
        if len(md) < 3: continue
        t0 = time.time(); n0 = len(rows)
        for ti in range(NT_CGL):
            A = s.phi(s.ic(80000+ti), 20000)
            V = A.copy(); MO = [V.copy()]
            for _ in range(NSEG): V = cseg(s, V, cbasis(s,V,md), ns); MO.append(V.copy())
            T = [A.copy()]; t = A.copy()
            for _ in range(NSEG): t = s.phi(t, ns); T.append(t.copy())
            E = np.zeros_like(MO[0]); EH = [E.copy()]
            for k in range(NSEG-FC):
                Phi = s.phi(MO[k], ns); eta = Phi - MO[k+1]
                if rnorm(E) < 1e-30: E = eta.copy()
                else:
                    ep = min(max(1e-5*rnorm(MO[k])/rnorm(E),1e-6),5e-2)
                    Pp = s.phi(MO[k]+ep*E, ns); Mm = s.phi(MO[k]-ep*E, ns)
                    E = eta + (Pp-Mm)/(2*ep) + 0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fwd(x, n=FC):
                a = x.copy()
                for _ in range(n): a = cseg(s, a, cbasis(s,a,md), ns)
                return a
            for k in (3,4):
                if k >= len(EH): continue
                Vk, TF = MO[k], T[k+FC]
                pe, qe = csplit(T[k]-Vk, md)
                qh = csplit(EH[k], md)[1]
                if not np.isfinite(rnorm(qh)) or rnorm(qh) > 0.5*rnorm(Vk): continue
                bvec = csplit(fwd(Vk)-TF, md)[0]
                if rnorm(bvec)/max(rnorm(csplit(TF,md)[0]),1e-30) > 0.50: continue
                r = block_kappa(fwd, Vk, pe, qe,
                                dict(split=csplit, m=md, vec=bvec), rnorm, rdot)
                if r: rows.append(dict(system="CGL", band=nm, **r))
        print(f"  CGL/{nm:<8} +{len(rows)-n0:>3} windows  [{time.time()-t0:.0f}s]",
              flush=True)

# ----------------------------------------------------------------- FD + sensors
def run_fd(rows):
    NX, LD, DT, NSTEP = 200, 100.0, 0.002, 125
    class KSFD:
        def __init__(s_):
            s_.n, s_.L, s_.h, s_.dt = NX, LD, LD/NX, DT
        def F(s_, u):
            h = s_.h
            um1=np.roll(u,1); up1=np.roll(u,-1); um2=np.roll(u,2); up2=np.roll(u,-2)
            return -u*((up1-um1)/(2*h)) - (up1-2*u+um1)/h**2 \
                   - (up2-4*up1+6*u-4*um1+um2)/h**4
        def DF(s_, u, v):
            h = s_.h
            vm1=np.roll(v,1); vp1=np.roll(v,-1); vm2=np.roll(v,2); vp2=np.roll(v,-2)
            um1=np.roll(u,1); up1=np.roll(u,-1)
            return -(v*((up1-um1)/(2*h)) + u*((vp1-vm1)/(2*h))) \
                   - (vp1-2*v+vm1)/h**2 - (vp2-4*vp1+6*v-4*vm1+vm2)/h**4
        def step(s_, u):
            dt=s_.dt
            k1=s_.F(u);k2=s_.F(u+.5*dt*k1);k3=s_.F(u+.5*dt*k2);k4=s_.F(u+dt*k3)
            return u + dt/6*(k1+2*k2+2*k3+k4)
        def phi(s_, u, n):
            for _ in range(n): u = s_.step(u)
            return u
        def ic(s_, seed):
            rng = np.random.default_rng(seed); x = np.arange(s_.n)*s_.h
            u = 0.1*rng.normal(size=s_.n)
            for m in range(1,5):
                u += (0.5/m)*np.cos(2*np.pi*m*x/s_.L + rng.uniform(0,2*np.pi))
            return u
    sysf = KSFD()
    for nsen in (8, 12, 20):
        sen = np.linspace(0, NX, nsen, endpoint=False).astype(int)
        def splitS(u, m): 
            P = np.zeros_like(u); P[m] = u[m]; return P, u-P
        def basisS(u, m):
            B = []
            for s_ in m:
                e = np.zeros_like(u); e[s_] = 1.0; B.append(e/np.linalg.norm(e))
            def orth(v):
                for f_ in B: v = v - np.dot(f_, v)*f_
                n = np.linalg.norm(v); return v/n if n > 1e-12 else None
            w = sysf.F(u)
            for _ in range(2):
                q = orth(splitS(w, m)[1])
                if q is None: break
                B.append(q); w = sysf.DF(u, w)
            return B
        def segS(anc, B, n):
            a = np.zeros(len(B)); dt = sysf.dt
            def rhs(a):
                u = anc + sum(a[i]*B[i] for i in range(len(B))); F = sysf.F(u)
                return np.array([np.dot(B[i], F) for i in range(len(B))])
            for _ in range(n):
                k1=rhs(a);k2=rhs(a+.5*dt*k1);k3=rhs(a+.5*dt*k2);k4=rhs(a+dt*k3)
                a = a + dt/6*(k1+2*k2+2*k3+k4)
            return anc + sum(a[i]*B[i] for i in range(len(B)))
        t0 = time.time(); n0 = len(rows)
        for ti in range(NT_FD):
            u = sysf.phi(sysf.ic(11000+ti), 60000)
            if not np.all(np.isfinite(u)): continue
            V = u.copy(); MO = [V.copy()]
            for _ in range(NSEG): V = segS(V, basisS(V,sen), NSTEP); MO.append(V.copy())
            T = [u.copy()]; t = u.copy()
            for _ in range(NSEG): t = sysf.phi(t, NSTEP); T.append(t.copy())
            E = np.zeros_like(MO[0]); EH = [E.copy()]
            for k in range(NSEG-FC):
                Phi = sysf.phi(MO[k], NSTEP); eta = Phi - MO[k+1]
                if np.linalg.norm(E) < 1e-30: E = eta.copy()
                else:
                    ep = min(max(1e-5*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
                    Pp = sysf.phi(MO[k]+ep*E, NSTEP); Mm = sysf.phi(MO[k]-ep*E, NSTEP)
                    E = eta + (Pp-Mm)/(2*ep) + 0.5*(Pp+Mm-2*Phi)/(ep*ep)
                EH.append(E.copy())
            def fwd(x, n=FC):
                a = x.copy()
                for _ in range(n): a = segS(a, basisS(a,sen), NSTEP)
                return a
            for k in (3,4):
                if k >= len(EH): continue
                Vk, TF = MO[k], T[k+FC]
                pe, qe = splitS(T[k]-Vk, sen)
                qh = splitS(EH[k], sen)[1]
                if not np.isfinite(np.linalg.norm(qh)) or \
                   np.linalg.norm(qh) > 0.5*np.linalg.norm(Vk): continue
                bvec = splitS(fwd(Vk)-TF, sen)[0]
                if np.linalg.norm(bvec)/max(np.linalg.norm(splitS(TF,sen)[0]),
                                            1e-30) > 0.50: continue
                r = block_kappa(fwd, Vk, pe, qe,
                                dict(split=splitS, m=sen, vec=bvec),
                                np.linalg.norm, lambda a_,b_: float(np.dot(a_,b_)))
                if r: rows.append(dict(system="FD", band=f"{nsen}sensors", **r))
        print(f"  FD/{nsen:>2}sensors +{len(rows)-n0:>3} windows  [{time.time()-t0:.0f}s]",
              flush=True)

# ----------------------------------------------------------------- the curve
def f(kap, p):
    kap = np.asarray(kap, float)
    return kap/np.power(np.power(kap, p) + 1.0, 1.0/p)

def fit_p(kap, lam):
    grid = np.logspace(-1.0, 1.2, 200)
    err = [np.median(np.abs(f(kap, p) - lam)) for p in grid]
    i = int(np.argmin(err)); return float(grid[i]), float(err[i])

def analyse(rows):
    if len(rows) < 20:
        print("\nToo few windows to analyse."); return
    kap = np.array([r["kappa"] for r in rows]); lam = np.array([r["lam"] for r in rows])
    sysv = np.array([r["system"] for r in rows])
    systems = sorted(set(sysv))
    print("\n" + "="*84); print(f"UNIVERSAL CURVE TEST, n={len(rows)}"); print("="*84)
    print(f"  systems: {systems}")
    print(f"  kappa {kap.min():.4f} .. {kap.max():.1f}   lam {lam.min():.4f} .. {lam.max():.4f}")
    p_all, e_all = fit_p(kap, lam)
    cf = np.median(np.abs(np.array([r["lam_cf"] for r in rows]) - lam))
    print(f"\nU1 pooled p = {p_all:.3f}   median |f-lam| = {e_all:.4f}"
          f"  -> {'PASS' if e_all < 0.10 else 'FAIL'}")
    print(f"U3 vs derived closed form ({cf:.4f})"
          f"  -> {'PASS' if e_all < cf else 'FAIL'}")
    print(f"\nU2 per-SYSTEM fits")
    print(f"   {'system':<8}{'n':>6}{'p':>9}{'median err':>13}")
    ps = {}
    for s_ in systems:
        m = sysv == s_
        p, e = fit_p(kap[m], lam[m]); ps[s_] = p
        print(f"   {s_:<8}{int(m.sum()):>6}{p:>9.3f}{e:>13.4f}")
    if len(ps) >= 2:
        sp_ = max(ps.values())/max(min(ps.values()), 1e-9)
        print(f"   spread {sp_:.2f}x  -> {'PASS' if sp_ < 2.0 else 'FAIL'}")
    print(f"\nU4 LEAVE-ONE-SYSTEM-OUT — the test that matters")
    print(f"   {'held out':<8}{'p from others':>15}{'median err':>13}")
    errs = []
    for s_ in systems:
        tr = sysv != s_; te = sysv == s_
        if tr.sum() < 10 or te.sum() < 5: continue
        p, _ = fit_p(kap[tr], lam[tr])
        e = float(np.median(np.abs(f(kap[te], p) - lam[te])))
        errs.append(e); print(f"   {s_:<8}{p:>15.3f}{e:>13.4f}")
    if errs:
        print(f"   median {np.median(errs):.4f}"
              f"  -> {'PASS' if np.median(errs) < 0.15 else 'FAIL'}")

def main():
    rows = []
    print("MASTER — kappa/lambda across KS, CGL (cubic), FD+sensors\n", flush=True)
    for nm, fn in (("KS", run_ks), ("CGL", run_cgl), ("FD", run_fd)):
        try: fn(rows)
        except Exception as exc: print(f"  {nm} FAILED: {exc!r}", flush=True)
        json.dump(rows, open("master_kappa.json", "w"), default=float)
    analyse(rows)
    print("\nsaved master_kappa.json")

if __name__ == "__main__": main()
