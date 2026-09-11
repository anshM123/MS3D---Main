"""Self-contained: NS + KS solvers, STAR2, W recurrence. Pure numpy, no deps."""
import numpy as np

# ============================== Navier-Stokes ==============================
class NS:
    def __init__(self, N=32, L=8.0, nu=2.5e-3, dt=0.1):
        self.N, self.L, self.nu, self.dt = N, L, nu, dt
        k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
        self.KX, self.KY, self.KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K2i = 1.0/np.where(self.K2 == 0, 1.0, self.K2)
        kmax = np.abs(k).max()
        self.deal = ((np.abs(self.KX) < 2/3*kmax) & (np.abs(self.KY) < 2/3*kmax)
                     & (np.abs(self.KZ) < 2/3*kmax))
        self.kmag = np.sqrt(self.K2)

    def leray(self, U):
        d = self.KX*U[0] + self.KY*U[1] + self.KZ*U[2]
        return np.stack([U[0]-self.KX*d*self.K2i, U[1]-self.KY*d*self.K2i,
                         U[2]-self.KZ*d*self.K2i])

    def clean(self, U):
        return self.leray(U*self.deal)

    def curl(self, U):
        return np.stack([1j*(self.KY*U[2]-self.KZ*U[1]),
                         1j*(self.KZ*U[0]-self.KX*U[2]),
                         1j*(self.KX*U[1]-self.KY*U[0])])

    def F(self, U):
        u = np.fft.ifftn(U, axes=(1,2,3)).real
        w = np.fft.ifftn(self.curl(U), axes=(1,2,3)).real
        c = np.stack([u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2],
                      u[0]*w[1]-u[1]*w[0]])
        nl = np.fft.fftn(c, axes=(1,2,3))
        return self.clean(nl - self.nu*self.K2*U)

    def DF(self, U, V):
        """Analytic JVP — exact, since F is quadratic."""
        u = np.fft.ifftn(U, axes=(1,2,3)).real; v = np.fft.ifftn(V, axes=(1,2,3)).real
        wu = np.fft.ifftn(self.curl(U), axes=(1,2,3)).real
        wv = np.fft.ifftn(self.curl(V), axes=(1,2,3)).real
        c = np.empty_like(u)
        for i,(a,b) in enumerate(((1,2),(2,0),(0,1))):
            c[i] = (u[a]*wv[b]-u[b]*wv[a]) + (v[a]*wu[b]-v[b]*wu[a])
        nl = np.fft.fftn(c, axes=(1,2,3))
        return self.clean(nl - self.nu*self.K2*V)

    def step(self, U, dt=None):
        dt = self.dt if dt is None else dt
        k1=self.F(U); k2=self.F(U+.5*dt*k1); k3=self.F(U+.5*dt*k2); k4=self.F(U+dt*k3)
        return self.clean(U + dt/6*(k1+2*k2+2*k3+k4))

    def phi(self, U, n):
        for _ in range(n): U = self.step(U)
        return U

    def ic(self, seed, amp=1.0, ic_khi=5.0):
        rng = np.random.default_rng(seed)
        U = np.zeros((3,)+self.KX.shape, complex)
        band = (self.kmag >= 1) & (self.kmag <= ic_khi)
        for c in range(3):
            r = rng.normal(size=self.KX.shape)
            h = np.fft.fftn(r)
            U[c] = h/(np.abs(h)+1e-300)*band
        U = self.clean(U)
        n = np.sqrt(np.sum(np.abs(U)**2))
        return U*(amp*self.N**1.5/max(n,1e-30))

    def target_ids(self, klo, khi, want=24):
        ok = self.deal.ravel().copy(); ok[0] = False
        km = self.kmag.ravel(); ctr = 0.5*(klo+khi)
        cand = np.nonzero(ok & (km>=klo) & (km<=khi))[0]
        if len(cand) < want:
            allm = np.nonzero(ok)[0]
            cand = allm[np.argsort(np.abs(km[allm]-ctr))[:want]]
        else:
            cand = cand[np.argsort(np.abs(km[cand]-ctr))[:want]]
        return np.sort(cand)

# ---- P/Q and STAR2, shared shape conventions ----
def splitP(U, ids):
    P = np.zeros_like(U); f = U.reshape(3,-1); Pf = P.reshape(3,-1)
    Pf[:, ids] = f[:, ids]
    return P, U-P

def star2_basis(sys, U, ids):
    B = []
    fl = U.reshape(3,-1)
    for m in ids:
        for c in range(3):
            for ph in (0,1):
                e = np.zeros_like(U); e.reshape(3,-1)[c, m] = 1.0 if ph==0 else 1.0j
                e = sys.clean(e)
                for b in B: e = e - np.vdot(b,e)*b
                n = np.linalg.norm(e)
                if n > 1e-10: B.append(e/n)
    def orth(v):
        for b in B: v = v - np.vdot(b,v)*b
        n = np.linalg.norm(v); return v/n if n > 1e-10 else None
    f = sys.F(U)
    q = orth(splitP(f, ids)[1])
    if q is not None: B.append(q)
    q = orth(splitP(sys.DF(U, f), ids)[1])
    if q is not None: B.append(q)
    return B

def rom_seg(sys, anchor, B, n):
    a = np.zeros(len(B), complex); dt = sys.dt
    def rhs(a):
        U = anchor + sum(a[i]*B[i] for i in range(len(B)))
        F = sys.F(U)
        return np.array([np.vdot(B[i],F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a); k2=rhs(a+.5*dt*k1); k3=rhs(a+.5*dt*k2); k4=rhs(a+dt*k3)
        a = a + dt/6*(k1+2*k2+2*k3+k4)
    return anchor + sum(a[i]*B[i] for i in range(len(B)))

def rom_run(sys, U0, ids, nseg, nstep):
    V = U0.copy(); out=[V.copy()]
    for _ in range(nseg):
        V = rom_seg(sys, V, star2_basis(sys,V,ids), nstep); out.append(V.copy())
    return out

def rom_fc(sys, U, ids, nseg, nstep):
    V = U.copy()
    for _ in range(nseg):
        V = rom_seg(sys, V, star2_basis(sys,V,ids), nstep)
    return V

def terr(A, T, ids):
    a = splitP(A-T, ids)[0]; t = splitP(T, ids)[0]
    return np.linalg.norm(a)/max(np.linalg.norm(t),1e-300)

def w_recurrence(sys, MO, nstep, kmax, deriv_rel=1e-5):
    """Truth-free: uses only the ROM's own defect history."""
    E = np.zeros_like(MO[0]); EH=[E.copy()]
    for k in range(kmax):
        Phi = sys.phi(MO[k], nstep); eta = Phi - MO[k+1]
        if np.linalg.norm(E) < 1e-30: E = eta.copy()
        else:
            eps = min(max(deriv_rel*np.linalg.norm(MO[k])/np.linalg.norm(E),1e-6),5e-2)
            Pp = sys.phi(MO[k]+eps*E, nstep); Mm = sys.phi(MO[k]-eps*E, nstep)
            E = eta + (Pp-Mm)/(2*eps) + 0.5*(Pp+Mm-2*Phi)/(eps*eps)
        EH.append(E.copy())
    return EH
