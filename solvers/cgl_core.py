"""Complex Ginzburg-Landau, 1D periodic. Self-contained, pure numpy.

    A_t = A + (1+i b) A_xx - (1+i c) |A|^2 A

CUBIC nonlinearity — unlike Navier-Stokes and Kuramoto-Sivashinsky, both of
which are exactly quadratic. This is the point of the test: it breaks the one
structural property the two development systems shared.

Consequences handled here:
  * central finite differences are NO LONGER exact for the Jacobian, so the
    analytic JVP is used and the FD path is only a check
  * the nonlinearity involves conj(A), so DF is REAL-linear, not complex-linear.
    The state is therefore treated as a real vector space: basis directions come
    in (1, i) pairs per mode and ROM coefficients are REAL.
"""
import numpy as np

class CGL:
    def __init__(self, N=256, L=100.0, b=2.0, c=-1.0, dt=0.002):
        self.N, self.L, self.b, self.c, self.dt = N, L, b, c, dt
        self.k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
        self.lin = 1.0 - (1+1j*b)*self.k**2      # growth rate per mode
        self.growth = np.real(self.lin)          # unstable where k^2 < 1
        kmax = np.abs(self.k).max()
        self.deal = np.abs(self.k) < (2/3)*kmax

    def dealias(self, Ah): return Ah*self.deal

    def F(self, Ah):
        A = np.fft.ifft(Ah)
        nl = -(1+1j*self.c)*np.fft.fft(np.abs(A)**2*A)
        return self.dealias(self.lin*Ah + nl)

    def DF(self, Ah, Vh):
        """Analytic JVP. d/de[|A|^2 A] = 2|A|^2 V + A^2 conj(V) — real-linear."""
        A = np.fft.ifft(Ah); V = np.fft.ifft(Vh)
        d = 2*np.abs(A)**2*V + A*A*np.conj(V)
        return self.dealias(self.lin*Vh - (1+1j*self.c)*np.fft.fft(d))

    def step(self, Ah, dt=None):
        dt = self.dt if dt is None else dt
        k1=self.F(Ah); k2=self.F(Ah+.5*dt*k1); k3=self.F(Ah+.5*dt*k2); k4=self.F(Ah+dt*k3)
        return self.dealias(Ah + dt/6*(k1+2*k2+2*k3+k4))

    def phi(self, Ah, n):
        for _ in range(n): Ah = self.step(Ah)
        return Ah

    def ic(self, seed):
        rng = np.random.default_rng(seed)
        x = np.arange(self.N)*self.L/self.N
        A = 0.5*(rng.normal(size=self.N) + 1j*rng.normal(size=self.N))
        for m in range(1, 5):
            A += (0.3/m)*np.exp(1j*(2*np.pi*m*x/self.L + rng.uniform(0, 2*np.pi)))
        return self.dealias(np.fft.fft(A))

# ---- real-vector-space helpers (state is complex but DF is only real-linear)
def rdot(u, v):  return float(np.real(np.vdot(u, v)))
def rnorm(u):    return float(np.sqrt(max(rdot(u, u), 0.0)))

def splitM(Ah, modes):
    P = np.zeros_like(Ah); P[modes] = Ah[modes]
    return P, Ah - P

def basis(sys, Ah, modes):
    B = []
    for m in modes:
        for ph in (1.0, 1.0j):
            e = np.zeros_like(Ah); e[m] = ph
            B.append(e/rnorm(e))
    def orth(v):
        for f in B: v = v - rdot(f, v)*f
        n = rnorm(v); return v/n if n > 1e-12 else None
    f = sys.F(Ah)
    q = orth(splitM(f, modes)[1])
    if q is not None: B.append(q)
    q = orth(splitM(sys.DF(Ah, f), modes)[1])
    if q is not None: B.append(q)
    return B

def seg(sys, anchor, B, n):
    a = np.zeros(len(B))                      # REAL coefficients
    dt = sys.dt
    def rhs(a):
        A = anchor + sum(a[i]*B[i] for i in range(len(B)))
        F = sys.F(A)
        return np.array([rdot(B[i], F) for i in range(len(B))])
    for _ in range(n):
        k1=rhs(a); k2=rhs(a+.5*dt*k1); k3=rhs(a+.5*dt*k2); k4=rhs(a+dt*k3)
        a = a + dt/6*(k1+2*k2+2*k3+k4)
    return anchor + sum(a[i]*B[i] for i in range(len(B)))

def terr(A, T, modes):
    return rnorm(splitM(A-T, modes)[0])/max(rnorm(splitM(T, modes)[0]), 1e-300)
