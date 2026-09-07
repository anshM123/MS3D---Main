"""Kuramoto-Sivashinsky: solver + the M3D framework, independent of the NS code.

    u_t = -u u_x - u_xx - u_xxxx          on [0,L), periodic

Fourier (rfft), exactly quadratic nonlinearity like NS, so the analytic
Jacobian is available in closed form and central differences are exact.
Attractor dimension grows ~linearly with L, which is the point: it buys
trajectory rank 10 -> 200+ without an 800^3 Navier-Stokes.
"""
import numpy as np

class KS:
    def __init__(self, L, N, dt=0.01, seed=0):
        self.L, self.N, self.dt = L, N, dt
        self.k = 2*np.pi*np.fft.rfftfreq(N, d=L/N)
        self.lin = self.k**2 - self.k**4                 # -u_xx - u_xxxx
        self.deal = np.abs(self.k) < (2/3)*self.k.max()  # 2/3 rule
        self.rng = np.random.default_rng(seed)

    def ic(self, seed):
        rng = np.random.default_rng(seed)
        x = np.arange(self.N)*self.L/self.N
        u = 0.1*rng.normal(size=self.N)
        for m in range(1, 6):
            u += (0.5/m)*np.cos(2*np.pi*m*x/self.L + rng.uniform(0, 2*np.pi))
        return self.dealias(np.fft.rfft(u))

    def dealias(self, uh):
        return uh*self.deal

    def F(self, uh):
        """Exactly quadratic in uh."""
        u = np.fft.irfft(uh, n=self.N)
        nl = -0.5j*self.k*np.fft.rfft(u*u)
        return self.dealias(nl + self.lin*uh)

    def DF(self, uh, vh):
        """Analytic Jacobian-vector product: the bilinear term polarised."""
        u = np.fft.irfft(uh, n=self.N); v = np.fft.irfft(vh, n=self.N)
        nl = -1.0j*self.k*np.fft.rfft(u*v)
        return self.dealias(nl + self.lin*vh)

    def step(self, uh, dt=None):
        dt = self.dt if dt is None else dt
        k1 = self.F(uh); k2 = self.F(uh+0.5*dt*k1)
        k3 = self.F(uh+0.5*dt*k2); k4 = self.F(uh+dt*k3)
        return self.dealias(uh + dt/6*(k1+2*k2+2*k3+k4))

    def phi(self, uh, nsteps):
        for _ in range(nsteps):
            uh = self.step(uh)
        return uh
