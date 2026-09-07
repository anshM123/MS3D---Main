"""The M3D framework on Kuramoto-Sivashinsky.

Independent reimplementation: no code shared with the Navier-Stokes bootstrap,
different equation, different discretisation. P is a set of low Fourier modes;
Q is everything else. Everything below mirrors Tests W/X/F exactly.
"""
import numpy as np
from ks_core import KS

# ---------------------------------------------------------------- P / Q
def split_PQ(uh, npm):
    """P = the npm lowest wavenumbers (mode 0 excluded: it is conserved)."""
    P = np.zeros_like(uh); P[1:npm+1] = uh[1:npm+1]
    return P, uh - P

# ---------------------------------------------------------------- ROM
def star2_basis(ks, uh, npm):
    """P + q0 + q1*, rebuilt at the anchor from the local trajectory jet.
    Same construction as STAR2 for NS, expressed for KS."""
    B = []
    for m in range(1, npm+1):                     # observable directions
        for ph in (0, 1):
            e = np.zeros_like(uh)
            e[m] = 1.0 if ph == 0 else 1.0j
            B.append(e/np.linalg.norm(e))
    def orth(v):
        for b in B:
            v = v - np.vdot(b, v)*b
        n = np.linalg.norm(v)
        return v/n if n > 1e-12 else None
    f = ks.F(uh)
    q0 = orth(split_PQ(f, npm)[1])
    if q0 is not None: B.append(q0)
    a = ks.DF(uh, f)
    q1 = orth(split_PQ(a, npm)[1])
    if q1 is not None: B.append(q1)
    return B

def rom_segment(ks, anchor, B, nsteps):
    """Anchored projection ROM: u = anchor + sum a_i b_i, a(0)=0."""
    a = np.zeros(len(B), dtype=complex)
    def rhs(a):
        u = anchor + sum(a[i]*B[i] for i in range(len(B)))
        F = ks.F(u)
        return np.array([np.vdot(B[i], F) for i in range(len(B))])
    dt = ks.dt
    for _ in range(nsteps):
        k1 = rhs(a); k2 = rhs(a+0.5*dt*k1)
        k3 = rhs(a+0.5*dt*k2); k4 = rhs(a+dt*k3)
        a = a + dt/6*(k1+2*k2+2*k3+k4)
    return anchor + sum(a[i]*B[i] for i in range(len(B)))

def rom_run(ks, u0, npm, nseg, nsteps):
    """Re-anchored ROM trajectory, rebuilding the jet at each anchor."""
    V = u0.copy(); out = [V.copy()]
    for _ in range(nseg):
        B = star2_basis(ks, V, npm)
        V = rom_segment(ks, V, B, nsteps)
        out.append(V.copy())
    return out

# ---------------------------------------------------------------- W recurrence
def w_estimate(ks, MO, nsteps, kmax, deriv_rel=1e-5):
    """Truth-free: uses only the ROM's own defect history."""
    E = np.zeros_like(MO[0]); EH = [E.copy()]
    for k in range(kmax):
        V, Vn = MO[k], MO[k+1]
        Phi = ks.phi(V, nsteps)
        eta = Phi - Vn
        if np.linalg.norm(E) < 1e-30:
            E = eta.copy()
        else:
            eps = min(max(deriv_rel*np.linalg.norm(V)/np.linalg.norm(E), 1e-6), 5e-2)
            Pp = ks.phi(V+eps*E, nsteps); Mm = ks.phi(V-eps*E, nsteps)
            E = eta + (Pp-Mm)/(2*eps) + 0.5*(Pp+Mm-2*Phi)/(eps*eps)
        EH.append(E.copy())
    return EH

# ---------------------------------------------------------------- diagnostics
def terr(A, T, npm):
    return np.linalg.norm(split_PQ(A-T, npm)[0])/np.linalg.norm(split_PQ(T, npm)[0])

def leakage(ks, V, dQs, npm, nsteps):
    """T = P DPhi Q restricted to the empirical Q-error subspace."""
    B = []
    for q in dQs:
        w = q.copy()
        for b in B:
            w = w - np.vdot(b, w)*b
        n = np.linalg.norm(w)
        if n > 1e-13*max(np.linalg.norm(q), 1e-300): B.append(w/n)
    if len(B) < 2: return None, 0, 0, np.nan
    base = ks.phi(V, nsteps); sc = np.linalg.norm(V)
    cols = []
    for b in B:
        eps = 1e-6*sc
        cols.append(split_PQ(ks.phi(V+eps*b, nsteps)-base, npm)[0]/eps)
    Y = np.array(cols).T
    sv = np.linalg.svd(Y, compute_uv=False); e = sv**2
    c = np.cumsum(e)/e.sum()
    return Y, len(B), int(np.searchsorted(c, 0.90))+1, float(e[:3].sum()/e.sum())

def reach(ks, B, bvec, npm):
    """Theorem F: rho and the ceiling 1/sqrt(1-rho^2)."""
    cols = [split_PQ(b, npm)[0] for b in B]
    M = np.array(cols).T
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sv > sv[0]*1e-10).sum()) if sv.size else 0
    if r == 0: return 0.0, 1.0, 0
    U = U[:, :r]
    nb = np.linalg.norm(bvec)
    if nb < 1e-300: return 0.0, 1.0, r
    rho = min(np.linalg.norm(U@(U.conj().T@bvec))/nb, 1.0)
    return rho, (np.inf if rho > 1-1e-9 else 1/np.sqrt(1-rho*rho)), r
