"""
M3D-2 reconstructed baseline for Colab / PyTorch
=================================================

This file recreates the runtime objects that the later M3D Test-D / H2 / I
cells expect:

    initial_flow_D
    initialize_model_D
    m3d_rhs_D
    m3d_rk4_D
    pack_modes_D
    velocity_rhs_D
    full_rk4_D

and also:

    TAU0_D, NU1_D, NU2_D
    ENVIRONMENTS_D
    small_env_D, large_env_D
    small_env, large_env

IMPORTANT
---------
This is a *reconstruction* of the lost baseline from the mathematical method
and the surviving benchmark conventions. It is not claimed to be bit-for-bit
identical to the original lost notebook. It intentionally preserves the key
interfaces and the same M3D-2 construction:

    q0 = Q F(U0) / ||Q F(U0)||
    q1 = orth_Q[ D F(U0) q0 ]

with a rotational-form incompressible pseudospectral Navier-Stokes RHS.

The target projector called "mixed" reproduces the 24 target mode convention
seen in the surviving N=40 H0/H2 diagnostics, including conjugate pairs. For
larger physical boxes at fixed dx, mode indices are scaled with box length so
that the physical wave numbers remain comparable.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

DEVICE_D = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DTYPE_D = torch.float32
COMPLEX_DTYPE_D = torch.complex64

# These values reproduce the normalization used in the surviving Test-G/H
# cells: dt=0.01 -> d_tau=0.00075, hence tau0=13.333333...
TAU0_D = 40.0 / 3.0
NU1_D = 2.5e-3
NU2_D = 2.5e-3

# Default physical discretization used by Test D: fixed dx=0.2.
DX_D = 0.2


# =============================================================================
# SPECTRAL ENVIRONMENT
# =============================================================================

def _fft_integer_modes_D(N: int, device=DEVICE_D) -> torch.Tensor:
    """Integer Fourier mode numbers in torch FFT ordering."""
    # np.fft.fftfreq(N) * N gives [0,1,...,-2,-1] robustly.
    vals = np.fft.fftfreq(N) * N
    return torch.as_tensor(vals, dtype=REAL_DTYPE_D, device=device)


def make_env_D(N: int, L: float | None = None, dx: float | None = None) -> Dict:
    """Create a cubic periodic pseudospectral environment."""
    if L is None and dx is None:
        dx = DX_D
    if L is None:
        L = float(N) * float(dx)
    if dx is None:
        dx = float(L) / float(N)

    N = int(N)
    L = float(L)
    dx = float(dx)

    m = _fft_integer_modes_D(N)
    k1 = (2.0 * math.pi / L) * m

    KX, KY, KZ = torch.meshgrid(k1, k1, k1, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ

    # Standard 2/3 dealiasing in integer-mode coordinates.
    cutoff = N // 3
    Mx, My, Mz = torch.meshgrid(m, m, m, indexing="ij")
    dealias = (
        (torch.abs(Mx) <= cutoff)
        & (torch.abs(My) <= cutoff)
        & (torch.abs(Mz) <= cutoff)
    )

    nonzero = K2 > 0
    invK2 = torch.zeros_like(K2)
    invK2[nonzero] = 1.0 / K2[nonzero]

    return {
        "N": N,
        "L": L,
        "dx": dx,
        "KX": KX,
        "KY": KY,
        "KZ": KZ,
        "K2": K2,
        "invK2": invK2,
        "dealias": dealias,
        "device": DEVICE_D,
    }


# Recreate the two environments that H2 previously discovered.
small_env_D = make_env_D(40, L=8.0)
large_env_D = make_env_D(80, L=16.0)
ENVIRONMENTS_D = {"small": small_env_D, "large": large_env_D}

# Aliases retained because the H2 environment discovery saw both conventions.
small_env = small_env_D
large_env = large_env_D


# =============================================================================
# BASIC SPECTRAL OPERATORS
# =============================================================================

def project_divfree_D(Uhat: torch.Tensor, env: Dict) -> torch.Tensor:
    """Leray projection of a Fourier-space vector field, shape (3,N,N,N)."""
    KX, KY, KZ = env["KX"], env["KY"], env["KZ"]
    invK2 = env["invK2"]

    dot = KX * Uhat[0] + KY * Uhat[1] + KZ * Uhat[2]
    out = Uhat.clone()
    out[0] = Uhat[0] - KX * dot * invK2
    out[1] = Uhat[1] - KY * dot * invK2
    out[2] = Uhat[2] - KZ * dot * invK2

    # Zero mode is harmless but explicitly remove it for reproducibility.
    out[:, 0, 0, 0] = 0
    return out


def dealias_D(Uhat: torch.Tensor, env: Dict) -> torch.Tensor:
    return Uhat * env["dealias"].to(Uhat.dtype).unsqueeze(0)


def curl_hat_D(Uhat: torch.Tensor, env: Dict) -> torch.Tensor:
    KX, KY, KZ = env["KX"], env["KY"], env["KZ"]
    wx = 1j * (KY * Uhat[2] - KZ * Uhat[1])
    wy = 1j * (KZ * Uhat[0] - KX * Uhat[2])
    wz = 1j * (KX * Uhat[1] - KY * Uhat[0])
    return torch.stack((wx, wy, wz), dim=0)


@torch.no_grad()
def velocity_rhs_D(Uhat: torch.Tensor, nu: float, env: Dict) -> torch.Tensor:
    """
    Rotational-form incompressible Navier-Stokes RHS in Fourier space:

        du/dt = nu Delta u + P_sigma(u x omega)

    Uhat is the raw torch FFT coefficient array (complex64).
    """
    Uhat = dealias_D(project_divfree_D(Uhat, env), env)

    u = torch.fft.ifftn(Uhat, dim=(-3, -2, -1)).real
    what = curl_hat_D(Uhat, env)
    w = torch.fft.ifftn(what, dim=(-3, -2, -1)).real

    # u x omega
    cross = torch.empty_like(u)
    cross[0] = u[1] * w[2] - u[2] * w[1]
    cross[1] = u[2] * w[0] - u[0] * w[2]
    cross[2] = u[0] * w[1] - u[1] * w[0]

    nhat = torch.fft.fftn(cross, dim=(-3, -2, -1))
    nhat = project_divfree_D(nhat, env)
    nhat = dealias_D(nhat, env)

    rhs = nhat - float(nu) * env["K2"].unsqueeze(0) * Uhat
    rhs = dealias_D(project_divfree_D(rhs, env), env)
    return rhs.to(COMPLEX_DTYPE_D)


@torch.no_grad()
def full_rk4_D(Uhat: torch.Tensor, dt: float, nu: float, env: Dict) -> torch.Tensor:
    k1 = velocity_rhs_D(Uhat, nu, env)
    k2 = velocity_rhs_D(Uhat + 0.5 * dt * k1, nu, env)
    k3 = velocity_rhs_D(Uhat + 0.5 * dt * k2, nu, env)
    k4 = velocity_rhs_D(Uhat + dt * k3, nu, env)
    return Uhat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =============================================================================
# DETERMINISTIC 3-D INITIAL FLOW
# =============================================================================

def _periodic_delta_D(x: torch.Tensor, x0: float, L: float) -> torch.Tensor:
    """Shortest signed periodic distance."""
    return torch.remainder(x - x0 + 0.5 * L, L) - 0.5 * L


def _make_mixed_velocity_D(env: Dict, variant: int) -> torch.Tensor:
    """Smooth localized 3-D vortical field in physical space."""
    N, L, dx = env["N"], env["L"], env["dx"]
    coord = (torch.arange(N, device=DEVICE_D, dtype=REAL_DTYPE_D) - N/2) * dx
    X, Y, Z = torch.meshgrid(coord, coord, coord, indexing="ij")

    u = torch.zeros((3, N, N, N), device=DEVICE_D, dtype=REAL_DTYPE_D)

    # Parameters are specified as fractions of L so the L=8 and L=16 fields
    # are physically similar after box enlargement.
    if variant == 0:
        tubes = [
            ("z", -0.16*L,  0.08*L, 0.11*L, +1.00),
            ("x",  0.10*L, -0.12*L, 0.095*L, -0.82),
            ("y", -0.08*L, -0.17*L, 0.085*L, +0.68),
        ]
    else:
        tubes = [
            ("z",  0.13*L, -0.10*L, 0.10*L, -0.92),
            ("x", -0.12*L,  0.14*L, 0.090*L, +0.77),
            ("y",  0.09*L,  0.16*L, 0.080*L, -0.64),
        ]

    for axis, a0, b0, sigma, strength in tubes:
        if axis == "z":
            dxp = _periodic_delta_D(X, a0, L)
            dyp = _periodic_delta_D(Y, b0, L)
            r2 = dxp*dxp + dyp*dyp
            g = strength * torch.exp(-0.5 * r2 / (sigma*sigma))
            u[0] += -dyp / sigma * g
            u[1] +=  dxp / sigma * g
            u[2] += 0.16 * g * torch.sin(2*math.pi*Z/L + 0.4*variant)
        elif axis == "x":
            dyp = _periodic_delta_D(Y, a0, L)
            dzp = _periodic_delta_D(Z, b0, L)
            r2 = dyp*dyp + dzp*dzp
            g = strength * torch.exp(-0.5 * r2 / (sigma*sigma))
            u[1] += -dzp / sigma * g
            u[2] +=  dyp / sigma * g
            u[0] += 0.13 * g * torch.sin(2*math.pi*X/L + 0.7 + 0.2*variant)
        else:  # y-axis
            dxp = _periodic_delta_D(X, a0, L)
            dzp = _periodic_delta_D(Z, b0, L)
            r2 = dxp*dxp + dzp*dzp
            g = strength * torch.exp(-0.5 * r2 / (sigma*sigma))
            u[0] +=  dzp / sigma * g
            u[2] += -dxp / sigma * g
            u[1] += 0.11 * g * torch.cos(2*math.pi*Y/L + 0.3*variant)

    # Add a weak broadband-but-smooth divergence-free precursor; Leray
    # projection below removes any residual divergence exactly in spectral form.
    u[0] += 0.08 * torch.sin(2*math.pi*(X + 2*Y)/L) * torch.cos(2*math.pi*Z/L)
    u[1] += 0.06 * torch.cos(2*math.pi*(Y - Z)/L) * torch.sin(4*math.pi*X/L)
    u[2] += 0.05 * torch.sin(2*math.pi*(Z + X)/L) * torch.cos(4*math.pi*Y/L)

    Uhat = torch.fft.fftn(u, dim=(-3, -2, -1)).to(COMPLEX_DTYPE_D)
    Uhat = dealias_D(project_divfree_D(Uhat, env), env)

    # Scale physical maximum speed to 0.18, matching the Test-G convention.
    up = torch.fft.ifftn(Uhat, dim=(-3, -2, -1)).real
    speed = torch.sqrt(torch.sum(up*up, dim=0))
    umax = torch.max(speed)
    Uhat = Uhat * (0.18 / torch.clamp(umax, min=1e-12))
    return Uhat


@torch.no_grad()
def initial_flow_D(env: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    return _make_mixed_velocity_D(env, 0), _make_mixed_velocity_D(env, 1)


# =============================================================================
# MODE / PROJECTOR UTILITIES
# =============================================================================

# Exact N=40 "mixed" mode list recovered from the surviving H0 diagnostic.
# Listed as integer Fourier wave-number triples, already including +/- pairs.
_MIXED_BASE_MODES_D: List[Tuple[int,int,int]] = [
    (0, 1, 0), (0,-1, 0),
    (0, 1,-1), (0,-1, 1),
    (1,-1, 0), (-1, 1, 0),
    (0, 0, 1), (0, 0,-1),
    (0, 1, 1), (0,-1,-1),
    (1, 0, 0), (-1, 0, 0),
    (0, 2, 4), (0,-2,-4),
    (1,-2,-4), (-1, 2, 4),
    (2, 0, 4), (-2, 0,-4),
    (0, 3, 4), (0,-3,-4),
    (2,-1, 4), (-2, 1,-4),
    (2, 1, 4), (-2,-1,-4),
]

_LOW_BASE_MODES_D: List[Tuple[int,int,int]] = [
    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
    (1,1,0),(-1,-1,0),(1,0,1),(-1,0,-1),(0,1,1),(0,-1,-1),
    (1,-1,0),(-1,1,0),(1,0,-1),(-1,0,1),(0,1,-1),(0,-1,1),
    (1,1,1),(-1,-1,-1),(1,-1,1),(-1,1,-1),(1,1,-1),(-1,-1,1),
]

_MID_BASE_MODES_D: List[Tuple[int,int,int]] = [
    (0,2,4),(0,-2,-4),(1,-2,-4),(-1,2,4),(2,0,4),(-2,0,-4),
    (0,3,4),(0,-3,-4),(2,-1,4),(-2,1,-4),(2,1,4),(-2,-1,-4),
    (3,1,4),(-3,-1,-4),(1,3,4),(-1,-3,-4),(3,-2,4),(-3,2,-4),
    (2,3,4),(-2,-3,-4),(4,1,3),(-4,-1,-3),(1,4,3),(-1,-4,-3),
]


def _mode_scale_for_box_D(env: Dict) -> int:
    # Test-D boxes used L=8 -> scale 1, L=16 -> 2, L=24 -> 3.
    return max(1, int(round(float(env["L"]) / 8.0)))


def _index_from_mode_D(mode: Sequence[int], N: int) -> Tuple[int,int,int]:
    return tuple(int(m) % N for m in mode)


def _flat_id_from_mode_D(mode: Sequence[int], N: int) -> int:
    idx = _index_from_mode_D(mode, N)
    return int(np.ravel_multi_index(idx, (N,N,N)))


def make_projector_D(name: str, env: Dict) -> Dict:
    key = str(name).lower()
    if key == "mixed":
        base = _MIXED_BASE_MODES_D
    elif key == "low":
        base = _LOW_BASE_MODES_D
    elif key == "mid":
        base = _MID_BASE_MODES_D
    else:
        raise ValueError(f"Unknown projector {name!r}; expected low/mid/mixed")

    s = _mode_scale_for_box_D(env)
    modes = [(s*a, s*b, s*c) for (a,b,c) in base]
    N = env["N"]
    ids = torch.tensor(
        [_flat_id_from_mode_D(m, N) for m in modes],
        device=DEVICE_D,
        dtype=torch.long,
    )

    # Full dealiased complement, excluding target modes and zero mode.
    mask = env["dealias"].reshape(-1).clone()
    mask[0] = False
    mask[ids] = False
    dep_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).to(torch.long)

    return {
        "name": key,
        "target_modes": modes,
        "target_ids": ids,
        "dep_ids": dep_ids,
    }


def pack_modes_D(U1: torch.Tensor, U2: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """
    Pack selected complex Fourier coefficients into one real vector.

    Ordering is self-consistent and fixed:
        wakes -> vector component -> mode -> (real, imag)
    """
    a = U1.reshape(3, -1)[:, ids]
    b = U2.reshape(3, -1)[:, ids]
    c = torch.stack((a, b), dim=0)  # (2,3,M), complex
    return torch.view_as_real(c).reshape(-1).to(REAL_DTYPE_D)


def unpack_modes_D(
    packed: torch.Tensor,
    U1: torch.Tensor,
    U2: torch.Tensor,
    ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return copies of U1/U2 with selected modes overwritten by packed."""
    M = int(ids.numel())
    vals = packed.reshape(2, 3, M, 2).contiguous()
    vals_c = torch.view_as_complex(vals.to(REAL_DTYPE_D)).to(COMPLEX_DTYPE_D)

    A = U1.clone()
    B = U2.clone()
    Af = A.reshape(3, -1)
    Bf = B.reshape(3, -1)
    Af[:, ids] = vals_c[0]
    Bf[:, ids] = vals_c[1]
    return A, B


def pair_inner_D(
    A1: torch.Tensor, A2: torch.Tensor,
    B1: torch.Tensor, B2: torch.Tensor,
) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(A1)*B1) + torch.sum(torch.conj(A2)*B2))


def pair_norm_D(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(pair_inner_D(A1,A2,A1,A2), min=0.0))


def zero_target_modes_D(U1: torch.Tensor, U2: torch.Tensor, ids: torch.Tensor):
    A, B = U1.clone(), U2.clone()
    A.reshape(3,-1)[:, ids] = 0
    B.reshape(3,-1)[:, ids] = 0
    return A, B


# =============================================================================
# M3D-2 BUILD
# =============================================================================

@torch.no_grad()
def _pair_rhs_D(U1: torch.Tensor, U2: torch.Tensor, env: Dict):
    return velocity_rhs_D(U1, NU1_D, env), velocity_rhs_D(U2, NU2_D, env)


@torch.no_grad()
def initialize_model_D(
    U10: torch.Tensor,
    U20: torch.Tensor,
    projector_name: str,
    env: Dict,
) -> Dict:
    """Construct the rank-2 hidden M3D model at the supplied anchor state."""
    proj = make_projector_D(projector_name, env)
    tids = proj["target_ids"]
    dids = proj["dep_ids"]

    x0_phys = pack_modes_D(U10, U20, tids)
    dep0_phys = pack_modes_D(U10, U20, dids)

    # Scalar normalizations keep the interface compatible with H2 while
    # keeping reduced coordinates O(1).
    x_scale = torch.clamp(
        torch.linalg.vector_norm(x0_phys) / math.sqrt(max(1, x0_phys.numel())),
        min=1e-7,
    )

    full_norm = pair_norm_D(U10, U20)
    hidden_scale = torch.clamp(0.05 * full_norm, min=1e-7)
    dep_scale = torch.clamp(
        torch.linalg.vector_norm(dep0_phys) / math.sqrt(max(1, dep0_phys.numel())),
        min=1e-7,
    )

    F10, F20 = _pair_rhs_D(U10, U20, env)
    QF1, QF2 = zero_target_modes_D(F10, F20, tids)
    n0 = pair_norm_D(QF1, QF2)
    if float(n0.detach().cpu()) < 1e-12:
        raise RuntimeError("QF(U0) is numerically zero; q0 is undefined.")

    q01, q02 = QF1 / n0, QF2 / n0

    # Centered JVP. For a quadratic vector field this is algebraically exact
    # apart from floating-point roundoff.
    eps = torch.clamp(1e-3 * full_norm, min=1e-5)
    e = float(eps.detach().cpu())

    Fp1, Fp2 = _pair_rhs_D(U10 + e*q01, U20 + e*q02, env)
    Fm1, Fm2 = _pair_rhs_D(U10 - e*q01, U20 - e*q02, env)
    Jq1 = (Fp1 - Fm1) / (2.0*e)
    Jq2 = (Fp2 - Fm2) / (2.0*e)
    Jq1, Jq2 = zero_target_modes_D(Jq1, Jq2, tids)

    # Orthogonalize against q0.
    beta = pair_inner_D(q01,q02,Jq1,Jq2)
    r1 = Jq1 - beta*q01
    r2 = Jq2 - beta*q02
    n1 = pair_norm_D(r1, r2)

    if float(n1.detach().cpu()) < 1e-12:
        # Degenerate case: q1 is unnecessary. Keep a zero vector but expose
        # the expected 2-D interface. This should be rare for the mixed flow.
        q11 = torch.zeros_like(q01)
        q12 = torch.zeros_like(q02)
    else:
        q11, q12 = r1 / n1, r2 / n1

    # Explicitly ensure hidden directions have zero target support.
    q01, q02 = zero_target_modes_D(q01, q02, tids)
    q11, q12 = zero_target_modes_D(q11, q12, tids)

    # Packed hidden basis in dependency coordinates, matching the historical
    # Qr key convention closely enough for diagnostics / future reanchoring.
    q0_dep = pack_modes_D(q01, q02, dids)
    q1_dep = pack_modes_D(q11, q12, dids)
    Qr = torch.stack((q0_dep, q1_dep), dim=1)

    gram = Qr.T @ Qr
    gram_error = torch.max(torch.abs(gram - torch.eye(2, device=gram.device, dtype=gram.dtype)))

    model = {
        "name": f"M3D-2-{projector_name}",
        "proj": proj,
        "x0_phys": x0_phys.clone(),
        "dep0_phys": dep0_phys.clone(),
        "x_scale": x_scale,
        "dep_scale": dep_scale,
        "hidden_scale": hidden_scale,
        "Qr": Qr,
        "gram_error": gram_error,
        "x": (x0_phys / x_scale).clone(),
        "z": torch.zeros(2, device=DEVICE_D, dtype=REAL_DTYPE_D),
        "anchor_U1": U10.clone(),
        "anchor_U2": U20.clone(),
        "q0_U1": q01,
        "q0_U2": q02,
        "q1_U1": q11,
        "q1_U2": q12,
        "jvp_eps": torch.tensor(e, device=DEVICE_D, dtype=REAL_DTYPE_D),
    }
    return model


# =============================================================================
# M3D STATE RECONSTRUCTION AND RHS
# =============================================================================

@torch.no_grad()
def reconstruct_m3d_state_D(
    x: torch.Tensor,
    z: torch.Tensor,
    U10: torch.Tensor,
    U20: torch.Tensor,
    model: Dict,
    env: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the ambient Fourier state represented by (x,z)."""
    tids = model["proj"]["target_ids"]
    x_phys = x * model["x_scale"]

    U1, U2 = unpack_modes_D(x_phys, U10, U20, tids)

    hs = model["hidden_scale"]
    U1 = U1 + hs * (z[0]*model["q0_U1"] + z[1]*model["q1_U1"])
    U2 = U2 + hs * (z[0]*model["q0_U2"] + z[1]*model["q1_U2"])

    U1 = dealias_D(project_divfree_D(U1, env), env)
    U2 = dealias_D(project_divfree_D(U2, env), env)
    return U1, U2


@torch.no_grad()
def m3d_rhs_D(
    x: torch.Tensor,
    z: torch.Tensor,
    U10: torch.Tensor,
    U20: torch.Tensor,
    model: Dict,
    env: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dimensionless-time M3D-2 RHS for target coordinates x and hidden z."""
    U1, U2 = reconstruct_m3d_state_D(x, z, U10, U20, model, env)
    F1, F2 = _pair_rhs_D(U1, U2, env)

    tids = model["proj"]["target_ids"]
    dx_phys_dt = pack_modes_D(F1, F2, tids)
    dx_dtau = TAU0_D * dx_phys_dt / model["x_scale"]

    # Galerkin projection onto the two hidden vectors.
    dz0_dt = pair_inner_D(model["q0_U1"], model["q0_U2"], F1, F2) / model["hidden_scale"]
    dz1_dt = pair_inner_D(model["q1_U1"], model["q1_U2"], F1, F2) / model["hidden_scale"]
    dz_dtau = TAU0_D * torch.stack((dz0_dt, dz1_dt)).to(REAL_DTYPE_D)

    return dx_dtau.to(REAL_DTYPE_D), dz_dtau


@torch.no_grad()
def m3d_rk4_D(
    model: Dict,
    d_tau: float,
    U10: torch.Tensor,
    U20: torch.Tensor,
    env: Dict,
):
    """In-place RK4 advance of model['x'], model['z']; returns (x,z)."""
    x0 = model["x"]
    z0 = model["z"]

    k1x, k1z = m3d_rhs_D(x0, z0, U10, U20, model, env)
    k2x, k2z = m3d_rhs_D(x0 + 0.5*d_tau*k1x, z0 + 0.5*d_tau*k1z, U10, U20, model, env)
    k3x, k3z = m3d_rhs_D(x0 + 0.5*d_tau*k2x, z0 + 0.5*d_tau*k2z, U10, U20, model, env)
    k4x, k4z = m3d_rhs_D(x0 + d_tau*k3x, z0 + d_tau*k3z, U10, U20, model, env)

    model["x"] = x0 + (d_tau/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    model["z"] = z0 + (d_tau/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return model["x"], model["z"]


# =============================================================================
# REANCHORING HELPERS FOR THE NEXT TEST
# =============================================================================

@torch.no_grad()
def current_full_state_D(model: Dict, env: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    return reconstruct_m3d_state_D(
        model["x"], model["z"],
        model["anchor_U1"], model["anchor_U2"],
        model, env,
    )


@torch.no_grad()
def reanchor_model_D(model: Dict, env: Dict) -> Dict:
    """Teacher-free reanchor from M3D's own current reconstructed state."""
    U1, U2 = current_full_state_D(model, env)
    return initialize_model_D(U1, U2, model["proj"]["name"], env)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

@torch.no_grad()
def m3d_baseline_selftest_D(env: Dict | None = None, verbose: bool = True) -> Dict:
    """Small consistency check; use small_env_D by default."""
    if env is None:
        env = small_env_D

    U1, U2 = initial_flow_D(env)
    model = initialize_model_D(U1, U2, "mixed", env)

    # Divergence ratio in Fourier space.
    KX, KY, KZ = env["KX"], env["KY"], env["KZ"]
    div1 = KX*U1[0] + KY*U1[1] + KZ*U1[2]
    div_ratio = float((torch.linalg.vector_norm(div1) / (torch.linalg.vector_norm(U1)+1e-30)).cpu())

    dx, dz = m3d_rhs_D(model["x"], model["z"], U1, U2, model, env)

    out = {
        "device": str(DEVICE_D),
        "N": env["N"],
        "target_modes": int(model["proj"]["target_ids"].numel()),
        "x_dim": int(model["x"].numel()),
        "z_dim": int(model["z"].numel()),
        "dep_modes": int(model["proj"]["dep_ids"].numel()),
        "Qr_shape": tuple(model["Qr"].shape),
        "gram_error": float(model["gram_error"].cpu()),
        "divergence_ratio": div_ratio,
        "dx_finite": bool(torch.isfinite(dx).all().item()),
        "dz_finite": bool(torch.isfinite(dz).all().item()),
    }

    if verbose:
        print("="*88)
        print("M3D-2 REBUILT BASELINE SELF-TEST")
        print("="*88)
        for k,v in out.items():
            print(f"{k:20s}: {v}")
        print("="*88)
        print("Required runtime names restored.")

    return out


if __name__ == "__main__":
    print("M3D-2 reconstructed baseline loaded.")
    print("Device:", DEVICE_D)
    print("Available environments:", {k: (v['N'],v['L'],v['dx']) for k,v in ENVIRONMENTS_D.items()})
    print("Run: m3d_baseline_selftest_D()")


# ===== Kaggle W bootstrap additions =====

from pathlib import Path
import numpy as np
import pandas as pd
import math, time
import torch

def finalize_velocity_L2(
    u,
    env,
    target_umax=0.18,
):
    U = torch.fft.fftn(
        u,
        dim=(-3, -2, -1),
    ).to(
        COMPLEX_DTYPE_D
    )

    U = dealias_D(
        project_divfree_D(
            U,
            env,
        ),
        env,
    )

    up = torch.fft.ifftn(
        U,
        dim=(-3, -2, -1),
    ).real

    speed = torch.sqrt(
        torch.sum(
            up * up,
            dim=0,
        )
    )

    umax = torch.max(
        speed
    )

    U = (
        U
        *
        (
            float(target_umax)
            /
            torch.clamp(
                umax,
                min=1e-12,
            )
        )
    )

    return U

def grid_L2(
    env,
):
    N = env["N"]
    L = env["L"]
    dx = env["dx"]

    coord = (
        torch.arange(
            N,
            device=DEVICE_D,
            dtype=REAL_DTYPE_D,
        )
        -
        N / 2
    ) * dx

    return torch.meshgrid(
        coord,
        coord,
        coord,
        indexing="ij",
    )

def vortex_ring_single_L2(
    env,
    variant,
):
    X, Y, Z = grid_L2(
        env
    )

    N = env["N"]
    L = env["L"]

    phase = (
        1.0
        if variant == 0
        else -1.0
    )

    x0 = (
        -0.07 * L
        if variant == 0
        else 0.09 * L
    )

    y0 = (
        0.06 * L
        if variant == 0
        else -0.08 * L
    )

    z0 = (
        -0.035 * L
        if variant == 0
        else 0.045 * L
    )

    dxp = _periodic_delta_D(
        X,
        x0,
        L,
    )

    dyp = _periodic_delta_D(
        Y,
        y0,
        L,
    )

    dzp = _periodic_delta_D(
        Z,
        z0,
        L,
    )

    rho = torch.sqrt(
        dxp * dxp
        +
        dyp * dyp
        +
        1e-10
    )

    R0 = 0.20 * L
    sigma = 0.070 * L

    ring = torch.exp(
        -(
            (rho - R0) ** 2
            +
            dzp ** 2
        )
        /
        (
            2.0
            *
            sigma ** 2
        )
    )

    u = torch.zeros(
        (
            3,
            N,
            N,
            N,
        ),
        device=DEVICE_D,
        dtype=REAL_DTYPE_D,
    )

    # Azimuthal ring circulation.
    u[0] = (
        phase
        *
        (
            -dyp
            /
            rho
        )
        *
        ring
    )

    u[1] = (
        phase
        *
        (
            dxp
            /
            rho
        )
        *
        ring
    )

    # Weak 3-D ring deformation.
    u[2] = (
        0.24
        *
        ring
        *
        torch.sin(
            2.0
            *
            math.pi
            *
            (
                X
                +
                0.6 * Y
            )
            /
            L
            +
            0.7 * variant
        )
    )

    return finalize_velocity_L2(
        u,
        env,
    )

def periodic_shear_single_L2(
    env,
    variant,
):
    X, Y, Z = grid_L2(
        env
    )

    N = env["N"]
    L = env["L"]

    p = (
        0.37
        if variant == 0
        else 1.11
    )

    u = torch.zeros(
        (
            3,
            N,
            N,
            N,
        ),
        device=DEVICE_D,
        dtype=REAL_DTYPE_D,
    )

    # Each dominant velocity component varies mostly perpendicular
    # to itself; Leray projection removes remaining divergence exactly.
    u[0] = (
        1.00
        *
        torch.sin(
            2 * math.pi * Y / L
            +
            p
        )
        +
        0.32
        *
        torch.sin(
            4 * math.pi * Z / L
            -
            0.4 * p
        )
    )

    u[1] = (
        -0.84
        *
        torch.sin(
            2 * math.pi * Z / L
            -
            0.6 * p
        )
        +
        0.27
        *
        torch.cos(
            4 * math.pi * X / L
            +
            p
        )
    )

    u[2] = (
        0.71
        *
        torch.sin(
            2 * math.pi * X / L
            +
            0.8 * p
        )
        +
        0.23
        *
        torch.cos(
            2 * math.pi
            *
            (
                X + Y
            )
            /
            L
        )
    )

    return finalize_velocity_L2(
        u,
        env,
    )

def skew_tubes_single_L2(
    env,
    variant,
):
    X, Y, Z = grid_L2(
        env
    )

    N = env["N"]
    L = env["L"]

    u = torch.zeros(
        (
            3,
            N,
            N,
            N,
        ),
        device=DEVICE_D,
        dtype=REAL_DTYPE_D,
    )

    if variant == 0:
        centers = [
            (
                -0.12 * L,
                0.08 * L,
                0.10 * L,
                +1.0,
            ),
            (
                0.13 * L,
                -0.09 * L,
                0.085 * L,
                -0.82,
            ),
        ]
    else:
        centers = [
            (
                0.11 * L,
                0.10 * L,
                0.095 * L,
                -0.93,
            ),
            (
                -0.14 * L,
                -0.07 * L,
                0.080 * L,
                +0.74,
            ),
        ]

    for j, (
        a,
        b,
        sigma,
        strength,
    ) in enumerate(
        centers
    ):

        # Rotated coordinates make the tubes oblique rather than
        # axis-aligned.
        theta = (
            0.43
            +
            0.51 * j
            +
            0.17 * variant
        )

        c = math.cos(
            theta
        )

        s = math.sin(
            theta
        )

        xp = (
            c * X
            +
            s * Z
        )

        zp = (
            -s * X
            +
            c * Z
        )

        da = _periodic_delta_D(
            xp,
            a,
            L,
        )

        db = _periodic_delta_D(
            Y,
            b,
            L,
        )

        r2 = (
            da * da
            +
            db * db
        )

        g = (
            strength
            *
            torch.exp(
                -0.5
                *
                r2
                /
                (
                    sigma
                    *
                    sigma
                )
            )
        )

        # Local tube swirl in rotated coordinates.
        ux_local = (
            -db
            /
            sigma
            *
            g
        )

        uy_local = (
            da
            /
            sigma
            *
            g
        )

        u[0] += (
            c
            *
            ux_local
        )

        u[2] += (
            s
            *
            ux_local
        )

        u[1] += (
            uy_local
        )

        u[2] += (
            0.11
            *
            g
            *
            torch.sin(
                2
                *
                math.pi
                *
                zp
                /
                L
                +
                0.6 * j
            )
        )

    # Weak interaction field.
    u[0] += (
        0.08
        *
        torch.sin(
            2
            *
            math.pi
            *
            (
                X + Y + Z
            )
            /
            L
        )
    )

    return finalize_velocity_L2(
        u,
        env,
    )

def topology_pair_L2(
    topology,
    env,
):
    if topology == "vortex_ring":
        return (
            vortex_ring_single_L2(
                env,
                0,
            ),
            vortex_ring_single_L2(
                env,
                1,
            ),
        )

    if topology == "periodic_shear":
        return (
            periodic_shear_single_L2(
                env,
                0,
            ),
            periodic_shear_single_L2(
                env,
                1,
            ),
        )

    if topology == "skew_tubes":
        return (
            skew_tubes_single_L2(
                env,
                0,
            ),
            skew_tubes_single_L2(
                env,
                1,
            ),
        )

    if topology == "mixed_vortices":
        # This one uses the preserved deterministic baseline generator.
        return initial_flow_D(
            env
        )

    raise ValueError(
        f"Unknown topology: {topology}"
    )

def pair_jvp_M(
    U1,
    U2,
    V1,
    V2,
    env,
):
    vn = pair_norm_D(
        V1,
        V2,
    )

    vn_float = float(
        vn.detach().cpu()
    )

    if (
        not np.isfinite(
            vn_float
        )
        or
        vn_float < 1e-13
    ):
        return (
            torch.zeros_like(U1),
            torch.zeros_like(U2),
        )

    v1 = (
        V1
        /
        vn
    )

    v2 = (
        V2
        /
        vn
    )

    un = pair_norm_D(
        U1,
        U2,
    )

    eps = max(
        1e-5,
        1e-3
        *
        float(
            un.detach().cpu()
        ),
    )

    Fp1, Fp2 = _pair_rhs_D(
        U1 + eps * v1,
        U2 + eps * v2,
        env,
    )

    Fm1, Fm2 = _pair_rhs_D(
        U1 - eps * v1,
        U2 - eps * v2,
        env,
    )

    J1 = (
        Fp1
        -
        Fm1
    ) / (
        2.0
        *
        eps
    )

    J2 = (
        Fp2
        -
        Fm2
    ) / (
        2.0
        *
        eps
    )

    return (
        vn * J1,
        vn * J2,
    )

def complement_M(
    V1,
    V2,
    tids,
    basis,
):
    R1, R2 = zero_target_modes_D(
        V1,
        V2,
        tids,
    )

    for B1, B2 in basis:

        bn = float(
            pair_norm_D(
                B1,
                B2,
            ).detach().cpu()
        )

        if bn < 1e-12:
            continue

        a = pair_inner_D(
            B1,
            B2,
            R1,
            R2,
        )

        R1 = (
            R1
            -
            a * B1
        )

        R2 = (
            R2
            -
            a * B2
        )

    R1, R2 = zero_target_modes_D(
        R1,
        R2,
        tids,
    )

    return (
        R1,
        R2,
    )

def reconstruct_M(
    model,
    env,
):
    tids = model[
        "proj"
    ][
        "target_ids"
    ]

    x_phys = (
        model[
            "x"
        ]
        *
        model[
            "x_scale"
        ]
    )

    U1, U2 = unpack_modes_D(
        x_phys,
        model[
            "anchor_U1"
        ],
        model[
            "anchor_U2"
        ],
        tids,
    )

    hs = model[
        "hidden_scale"
    ]

    for i, (
        B1,
        B2,
    ) in enumerate(
        model[
            "basis"
        ]
    ):

        U1 = (
            U1
            +
            hs
            *
            model["z"][i]
            *
            B1
        )

        U2 = (
            U2
            +
            hs
            *
            model["z"][i]
            *
            B2
        )

    U1 = dealias_D(
        project_divfree_D(
            U1,
            env,
        ),
        env,
    )

    U2 = dealias_D(
        project_divfree_D(
            U2,
            env,
        ),
        env,
    )

    return (
        U1,
        U2,
    )

def rhs_M(
    x,
    z,
    model,
    env,
):
    # Temporarily reconstruct from supplied RK stage state.
    old_x = model[
        "x"
    ]

    old_z = model[
        "z"
    ]

    model[
        "x"
    ] = x

    model[
        "z"
    ] = z

    U1, U2 = reconstruct_M(
        model,
        env,
    )

    model[
        "x"
    ] = old_x

    model[
        "z"
    ] = old_z

    F1, F2 = _pair_rhs_D(
        U1,
        U2,
        env,
    )

    tids = model[
        "proj"
    ][
        "target_ids"
    ]

    dx_phys_dt = pack_modes_D(
        F1,
        F2,
        tids,
    )

    dx_dtau = (
        TAU0_D
        *
        dx_phys_dt
        /
        model[
            "x_scale"
        ]
    )

    if len(
        model[
            "basis"
        ]
    ) == 0:

        dz_dtau = torch.empty(
            0,
            device=DEVICE_D,
            dtype=REAL_DTYPE_D,
        )

    else:

        dz_phys = []

        hs = model[
            "hidden_scale"
        ]

        for B1, B2 in model[
            "basis"
        ]:

            dz_phys.append(
                pair_inner_D(
                    B1,
                    B2,
                    F1,
                    F2,
                )
                /
                hs
            )

        dz_dtau = (
            TAU0_D
            *
            torch.stack(
                dz_phys
            ).to(
                REAL_DTYPE_D
            )
        )

    return (
        dx_dtau.to(
            REAL_DTYPE_D
        ),
        dz_dtau,
    )

def rk4_M(
    model,
    d_tau,
    env,
):
    x0 = model[
        "x"
    ]

    z0 = model[
        "z"
    ]

    k1x, k1z = rhs_M(
        x0,
        z0,
        model,
        env,
    )

    k2x, k2z = rhs_M(
        x0
        +
        0.5
        *
        d_tau
        *
        k1x,
        z0
        +
        0.5
        *
        d_tau
        *
        k1z,
        model,
        env,
    )

    k3x, k3z = rhs_M(
        x0
        +
        0.5
        *
        d_tau
        *
        k2x,
        z0
        +
        0.5
        *
        d_tau
        *
        k2z,
        model,
        env,
    )

    k4x, k4z = rhs_M(
        x0
        +
        d_tau
        *
        k3x,
        z0
        +
        d_tau
        *
        k3z,
        model,
        env,
    )

    model[
        "x"
    ] = (
        x0
        +
        d_tau
        /
        6.0
        *
        (
            k1x
            +
            2.0 * k2x
            +
            2.0 * k3x
            +
            k4x
        )
    )

    model[
        "z"
    ] = (
        z0
        +
        d_tau
        /
        6.0
        *
        (
            k1z
            +
            2.0 * k2z
            +
            2.0 * k3z
            +
            k4z
        )
    )

def initialize_star2_N(
    U10,
    U20,
    env,
):
    # Use baseline initialization so P, q0, coordinate packing,
    # x_scale and hidden_scale are identical to OLD2.

    base = initialize_model_D(
        U10,
        U20,
        PROJECTOR_N,
        env,
    )

    tids = base["proj"]["target_ids"]

    q0 = (
        base["q0_U1"].clone(),
        base["q0_U2"].clone(),
    )

    # f = F(U0)
    f1, f2 = _pair_rhs_D(
        U10,
        U20,
        env,
    )

    # Af = DF(U0) f
    Af1, Af2 = pair_jvp_M(
        U10,
        U20,
        f1,
        f2,
        env,
    )

    # q1_star = normalized R1 Af
    r1, r2 = complement_M(
        Af1,
        Af2,
        tids,
        [q0],
    )

    raw_norm = pair_norm_D(
        r1,
        r2,
    )

    raw_norm_value = float(
        raw_norm.detach().cpu()
    )

    if (
        not np.isfinite(raw_norm_value)
        or raw_norm_value < 1e-12
    ):
        raise RuntimeError(
            "STAR2 q1* became numerically degenerate."
        )

    q1_star = (
        r1 / raw_norm,
        r2 / raw_norm,
    )

    # Check residual acceleration outside STAR2 span.
    RAf1, RAf2 = complement_M(
        Af1,
        Af2,
        tids,
        [
            q0,
            q1_star,
        ],
    )

    Af_norm = float(
        pair_norm_D(
            Af1,
            Af2,
        ).detach().cpu()
    )

    RAf_ratio = float(
        (
            pair_norm_D(
                RAf1,
                RAf2,
            )
            /
            (
                Af_norm
                +
                1e-30
            )
        ).detach().cpu()
    )

    model = {
        "name": "STAR2",

        "proj": base["proj"],

        "x_scale":
            base["x_scale"].clone(),

        "hidden_scale":
            base["hidden_scale"].clone(),

        "x":
            base["x"].clone(),

        "z":
            torch.zeros(
                2,
                device=DEVICE_D,
                dtype=REAL_DTYPE_D,
            ),

        "anchor_U1":
            U10.clone(),

        "anchor_U2":
            U20.clone(),

        "basis": [
            (
                q0[0].clone(),
                q0[1].clone(),
            ),
            (
                q1_star[0].clone(),
                q1_star[1].clone(),
            ),
        ],
    }

    diagnostics = {
        "RAf_ratio":
            RAf_ratio,

        "Af_norm":
            Af_norm,

        "q1star_raw_norm":
            raw_norm_value,
    }

    return model, diagnostics

def reanchor_star2_N(
    model,
    env,
):
    # STAR2's OWN current predicted state.
    before1, before2 = reconstruct_M(
        model,
        env,
    )

    t0 = time.perf_counter()

    new_model, diagnostics = initialize_star2_N(
        before1,
        before2,
        env,
    )

    build_time = (
        time.perf_counter()
        -
        t0
    )

    after1, after2 = reconstruct_M(
        new_model,
        env,
    )

    jump = float(
        (
            pair_norm_D(
                after1 - before1,
                after2 - before2,
            )
            /
            (
                pair_norm_D(
                    before1,
                    before2,
                )
                +
                1e-30
            )
        ).detach().cpu()
    )

    return (
        new_model,
        diagnostics,
        jump,
        build_time,
    )


PROJECTOR_N = "mixed"

# Build namespace synchronization AFTER all required functions exist.
FUNCTIONS_TO_SYNC_MB = []
for _name in [
    "_fft_integer_modes_D","make_env_D","project_divfree_D","dealias_D","curl_hat_D",
    "velocity_rhs_D","full_rk4_D","_periodic_delta_D","_make_mixed_velocity_D",
    "initial_flow_D","make_projector_D","pack_modes_D","unpack_modes_D","pair_inner_D",
    "pair_norm_D","zero_target_modes_D","_pair_rhs_D","initialize_model_D",
    "finalize_velocity_L2","grid_L2","vortex_ring_single_L2","periodic_shear_single_L2",
    "skew_tubes_single_L2","topology_pair_L2","pair_jvp_M","complement_M",
    "reconstruct_M","rhs_M","rk4_M","initialize_star2_N","reanchor_star2_N",
]:
    obj = globals().get(_name)
    if callable(obj) and hasattr(obj, "__globals__"):
        FUNCTIONS_TO_SYNC_MB.append(obj)

NAMESPACES_MB = []
_seen_ids_MB = set()
for fn in FUNCTIONS_TO_SYNC_MB:
    gd = fn.__globals__
    if id(gd) not in _seen_ids_MB:
        _seen_ids_MB.add(id(gd))
        NAMESPACES_MB.append(gd)

def sync_m3d_globals_MB(real_dtype, complex_dtype, nu, tau0):
    globals()["REAL_DTYPE_D"] = real_dtype
    globals()["COMPLEX_DTYPE_D"] = complex_dtype
    globals()["nu1_D"] = float(nu)
    globals()["nu2_D"] = float(nu)
    globals()["TAU0_D"] = float(tau0)
    for gd in NAMESPACES_MB:
        gd["REAL_DTYPE_D"] = real_dtype
        gd["COMPLEX_DTYPE_D"] = complex_dtype
        gd["nu1_D"] = float(nu)
        gd["nu2_D"] = float(nu)
        gd["TAU0_D"] = float(tau0)

SAVE_DIR = Path("/kaggle/working/M3D")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Match the established normal regime.
sync_m3d_globals_MB(torch.float32, torch.complex64, 2.5e-3, 40.0/3.0)

_required = [
    "make_env_D","topology_pair_L2","initialize_star2_N","reanchor_star2_N",
    "reconstruct_M","rk4_M","full_rk4_D","pack_modes_D","sync_m3d_globals_MB",
]
_missing = [x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError(f"Kaggle bootstrap incomplete: {_missing}")

print("="*88)
print("M3D KAGGLE BOOTSTRAP READY")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("CUDA:", torch.cuda.is_available())
print("SAVE_DIR:", SAVE_DIR)
print("Required W dependencies: ALL PRESENT")
print("="*88)
