# ============================================================================
# M3D — WHAT ARE THE THREE LEAKAGE DIRECTIONS?
# ============================================================================
# R3-fix showed 3 of 9 empirical directions carry 92.7% of all unresolved ->
# resolved feedback, flat across a 14x span in dim(Q) and 16x in viscosity.
# This asks what those directions ARE in physical space.
#
# The decisive question is spectral. P is a fixed 24-mode set at the lowest
# wavenumbers, so Q is everything above it. If the leakage directions sit just
# above the P cutoff, the dangerous unresolved content is LARGE-SCALE, which
# is contrary to the usual closure intuition AND would mean a coarse-grid
# Phi_h could estimate the defect -- i.e. it would also solve the cost problem.
# If they sit at high k, the conventional picture holds.
# ============================================================================
import glob, math
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = Path("/kaggle/working/M3D")
Z = np.load(SAVE / "m3d_R3fix_directions.npz")
print("directions found:", len(Z.files))
L_BOX = 8.0

# --- where does P actually sit in wavenumber? recompute the target set ---
import torch
BOOT = glob.glob("/kaggle/input/**/m3d_kaggle_bootstrap.py", recursive=True)[0]
G = {"__name__": "__m3d__"}
exec(compile(open(BOOT).read(), BOOT, "exec"), G, G)
env40 = G["make_env_D"](40, L=L_BOX)
U1, U2 = G["topology_pair_L2"]("skew_tubes", env40)
star, _ = G["initialize_star2_N"](U1, U2, env40)
tids = star["proj"]["target_ids"].detach().cpu().numpy()
n40 = 40
ii = np.stack(np.unravel_index(tids, (n40, n40, n40)))
kk = np.where(ii > n40 // 2, ii - n40, ii)
P_kmax = float(np.max(np.sqrt((kk ** 2).sum(axis=0))))
print(f"P is {len(tids)} modes, max |k| index = {P_kmax:.2f}  "
      f"(physical k = {2*math.pi*P_kmax/L_BOX:.3f})")
del env40, U1, U2, star


def spectrum(u):
    """Shell-averaged energy spectrum of a real vector field (3,N,N,N)."""
    N = u.shape[-1]
    uh = np.fft.fftn(u, axes=(1, 2, 3)) / N ** 3
    e = 0.5 * np.sum(np.abs(uh) ** 2, axis=0)
    f = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(f, f, f, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kbin = np.rint(kmag).astype(int)
    nb = N // 2 + 1
    Ek = np.bincount(kbin.ravel(), weights=e.ravel(), minlength=nb)[:nb]
    return np.arange(nb), Ek


def vort_stats(u, N):
    """Vorticity, enstrophy and relative helicity from the velocity field."""
    uh = np.fft.fftn(u, axes=(1, 2, 3))
    f = np.fft.fftfreq(N) * N * (2 * math.pi / L_BOX)
    KX, KY, KZ = np.meshgrid(f, f, f, indexing="ij")
    wh = np.empty_like(uh)
    wh[0] = 1j * (KY * uh[2] - KZ * uh[1])
    wh[1] = 1j * (KZ * uh[0] - KX * uh[2])
    wh[2] = 1j * (KX * uh[1] - KY * uh[0])
    w = np.real(np.fft.ifftn(wh, axes=(1, 2, 3)))
    ke = 0.5 * np.mean(np.sum(u ** 2, axis=0))
    ens = 0.5 * np.mean(np.sum(w ** 2, axis=0))
    hel = np.mean(np.sum(u * w, axis=0))
    nu_ = np.sqrt(np.mean(np.sum(u ** 2, axis=0)))
    nw_ = np.sqrt(np.mean(np.sum(w ** 2, axis=0)))
    return ke, ens, hel / max(nu_ * nw_, 1e-30)


rows, curves = [], {}
for key in sorted(Z.files):
    u = Z[key].astype(np.float64)
    N = u.shape[-1]
    k, Ek = spectrum(u)
    tot = Ek.sum()
    if tot <= 0:
        continue
    cum = np.cumsum(Ek) / tot
    kpeak = int(np.argmax(Ek))
    kcent = float((k * Ek).sum() / tot)
    # fraction of the direction's energy inside / just outside the P shell
    f_inP = float(Ek[:int(P_kmax) + 1].sum() / tot)
    f_near = float(Ek[:int(2 * P_kmax) + 1].sum() / tot)
    k50 = int(np.searchsorted(cum, 0.50))
    k90 = int(np.searchsorted(cum, 0.90))
    ke, ens, relhel = vort_stats(u, N)
    cfg, d = key.rsplit("_dir", 1)
    rows.append(dict(config=cfg, direction=int(d), N=N,
                     k_peak=kpeak, k_centroid=kcent, k50=k50, k90=k90,
                     frac_within_P_shell=f_inP, frac_within_2P=f_near,
                     enstrophy_over_energy=ens / max(ke, 1e-30),
                     relative_helicity=relhel))
    curves[key] = (k, Ek / tot)

df = pd.DataFrame(rows).sort_values(["config", "direction"])
df.to_csv(SAVE / "m3d_leakdir_stats.csv", index=False)

print("\n" + "=" * 104)
print("PHYSICAL CHARACTER OF THE TOP-3 LEAKAGE DIRECTIONS")
print("=" * 104)
print(df.to_string(index=False))

print("\n--- the decisive number: where does leakage energy live? ---")
print(f"P shell extends to |k| = {P_kmax:.1f}")
for d in (0, 1, 2):
    s = df[df.direction == d]
    print(f"  dir{d}: median k_centroid = {s.k_centroid.median():6.2f}   "
          f"median k90 = {s.k90.median():5.1f}   "
          f"median energy within 2x P shell = {s.frac_within_2P.median():.4f}")
allf = df.frac_within_2P.median()
print(f"\n  median energy inside 2x the P cutoff, all directions: {allf:.4f}")
if allf > 0.5:
    print("  -> LARGE-SCALE. The dangerous unresolved content sits just above")
    print("     the resolved cutoff, not in the far dissipation range.")
    print("     This is the coarse-Phi_h opening: a low-resolution solver would")
    print("     capture most of the defect, which attacks the cost problem.")
else:
    print("  -> SMALL-SCALE. Leakage is dominated by far-field wavenumbers;")
    print("     a coarse Phi_h would miss it and the cost problem stands.")

# --- do the directions look the same across resolution and Reynolds? ---
print("\n--- shape stability of the spectra across configs ---")
ref = "N40_nu1"
kmaxc = 24
for d in (0, 1, 2):
    base = curves.get(f"{ref}_dir{d}")
    if base is None:
        continue
    b = base[1][:kmaxc]
    b = b / max(np.linalg.norm(b), 1e-30)
    sims = []
    for key, (k, E) in curves.items():
        if not key.endswith(f"_dir{d}") or key.startswith(ref):
            continue
        e = E[:kmaxc]
        e = e / max(np.linalg.norm(e), 1e-30)
        sims.append((key.rsplit("_dir", 1)[0], float(np.dot(b, e))))
    print(f"  dir{d} spectral cosine vs {ref}: " +
          "  ".join(f"{c}={v:.3f}" for c, v in sims))

# --- figures ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for d in range(3):
    for key, (k, E) in curves.items():
        if key.endswith(f"_dir{d}"):
            ax[d].loglog(np.maximum(k, 1), np.maximum(E, 1e-16),
                         label=key.rsplit("_dir", 1)[0], lw=1.4)
    ax[d].axvline(P_kmax, color="k", ls="--", lw=1, label="P cutoff")
    ax[d].set_title(f"leakage direction {d}")
    ax[d].set_xlabel("|k|")
    ax[d].set_xlim(1, 40)
ax[0].set_ylabel("normalised E(k)")
ax[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(SAVE / "m3d_leakdir_spectra.png", dpi=140)
print("\nsaved:", SAVE / "m3d_leakdir_spectra.png")

# mid-plane slice of the dominant direction at the richest config
rich = [k for k in Z.files if k.startswith("N96")]
if rich:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for i, key in enumerate(sorted(rich)[:3]):
        u = Z[key]
        N = u.shape[-1]
        sl = np.sqrt(np.sum(u[:, :, :, N // 2] ** 2, axis=0))
        im = ax[i].imshow(sl.T, origin="lower", cmap="magma")
        ax[i].set_title(key)
        plt.colorbar(im, ax=ax[i], fraction=0.046)
    plt.tight_layout()
    plt.savefig(SAVE / "m3d_leakdir_slices.png", dpi=140)
    print("saved:", SAVE / "m3d_leakdir_slices.png")
print("saved:", SAVE / "m3d_leakdir_stats.csv")
