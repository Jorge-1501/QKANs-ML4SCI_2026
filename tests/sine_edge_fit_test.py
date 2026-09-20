"""
Small-sample sanity test: fitting a KAN edge with the SineKAN basis
(Reinhardt et al. 2024, "SineKAN: Kolmogorov-Arnold Networks Using
Sinusoidal Activation Functions", arXiv:2407.04149) instead of the
Chebyshev basis currently used in extractor.py.

Why this basis and not a generic Fourier series
-------------------------------------------------
This mirrors the reference SineKAN layer (the same functional form used
in Ria Khatoniar's GSoC 2025 QKAN code, itself a reimplementation of the
official SineKAN layer), not an arbitrary sin/cos fit:

    y = sum_k  A_k * sin(freq_k * x + phase_k)

with grid_size = "degree" = number of sine terms in the sum -- exactly
the same role `degree` plays in your current Chebyshev `_fit_edge`.
freq_k and phase_k are NOT free parameters here: in the reference
implementation they come from a fixed, deterministic grid (freq_k ~
k / (grid_size+1), then phase is warped through a small fixed-point
iteration -- see `build_sine_grid` below, copied 1:1 from the reference
`SineKANLayer.__init__`). Only the amplitudes A_k are trained.

That is the key practical point for your pipeline: because freq_k and
phase_k are fixed once grid_size is chosen, sin(freq_k*x + phase_k) is
just another set of *fixed basis functions* of x -- exactly like
T_n(x) is for Chebyshev. So fitting A_k is a plain linear least-squares
problem, solvable with np.linalg.lstsq, the same way `chebfit` solves
for Chebyshev coefficients. No autograd, no iterative training needed
for this warm-start step -- and it reuses the exact
"fixed degree, no R^2 gate" strategy you already validated for Chebyshev.

This script:
  1. Rebuilds the reference SineKAN frequency/phase grid (from Ria's
     working code / the original paper's reference implementation).
  2. Fits a small sample of one KAN edge with that basis via lstsq,
     fixed degree (grid_size), no early-acceptance gate.
  3. Fits the same sample with your current Chebyshev basis at the
     same degree, for a direct comparison.
  4. Reports R^2/MSE for both and plots the fits.

Replace `make_demo_edge_sample()` with a real (x, y) pair pulled from
one edge of your trained classical KAN (e.g. whatever `_fit_edge`
currently receives) before drawing conclusions -- the function below is
a synthetic stand-in only, so this script runs standalone.
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(10)  # match your SEED=10 convention


# ---------------------------------------------------------------------
# 1. Reference SineKAN frequency / phase grid
#    (ported directly from SineKANLayer.__init__ in the uploaded
#    SineKAN_quarkgluon.py / the official reference implementation)
# ---------------------------------------------------------------------
def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size ** (-K) + C
    return ratio * i_n


def build_sine_grid(grid_size: int, is_first: bool = True):
    """Returns freq (grid_size,) and phase (grid_size,) as fixed 1-D
    input, univariate-edge version of the reference layer (input_dim=1,
    so the input_phase offset -- linspace(0, pi, input_dim) -- is a
    single 0.0 and drops out)."""
    A, K, C = 0.9724, 0.9884, 0.9994

    grid_phase = np.arange(1, grid_size + 1) / (grid_size + 1)  # (grid_size,)
    input_phase = 0.0  # input_dim = 1 -> linspace(0, pi, 1) = [0.0]
    phase = grid_phase + input_phase

    for i in range(1, grid_size):
        phase = forward_step(phase, i, A, K, C)

    exponent = 1 - int(is_first)
    freq = np.arange(1, grid_size + 1) / (grid_size + 1) ** exponent

    return freq, phase


# ---------------------------------------------------------------------
# 2. Basis-fit helpers (fixed degree, no R^2 gate -- same policy as the
#    validated Chebyshev fix)
# ---------------------------------------------------------------------
def fit_sine_edge(x, y, grid_size):
    freq, phase = build_sine_grid(grid_size, is_first=True)
    # design matrix: columns = sin(freq_k * x + phase_k)
    Phi = np.sin(np.outer(x, freq) + phase[None, :])  # (N, grid_size)
    amplitudes, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    y_hat = Phi @ amplitudes
    return y_hat, amplitudes, freq, phase


def fit_chebyshev_edge(x, y, degree):
    # same call your extractor.py already makes (fixed degree, no gate)
    coeffs = np.polynomial.chebyshev.chebfit(x, y, deg=degree)
    y_hat = np.polynomial.chebyshev.chebval(x, coeffs)
    return y_hat, coeffs


def r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


# ---------------------------------------------------------------------
# 3. Demo edge sample -- REPLACE with a real edge from your trained
#    classical KAN before drawing any conclusions from this.
# ---------------------------------------------------------------------
def make_demo_edge_sample(n_points=40):
    x = np.sort(RNG.uniform(-1, 1, n_points))
    y_true = 0.5 * np.sin(3.1 * x + 0.4) + 0.3 * x ** 2 - 0.4 * x
    y = y_true + RNG.normal(0, 0.02, n_points)  # small observation noise
    return x, y


# ---------------------------------------------------------------------
# 4. Run the comparison
# ---------------------------------------------------------------------
def main():
    degree = 4  # same fixed degree validated for the Chebyshev fix

    x, y = make_demo_edge_sample()

    y_hat_sine, amps, freq, phase = fit_sine_edge(x, y, grid_size=degree)
    y_hat_cheb, cheb_coeffs = fit_chebyshev_edge(x, y, degree=degree)

    print(f"Sample size: {len(x)} points, fixed degree/grid_size = {degree}\n")

    print("SineKAN-basis fit (amplitudes trained, freq/phase fixed by the grid):")
    for k in range(degree):
        print(f"  k={k+1}:  A={amps[k]: .4f}   freq={freq[k]: .4f}   phase={phase[k]: .4f}")
    print(f"  R^2 = {r2(y, y_hat_sine):.4f}   MSE = {mse(y, y_hat_sine):.6f}\n")

    print("Chebyshev-basis fit (current pipeline, same degree):")
    print(f"  coeffs = {np.array2string(cheb_coeffs, precision=4)}")
    print(f"  R^2 = {r2(y, y_hat_cheb):.4f}   MSE = {mse(y, y_hat_cheb):.6f}\n")

    # ---- plot ----
    x_dense = np.linspace(x.min(), x.max(), 400)
    Phi_dense = np.sin(np.outer(x_dense, freq) + phase[None, :])
    y_dense_sine = Phi_dense @ amps
    y_dense_cheb = np.polynomial.chebyshev.chebval(x_dense, cheb_coeffs)

    plt.figure(figsize=(7, 4.5))
    plt.scatter(x, y, s=18, color="#5B6B76", label="Edge sample (x, y)", zorder=3)
    plt.plot(x_dense, y_dense_sine, color="#1C7293", lw=2, label=f"SineKAN fit (grid_size={degree})")
    plt.plot(x_dense, y_dense_cheb, color="#E8871E", lw=2, ls="--", label=f"Chebyshev fit (degree={degree})")
    plt.xlabel("x")
    plt.ylabel("edge output")
    plt.title("Sine-basis vs. Chebyshev-basis edge fit (small sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sine_vs_chebyshev_edge_fit.png", dpi=150)
    print("Saved plot: sine_vs_chebyshev_edge_fit.png")

    # ---- circuit-facing parameters ----
    # freq_k*x + phase_k is exactly the rotation angle you'd re-upload
    # per harmonic k in a DRU-style circuit; amps[k] becomes the
    # trainable weight combining the per-harmonic expectation values
    # (analogous to how Chebyshev coefficients currently seed the
    # circuit's initial weights).
    print("\nAngles for a k-th re-uploading gate: theta_k(x) = freq_k * x + phase_k")
    print("These, together with amps[k], are what you'd hand to plot_circuit()")
    print("for a small SineKAN-warm-start circuit diagram.")


if __name__ == "__main__":
    main()
