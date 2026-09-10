#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-D steady conduction: carbon paper + Mg-Al silicate powder + quartz tube in Cl2.
Based on the original "carbon paper + glass cover + argon" model, with the following changes:
- Carbon paper dimensions and position unchanged.
- 1.1 mm glass layer above carbon paper replaced by 1.1 mm Mg-Al silicate powder layer.
- Quartz tube (inner diameter 22 mm / outer diameter 25 mm) added as an annular layer at
  the centre of the domain.
- Gas-region thermal conductivity replaced with that of Cl2.
- 2-D steady-state conduction with Robin (convective) boundary conditions retained.
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import time

# ---------- 1. Physics and mesh ----------
# Domain changed to 3 cm x 3 cm to fully enclose the 25 mm quartz tube
# plus a small gas margin on the outside.
L, N = 0.03, 201          # 3 cm square domain, 201x201 nodes
dx = L/(N-1)
font_family = "Arial"

# Thermal conductivities (adjust as needed)
k_gas    = 0.010          # W m-1 K-1, Cl2 (approximate constant value)
k_powder = 0.30           # W m-1 K-1, Mg-Al silicate powder (dense; roughly 0.2-0.5 range)
k_quartz = 1.40           # W m-1 K-1, quartz tube
k_cp     = 0.25           # Effective thermal conductivity of carbon paper (same as original script)

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 18

T_inf = 25.0              # Ambient temperature (°C)
T_hot_list = [576.6, 637.8, 670, 724.3, 762.9, 798.5, 835.3, 894, 933.8, 987.6]
h_eff = 10.0              # Effective convective heat transfer coefficient at outer boundaries (W m-2 K-1)
tol, max_iter = 1e-3, 20000

# ---------- 2. Geometry ----------
# 2.1 Carbon paper geometry (kept identical to the original script)
cp_w, cp_th = 0.015, 0.000370                   # Carbon paper width and thickness (m)
ix0 = int((L/2-cp_w/2)/dx); ix1 = int((L/2+cp_w/2)/dx)
iy_cp0 = int((L/2-cp_th/2)/dx); iy_cp1 = int((L/2+cp_th/2)/dx)

# 2.2 Mg-Al silicate powder layer: placed immediately above the carbon paper,
#     thickness initially set to 1.1 mm (adjustable)
pow_th = 0.0011                                 # Powder layer thickness: 1.1 mm
iy_pow0 = iy_cp1 + 1
iy_pow1 = iy_pow0 + int(pow_th/dx)

# 2.3 Quartz tube geometry (axis perpendicular to screen; current view is the radial cross-section)
R_in  = 0.011    # m, inner radius of quartz tube (22 mm inner diameter)
R_out = 0.0125   # m, outer radius of quartz tube (25 mm outer diameter)

# Grid coordinates (used to determine radial position of each node)
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y, indexing='xy')
r = np.sqrt((X - L/2)**2 + (Y - L/2)**2)

# ---------- 3. Thermal conductivity field ----------
K = k_gas * np.ones((N, N), dtype=np.float64)   # Default: Cl2 gas everywhere

# Assign powder and carbon paper to their rectangular regions
K[iy_pow0:iy_pow1+1, ix0:ix1+1] = k_powder
K[iy_cp0:iy_cp1+1, ix0:ix1+1]   = k_cp

# Quartz tube annular region: R_in <= r <= R_out
tube_mask = (r >= R_in) & (r <= R_out)
K[tube_mask] = k_quartz

# Carbon paper region uses Dirichlet condition (T = T_hot)
mask_cp = np.zeros_like(K, dtype=bool)
mask_cp[iy_cp0:iy_cp1+1, ix0:ix1+1] = True

beta = K/dx
coef = h_eff + beta

extent_cm = [0, L*100, 0, L*100]
vmin_global, vmax_global = T_inf, max(T_hot_list)

# ---------- 4. Solver ----------
def solve_case(T_hot: float) -> np.ndarray:
    T = np.full((N, N), T_inf, dtype=np.float64)
    T[mask_cp] = T_hot

    for _ in range(max_iter):
        T_old = T.copy()

        # Harmonic-mean interface conductivities (east, west, north, south)
        k_e = 0.5*(K[1:-1,1:-1] + K[2:,1:-1])
        k_w = 0.5*(K[1:-1,1:-1] + K[:-2,1:-1])
        k_n = 0.5*(K[1:-1,1:-1] + K[1:-1,2:])
        k_s = 0.5*(K[1:-1,1:-1] + K[1:-1,:-2])
        denom = k_e + k_w + k_n + k_s

        T_mid = (k_e*T_old[2:,1:-1] + k_w*T_old[:-2,1:-1] +
                 k_n*T_old[1:-1,2:] + k_s*T_old[1:-1,:-2]) / denom
        T[1:-1,1:-1] = T_mid
        T[mask_cp] = T_hot

        # Robin boundary conditions on all four outer edges (convection to ambient)
        T[:,0]  = (beta[:,0]*T[:,1]   + h_eff*T_inf) / coef[:,0]
        T[:,-1] = (beta[:,-1]*T[:,-2] + h_eff*T_inf) / coef[:,-1]
        T[0,:]  = (beta[0,:]*T[1,:]   + h_eff*T_inf) / coef[0,:]
        T[-1,:] = (beta[-1,:]*T[-2,:] + h_eff*T_inf) / coef[-1,:]

        if np.max(np.abs(T - T_old)) < tol:
            break
    return T

# ---------- 5. Main loop ----------
if __name__ == "__main__":
    t0 = time()

    # Carbon paper rectangle in cm coordinates
    x_cp, y_cp = ix0 * dx * 100, iy_cp0 * dx * 100  # cm
    w_cp, h_cp = cp_w * 100, cp_th * 100            # cm

    # Powder rectangle immediately above the carbon paper top surface
    x_pow = x_cp
    y_pow = y_cp + h_cp
    w_pow = w_cp
    h_pow = pow_th * 100

    # Quartz tube centre and radii for visualisation (cm)
    x_c_cm = (L/2) * 100
    y_c_cm = (L/2) * 100
    R_in_cm  = R_in  * 100
    R_out_cm = R_out * 100

    print("Temperature at the top surface of the powder layer (centre point & mean):")
    print("T_hot (°C) |  Center (°C)  |  Mean (°C)")
    print("-----------------------------------------")

    # Set global font to Arial
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.unicode_minus'] = False

    for Th in T_hot_list:
        Tfield = solve_case(Th)

        # Extract temperature along the top surface of the powder layer
        center_idx = (ix0 + ix1) // 2
        T_center = Tfield[iy_pow1, center_idx]               # At x-centre of the powder region
        T_mean   = np.mean(Tfield[iy_pow1, ix0:ix1+1])       # Averaged across the full powder width

        print(f"{Th:9.0f} | {T_center:13.2f} | {T_mean:9.2f}")

        # Plot temperature field (no contour lines)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(Tfield, origin='lower', cmap='inferno',
                       extent=extent_cm, vmin=vmin_global, vmax=vmax_global)
        cbar = fig.colorbar(im, ax=ax, label='Temperature (°C)')

        # Outline carbon paper and powder layer with black rectangles
        ax.add_patch(patches.Rectangle((x_cp,  y_cp),  w_cp,  h_cp,
                                       fill=False, edgecolor='black', linewidth=1.2))
        ax.add_patch(patches.Rectangle((x_pow, y_pow), w_pow, h_pow,
                                       fill=False, edgecolor='black', linewidth=1.2))

        # Mark inner and outer walls of the quartz tube with white circles
        ax.add_patch(patches.Circle((x_c_cm, y_c_cm), R_out_cm,
                                    fill=False, edgecolor='white', linewidth=1.2))
        ax.add_patch(patches.Circle((x_c_cm, y_c_cm), R_in_cm,
                                    fill=False, edgecolor='white', linewidth=1.2, linestyle='--'))

        ax.set_xlabel('x (cm)');  ax.set_ylabel('y (cm)')
        plt.tight_layout()
        fig.savefig(f"cp_pow_quartz_{Th:.0f}C.png", dpi=300)
        plt.close(fig)

    print(f"\nAll cases completed in {time()-t0:.2f} s  (grid {N}x{N})")