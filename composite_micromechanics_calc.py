"""
Composite Micromechanics Calculator 

moduli in GPa, stress in MPa, strain in microstrain
"""
import numpy as np

# ------------------------- INPUTS  -------------------------
E1, E2, nu12, G12 = 19.755, 1.9755, 0.35, 0.7   # GPa 
theta_deg = 60                                  # degrees
known  = "stress"            # "stress" or "strain"
system = "xy"                # coordinate system: "xy" or "12"
load   = [-3.5, 7, -1.4]     # stress[MPa] or strain[microstrain]
# -----------------------------------------------------------------------


def transforms(theta_deg):
    """Transformation T: xy to 12"""
    t = np.radians(theta_deg)
    m, n = np.cos(t), np.sin(t)
    Ts = np.array([[m * m, n * n, 2 * m * n],
                   [n * n, m * m, -2 * m * n],
                   [-m * n, m * n, m * m - n * n]])
    Te = np.array([[m * m, n * n, m * n],
                   [n * n, m * m, -m * n],
                   [-2 * m * n, 2 * m * n, m * m - n * n]])
    return Ts, Te


def material_SQ(E1, E2, nu12, G12):
    """Compliance S and stiffness Q in 12 system"""
    nu21 = nu12 * E2 / E1
    d = 1 - nu12 * nu21
    S = np.array([[1 / E1, -nu12 / E1, 0],
                  [-nu12 / E1, 1 / E2, 0],
                  [0, 0, 1 / G12]])
    Q = np.array([[E1 / d, nu12 * E2 / d, 0],
                  [nu12 * E2 / d, E2 / d, 0],
                  [0, 0, G12]])
    return S, Q


def global_SQ(S, Q, theta_deg):
    """Qbar and Sbar in xy system"""
    Ts, Te = transforms(theta_deg)
    Qb = np.linalg.inv(Ts) @ Q @ Te
    Sb = np.linalg.inv(Qb)
    return Qb, Sb


def solver(known, system, load, S, Q, theta_deg):
    """Return stress [MPa] and strain in both systems"""
    Ts, Te = transforms(theta_deg)
    Qb, Sb = global_SQ(S, Q, theta_deg)
    Qb, Sb = Qb * 1e3, Sb / 1e3            # GPa to MPa
    v = np.array(load, dtype=float)
    if known == "strain":
        v = v * 1e-6                       # microstrain to strain
        exy = v if system == "xy" else np.linalg.inv(Te) @ v
        sxy = Qb @ exy
    elif known == "stress":
        sxy = v if system == "xy" else np.linalg.inv(Ts) @ v
        exy = Sb @ sxy
    else:
        raise ValueError("known must be either 'stress' or 'strain'")
    return sxy, Ts @ sxy, exy, Te @ exy


def show(name, M):
    print(f"\n{name}\n{np.array2string(np.asarray(M), precision=4, suppress_small=True)}")
    

S, Q = material_SQ(E1, E2, nu12, G12)
Qb, Sb = global_SQ(S, Q, theta_deg)
show("Q [GPa]", Q)
show("S [1/GPa]", S)
show("Qbar [GPa]", Qb)
show("Sbar [1/GPa]", Sb)

sxy, s12, exy, e12 = solver(known, system, load, S, Q, theta_deg)
show("stress_xy [MPa]", sxy)
show("stress_12 [MPa]", s12)
show("strain_xy [microstrain]", exy * 1e6)
show("strain_12 [microstrain]", e12 * 1e6)
    
    
