import math
import numpy as np
from scipy.optimize import fsolve

# Surface plot for 3D plot of horizon:
def sphere_plot(r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Circle plot for horizon:
def circle_plot(r):
    n_points = 200
    angles = [2*math.pi*float(i)/(n_points-1) for i in range(n_points)]
    x = []
    y = []
    for a in angles:
        x.append(r*math.cos(a))
        y.append(r*math.sin(a))
    return x, y

# Get Mass of BH from measured diameter and spin (in units of M).
def M_from_diam_a(d, a):
    # Peak brightness aka ring location is not identical with photon ring.
    # Paper VI: ring location is about 10% larger than photon ring, hence
    # rescale factor:
    ring_to_peakbrightness = 1.1
    alpha, _ = kerr_photon_ring_radius(1., a)
    alpha = 2.*alpha
    M = d/ring_to_peakbrightness/alpha
    return M

# Radius of the photon ring for given mass and spin according to 0812.1328.
# THIS IS ONLY VALID WHEN BH SHADE IS APPROXIMATELY CIRCULAR, I.E. OBS ANGLE
# CLOSE TO ZERO AND/OR SPIN CLOSE TO ZERO.
def kerr_photon_ring_radius(M, J):
    r0_guess = 3.*M
    r0 = fsolve(xi_zero, r0_guess, args=(M, J))[0]
    eta0 = (4.*J**2*M*r0**3-r0**4*(r0-3.*M)**2)/J**2/(r0-M)**2 # Eq. 9.
    r_ring = math.sqrt(eta0 + J**2)
    return r_ring, r0

# Find where xi_c is zero (eq. 9 in 0812.1328) because there x is zero.
def xi_zero(r, *params):
    M, J = params
    return M*(r**2-J**2)-r*(r**2-2.*M*r+J**2) 

# Schwarzschild metric:
def f_schwarz(r, M):
    return 1. - 2.*M/r

# Filename extension characterizing multipole l, reflection coefficient R0 and
# spin a in units of M:
def plot_name_extension(l, R0, a):
    name_app = "_l"+str(round(l, 2))+ \
            "_R"+str(round(R0, 4))+ \
            "_a"+str(round(a, 2))
    return name_app

# Compare original image diagnostics and modified image diagnostics based on
# n-sigma:
def check_exclusion(n, orig, modified):
    diam, diam_std, circ, width, width_std, fc, oa, oa_std, A, A_std = orig
    diam_re, diam_re_std, circ_re, width_re, width_re_std, fc_re, oa_re, \
            oa_re_std, A_re, A_re_std = modified
    xs = [diam, width, oa, A]
    sigmaxs = [diam_std, width_std, oa_std, A_std]
    ys = [diam_re, width_re, oa_re, A_re]
    sigmays = [diam_re_std, width_re_std, oa_re_std, A_re_std]
    names = ["diam", "width", "oa", "A"]
    res = {'circ_val': circ, 'circ_val_mod': circ_re, 'fc_val': fc, \
            'fc_val_mod': fc_re}
    if circ_re > n*circ:
        res['circ'] = "Fail"
    else:
        res['circ'] = "Ok"
    if fc_re > n*fc:
        res['fc'] = "Fail"
    else:
        res['fc'] = "Ok"
    for x, sigma_x, y, sigma_y, name in zip(xs, sigmaxs, ys, sigmays, names):
        if not compatible(n, x, sigma_x, y, sigma_y):
            res[name] = "Fail"
        else:
            res[name] ="Ok"
    return res

# Function to check if to measurements are compatible with each other at the
# n-sigma level:
def compatible(n, x, sigma_x, y, sigma_y):
    sigma_distance = np.sqrt(sigma_x**2+sigma_y**2)
    comp = True
    if x < y:
        if x + n*sigma_distance < y:
            comp = False
    elif x >= y:
        if x - n*sigma_distance > y:
            comp = False
    return comp

# Merge two dictionaries:
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
