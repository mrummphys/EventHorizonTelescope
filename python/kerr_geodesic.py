import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sys
from utilities import sphere_plot
from kerr_dgl import kerr_functions
from reflection_coefficient import reflection

# Get Kerr geodesic.
def kerr_geodesic(l, R0, M, a, r_reflect, r_obs, Theta_obs, ang_size, E, h, v, \
        r_cth=0., d_closeenough=0., cth_map=np.array([]), mode="no_plot"):

    # Outer horizon:
    r_horizon = M + math.sqrt(M**2 - a**2)

    # Radius at which we consider photon to be far enough from black hole to
    # use flat space approximation, i.e. no more light bending.
    r_flat = 100.*M

    # ODE solver parameters:
    abserr = 1.0e-8
    relerr = 1.0e-6
    s_stop = 2000.0
    numpoints = 3000

    # Affine parameter sampling:
    s = [s_stop * float(i) / (numpoints - 1) for i in range(numpoints)]

    # Get initial position in BL coordinates:
    r0, Theta0, phi0 = x_cam_to_x_BL(a, r_obs, Theta_obs, ang_size, h, v)

    # Inverse Matrix for intitial momentum in BL coordinates:
    Minv = inv_p_matrix(a, r0, Theta0, phi0)

    # Get initial momentum from E, h and v in flat space approximation for
    # camera location:
    p_r0, p_Theta0, p_phi0 = initial_p_BL(Theta_obs, E, Minv)

    # Pack up the parameters and initial conditions:
    m = [M, a, E]
    x0 = [r0, Theta0, phi0, p_r0, p_Theta0, p_phi0]
    x0_reflect = x0

    # Unreflected rays get dummy default intensity to filter for them later.
    I = -1.

    # Get solutions from camera to reflection surface:
    reflected = False
    kerr_sol = odeint(kerr_dgl, x0, s, args=(m,), atol=abserr, \
            rtol=relerr)
    x = []
    y = []
    z = []
    for si, xi in zip(s, kerr_sol):
        x_cam = BL_to_cam(a, Theta_obs, xi)
        if xi[0] < r_flat:
            if xi[0] > r_reflect:
                x.append(x_cam[0])
                y.append(x_cam[1])
                z.append(x_cam[2])
                if si > 0.95*s_stop:
                    print("Running out of parameter sampling. Abort!")
                    sys.exit(0)
            else:
                # Only explore reflection if cth_map is provided. Otherwise
                # intensity can't be determined.
                if cth_map.size != 0:

                    # BL coordinates where the light ray got reflected.
                    # Important to determine how intensity gets modified by
                    # reflection coefficient.
                    x0_reflect = xi
                    reflected = True

                break

    # Calculate reflected geodesic and use cth map:
    if cth_map.size != 0 and reflected:

        # Get solutions from the reflection surface to twice r_cth in case there
        # was a reflection:
        kerr_sol_reflect = odeint(kerr_dgl_reflect, \
                x0_reflect, s, args=(m,), atol=abserr, rtol=relerr)
        for si, xi in zip(s, kerr_sol_reflect):
            if xi[0] < 2.*r_cth and xi[0] > r_horizon:
                x_cam = BL_to_cam(a, Theta_obs, xi)
                x.append(x_cam[0])
                y.append(x_cam[1])
                z.append(x_cam[2])
            else:
                break

        # Reverse order list of coordinates produced by dgl solver, IF cth map
        # should be used BEFORE reflection:
        x = list(reversed(x))
        y = list(reversed(y))
        z = list(reversed(z))

        # Get coordinates of geodesic that are at same distance as the cth map:
        for i in range(len(x)):
            d = math.sqrt(x[i]**2+y[i]**2+z[i]**2)
            if d < r_cth:

                # Get intensity via cth map and reflection:
                I = I_cth(cth_map, np.array([x[i], y[i], z[i]]), d_closeenough)

                # Plot geodesic if it will contribute to the total flux to
                # check where the light rays come from:
                """
                if I > 0.:
                #if I > 0. and abs(h-0.5) < 0.01 and abs(v-0.5) < 0.01:
                    print((2*h-1)*ang_size, (2*v-1)*ang_size)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    x_refl, y_refl, z_refl = sphere_plot(r_reflect)
                    x_cth, y_cth, z_cth = sphere_plot(r_cth)
                    r_3Dplot = 1.5*r_cth
                    ax.set_xlim([-r_3Dplot, r_3Dplot])
                    ax.set_ylim([-r_3Dplot, r_3Dplot])
                    ax.set_zlim([-r_3Dplot, r_3Dplot])
                    ax.plot_surface(x_refl, y_refl, z_refl, color='black', \
                            alpha=0.3)
                    ax.plot_surface(x_cth, y_cth, z_cth, color='black', \
                            alpha=0.1)
                    ax.scatter(x, y, z, s=10, color='blue')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    plt.show()
                """

                # Modify intensity by reflection coefficient at reflection
                # surface:
                I = I*reflection(l, R0, x0_reflect[1])

                break

    # Plot geodesic:
    if mode == "geo_plot":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_refl, y_refl, z_refl = sphere_plot(r_reflect)
        ax.set_xlim([-r_obs, 1.05*r_obs])
        ax.set_ylim([-r_obs, r_obs])
        ax.set_zlim([-r_obs, r_obs])
        ax.plot_surface(x_refl, y_refl, z_refl, color='black', alpha=0.2, \
                label="r=r_h + epsilon")
        ax.scatter(x, y, z, s=10, color='blue', label="Photon Geodesic")
        ax.scatter(x[0], y[0], z[0], s=200, marker="o", color="black")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #plt.legend()
        plt.show()
        plt.savefig("figures/kerr_geodesic.pdf", \
                bbox_inches='tight')
        plt.close()
    
    return x, y, z, I

# Defines the differential equation for the Kerr geoesic. s is the affine
# parameter of the geodesic. 
def kerr_dgl(x, s, m):
    
    # Coordinates:
    r, Theta, phi, p_r, p_Theta, p_phi  = x

    # Parameters:
    M, a, E = m

    # Get Kerr functions:
    k, dkdr, dkdT = kerr_functions(M, a, r, Theta)
    A, B, C, D, F, G, H = k
    dAdr, dBdr, dCdr, dDdr, dFdr, dGdr, dHdr = dkdr
    dAdT, dBdT, dCdT, dDdT, dFdT, dGdT, dHdT = dkdT

    # Define dt/dlambda which is a function of the Kerr functions:
    tdot = E/A + H*p_phi

    # Create f = (r', Theta', phi', p_r', p_Theta', p_phi'):
    f = [ p_r, \
            p_Theta, \
            p_phi, \
            1/2./C*(dAdr*tdot**2 - 2*dBdr*tdot*p_phi - dCdr*p_r**2 - \
            2*dCdT*p_Theta*p_r + dDdr*p_Theta**2 + dFdr*p_phi**2), \
            1/2./D*(dAdT*tdot**2 - 2*dBdT*tdot*p_phi + dCdT*p_r**2 - \
            2*dDdr*p_Theta*p_r - dDdT*p_Theta**2 + dFdT*p_phi**2), \
            1/G*(E*dHdr*p_r + E*dHdT*p_Theta - dGdr*p_r*p_phi - \
            dGdT*p_Theta*p_phi)]

    return f

# Define reflected differential equation for Kerr:
def kerr_dgl_reflect(x, s, m):
    f = kerr_dgl(x, s, m)
    return [f[0], f[1], f[2], -f[3], f[4], f[5]]

# Calculate the initial momentum from the camera pointing and photon energy.
# This is done in the flat space approximation.
def initial_p_BL(Theta_obs, E, Minv):

    # Initial momentum in camera coordinate system. Due to extreme hierachy
    # between distance to M87 (Mpc) and size of BH (AU) the light rays are
    # parallel to the x-axis for all practical purposes.
    p0_cam = [-E, 0., 0.]

    # Rotate to Kerr cartesian momentum:
    R = R_cam_to_kerr(Theta_obs)
    p0_kerr = R.dot(p0_cam)

    # Transform Kerr cartesian momentum to Boyer-Lindquist momentum:
    p_BL = Minv.dot(p0_kerr)

    return p_BL

# Inverse matrix for Boyer-Lindquist momentum variables, calcuated analytically
# in ../mathematica/initial_momentum_matrix_inverse.nb.
def inv_p_matrix(a, r0, Theta0, phi0):
    denom = a**2 + 2.*r0**2 + a**2*np.cos(2.*Theta0)
    sqrtfac = np.sqrt(a**2 + r0**2)
    Minv = np.array([[2.*r0*sqrtfac*np.sin(Theta0)*np.cos(phi0)/denom, \
            2.*r0*sqrtfac*np.sin(Theta0)*np.sin(phi0)/denom, \
            2.*(a**2+r0**2)*np.cos(Theta0)/denom], \
            [2.*sqrtfac*np.cos(Theta0)*np.cos(phi0)/denom, \
            2.*sqrtfac*np.cos(Theta0)*np.sin(phi0)/denom, \
            -2.*r0*np.sin(Theta0)/denom], \
            [-np.sin(phi0)/np.sin(Theta0)/sqrtfac, \
            np.cos(phi0)/np.sin(Theta0)/sqrtfac, \
            0.]])
    return Minv

# Transfer Boyer-Lindquist to cartesian camera system: 
def BL_to_cam(a, Theta_obs, x_BL):
    # Transfer from Boyer-Lindquist to cartesian Kerr:
    x_kerr = [np.sqrt(x_BL[0]**2+a**2)*math.cos(x_BL[2])*math.sin(x_BL[1]), \
            np.sqrt(x_BL[0]**2+a**2)*math.sin(x_BL[2])*math.sin(x_BL[1]), \
            x_BL[0]*math.cos(x_BL[1])]
    # Transfer cartesian Kerr to cartesian camera: 
    R_inv = R_kerr_to_cam(Theta_obs)
    x_cam = R_inv.dot(np.array(x_kerr))
    return x_cam

# Rotate Cartesian camera coordinate system to cartesian coordinate system in
# which the z-axis is aligned with the black hole spin. The camera is located
# along the x-axis of the camera system.
def x_cam_to_x_kerr(r0_cam, Theta_obs, ang_size, h, v):
    x_cam = np.array([r0_cam, (2*h-1)*ang_size, (2*v-1)*ang_size])
    R = R_cam_to_kerr(Theta_obs)
    x_kerr = R.dot(x_cam)
    return x_kerr

# Rotation matrix that rotates camera cartesian coordinates to Kerr cartesian
# coordinates:
def R_cam_to_kerr(Theta_obs):
    if Theta_obs > math.pi/2. or Theta_obs < 0.:
        print("Theta_obs is out of range [0,pi/2]. Abort!")
        sys.exit(0)
    R = np.array([[math.sin(Theta_obs), 0., -math.cos(Theta_obs)], \
            [0., 1., 0.], \
            [math.cos(Theta_obs), 0., math.sin(Theta_obs)]])
    return R

# Rotation matrix that rotates Kerr cartesian coordinates to camera cartesian
# coordinates:
def R_kerr_to_cam(Theta_obs):
    if Theta_obs > math.pi/2. or Theta_obs < 0.:
        print("Theta_obs is out of range [0,pi/2]. Abort!")
        sys.exit(0)
    R = np.array([[math.sin(Theta_obs), 0., math.cos(Theta_obs)], \
            [0., 1., 0.], \
            [-math.cos(Theta_obs), 0., math.sin(Theta_obs)]])
    return R

# Transform camera cartesian coordinates to Boyer-Lindquist coordinates:
def x_cam_to_x_BL(a, r0_cam, Theta_obs, ang_size, h, v):
    x_kerr = x_cam_to_x_kerr(r0_cam, Theta_obs, ang_size, h, v)
    v_guess = [r0_cam, Theta_obs, 0.]
    BL = fsolve(eqs_x_cam_to_x_BL , v_guess, \
            args=(a, x_kerr[0], x_kerr[1], x_kerr[2]))
    r0, Theta0, phi0 = BL[0], BL[1], BL[2]
    if r0 < 0.:
        print("Negative Boyer-Lindquist radial coordinate. Abort!")
        sys,exit(0)
    return r0, Theta0, phi0

# Equations for fsolve to transform cartesian to Boyer-Lindquist. z is the
# parameter vector build by [a, x, y, z].
def eqs_x_cam_to_x_BL(v, *z):
    r0, Theta0, phi0 = v
    return (math.sqrt(r0**2+z[0]**2)*math.sin(Theta0)*math.cos(phi0) - z[1], \
            math.sqrt(r0**2+z[0]**2)*math.sin(Theta0)*math.sin(phi0) - z[2], \
            r0*math.cos(Theta0) - z[3])

# Get intensity from point closests to xyz on cth map sphere.
def I_cth(cth_map, xyz, d_closeenough):
    xyz = np.asarray(xyz)
    cth_xyzs = cth_map[:,:-1]
    deltas = cth_xyzs - xyz
    dists_to_point = np.einsum('ij,ij->i', deltas, deltas)
    idx = dists_to_point.argmin()
    # Only return intensity of nearest point of cth map if xyz is not too far
    # away:
    if np.sqrt(dists_to_point[idx]) < d_closeenough:
        I = cth_map[idx][-1]
    else:
        I = 0.
    return I
