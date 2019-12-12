import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os.path
import sys
from image_analysis import find_nearest
from kerr_geodesic import kerr_geodesic
from utilities import M_from_diam_a, kerr_photon_ring_radius, \
        plot_name_extension

# Create image of Kerr black hole. a is dimensionless as an argument and then
# gets rescaled to a*M.
def kerr_reflected_intensities(l, R0, diam, x_c, y_c, a, Theta_obs, pix_size):

    # Initial conditions:
    E = 1.
    r_obs = 100.

    # Fix angular size for region from where reflection is considered.
    # Typically 1.5 times the size of the photon ring should be sufficient.
    ang_size = 30.

    # Number of pixels:
    n_pix = int((2.*ang_size/pix_size)**2)

    # Determine mass of black hole from given size of the eht shadow diameter:
    # THIS IS ONLY VALID WHEN BH SHADE IS APPROXIMATELY CIRCULAR, I.E. OBS ANGLE
    # CLOSE TO ZERO AND/OR SPIN CLOSE TO ZERO.
    M = M_from_diam_a(diam, a)

    # Rescale spin to dimensionfull angular momentum:
    a = a*M

    # Define surface where light rays get reflected:
    r_horizon = M + math.sqrt(M**2 - a**2)
    r_reflect = 1.2*r_horizon

    # Define radius where the cth map is created:
    r_photonring, r_BL_photonring = kerr_photon_ring_radius(M, a)
    r_cth = r_BL_photonring

    # Define how close points have to be to points of cth map to apply cth map:
    d_closeenough = 1.5*pix_size #Resolution of EHT choice.

    # Check if csv file with close to horizon (cth) map exists:
    cth_map_exists = False
    cth_path = "cth_maps/map_M"+str(round(M, 2))+"_a"+str(round(a, 2))+ \
            "_r"+str(round(r_obs, 2))+"_Theta"+str(round(Theta_obs, 2))+ \
            "_angsize"+str(round(ang_size, 2))+"_rcth"+str(round(r_cth, 2))+ \
            "_dclose"+str(round(d_closeenough, 2))+ \
            "_npix"+str(round(n_pix, 2))+".csv"
    if os.path.isfile(cth_path):
        cth_map_exists = True


    # Name of plot depending on key parameters.
    name_app = plot_name_extension(l, R0, a/M)

    # Sample list of h and v:
    n_sample = int(np.sqrt(n_pix))
    hv_set = [float(i) / (n_sample - 1) for i in range(n_sample)]

    # Create cth map if it doesn't exist yet:
    if not cth_map_exists:
    
        # Load csv with intensities from digitized EHT image:
        I_EHT = np.genfromtxt( \
                "../digitize_eht/EHT_xyT_interpolated_centered.csv", \
                delimiter=",")

        # Scan over hv set to create the cth map:
        map_cth = []
        hvI_cth = []
        for h in hv_set:
            for v in hv_set:

                # Find cartesian coordinates along geodesic:
                x_geo, y_geo, z_geo, I = kerr_geodesic(l, R0, M, a, r_reflect, \
                        r_obs, Theta_obs, ang_size, E, h, v)

                # Create cth map: get coordinates at r_cth of unreflected
                # geodesics when they exit close to horizon region:
                if I == -1.:
                    for i in range(len(x_geo)):
                        d = math.sqrt(x_geo[i]**2+y_geo[i]**2+z_geo[i]**2)
                        if d < r_cth:

                            # Get angular coordinates of starting point of ray.
                            x_ang, y_ang, _ = xy_from_hv([[h, v, 1.]], ang_size)
                            x_ang = x_ang[0]
                            y_ang = y_ang[0]

                            # Get nearest point in EHT image and take its
                            # intensity.
                            I_ray = find_nearest(I_EHT, x_ang, y_ang)[-1]

                            # Add to close to horizon map, i.e. where did the
                            # light ray come from in the close to horizon
                            # region.
                            map_cth.append( \
                                    [x_geo[i], y_geo[i], z_geo[i], I_ray])
                            hvI_cth.append([h, v, I_ray])
                            break

        np.savetxt(cth_path, np.array(map_cth), delimiter=',')

        # Create intensity plot for points used in cth map:
        if len(map_cth) > 0:
            intensity_plot(hvI_cth, ang_size, \
                    "figures/kerr_intensity_cth_used", name_app)

    # Scan over hv set to find intensities of reflected rays:
    map_cth = np.genfromtxt(cth_path, delimiter=",")
    xyzI_reflected = []
    hvI_reflected = []
    for h in hv_set:
        for v in hv_set:

            # Append intensities of reflected rays that came close enough to
            # cth map:
            x_geo, y_geo, z_geo, I = kerr_geodesic(l, R0, M, a, r_reflect, \
                    r_obs, Theta_obs, ang_size, E, h, v, r_cth, d_closeenough, \
                    map_cth)
            if I > 0.:
                hvI_reflected.append([h, v, I])
                for i in range(len(x_geo)):
                    d = math.sqrt(x_geo[i]**2+y_geo[i]**2+z_geo[i]**2)
                    if d < r_cth:
                        xyzI_reflected.append([x_geo[i], y_geo[i], z_geo[i], I])
                        break

    # Plot cth map and reflected end points of geodesics at r_reflect:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-r_cth, r_cth])
    ax.set_ylim([-r_cth, r_cth])
    ax.set_zlim([-r_cth, r_cth])
    xyz_cth = np.array(map_cth)
    ax.scatter(xyz_cth[:,0], xyz_cth[:,1], xyz_cth[:,2], s=10, color='black')
    if len(xyzI_reflected) > 0:
        xyz_re = np.array(xyzI_reflected)
        ax.scatter(xyz_re[:,0], xyz_re[:,1], xyz_re[:,2], s=10, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #plt.show()
    plt.savefig("figures/kerr_close_to_horizon_map.pdf", \
            bbox_inches='tight')
    plt.savefig("figures/kerr_close_to_horizon_map"+name_app+".pdf", \
            bbox_inches='tight')
    plt.close()

    # Plot intensity of reflected rays:
    intensity_plot(hvI_reflected, ang_size, \
            "figures/kerr_reflected_intensity", name_app)

    # Return list of reflected angular x, y coordinates and intensities:
    xyI_reflected = xy_from_hv_np(hvI_reflected, ang_size)

    # Write reflected xyT to file:
    np.savetxt("reflected_intensities/reflected_xyT"+name_app+".csv", \
            xyI_reflected, delimiter=',')

    # Print a few important length scales:
    print("\nBlack hole spin: ", a/M)
    print("Black hole mass: ", M)
    print("Horizon: ", r_horizon)
    print("Reflection surface: ", r_reflect)
    print("Close to horizon map: ", r_cth)
    print("Photon ring Schwarzschild (3M): ", 3.*M)
    print("Photon ring Kerr: ", r_photonring)

    return xyI_reflected

# Get x, y coordinates from h and v angle variables as separate lists. 0.5
# shift is to center origin.
def xy_from_hv(hvI, ang_size):
    x = [(2.*s[0]-1.)*ang_size for s in hvI if s[2] > 0]
    y = [(2.*s[1]-1.)*ang_size for s in hvI if s[2] > 0]
    I = [s[2] for s in hvI if s[2] > 0]
    return x, y, I

# Get x, y coordinates and intensity from h and v angle variables as numpy
# array.
def xy_from_hv_np(hvI, ang_size):
    xyI = [[(2.*s[0]-1.)*ang_size, (2.*s[1]-1.)*ang_size, s[2]] \
            for s in hvI if s[2] > 0]
    xyI = np.asarray(xyI)
    return xyI

# Intensity plot routine:
def intensity_plot(hvI, ang_size, path, name_app):
    x, y, I = xy_from_hv(hvI, ang_size)
    plt.figure(figsize=(18, 15))
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.scatter(x, y, c=I, cmap="hot")
    plt.colorbar()
    plt.xlabel(r'$x \,\,\, [\mu{\rm as}]$', fontsize=20)
    plt.ylabel(r'$y \,\,\, [\mu{\rm as}]$', fontsize=20)
    plt.savefig(path+".pdf", bbox_inches='tight')
    plt.savefig(path+name_app+".pdf", bbox_inches='tight')
    plt.close()

