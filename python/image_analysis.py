import numpy as np
import sys
from temperature_map_plot import plot_xyT
from utilities import plot_name_extension

# Analyze image = table of x, y, T on 
# - center position, 
# - diameter, 
# - circularity, 
# - width, 
# - ring orientation, 
# - azimuthal asymmetry, 
# - fractional central brightness.
def image_analysis(l, R0, a, csv_path, x_c=0., y_c=0., vary_small=False):

    # Path for centered image:
    cent_path = csv_path[:-4]+"_centered.csv"

    # Load image csv:
    im = np.genfromtxt(csv_path, delimiter=",")

    # Only take datapoints up to 60% of peak brightness (EHT uses only 95% of
    # peak brightness which we can't accomplish here because temperature
    # resolution of digitized image isn't good enough and reflected images have
    # higher peak brightness than the original image).
    max_I = np.amax(im[:,-1])
    im_red = np.asarray([x for x in im if x[-1] >= 0.6*max_I])

    # If central coordinates not given, find them (takes long).
    if x_c == 0. and y_c == 0.:
        x_c, y_c, _, _, _, _ = find_ring_center(im_red, vary_small)

        # Create centered version of the image and save:
        im_centered = np.copy(im)
        im_centered[:,0] -= x_c
        im_centered[:,1] -= y_c
        np.savetxt(cent_path, im_centered, delimiter=",")

    # Calculate Intensity floor:
    I_floor = I_at_radius(im, 50., x_c, y_c)

    # Calculate diameter and width with errors:
    rpkm, rpkstd, width, width_std = rpeak_width(im_red, x_c, y_c, I_floor)
    diam, diam_std = 2.*rpkm, 2.*rpkstd

    # Calculate deviation from circularity:
    circ = diam_std/diam

    # Calculate fractional central brightness:
    fc = frac_central_brightness(im, diam/2., x_c, y_c)

    # Calculate ring orientation in degrees:
    oa, oa_std = \
            orientation_angle(im, (diam-width)/2., (diam+width)/2., x_c, y_c)

    # Calculate azimuthal asymmetry:
    A, A_std = az_asymmetry(im, (diam-width)/2., (diam+width)/2., x_c, y_c)

    # Display analysis results:
    #print_image_probs(x_c, y_c, diam, diam_std, width, width_std, fc, oa, \
    #    oa_std, A, A_std)

    # Plot temperature map and diameter:
    name_app = plot_name_extension(l, R0, a)
    plot_path = "figures/"+csv_path.split("/")[-1][:-4]+"_diameter.pdf"
    plot_xyT(im, plot_path, diam/2., diam_std/2., x_c, y_c)
    plot_path_name_app = plot_path[:-4]+name_app+".pdf"
    plot_xyT(im, plot_path_name_app, diam/2., diam_std/2., x_c, y_c)

    return x_c, y_c, diam, diam_std, circ, width, width_std, fc, oa, oa_std, \
            A, A_std, cent_path

# Find the center of the ring:
def find_ring_center(im, vary_small):

    # Scan through candidates for ring center. Vary small should be used for
    # the reflected image added to the originial EHT image because original
    # image was centered so new center should be close to zero.
    if vary_small:
        search_lim = 4.
        xyc_cands = np.linspace(-search_lim, search_lim, 10)
    else:
        search_lim = 8.
        xyc_cands = np.linspace(-search_lim, search_lim, 20)
    def_val = 10000.
    circ_min = def_val
    xc, yc = def_val, def_val
    diam, diam_std = def_val, def_val
    width, width_std = def_val, def_val
    for x in xyc_cands:
        for y in xyc_cands:
            rpkm, rpkstd, FWHMm, FWHMstd = rpeak_width(im, x, y, 0., "no_width")
            #print(x, y, rpkm, rpkstd)
            if rpkm > 20. and rpkm < 27. and rpkstd > 0.:
                circ = rpkstd/rpkm
                if circ < circ_min and circ > 0.01:
                    circ_min = circ
                    xc, yc = x, y
                    diam, diam_std = 2.*rpkm, 2.*rpkstd
                    width, width_std = FWHMm, FWHMstd

    if abs(xc) == search_lim or abs(yc) == search_lim:
        print("Warning! Image center is found to be at limits of " \
                +"coordinate search. Increase search limits!")
        print("Setting x_c = y_c = 0 manually!")
        if not vary_small:
            xc = 0
            yc = 0

    return xc, yc, diam, diam_std, width, width_std

# Get mean/std of the radii of peak brightness:
def rpeak_width(im, x_c, y_c, I_floor, calc_width="with_width"):

    theta_scan = theta_scanset()
    r_scan = r_scanset()
    rpks = []
    FWHMs = []
    for t in theta_scan:
        n_r = r_scan.shape[0]
        rI = np.empty([n_r, 2], np.float64)
        for i in range(n_r):
            x = x_c + r_scan[i]*np.cos(t)
            y = y_c + r_scan[i]*np.sin(t)
            # Find point in im nearests to x, y:
            xyT_nearest = find_nearest(im, x, y)
            rI[i, 0] = r_scan[i]
            rI[i, 1] = xyT_nearest[-1]

        # Find rpk where I is maximal. rI is rounded because otherwise argmax
        # picks different index for different versions of numpy.
        # Decimals should be determined by the T-resolution of the digitized
        # image, currently 0.5*10^9 K.
        I_theta_rounded = np.around(0.2*rI[:,1], decimals=1)

        # Take average of all the indices where Temperature is max, to make
        # peak finder less likely to be unstable:
        indices_max = np.argwhere(I_theta_rounded == np.amax(I_theta_rounded))
        idx = int(np.mean(indices_max))
        #idx = np.around(rI[:,1], decimals=2).argmax()
        rpk = rI[idx][0]

        # Calculate width via FWHM if desired:
        if calc_width == "with_width":
            rI_nobias = np.asarray([[x[0], x[1] - I_floor] for x in rI])
            Ipk = rI_nobias[idx][1]
            rI_left = rI_nobias[rI_nobias[:,0] < rpk]
            rI_right = rI_nobias[rI_nobias[:,0] > rpk]
            if rI_left.size > 0 and rI_right.size > 0:
                r_left = find_nearest_rI(rI_left, Ipk/2.)
                r_right = find_nearest_rI(rI_right, Ipk/2.)
                FWHM = r_right - r_left
                FWHMs.append(FWHM)

        # Keep only rpks(/FWHMs) that are within 5 and 50 muas, as in paper IV:
        if rpk > 5. and rpk < 50.:
            rpks.append(rpk)

    # Find mean and std of rpks and FWHMs:
    rpkm, rpkstd = mean_std(rpks)
    if calc_width == "with_width":
        FWHMm, FWHMstd = mean_std(FWHMs)
    else:
        FWHMm, FWHMstd = 0., 0.
        
    return rpkm, rpkstd, FWHMm, FWHMstd

# Calculate fractional central brightness:
def frac_central_brightness(im, r_ring, x_c, y_c):
    I_innerdisk = I_within_radius(im, 5., x_c, y_c)
    I_ring = I_at_radius(im, r_ring, x_c, y_c)
    return I_innerdisk/I_ring

# Calculate orientation angle in degrees:
def orientation_angle(im, r_in, r_out, x_c, y_c):
    r_in_out = r_inout(r_in, r_out)
    v_int = np.vectorize(int_Iexptheta_at_radius)
    v_int.excluded.add(0)
    v_int.excluded.add(2)
    v_int.excluded.add(3)
    angles_in_out = np.angle(v_int(im, r_in_out, x_c, y_c))
    angle_m, angle_std = mean_std(np.asarray(angles_in_out))
    if  angle_m < 0.:
        angle_m = 2*np.pi + angle_m
    angle_m_deg = angle_m*180./np.pi
    angle_std_deg = angle_std*180./np.pi
    # In EHT paper eta is measured from y-axis counterclockwise:
    angle_m_deg = angle_m_deg - 90.
    return angle_m_deg, angle_std_deg

# Calculate azimuthal asymmetry:
def az_asymmetry(im, r_in, r_out, x_c, y_c):
    r_in_out = r_inout(r_in, r_out)
    theta_scan = theta_scanset()
    n_theta = len(theta_scan)
    Delta_theta = theta_scan[1] - theta_scan[0]
    As = []
    for r in r_in_out:
        numerator = np.absolute(int_Iexptheta_at_radius(im, r, x_c, y_c))
        # Get integral over I from average of I:
        denominator = n_theta*Delta_theta*I_at_radius(im, r, x_c, y_c)
        As.append(numerator/denominator)
    return mean_std(np.asarray(As))

# Find average value of intensity at a given radius:
def I_at_radius(im, r, x_c, y_c):
    theta_scan = theta_scanset()
    T_at_radius = []
    for t in theta_scan:
        x = x_c + r*np.cos(t)
        y = y_c + r*np.sin(t)
        xyT_nearest = find_nearest(im, x, y)
        T_at_radius.append(xyT_nearest[-1])
    return np.mean(np.asarray(T_at_radius))

# Find absolute value and argument of integral over exp(I*Theta) I(Theta) via
# sum approximation. Used to find orientation angle and azimuthal asymmetry.
def int_Iexptheta_at_radius(im, r, x_c, y_c):
    theta_scan = theta_scanset()
    Iexptheta_at_radius = []
    for t in theta_scan:
        x = x_c + r*np.cos(t)
        y = y_c + r*np.sin(t)
        xyT_nearest = find_nearest(im, x, y)
        Iexptheta_at_radius.append(xyT_nearest[-1]*np.exp(complex(0, t)))
    Delta_theta = theta_scan[1] - theta_scan[0] 
    int_Iexptheta = Delta_theta*np.sum(np.asarray(Iexptheta_at_radius))
    return int_Iexptheta

# Find average value of intensity within a given radius:
def I_within_radius(im, r, x_c, y_c):
    im_inr = im[ (im[:,0] - x_c)**2 + (im[:,1] - y_c)**2 < r**2 ]
    return np.mean(im_inr[:,-1])
        
# Calculate mean and std of list:
def mean_std(array):
    array = np.array(array)
    return np.mean(array), np.std(array)

# Values of theta that are scanned:
def theta_scanset():
    return np.arange(0., 2.*np.pi, 3./360.*2.*np.pi)
    #return np.arange(0., 2.*np.pi, 1./360.*2.*np.pi) # EHT used.

# Values of r that are scanned:
def r_scanset():
    return np.arange(0., 50.5, 1.5)
    #return np.arange(0., 50.5, 0.5) # EHT used.

# Find radii from the scanset that are between r_in and r_out:
def r_inout(r_in, r_out):
    r_scan = r_scanset()
    r_less_out = r_scan[r_scan < r_out]
    r_in_out = r_less_out[r_less_out > r_in]
    return r_in_out

# Display analysis results:
def print_image_probs(x_c, y_c, diam, diam_std, width, width_std, fc, oa, \
        oa_std, A, A_std):
    print("\n Image center: x =", x_c, ", y =", y_c)
    print("\n Image diameter:", diam, "pm", diam_std, "muas")
    print("\n Circularity:", diam_std/diam)
    print("\n Ring width:", width, "pm", width_std, "muas")
    print("\n Fractional central brightness:", fc)
    print("\n Orientation angle:", oa, "pm", oa_std, "deg")
    print("\n Azymuthal asymmetry:", A, "pm", A_std)

# Find nearest point in im to given x and y coordinates:
def find_nearest(im, x, y):
    point = np.asarray([x, y])
    xys = im[:,:-1]
    #dists_to_point = np.sum((xys - point)**2, axis=1)
    deltas = xys - point
    dists_to_point = np.einsum('ij,ij->i', deltas, deltas)
    idx = dists_to_point.argmin()
    return im[idx]

# Find nearest r for a given rI 2D array. Used to find FWHM.
def find_nearest_rI(rI, I_val):
    Is = rI[:,1]
    idx = (np.abs(Is - I_val)).argmin()
    r_nearest = rI[idx][0]
    return r_nearest
