from interpolate_eht_image import interpolate_eht_image
from image_analysis import image_analysis, print_image_probs
from kerr_reflected_intensities import kerr_reflected_intensities
from eht_add_reflected import xyI_add_reflected_to_eht
from exclusion_plot import exclusion_plots
from utilities import check_exclusion, merge_two_dicts
import numpy as np
import sys

# Multipole moments:
l_set = [0, 1, 2]

# Reflection coefficients:
R0_set = [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.2, 0.4, 1.0]

# BH spins in units of M in (-1, 1):
a_set = [-0.94, -0.5, 0.01, 0.5, 0.94]

# Pixel size in muas:
pix_size = 1.

# Initial conditions: Theta_obs = 0 means the spin of the black hole and the
# camera pointing are aligned pointing in opposite directions. Maximum angle
# for Theta_obs is pi/2.
Theta_obs = 17./360.*2.*np.pi # M87 value.

# Maximally allowed circulartiy and fractional brightness for exclusions and
# number of sigmas at which the exclusions are calculated:
circ_max = 0.2
fc_max = 0.7
n_sigma = 1.
save_root = "exclusion_results/exclusions"

# Create interpolated EHT image:
image_path_pure = interpolate_eht_image(pix_size)

# Find properties of interpolated pure(=no new physics) EHT image:
x_c, y_c, diam, diam_std, circ, width, width_std, fc, oa, oa_std, A, A_std, \
        image_path_pure_centered = image_analysis(0, 0, 0, image_path_pure)
orig_prop = [diam, diam_std, circ, width, width_std, fc, oa, oa_std, A, A_std]

# Scan through all multipoles, reflection coefficients and BH spins:
exclusion_results = []
for l in l_set:
    for R0 in R0_set:
        for a in a_set:

            # Create reflected image:
            im_refl = kerr_reflected_intensities(l, R0, diam, x_c, y_c, a, \
                    Theta_obs, pix_size)

            # Add reflected image to centered EHT image:
            image_path_re = xyI_add_reflected_to_eht(image_path_pure_centered, \
                    im_refl)

            # Find properties of modified(=including reflection) image using the
            # 'vary_small' option for image_analysis:
            x_c_re, y_c_re, diam_re, diam_re_std, circ_re, width_re, \
                    width_re_std, fc_re, oa_re, oa_re_std, A_re, A_re_std, \
                    image_path_re = image_analysis(l, R0, a, image_path_re, \
                    0.000001, 0.000001, True)
            re_prop = diam_re, diam_re_std, circ_re, width_re, width_re_std, \
                    fc_re, oa_re, oa_re_std, A_re, A_re_std

            # Print physical parameters:
            print("\n PARAMETERS:")
            print("\n l="+str(l)+", R0="+str(R0)+", a="+str(a))

            # Print properties of original image:
            print("\n ORIGINAL EHT IMAGE:")
            print_image_probs(x_c, y_c, diam, diam_std, width, width_std, fc, \
                    oa, oa_std, A, A_std)

            # Print properties of modified image:
            print("\n MODIFIED EHT IMAGE:")
            print_image_probs(x_c_re, y_c_re, diam_re, diam_re_std, width_re, \
                    width_re_std, fc_re, oa_re, oa_re_std, A_re, A_re_std)

            # Check if modified image is consistent at level of image
            # diagnostics:
            compare_res = check_exclusion(n_sigma, orig_prop, re_prop)
            exclusion_results.append(merge_two_dicts(\
                    {'l': l, 'R0': R0, 'a': a}, compare_res))


# Write scan results to file:
exclusion_save_path = 'exclusion_results/exclusions_l'+str(l_set)+'_R0'+ \
        str(R0_set)+'_a'+str(a_set)+'.txt'
with open(exclusion_save_path, 'w') as f:
    for item in exclusion_results:
        f.write("%s\n" % item)

# Create exclusion plots:
exclusion_plots(circ_max, fc_max, exclusion_save_path, save_root, \
        with_scatter=False)
