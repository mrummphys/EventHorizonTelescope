from temperature_map_plot import plot_xyT
import numpy as np
import sys

# Routine that adds the reflected intensities to the eht measured intensities.
# The added picture is normalized to the same normalization as the pure eht
# picture.
def xyI_add_reflected_to_eht(eht_path, im_reflected):

    im_eht = np.genfromtxt(eht_path, delimiter=",")

    # Find normalization of original eht image:
    eht_norm = im_norm(im_eht)

    # Add intensities of reflected light rays to closest pixel point in eht
    # image:
    for xyI_re in im_reflected:
        idx_nearest = find_nearest_idx(im_eht, xyI_re[0], xyI_re[1])
        im_eht[idx_nearest, 2] += xyI_re[2]

    # Normalize image to original normalization:
    im_added = normalize_im(im_eht, eht_norm)

    # Create picture of added image:
    plot_xyT(im_added, "figures/xyT_added.pdf")

    # Save csv for later use:
    im_added_path = 'reflected_intensities/xyT_added.csv'
    np.savetxt(im_added_path, im_added, delimiter=',')

    return im_added_path

# Normalize image to given normalization:
def normalize_im(im, norm):
    current_norm = im_norm(im)
    if current_norm != 0.:
        im[:,2] = norm/current_norm*im[:,2]
    else:
        print("Error in normalize_im: Can't normalize if current norm is zero!")
        sys.exit(0)
    return im

# Normalization, i.e. sum of all intensities of an image:
def im_norm(im):
    return np.sum(im[:,2])

# Find nearest point in im to given x and y coordinates:
def find_nearest_idx(im, x, y):
    point = np.asarray([x, y])
    xys = im[:,:-1]
    #dists_to_point = np.sum((xys - point)**2, axis=1)
    deltas = xys - point
    dists_to_point = np.einsum('ij,ij->i', deltas, deltas)
    idx = dists_to_point.argmin()
    return idx

