# Constraining Fundamental Physics with the Event Horizon Telescope

We constrain extensions of general relativity that can be described by a
reflection coefficient as described in INSERT LINK.

## Requirements

Python 3.7, SciPy 1.2

## Usage

python/main.py includes all the relevant routines. These are:

### interpolate_eht_image 
Create interpolated digitized EHT image. The csv
files characterizing the digitized EHT image are stored in digitize_eht/.

### image_analysis
Determine properties of image such as ring diameter, width,
deviation from circularity, fractional central brightness, etc.

### kerr_reflected_intensities
Get intensities of reflected light rays.
Includes kerr_geodesic which calculates the Kerr geodesic of a light ray
using odeint.

### xyI_add_reflected_to_eht
Add reflected image to the original EHT image.

### check_exclusion
Check if a modified gravity image is consistent with the
EHT observtions.

### exclusion_plots
Creates exclusion plots.



