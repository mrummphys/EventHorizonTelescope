import math
import sys

# Reflection coefficient with arguments:
# - multipole moment: l=0,1,2
# - reflection coefficient R0 between 0 and 1
# - polar angle theta.
def reflection(l, R0, Theta):

    # Monopole:
    if l == 0:
        R = R0

    # Dipole:
    elif l == 1:
        R = R0*abs(math.cos(Theta))

    # Quadrupole:
    elif l == 2:
        R = R0*abs(math.sin(Theta)*math.cos(Theta))

    else:
        print("Can only handle multipoles up to l=2 right now. Abort!")
        sys.exit(0)
    if R < 0. or R > 1.:
        print("Invalid reflection coefficient. Abort!")
        sys.exit(0)
    return R
