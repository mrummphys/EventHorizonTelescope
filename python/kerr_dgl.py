import numpy as np

# Define expressions necessary for the Kerr differential equations:
def kerr_functions(M, a, r, Theta):
    T = Theta
    rs = 2*M
    rho2 = r**2 + a**2*np.cos(T)**2
    Delta = r**2 - rs*r + a**2

    # Functions that appear in the Kerr metric and Kerr geodesic equations:
    A = -1.*(1 - rs*r/rho2)
    B = a*r*rs*np.sin(T)**2/rho2
    C = rho2/Delta
    D = rho2
    F = np.sin(T)**2*(r**2+a**2+a*B)
    G = F - B**2/A
    H = B/A

    # Derivates of those functions:
    dDdr = 2*r
    dDdT = -a**2*np.sin(2*T)
    dAdr = -1.*(-rs*(1/D - r/D**2*dDdr))
    dAdT = -1.*(rs*r/D**2*dDdT)
    dBdr = a*rs*np.sin(T)**2*(1/D - r/D**2*dDdr)
    dBdT = a*rs*r*(np.sin(2*T)/D - np.sin(T)**2/D**2*dDdT)
    dCdr = 1/Delta*dDdr - D/Delta**2*(2*r-rs)
    dCdT = 1/Delta*dDdT
    dFdr = np.sin(T)**2*(2*r + a*dBdr)
    dFdT = np.sin(2*T)*(r**2+a**2+a*B) + np.sin(T)**2*a*dBdT
    dGdr = dFdr - 2*B/A*dBdr + B**2/A**2*dAdr
    dGdT = dFdT - 2*B/A*dBdT + B**2/A**2*dAdT
    dHdr = 1/A*dBdr - B/A**2*dAdr
    dHdT = 1/A*dBdT - B/A**2*dAdT

    # k and derivatives are composed of the functions A, B, C, D, F, G, H:
    k = [A, B, C, D, F, G, H]
    dkdr = [dAdr, dBdr, dCdr, dDdr, dFdr, dGdr, dHdr]
    dkdT = [dAdT, dBdT, dCdT, dDdT, dFdT, dGdT, dHdT]

    return k, dkdr, dkdT
