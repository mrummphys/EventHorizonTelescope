import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from temperature_map_plot import plot_xyT

# Create interpolated image of WebPlotDigitizer image. pix_size is given in
# micro arcseconds.
def interpolate_eht_image(pix_size):

    # Maxium values for x and y pixel coordinates in muas: 
    lim = 55

    # Combine all temperature csv files into one list:
    files = glob.glob("../digitize_eht/T*_DeltaXY4_DeltaColor20.csv")
    xyT = []
    for f in files:
        T = float(f[17:20])
        data = np.genfromtxt(f, delimiter=",")
        for d in data:
            if d[0] < 10.: # Throw away legend points.
                xyT.append([d[0], d[1], T])
    xyT = np.array(xyT)

    # Find temperature center:
    T_sum = sum(xyT[:,2])
    x_c = 1./T_sum * sum([d[0]*d[2] for d in xyT])
    y_c = 1./T_sum * sum([d[1]*d[2] for d in xyT])

    # Center temperature map around zero:
    xyT = np.array([[d[0] - x_c, d[1] - y_c, d[2]] for d in xyT])

    # Save csv for later use:
    np.savetxt('../digitize_eht/EHT_xyT_digitized.csv', xyT, delimiter=',')

    # Plot temperature map as digitized:
    plot_xyT(xyT, "../digitize_eht/EHT_xyT_digitized.pdf")

    # Create sampled interpolated plot:
    n_grid = int(2*lim/pix_size) 
    grid_x, grid_y = \
            np.mgrid[-lim:lim:complex(0, n_grid), -lim:lim:complex(0, n_grid)]
    xys_grid = grid_x[:,0]
    xys = xyT[:,:-1]
    Ts = xyT[:,-1]
    T_int = griddata(xys, Ts, (grid_x, grid_y), method='linear')
    xyT_int = []
    for i in range(len(xys_grid)):
        for j in range(len(xys_grid)):
            xyT_int.append([xys_grid[i], xys_grid[j], T_int[i, j]])
    xyT_int = np.array(xyT_int)
    plot_xyT(xyT_int, "../digitize_eht/EHT_xyT_interpolated.pdf")

    # Save csv for later use:
    im_interpolated_path = '../digitize_eht/EHT_xyT_interpolated.csv'
    np.savetxt(im_interpolated_path, xyT_int, delimiter=',')

    return im_interpolated_path
