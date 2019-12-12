import matplotlib.pyplot as plt
import numpy as np

# Temperature map plot:
def plot_xyT(xyT, save_path, r_circ=0., std_circ=0., x_c=0., y_c=0.):
    xyT = np.array(xyT)
    plt.figure(figsize=(18, 15))
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tick_params(axis='both', which='minor', labelsize=21)
    plt.scatter(xyT[:,0], xyT[:,1], c=xyT[:,2], s=80, cmap="hot")
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=24)
    clb.ax.set_title(r'$T \,\,\, [10^9\,{\rm K}]$', fontsize=30)
    plt.xlabel(r'$x \,\,\, [\mu{\rm as}]$', fontsize=30)
    plt.ylabel(r'$y \,\,\, [\mu{\rm as}]$', fontsize=30)
    if r_circ > 0. and std_circ > 0.:
        xs_circ, ys_circ = circle_plot_offset(r_circ, x_c, y_c)
        xs_circ_p, ys_circ_p = circle_plot_offset(r_circ + std_circ, x_c, y_c)
        xs_circ_m, ys_circ_m = circle_plot_offset(r_circ - std_circ, x_c, y_c)
        plt.plot(xs_circ, ys_circ, color="black")
        plt.plot(xs_circ_p, ys_circ_p, '--', color="black")
        plt.plot(xs_circ_m, ys_circ_m, '--', color="black")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Circle plot for horizon:
def circle_plot_offset(r, x0, y0):
    n_points = 200
    angles = [2*np.pi*float(i)/(n_points-1) for i in range(n_points)]
    x = []
    y = []
    for a in angles:
        x.append(x0 + r*np.cos(a))
        y.append(y0 + r*np.sin(a))
    return x, y

