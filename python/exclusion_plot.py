import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Plot the exclusions for all multipoles:
def exclusion_plots(circ_max, fc_max, read_path, save_root, with_scatter=False):
    for l in [0, 1, 2]:
        save_path = save_root+"_l"+str(l)+".pdf"
        exclusion_plot_multipole(l, circ_max, fc_max, read_path, save_path, \
                with_scatter)

# Plot the exclusions for one multipole:
def exclusion_plot_multipole(l, circ_max, fc_max, read_path, save_path, \
        with_scatter):

    # Read results from scan:
    with open(read_path) as f:
         exclusions = f.readlines()

    # Build list of excluded/not-excluded parameter points:
    avals, aR0_excluded_circ, aR0_not_excluded_circ, aR0_bound_circ = \
            get_excluded_not_excluded_list(l, exclusions, "circ_val_mod", \
            circ_max)
    _, aR0_excluded_fc, aR0_not_excluded_fc, aR0_bound_fc = \
            get_excluded_not_excluded_list(l, exclusions, "fc_val_mod", fc_max)

    # Build plot:
    make_plot(l, circ_max, fc_max, avals, aR0_excluded_circ, \
            aR0_not_excluded_circ, aR0_bound_circ, aR0_excluded_fc, \
            aR0_not_excluded_fc, aR0_bound_fc, save_path, with_scatter)

# Build plot:
def make_plot(l, circ_max, fc_max, avals, aR0_excluded_circ, \
        aR0_not_excluded_circ, aR0_bound_circ, aR0_excluded_fc, \
        aR0_not_excluded_fc, aR0_bound_fc, save_path, with_scatter):
    if l == 0:
        title = "Exclusion for constant reflection coefficient " + \
                r'$R=R_0\, (l=0)$'
    elif l == 1:
        title = "Exclusion for reflection coefficient " + \
                r'$R=R_0 \|\cos (\Theta) \|\, (l=1)$'
    elif l == 2:
        title = "Exclusion for reflection coefficient " + \
                r'$R=R_0\|\sin (\Theta) \cos (\Theta) \|\, (l=2)$'
    plt.figure(figsize=(15, 10))
    plt.title(title, fontsize=25)
    ax = plt.subplot()
    ax.set_yscale('log')
    y_min = 0.008
    y_max = 1.0
    ax.set_xlim([-0.94, 0.94])
    ax.set_ylim([y_min, y_max])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xticks([-0.94, -0.5, 0., 0.5, 0.94])
    ax.set_yticks([0.01, 0.05, 0.1, 0.5, 1.0])
    # Circularity bounds:
    ax.fill_between(aR0_bound_circ[:,0], aR0_bound_circ[:,1], 10, \
            facecolor='orange', alpha=0.3, \
            label=r'$\Delta_c >\,$'+str(circ_max))
    if with_scatter:
        ax.scatter(aR0_excluded_circ[:,0], aR0_excluded_circ[:,1])
    # fc bounds:
    ax.fill_between(aR0_bound_fc[:,0], aR0_bound_fc[:,1], 10, \
            facecolor='green', alpha=0.3, label=r'$f_c >\,$'+str(fc_max))
    if with_scatter:
        ax.scatter(aR0_excluded_fc[:,0], aR0_excluded_fc[:,1], color="black")
    ax.legend(loc="upper right", fontsize=20)
    ax.set_xlabel("Black hole spin "+r'$a$', fontsize=20)
    ax.set_ylabel("Reflection coefficient "+r'$R_0$', fontsize=20)
    plt.savefig(save_path, bbox_inches='tight')

# Build list of excluded/not-excluded parameter points:
def get_excluded_not_excluded_list(l, exclusions, criterium, \
        criterium_max_value):
    R_max = 1.
    avals = []
    aR0_excluded = []
    aR0_not_excluded = []
    for e in exclusions:
        edict = eval(e)
        if edict["l"] == l:
            avals.append(edict["a"])
            aR0 = [edict["a"], edict["R0"]]
            excluded = False

            # Exclusion based on value of circ:
            if edict[criterium] > criterium_max_value:
                excluded = True

            """
            # Exclusion based on Fail/Ok in criteria:
            for c in criteria:
                if edict[c] == "Fail":
                    excluded = True
            """

            # Append to exclusion or no-excluded lists:
            if excluded:
                aR0_excluded.append(aR0)
            else:
                aR0_not_excluded.append(aR0)

    # Numpy manipulations to get bounds:
    avals = np.unique(np.array(avals))
    aR0_excluded = np.array(aR0_excluded)
    aR0_not_excluded = np.array(aR0_not_excluded)
    aR0_bound = []
    for a in avals:
        aR0_excluded_at_a = aR0_excluded[aR0_excluded[:,0] == a]
        if aR0_excluded_at_a.shape[0] != 0:
            R0min = np.amin(aR0_excluded_at_a[:,1])
        else:
            R0min = R_max
        aR0_bound.append([a, R0min])
    aR0_bound = np.array(aR0_bound)
    
    return avals, aR0_excluded, aR0_not_excluded, aR0_bound


