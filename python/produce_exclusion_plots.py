from exclusion_plot import exclusion_plots

read_path = "exclusion_results/exclusions_l[0, 1, 2]_R0[0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.2, 0.4, 1.0]_a[-0.94, -0.5, 0.01, 0.5, 0.94].txt"
save_root = "exclusion_results/exclusions"
circ_max = 0.2
fc_max = 0.7

exclusion_plots(circ_max, fc_max, read_path, save_root, with_scatter=False)
