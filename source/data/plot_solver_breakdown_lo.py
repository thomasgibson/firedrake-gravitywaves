import os
import pandas as pd
import seaborn
import numpy as np

from matplotlib import pyplot as plt


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3
WIDTH = 0.35


cfls = [8]
gmres_lo = ["gmres_cg_gamg_lo/gmres_order1_profile_GW_ref6_cfl%s_NS1.csv"
            % cfl for cfl in cfls]
gmres_lo_preonly = ["gmres_preonly_gamg_lo/gmres_order1_profile_GW_ref6_cfl%s_NS1.csv"
                    % cfl for cfl in cfls]
hybrid_lo = ["hybrid_cg_gamg_lo/hybrid_order1_profile_GW_ref6_cfl%s_NS1.csv"
             % cfl for cfl in cfls]


for d in (gmres_lo + gmres_lo_preonly + hybrid_lo):
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)


colors = list(seaborn.color_palette(n_colors=3))
fig, axes = plt.subplots(1, 2, figsize=(15, 7), squeeze=False)
axes = axes.flatten()
ax0, sax = axes

ind = np.arange(3)
ax0.set_xticks(ind)
sax.set_xticks([0, 1])

for ax in axes:
    ax.set_ylabel("Time [s]/ Total hybrid time [s]", fontsize=FONTSIZE)

gmres_lo_df, = [pd.read_csv(d) for d in gmres_lo]
gmres_pre_lo_df, = [pd.read_csv(d) for d in gmres_lo_preonly]
hybrid_lo_df, = [pd.read_csv(d) for d in hybrid_lo]


gmres_lo_schur_solve = gmres_lo_df.KSPSchur.values[0]
gmres_lo_low = gmres_lo_df.KSPFSLow.values[0]
gmres_lo_mass_invert = gmres_lo_df.KSPF0.values[0]
gmres_lo_total = gmres_lo_df.SNESSolve.values[0]
gmres_lo_other = gmres_lo_total - (gmres_lo_schur_solve +
                                   gmres_lo_low +
                                   gmres_lo_mass_invert)
gmres_pre_lo_schur_solve = gmres_pre_lo_df.KSPSchur.values[0]
gmres_pre_lo_low = gmres_pre_lo_df.KSPFSLow.values[0]
gmres_pre_lo_mass_invert = gmres_pre_lo_df.KSPF0.values[0]
gmres_pre_lo_total = gmres_pre_lo_df.SNESSolve.values[0]
gmres_pre_lo_other = gmres_pre_lo_total - (gmres_pre_lo_schur_solve +
                                           gmres_pre_lo_low +
                                           gmres_pre_lo_mass_invert)
hybrid_lo_break_time = hybrid_lo_df.HybridBreak.values[0]
hybrid_lo_rhs_time = hybrid_lo_df.HybridRHS.values[0]
hybrid_lo_trace_solve = hybrid_lo_df.HybridTraceSolve.values[0]
hybrid_lo_recon_time = hybrid_lo_df.HybridReconstruction.values[0]
hybrid_lo_proj_time = hybrid_lo_df.HybridProjection.values[0]
hybrid_lo_total = (hybrid_lo_break_time + hybrid_lo_rhs_time +
                   hybrid_lo_proj_time + hybrid_lo_recon_time +
                   hybrid_lo_trace_solve)

a1 = ax0.bar(ind[0], gmres_lo_schur_solve/hybrid_lo_total, WIDTH,
             color=colors[0])
a2 = ax0.bar(ind[0], (gmres_lo_low + gmres_lo_mass_invert)/hybrid_lo_total,
             WIDTH,
             bottom=gmres_lo_schur_solve/hybrid_lo_total,
             color=colors[1])
a4 = ax0.bar(ind[0], gmres_lo_other/hybrid_lo_total, WIDTH,
             bottom=(gmres_lo_schur_solve + gmres_lo_low +
                     gmres_lo_mass_invert)/hybrid_lo_total,
             color="k")

ax0.bar(ind[1], gmres_pre_lo_schur_solve/hybrid_lo_total, WIDTH,
        color=colors[0])
ax0.bar(ind[1], (gmres_pre_lo_low + gmres_pre_lo_mass_invert)/hybrid_lo_total,
        WIDTH,
        bottom=gmres_pre_lo_schur_solve/hybrid_lo_total,
        color=colors[1])
ax0.bar(ind[1], gmres_pre_lo_other/hybrid_lo_total, WIDTH,
        bottom=(gmres_pre_lo_schur_solve + gmres_pre_lo_low +
                gmres_pre_lo_mass_invert)/hybrid_lo_total,
        color="k")

sax.bar(0, gmres_pre_lo_schur_solve/hybrid_lo_total, WIDTH,
        color=colors[0])
sax.bar(0, (gmres_pre_lo_low + gmres_pre_lo_mass_invert)/hybrid_lo_total,
        WIDTH,
        bottom=gmres_pre_lo_schur_solve/hybrid_lo_total,
        color=colors[1])
sax.bar(0, gmres_pre_lo_other/hybrid_lo_total, WIDTH,
        bottom=(gmres_pre_lo_schur_solve + gmres_pre_lo_low +
                gmres_pre_lo_mass_invert)/hybrid_lo_total,
        color="k")

ax5 = ax0.bar(ind[2], hybrid_lo_trace_solve/hybrid_lo_total, WIDTH,
              color="#96595A")
ax6 = ax0.bar(ind[2], hybrid_lo_rhs_time/hybrid_lo_total, WIDTH,
              bottom=hybrid_lo_trace_solve/hybrid_lo_total,
              color="#B2E4CF")
ax7 = ax0.bar(ind[2], hybrid_lo_recon_time/hybrid_lo_total, WIDTH,
              bottom=(hybrid_lo_trace_solve +
                      hybrid_lo_rhs_time)/hybrid_lo_total,
              color="#E4E4B2")
ax8 = ax0.bar(ind[2], (hybrid_lo_proj_time +
                       hybrid_lo_break_time)/hybrid_lo_total,
              WIDTH,
              bottom=(hybrid_lo_trace_solve + hybrid_lo_rhs_time +
                      hybrid_lo_recon_time)/hybrid_lo_total,
              color="#DA897C")

sax.bar(1, hybrid_lo_trace_solve/hybrid_lo_total, WIDTH,
        color="#96595A")
sax.bar(1, hybrid_lo_rhs_time/hybrid_lo_total, WIDTH,
        bottom=hybrid_lo_trace_solve/hybrid_lo_total,
        color="#B2E4CF")
sax.bar(1, hybrid_lo_recon_time/hybrid_lo_total, WIDTH,
        bottom=(hybrid_lo_trace_solve +
                hybrid_lo_rhs_time)/hybrid_lo_total,
        color="#E4E4B2")
sax.bar(1, (hybrid_lo_proj_time +
            hybrid_lo_break_time)/hybrid_lo_total,
        WIDTH,
        bottom=(hybrid_lo_trace_solve + hybrid_lo_rhs_time +
                hybrid_lo_recon_time)/hybrid_lo_total,
        color="#DA897C")


for ax in axes:
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FONTSIZE-2)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(FONTSIZE-2)

labels = ["gmres\n cg+gamg\n LO",
          "gmres\n preonly gamg\n LO",
          "hybrid\n cg+gamg\n LO"]
ax0.set_xticklabels(labels, fontsize=FONTSIZE)
slabels = ["gmres\n preonly gamg\n LO",
           "hybrid\n cg+gamg\n LO"]
sax.set_xticklabels(slabels, fontsize=FONTSIZE)

legend1 = plt.legend((a4[0], a2[0], a1[0]),
                     ("other (gmres)", "inv mass and apply (gmres)",
                      "Schur solve (gmres)"),
                     fontsize=FONTSIZE,
                     ncol=1,
                     bbox_to_anchor=(-0.25, 0.9))
legend2 = plt.legend((ax8[0], ax7[0], ax6[0], ax5[0]),
                     ("Break and proj. (hybrid.)", "Back sub. (hybrid.)",
                      "Forward elim. (hybrid.)", "Trace solve (hybrid.)"),
                     fontsize=FONTSIZE,
                     ncol=1,
                     bbox_to_anchor=(-0.25, 0.6))

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
fig.subplots_adjust(wspace=0.25)
fig.savefig("solver_compare_lo.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight")
