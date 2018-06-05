import os
import pandas as pd
import seaborn
import numpy as np

from matplotlib import pyplot as plt


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3


cfls = [1, 2, 4, 8, 10, 16, 24, 32]
gmres_lo = ["gmres_cg_gamg_lo/gmres_order1_profile_GW_ref6_cfl%s_NS1.csv"
            % cfl for cfl in cfls]
gmres_nlo = ["gmres_cg_gamg_nlo/gmres_order2_profile_GW_ref4_cfl%s_NS1.csv"
             % cfl for cfl in cfls]
gmres_lo_preonly = ["gmres_preonly_gamg_lo/gmres_order1_profile_GW_ref6_cfl%s_NS1.csv"
                    % cfl for cfl in cfls]
gmres_nlo_preonly = ["gmres_preonly_gamg_nlo/gmres_order2_profile_GW_ref4_cfl%s_NS1.csv"
                     % cfl for cfl in cfls]
hybrid_lo = ["hybrid_cg_gamg_lo/hybrid_order1_profile_GW_ref6_cfl%s_NS1.csv"
             % cfl for cfl in cfls]
hybrid_nlo = ["hybrid_cg_gamg_nlo/hybrid_order2_profile_GW_ref4_cfl%s_NS1.csv"
              % cfl for cfl in cfls]


for d in (gmres_lo + gmres_nlo + gmres_lo_preonly +
          gmres_nlo_preonly + hybrid_lo + hybrid_nlo):
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)


markers = ["o", "s", "^", "v", ">", "<", "D", "p", "h", "*"]
colors = list(seaborn.color_palette(n_colors=3))
linestyles = ["solid", "dashdot"]
seaborn.set(style="ticks")

fig, (axes,) = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
ax, = axes

ax.set_ylabel("Time [s]", fontsize=FONTSIZE+2)
ax.set_xticks(cfls)
yrange = np.arange(0, 75, 5)
ax.set_ylim([0, 75])
ax.set_yticks(yrange)

gmres_lo_dfs = [pd.read_csv(d) for d in gmres_lo]
gmres_nlo_dfs = [pd.read_csv(d) for d in gmres_nlo]
gmres_pre_lo_dfs = [pd.read_csv(d) for d in gmres_lo_preonly]
gmres_pre_nlo_dfs = [pd.read_csv(d) for d in gmres_nlo_preonly]
hybrid_lo_dfs = [pd.read_csv(d) for d in hybrid_lo]
hybrid_nlo_dfs = [pd.read_csv(d) for d in hybrid_nlo]

gmres_lo_times = []
gmres_lo_cfls = []
for df in gmres_lo_dfs:
    gmres_lo_times.append(df.SNESSolve.values[0])
    gmres_lo_cfls.append(df.NuCFL.values[0])

ax.plot(gmres_lo_cfls, gmres_lo_times,
        label="gmres (CG + AMG) LO",
        color=colors[0],
        marker=markers[0],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="solid",
        clip_on=True)

gmres_nlo_times = []
gmres_nlo_cfls = []
for df in gmres_nlo_dfs:
    gmres_nlo_times.append(df.SNESSolve.values[0])
    gmres_nlo_cfls.append(df.NuCFL.values[0])

ax.plot(gmres_nlo_cfls, gmres_nlo_times,
        label="gmres (CG + AMG) NLO",
        color=colors[0],
        marker=markers[0],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="dotted",
        clip_on=True)

gmres_pre_lo_times = []
gmres_pre_lo_cfls = []
for df in gmres_pre_lo_dfs:
    gmres_pre_lo_times.append(df.SNESSolve.values[0])
    gmres_pre_lo_cfls.append(df.NuCFL.values[0])

ax.plot(gmres_pre_lo_cfls, gmres_pre_lo_times,
        label="gmres (preonly AMG) LO",
        color=colors[1],
        marker=markers[1],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="solid",
        clip_on=True)

gmres_pre_nlo_times = []
gmres_pre_nlo_cfls = []
for df in gmres_pre_nlo_dfs:
    gmres_pre_nlo_times.append(df.SNESSolve.values[0])
    gmres_pre_nlo_cfls.append(df.NuCFL.values[0])

ax.plot(gmres_pre_nlo_cfls, gmres_pre_nlo_times,
        label="gmres (preonly AMG) NLO",
        color=colors[1],
        marker=markers[1],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="dotted",
        clip_on=True)

hybrid_lo_times = []
hybrid_lo_cfls = []
for df in hybrid_lo_dfs:
    hybrid_lo_times.append(df.SNESSolve.values[0])
    hybrid_lo_cfls.append(df.NuCFL.values[0])

ax.plot(hybrid_lo_cfls, hybrid_lo_times,
        label="hybrid. (CG + AMG) LO",
        color=colors[2],
        marker=markers[2],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="solid",
        clip_on=True)

hybrid_nlo_times = []
hybrid_nlo_cfls = []
for df in hybrid_nlo_dfs:
    hybrid_nlo_times.append(df.SNESSolve.values[0])
    hybrid_nlo_cfls.append(df.NuCFL.values[0])

ax.plot(hybrid_nlo_cfls, hybrid_nlo_times,
        label="hybrid. (CG + AMG) NLO",
        color=colors[2],
        marker=markers[2],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="dotted",
        clip_on=True)

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE-2)

for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE-2)

xlabel = fig.text(0.5, -0.05,
                  "$\\nu_{CFL} = c \\Delta t /\\Delta x$",
                  ha='center',
                  fontsize=FONTSIZE+2)

handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.1),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    handlelength=1.5,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

fig.savefig("times.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
