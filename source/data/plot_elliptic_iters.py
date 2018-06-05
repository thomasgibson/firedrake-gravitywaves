import os
import pandas as pd
import seaborn
import numpy as np

from matplotlib import pyplot as plt


FONTSIZE = 16
MARKERSIZE = 10
LINEWIDTH = 3


cfls = [1, 2, 4, 8, 10, 16, 24, 32]
gmres_lo = ["gmres_cg_gamg_lo/gmres_order1_data_GW_ref6_cfl%s_NS1.csv"
            % cfl for cfl in cfls]
gmres_nlo = ["gmres_cg_gamg_nlo/gmres_order2_data_GW_ref4_cfl%s_NS1.csv"
             % cfl for cfl in cfls]
hybrid_lo = ["hybrid_cg_gamg_lo/hybrid_order1_data_GW_ref6_cfl%s_NS1.csv"
             % cfl for cfl in cfls]
hybrid_nlo = ["hybrid_cg_gamg_nlo/hybrid_order2_data_GW_ref4_cfl%s_NS1.csv"
              % cfl for cfl in cfls]


for d in gmres_lo + gmres_nlo + hybrid_lo + hybrid_nlo:
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

ax.set_ylabel("CG iterations", fontsize=FONTSIZE+2)
ax.set_xticks(cfls)
yrange = np.arange(0, 15, 1)
ax.set_ylim([0, 15])
ax.set_yticks(yrange)

gmres_lo_dfs = [pd.read_csv(d) for d in gmres_lo]
gmres_nlo_dfs = [pd.read_csv(d) for d in gmres_nlo]
hybrid_lo_dfs = [pd.read_csv(d) for d in hybrid_lo]
hybrid_nlo_dfs = [pd.read_csv(d) for d in hybrid_nlo]

elliptic_lo_iters = []
for df in gmres_lo_dfs:
    elliptic_lo_iters.append(df.InnerIters.values[0])

ax.plot(cfls, elliptic_lo_iters,
        label="KSP($\\widehat{S}$, AMG) LO",
        color=colors[0],
        marker=markers[0],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="solid",
        clip_on=True)

elliptic_nlo_iters = []
for df in gmres_nlo_dfs:
    elliptic_nlo_iters.append(df.InnerIters.values[0])

ax.plot(cfls, elliptic_nlo_iters,
        label="KSP($\\widehat{S}$, AMG) NLO",
        color=colors[0],
        marker=markers[0],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="dotted",
        clip_on=True)

hybrid_lo_iters = []
for df in hybrid_lo_dfs:
    hybrid_lo_iters.append(df.InnerIters.values[0])

ax.plot(cfls, hybrid_lo_iters,
        label="KSP($S_{\\lambda}$, AMG) LO",
        color=colors[2],
        marker=markers[2],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        linestyle="solid",
        clip_on=True)

hybrid_nlo_iters = []
for df in hybrid_nlo_dfs:
    hybrid_nlo_iters.append(df.InnerIters.values[0])

ax.plot(cfls, hybrid_nlo_iters,
        label="KSP($S_{\\lambda}$, AMG) NLO",
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
                    ncol=2,
                    handlelength=1.5,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

fig.savefig("elliptic_iters.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[xlabel, legend])
