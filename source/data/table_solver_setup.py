import os
import pandas as pd


cfls = [8]

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

gmres_lo_data = ["gmres_cg_gamg_lo/gmres_order1_data_GW_ref6_cfl%s_NS1.csv"
                 % cfl for cfl in cfls]
gmres_nlo_data = ["gmres_cg_gamg_nlo/gmres_order2_data_GW_ref4_cfl%s_NS1.csv"
                  % cfl for cfl in cfls]
gmres_lo_preonly_data = ["gmres_preonly_gamg_lo/gmres_order1_data_GW_ref6_cfl%s_NS1.csv"
                         % cfl for cfl in cfls]
gmres_nlo_preonly_data = ["gmres_preonly_gamg_nlo/gmres_order2_data_GW_ref4_cfl%s_NS1.csv"
                          % cfl for cfl in cfls]
hybrid_lo_data = ["hybrid_cg_gamg_lo/hybrid_order1_data_GW_ref6_cfl%s_NS1.csv"
                  % cfl for cfl in cfls]
hybrid_nlo_data = ["hybrid_cg_gamg_nlo/hybrid_order2_data_GW_ref4_cfl%s_NS1.csv"
                   % cfl for cfl in cfls]


for d in (gmres_lo + gmres_nlo + gmres_lo_preonly +
          gmres_nlo_preonly + hybrid_lo + hybrid_nlo +
          gmres_lo_data + gmres_nlo_data + gmres_lo_preonly_data +
          gmres_nlo_preonly_data + hybrid_lo_data + hybrid_nlo_data):
    if not os.path.exists(d):
        import sys
        print("Cannot find data file '%s'" % d)
        sys.exit(1)


gmres_lo_df, = [pd.read_csv(d) for d in gmres_lo]
gmres_pre_lo_df, = [pd.read_csv(d) for d in gmres_lo_preonly]
hybrid_lo_df, = [pd.read_csv(d) for d in hybrid_lo]
gmres_nlo_df, = [pd.read_csv(d) for d in gmres_nlo]
gmres_pre_nlo_df, = [pd.read_csv(d) for d in gmres_nlo_preonly]
hybrid_nlo_df, = [pd.read_csv(d) for d in hybrid_nlo]

gmres_lo_data_df, = [pd.read_csv(d) for d in gmres_lo_data]
gmres_pre_lo_data_df, = [pd.read_csv(d) for d in gmres_lo_preonly_data]
hybrid_lo_data_df, = [pd.read_csv(d) for d in hybrid_lo_data]
gmres_nlo_data_df, = [pd.read_csv(d) for d in gmres_nlo_data]
gmres_pre_nlo_data_df, = [pd.read_csv(d) for d in gmres_nlo_preonly_data]
hybrid_nlo_data_df, = [pd.read_csv(d) for d in hybrid_nlo_data]


table = r"""\begin{tabular}{ccccc}\hline
\multicolumn{5}{c}{\textbf{Preconditioner and solver setup}} \\
Order ($\Delta x$)&
PC &
inner config. &
$t_{\text{setup}}$ [s] &
(trace assembly [s])\\
\hline
"""

lformat = r"""{setup: .3f} & \\
"""

hformat = r"""{setup: .3f} & {trace: .3f} \\ \hline
"""

table += r"""\multirow{3}{*}{LO ($\Delta x \approx 87$km)} &
FS Schur (full) &
CG + AMG &
"""
table += lformat.format(setup=gmres_lo_df.PETSCLogPreSetup.values[0])
table += r"""&
FS Schur (full) &
AMG only &
"""

table += lformat.format(setup=gmres_pre_lo_df.PETSCLogPreSetup.values[0])
table += r"""&
hybridization &
CG + AMG &
"""

table += hformat.format(setup=hybrid_lo_df.PETSCLogPreSetup.values[0],
                        trace=hybrid_lo_df.PreHybridInit.values[0])

table += r"""\multirow{3}{*}{NLO ($\Delta x \approx 350$km)} &
FS Schur (full) &
CG + AMG &
"""
table += lformat.format(setup=gmres_nlo_df.PETSCLogPreSetup.values[0])
table += r"""&
FS Schur (full) &
AMG only &
"""

table += lformat.format(setup=gmres_pre_nlo_df.PETSCLogPreSetup.values[0])
table += r"""&
hybridization &
CG + AMG &
"""

table += hformat.format(setup=hybrid_nlo_df.PETSCLogPreSetup.values[0],
                        trace=hybrid_nlo_df.PreHybridInit.values[0])

table += """
\end{tabular}"""

print(table)
