from firedrake import COMM_WORLD, parameters
from firedrake.petsc import PETSc
from pyop2.profiling import timed_stage
from argparse import ArgumentParser
from mpi4py import MPI
import pandas as pd

import solver as module


parameters["pyop2_options"]["lazy_evaluation"] = False


parser = ArgumentParser(description="""Linear gravity wave system.""",
                        add_help=False)

parser.add_argument("--refinements",
                    default=4,
                    type=int,
                    help=("Number of refinements when generating the "
                          "spherical base mesh."))

parser.add_argument("--num_layers",
                    default=20,
                    type=int,
                    help="Number of vertical levels in the extruded mesh.")

parser.add_argument("--hexes",
                    action="store_true",
                    help="Use hexahedral elements.")

parser.add_argument("--hybridization",
                    action="store_true",
                    help=("Use a hybridized mixed method to solve the "
                          "gravity wave equations."))

parser.add_argument("--inner_pc_type",
                    default="gamg",
                    choices=["hypre", "gamg", "direct"],
                    help="Solver type for the linear solver.")

parser.add_argument("--nu_cfl",
                    default=1,
                    type=int,
                    help="Value for the horizontal courant number.")

parser.add_argument("--nsteps",
                    default=1,
                    type=int,
                    help="Number of time steps to take.")

parser.add_argument("--order",
                    default=1,
                    type=int,
                    help="Order of the compatible mixed method.")

parser.add_argument("--rtol",
                    default=1.0e-6,
                    type=float,
                    help="Rtolerance for the linear solve.")

parser.add_argument("--test",
                    action="store_true",
                    help="Enable a quick test run with ksp monitors.")

parser.add_argument("--profile",
                    action="store_true",
                    help="Turn on profiler for simulation timings.")

parser.add_argument("--write_output",
                    action="store_true",
                    help="Write output.")

parser.add_argument("--dumpfreq",
                    default=10,
                    type=int,
                    action="store",
                    help="Dump frequency of output.")

parser.add_argument("--monitor",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help.")

args, _ = parser.parse_known_args()

if args.help:
    import sys
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)

PETSc.Log.begin()


def run_gravity_waves(problem_cls, nu_cfl, refinements, num_layers, hexes,
                      order, nsteps, hybridization, inner_pc_type, rtol,
                      monitor=False, write=False, cold=False):

    # Radius of earth (scaled, m)
    r_earth = 6.371e6/125.0

    # Speed of sound (m/s)
    c = 300.0

    # Buoyancy frequency
    N = 0.01

    # Angular rotation rate
    Omega = 7.292e-5

    if cold:
        PETSc.Sys.Print("""
        Running cold initialization with parameters:\n
        Horizontal Courant number: %s,\n
        Horizontal refinements: %s,\n
        Vertical layers: %s,\n
        Hexes: %s,\n
        Discretization order: %s,\n
        Hybridization: %s,\n
        Inner PC type: %s,\n
        rtol: %s.
        """ % (nu_cfl, refinements, num_layers, hexes,
               order, hybridization, inner_pc_type, rtol))
        problem = problem_cls(order=order,
                              refinements=refinements,
                              num_layers=num_layers,
                              nu_cfl=nu_cfl,
                              c=c,
                              N=N,
                              Omega=Omega,
                              R=r_earth,
                              rtol=rtol,
                              hexes=hexes,
                              inner_pc_type=inner_pc_type,
                              hybridization=hybridization,
                              monitor=monitor)
        problem.warmup()
        return

    problem = problem_cls(order=order,
                          refinements=refinements,
                          num_layers=num_layers,
                          nu_cfl=nu_cfl,
                          c=c,
                          N=N,
                          Omega=Omega,
                          R=r_earth,
                          rtol=rtol,
                          hexes=hexes,
                          inner_pc_type=inner_pc_type,
                          hybridization=hybridization,
                          monitor=monitor)
    comm = problem.comm

    PETSc.Sys.Print("Warmup with one step.\n")
    with timed_stage("Warmup %s" % problem.name):
        problem.warmup()
        PETSc.Log.Stage("Warmup: Velocity-Pressure-Solve").push()
        prepcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pre_res_eval = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        pre_jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()

        pre_res_eval_time = comm.allreduce(pre_res_eval["time"],
                                           op=MPI.SUM) / comm.size
        pre_jac_eval_time = comm.allreduce(pre_jac_eval["time"],
                                           op=MPI.SUM) / comm.size
        pre_setup_time = comm.allreduce(prepcsetup["time"],
                                        op=MPI.SUM) / comm.size

        if problem._hybridization:
            prehybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()
            prehybridinit_time = comm.allreduce(prehybridinit["time"],
                                                op=MPI.SUM) / comm.size

        PETSc.Log.Stage("Warmup: Velocity-Pressure-Solve").pop()

    PETSc.Sys.Print("Running simulation for %s time-steps\n" % nsteps)
    tmax = nsteps * problem._dt
    problem.run_simulation(tmax, write=write, dumpfreq=args.dumpfreq)
    PETSc.Sys.Print("Simulation complete.\n")

    PETSc.Log.Stage("Velocity-Pressure-Solve").push()

    snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
    ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
    pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
    pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
    jac_eval = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
    residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

    snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
    ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
    pc_setup_time = comm.allreduce(pcsetup["time"], op=MPI.SUM) / comm.size
    pc_apply_time = comm.allreduce(pcapply["time"], op=MPI.SUM) / comm.size
    jac_eval_time = comm.allreduce(jac_eval["time"], op=MPI.SUM) / comm.size
    res_eval_time = comm.allreduce(residual["time"], op=MPI.SUM) / comm.size

    ref = problem._refinements
    num_cells = comm.allreduce(problem.num_cells, op=MPI.SUM)

    if problem._hybridization:
        results_data = "hybrid_order%s_data_GW_ref%d_cfl%s_NS%d" % (
            problem._order,
            ref,
            nu_cfl,
            nsteps
        )
        results_timings = "hybrid_order%s_profile_GW_ref%d_cfl%s_NS%d" % (
            problem._order,
            ref,
            nu_cfl,
            nsteps
        )

        RHS = PETSc.Log.Event("HybridRHS").getPerfInfo()
        trace = PETSc.Log.Event("HybridSolve").getPerfInfo()
        recover = PETSc.Log.Event("HybridRecover").getPerfInfo()
        recon = PETSc.Log.Event("HybridRecon").getPerfInfo()
        hybridbreak = PETSc.Log.Event("HybridBreak").getPerfInfo()
        hybridupdate = PETSc.Log.Event("HybridUpdate").getPerfInfo()
        hybridinit = PETSc.Log.Event("HybridInit").getPerfInfo()

        recon_time = comm.allreduce(recon["time"], op=MPI.SUM) / comm.size
        projection = comm.allreduce(recover["time"], op=MPI.SUM) / comm.size
        transfer = comm.allreduce(hybridbreak["time"], op=MPI.SUM) / comm.size
        full_recon = projection + recon_time
        update_time = comm.allreduce(hybridupdate["time"],
                                     op=MPI.SUM) / comm.size
        trace_solve = comm.allreduce(trace["time"], op=MPI.SUM) / comm.size
        rhstime = comm.allreduce(RHS["time"], op=MPI.SUM) / comm.size
        inittime = comm.allreduce(hybridinit["time"],
                                  op=MPI.SUM) / comm.size
        other = ksp_time - (trace_solve + transfer
                            + projection + recon_time + rhstime)
        full_solve = (transfer + trace_solve + rhstime
                      + recon_time + projection + update_time)
    else:
        results_data = "gmres_order%s_data_GW_ref%d_cfl%s_NS%d" % (
            problem._order,
            ref,
            nu_cfl,
            nsteps
        )
        results_timings = "gmres_order%s_profile_GW_ref%d_cfl%s_NS%d" % (
            problem._order,
            ref,
            nu_cfl,
            nsteps
        )

        KSPSchur = PETSc.Log.Event("KSPSolve_FS_Schu").getPerfInfo()
        KSPF0 = PETSc.Log.Event("KSPSolve_FS_0").getPerfInfo()
        KSPLow = PETSc.Log.Event("KSPSolve_FS_Low").getPerfInfo()

        schur_time = comm.allreduce(KSPSchur["time"], op=MPI.SUM) / comm.size
        f0_time = comm.allreduce(KSPF0["time"], op=MPI.SUM) / comm.size
        ksplow_time = comm.allreduce(KSPLow["time"], op=MPI.SUM) / comm.size
        other = ksp_time - (schur_time + f0_time + ksplow_time)

    PETSc.Log.Stage("Velocity-Pressure-Solve").pop()

    if hexes:
        results_data += "_hexes.csv"
        results_timings += "_hexes.csv"
    else:
        results_data += ".csv"
        results_timings += ".csv"

    if COMM_WORLD.rank == 0:
        data = {"OuterIters": problem._outer_ksp_iterations,
                "InnerIters": problem._inner_ksp_iterations,
                "SimTime": problem._sim_time,
                "ResidualReductions": problem._up_residual_reductions}

        up_dofs = problem._up.dof_dset.layout_vec.getSize()
        b_dofs = problem._btmp.dof_dset.layout_vec.getSize()
        total_dofs = up_dofs + b_dofs

        time_data = {"PETSCLogKSPSolve": ksp_time,
                     "PETSCLogPCApply": pc_apply_time,
                     "PETSCLogPCSetup": pc_setup_time,
                     "PETSCLogPreSetup": pre_setup_time,
                     "PETSCLogPreSNESJacobianEval": pre_jac_eval_time,
                     "PETSCLogPreSNESFunctionEval": pre_res_eval_time,
                     "SNESSolve": snes_time,
                     "SNESFunctionEval": res_eval_time,
                     "SNESJacobianEval": jac_eval_time,
                     "num_processes": problem.comm.size,
                     "order": problem._order,
                     "refinement_level": problem._refinements,
                     "vertical_layers": problem._nlayers,
                     "total_dofs": total_dofs,
                     "velocity_pressure_dofs": up_dofs,
                     "num_cells": num_cells,
                     "Dt": problem._dt,
                     "NuCFL": nu_cfl,
                     "DxMin": problem._dx_min,
                     "DxMax": problem._dx_max,
                     "DxAvg": problem._dx_avg}

        if problem._hybridization:
            updates = {"HybridTraceSolve": trace_solve,
                       "HybridRHS": rhstime,
                       "HybridBreak": transfer,
                       "HybridReconstruction": recon_time,
                       "HybridProjection": projection,
                       "HybridFullRecovery": full_recon,
                       "HybridUpdate": update_time,
                       "HybridInit": inittime,
                       "PreHybridInit": prehybridinit_time,
                       "HybridFullSolveTime": full_solve,
                       "HybridKSPOther": other}

        else:
            updates = {"KSPSchur": schur_time,
                       "KSPF0": f0_time,
                       "KSPFSLow": ksplow_time,
                       "KSPother": other}

        time_data.update(updates)

        df_data = pd.DataFrame(data)
        df_data.to_csv(results_data, index=False,
                       mode="w", header=True)

        df_time = pd.DataFrame(time_data, index=[0])
        df_time.to_csv(results_timings, index=False,
                       mode="w", header=True)


GravityWaveProblem = module.GravityWaveProblem
if args.profile:

    # Cold run
    run_gravity_waves(problem_cls=GravityWaveProblem,
                      nu_cfl=args.nu_cfl,
                      refinements=args.refinements,
                      num_layers=args.num_layers,
                      hexes=args.hexes,
                      order=args.order,
                      nsteps=args.nsteps,
                      hybridization=args.hybridization,
                      inner_pc_type=args.inner_pc_type,
                      rtol=args.rtol,
                      monitor=False,
                      write=False,
                      cold=True)

    # Now start profiler
    run_gravity_waves(problem_cls=GravityWaveProblem,
                      nu_cfl=args.nu_cfl,
                      refinements=args.refinements,
                      num_layers=args.num_layers,
                      hexes=args.hexes,
                      order=args.order,
                      nsteps=args.nsteps,
                      hybridization=args.hybridization,
                      inner_pc_type=args.inner_pc_type,
                      rtol=args.rtol,
                      monitor=False,
                      write=False,
                      cold=False)
else:
    run_gravity_waves(problem_cls=GravityWaveProblem,
                      nu_cfl=args.nu_cfl,
                      refinements=args.refinements,
                      num_layers=args.num_layers,
                      hexes=args.hexes,
                      order=args.order,
                      nsteps=args.nsteps,
                      hybridization=args.hybridization,
                      inner_pc_type=args.inner_pc_type,
                      rtol=args.rtol,
                      monitor=args.monitor,
                      write=args.write_output,
                      cold=False)
