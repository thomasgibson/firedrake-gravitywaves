from firedrake import *
from function_spaces import construct_spaces
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage

import numpy as np


__all__ = ["GravityWaveProblem"]


class GravityWaveProblem(object):
    """Problem context for the linearized compressible Boussinesq equations
    (includes Coriolis term). The solver uses an explicit matrix solver
    and algebraic multigrid preconditioning. The equations are solved in
    three stages:

    (1) First analytically eliminate the buoyancy perturbation term from
        the discrete equations. This is possible since there is currently
        no orography. Note that it is indeed possible to eliminate buoyancy
        when orography is present, however this must be done at the continuous
        level first.

    (2) Eliminating buoyancy produces a saddle-point system for the velocity
        and pressure perturbations. The resulting system is solved using
        either an approximate full Schur-complement procedure or a
        hybridized mixed method.

    (3) Once the velocity and perturbation fields are computed from the
        previous step, the buoyancy term is reconstructed.
    """

    def __init__(self, order, refinements, num_layers,
                 nu_cfl, c, N, Omega, R, H, rtol=1.0E-8, mesh_degree=1,
                 hexes=False, inner_pc_type="gamg", inner_preonly=False,
                 hybridization=False,
                 monitor=False):
        """The constructor for the GravityWaveProblem.

        :arg order: A positive integer denoting the order of finite
            elements to use in the spatial discretization.
        :arg refinements: A positive integer describing the number of
            horizontal refinements to make.
        :arg num_layers: An integer describing the number of vertical
            levels to use.
        :arg nu_cfl: The acoustic horizontal Courant number. This determines
            the time-step used in the simulation.
        :arg c: A positive real number denoting the speed of sound waves
            in dry air.
        :arg N: A positive real number describing the Brunt–Väisälä frequency.
        :arg Omega: A positive real number; the angular rotation rate of the
            Earth.
        :arg R: A positive real number denoting the radius of the spherical
            mesh (Earth-size).
        :arg H: Lid height (m).
        :arg hexes: A boolean switch which determines if hexahedral elements
            are to be used. Default is `False` (triangular prism elements).
        :arg rtol: The relative tolerance for the solver.
        :arg mesh_degree: The degree of the coordinate field.
        :arg inner_pc_type: A string describing which inner-most preconditioner
            to use on the pressure space (approximate Schur-complement) or the
            trace space (hybridization).
        :arg inner_preonly: A boolean denoting whether to use a preonly
            configuration for the inner solve.
        :arg hybridization: A boolean switch between using a hybridized
            mixed method (True) on the velocity-pressure system, or GMRES
            with an approximate Schur-complement preconditioner (False).
        :arg monitor: A boolean switch with turns on/off KSP monitoring
            of the problem residuals (primarily for debugging and checking
            convergence of the solver). When profiling, keep this set
            to `False`.
        """

        self._R = R
        self._refinements = refinements
        self._nlayers = num_layers
        self._hexes = hexes

        # Create horizontal base mesh
        self._mesh_degree = mesh_degree
        if self._hexes:
            base = CubedSphereMesh(self._R,
                                   refinement_level=self._refinements,
                                   degree=self._mesh_degree)
        else:
            base = IcosahedralSphereMesh(self._R,
                                         refinement_level=self._refinements,
                                         degree=self._mesh_degree)

        global_normal = Expression(("x[0]", "x[1]", "x[2]"))
        base.init_cell_orientations(global_normal)
        self._base = base

        # Thickness of spherical shell (m)
        thickness = H
        self._H = thickness

        # Create extruded sphere mesh
        mesh = ExtrudedMesh(self._base,
                            extrusion_type="radial",
                            layers=self._nlayers,
                            layer_height=thickness/self._nlayers)
        self._mesh = mesh

        # Get horizontal Dx information (this is approximate).
        # We compute the area (m^2) of each cell in the mesh,
        # then take the square root to get the right units.
        cell_vs = interpolate(CellVolume(self._base),
                              FunctionSpace(self._base, "DG", 0))

        a_min = cell_vs.dat.data.min()
        a_max = cell_vs.dat.data.max()
        dx_min = sqrt(a_min)
        dx_max = sqrt(a_max)
        dx_avg = (dx_min + dx_max)/2.0
        self._dx_min = dx_min
        self._dx_max = dx_max
        self._dx_avg = dx_avg

        # Speed of sound (m/s)
        self._c = c

        # Horizontal acoustic Courant number
        self._nu_cfl = nu_cfl

        # Compute time-step size (s)
        dt = (self._nu_cfl / self._c) * self._dx_max
        self._dt = dt

        # Physical constants and timestepping parameters
        self._N = N            # Buoyancy frequency
        self._Omega = Omega    # Angular rotation rate
        self._dt_half = Constant(0.5*self._dt)
        self._dt_half_N2 = Constant(0.5*self._dt*self._N**2)
        self._dt_half_c2 = Constant(0.5*self._dt*self._c**2)
        self._omega_N2 = Constant((0.5*self._dt*self._N)**2)
        self._omega_c2 = Constant((0.5*self._dt*self._c)**2)

        # Build compatible finite element function spaces
        self._order = order
        W2, W3, Wb, W2v = construct_spaces(mesh=self._mesh,
                                           order=self._order,
                                           hexes=self._hexes)
        self._Wmixed = W2 * W3
        self._W2 = self._Wmixed.sub(0)
        self._W3 = self._Wmixed.sub(1)
        self._Wb = Wb
        self._W2v = W2v

        # Outward normal vector field
        x = SpatialCoordinate(self._mesh)
        xnorm = sqrt(inner(x, x))
        self._khat = interpolate(x/xnorm, mesh.coordinates.function_space())

        # Coriolis term
        fexpr = 2*self._Omega*x[2]/xnorm
        Vcg = FunctionSpace(self._mesh, "CG", self._mesh_degree)
        self._f = interpolate(fexpr, Vcg)

        # Solver details
        self._hybridization = hybridization
        self._monitor = monitor
        self._rtol = rtol

        # Solver parameters
        self._inner_preonly = inner_preonly
        if inner_pc_type == "gamg":
            self._params = self.gamg_paramters
        elif inner_pc_type == "hypre":
            self._params = self.hypre_parameters
        elif inner_pc_type == "direct":
            self._params = self.direct_parameters
        else:
            raise ValueError("Unknown inner PC type")

        # Functions for state solutions
        self._up = Function(self._Wmixed, name="Velocity-Pressure")
        self._btmp = Function(self._Wb, name="Buoyancy update")
        self._state = Function(self._W2 * self._W3 * self._Wb, name="State")

        # Function to store u-p residual
        self._up_residual = Function(self._Wmixed, name="U-P-Residual")

        # Build initial conditions for this particular problem
        self._build_initial_conditions()

        # Construct linear solvers
        self._build_up_solver()
        self._build_b_solver()

        # Keep record of iterations and reductions
        self._sim_time = []
        self._up_residual_reductions = []
        self._outer_ksp_iterations = []
        self._inner_ksp_iterations = []

    @cached_property
    def comm(self):
        return self._mesh.comm

    @cached_property
    def num_horizontal_cells(self):
        return self._base.cell_set.size

    @cached_property
    def num_cells(self):
        return self._mesh.cell_set.size

    @cached_property
    def name(self):
        name = "GW(cfl=%s, hybrid=%s, hexes=%s, cells=%s)" % (
            self._nu_cfl, self._hybridization,
            self._hexes, self.num_cells
        )
        return name

    @cached_property
    def output_file(self):
        dirname = "results/GW_r%s_nl%s_cfl%s" % (self._refinements,
                                                 self._nlayers,
                                                 self._nu_cfl)
        if self._hybridization:
            dirname += "_hybridization"

        if self._hexes:
            dirname += "_hexes"

        return File(dirname + "/lgw_" + str(self._refinements) + ".pvd")

    def write(self, dumpcount, dumpfreq):
        dumpcount += 1
        un, pn, bn = self._state.split()
        if dumpcount > dumpfreq:
            self.output_file.write(un, pn, bn)
            dumpcount -= dumpfreq
        return dumpcount

    @cached_property
    def direct_parameters(self):
        """Solver parameters using a direct method (LU)"""

        inner_params = {'ksp_type': 'preonly',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_package': 'mumps'}

        if self._hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': inner_params}
        else:
            params = {'ksp_type': 'preonly',
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'fieldsplit_0': inner_params,
                      'fieldsplit_1': inner_params}

        return params

    @cached_property
    def hypre_parameters(self):
        """Solver parameters using hypre's boomeramg
        implementation of AMG.
        """

        inner_params = {'ksp_type': 'cg',
                        'ksp_rtol': self._rtol,
                        'pc_type': 'hypre',
                        'pc_hypre_type': 'boomeramg',
                        'pc_hypre_boomeramg_no_CF': False,
                        'pc_hypre_boomeramg_coarsen_type': 'HMIS',
                        'pc_hypre_boomeramg_interp_type': 'ext+i',
                        'pc_hypre_boomeramg_P_max': 0,
                        'pc_hypre_boomeramg_agg_nl': 0,
                        'pc_hypre_boomeramg_max_level': 5,
                        'pc_hypre_boomeramg_strong_threshold': 0.25}

        if self._monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self._hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': inner_params}
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': self._rtol,
                      'pc_type': 'fieldsplit',
                      'pc_fieldsplit_type': 'schur',
                      'ksp_max_it': 100,
                      'ksp_gmres_restart': 50,
                      'pc_fieldsplit_schur_fact_type': 'FULL',
                      'pc_fieldsplit_schur_precondition': 'selfp',
                      'fieldsplit_0': {'ksp_type': 'preonly',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'},
                      'fieldsplit_1': inner_params}
            if self._monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    @cached_property
    def gamg_paramters(self):
        """Solver parameters for the velocity-pressure system using
        algebraic multigrid.
        """

        inner_params = {'ksp_type': 'bcgs',
                        'pc_type': 'gamg',
                        'pc_gamg_reuse_interpolation': True,
                        'pc_gamg_sym_graph': True,
                        'ksp_rtol': self._rtol,
                        'mg_levels': {'ksp_type': 'richardson',
                                      'ksp_richardson_self_scale': True,
                                      'ksp_max_it': 5,
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}}
        if self._monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self._hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': inner_params}
        else:
            if self._inner_preonly:
                params = {'ksp_type': 'gmres',
                          'ksp_rtol': self._rtol,
                          'pc_type': 'fieldsplit',
                          'pc_fieldsplit_type': 'schur',
                          'ksp_max_it': 100,
                          'ksp_gmres_restart': 50,
                          'pc_fieldsplit_schur_fact_type': 'FULL',
                          'pc_fieldsplit_schur_precondition': 'selfp',
                          'fieldsplit_0': {'ksp_type': 'preonly',
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'ilu'},
                          'fieldsplit_1': {'ksp_type': 'preonly',
                                           'pc_type': 'gamg',
                                           'pc_gamg_reuse_interpolation': True,
                                           'pc_gamg_sym_graph': True,
                                           'mg_levels': {'ksp_type': 'richardson',
                                                         'ksp_richardson_self_scale': True,
                                                         'ksp_max_it': 5,
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'}}}
            else:
                params = {'ksp_type': 'gmres',
                          'ksp_rtol': self._rtol,
                          'pc_type': 'fieldsplit',
                          'pc_fieldsplit_type': 'schur',
                          'ksp_max_it': 100,
                          'ksp_gmres_restart': 50,
                          'pc_fieldsplit_schur_fact_type': 'FULL',
                          'pc_fieldsplit_schur_precondition': 'selfp',
                          'fieldsplit_0': {'ksp_type': 'preonly',
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'ilu'},
                          'fieldsplit_1': {'ksp_type': 'bcgs',
                                           'pc_type': 'gamg',
                                           'pc_gamg_reuse_interpolation': True,
                                           'pc_gamg_sym_graph': True,
                                           'ksp_rtol': self._rtol,
                                           'mg_levels': {'ksp_type': 'richardson',
                                                         'ksp_richardson_self_scale': True,
                                                         'ksp_max_it': 5,
                                                         'pc_type': 'bjacobi',
                                                         'sub_pc_type': 'ilu'}}}
            if self._monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    def _build_initial_conditions(self):
        """Constructs initial conditions."""

        # Initial condition for velocity
        u0 = Function(self._W2)
        x = SpatialCoordinate(self._mesh)
        u_max = 20.    # Maximal zonal wind (m/s)
        uexpr = as_vector([-u_max*x[1]/self._R,
                           u_max*x[0]/self._R, 0.0])
        u0.project(uexpr)

        # Initial condition for buoyancy
        lamda_c = 2.0*np.pi/3.0
        phi_c = 0.0
        W_CG1 = FunctionSpace(self._mesh, "CG", 1)

        z_expr = Expression("sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) - a",
                            a=self._R)
        z = Function(W_CG1).interpolate(z_expr)

        lat_expr = Expression("asin(x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))")
        lat = Function(W_CG1).interpolate(lat_expr)
        lon = Function(W_CG1).interpolate(Expression("atan2(x[1], x[0])"))
        b0 = Function(self._Wb)
        deltaTheta = 1.0
        L_z = 20000.0
        d = 5000.0

        sin_tmp = sin(lat) * sin(phi_c)
        cos_tmp = cos(lat) * cos(phi_c)

        r = self._R*acos(sin_tmp + cos_tmp*cos(lon-lamda_c))
        s = (d**2)/(d**2 + r**2)

        bexpr = deltaTheta*s*sin(2*np.pi*z/L_z)
        b0.interpolate(bexpr)

        # Initial condition for pressure
        p0 = Function(self._W3).assign(0.0)
        p_eq = 1000.0 * 100.0
        g = 9.810616
        R_d = 287.0
        T_eq = 300.0
        c_p = 1004.5
        kappa = R_d / c_p
        G = g**2/(self._N**2*c_p)
        tsexpr = G + (T_eq - G)*exp(
            -(u_max*self._N**2/(4*g*g))*u_max*(cos(2.0*lat) - 1.0)
        )
        Ts = Function(W_CG1).interpolate(tsexpr)

        tk = (Ts/T_eq)**(1.0/kappa)
        psexp = p_eq*exp((u_max/(4.0*G*R_d))*u_max*(cos(2.0*lat)-1.0))*tk
        p0.interpolate(psexp)

        self._initial_state = (u0, p0, b0)

    def _build_up_solver(self):
        """Constructs the solver for the velocity-pressure increments."""

        from firedrake.assemble import create_assembly_callable

        # strong no-slip boundary conditions on the top
        # and bottom of the atmospheric domain
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        un, pn, bn = self._state.split()

        utest, ptest = TestFunctions(self._Wmixed)
        u, p = TrialFunctions(self._Wmixed)

        def outward(arg):
            return cross(self._khat, arg)

        # Linear gravity wave system for the velocity and pressure
        # increments (buoyancy has been eliminated in the discrete
        # equations since there is no orography)
        a_up = (ptest*p
                + self._dt_half_c2*ptest*div(u)
                - self._dt_half*div(utest)*p
                + (dot(utest, u)
                   + self._dt_half*dot(utest, self._f*outward(u))
                   + self._omega_N2
                   * dot(utest, self._khat)
                   * dot(u, self._khat))) * dx

        L_up = (dot(utest, un)
                + self._dt_half*dot(utest, self._f*outward(un))
                + self._dt_half*dot(utest, self._khat*bn)
                + ptest*pn) * dx

        # Set up linear solver
        up_problem = LinearVariationalProblem(a_up, L_up,
                                              self._up, bcs=bcs)
        params = self._params
        solver = LinearVariationalSolver(up_problem, solver_parameters=params,
                                         options_prefix='up-implicit-solver')
        self.up_solver = solver

        r = action(a_up, self._up) - L_up
        self._assemble_upr = create_assembly_callable(r,
                                                      tensor=self._up_residual)

    def _build_b_solver(self):
        """Constructs the solver for the buoyancy update."""

        # Computed velocity perturbation
        u0, _, _ = self._state.split()

        # Expression for buoyancy reconstruction
        btest = TestFunction(self._Wb)
        L_b = dot(btest*self._khat, u0) * dx
        a_b = btest*TrialFunction(self._Wb) * dx
        b_problem = LinearVariationalProblem(a_b, L_b, self._btmp)

        b_params = {'ksp_type': 'cg',
                    'pc_type': 'bjacobi',
                    'sub_pc_type': 'ilu'}
        if self._monitor:
            b_params['ksp_monitor_true_residual'] = True

        # Solver for buoyancy update
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters=b_params)
        self.b_solver = b_solver

    def _initialize(self):
        """Initialized the solver state with initial conditions
        for the velocity, pressure, and buoyancy fields.
        """

        u, p, b = self._state.split()
        u0, p0, b0 = self._initial_state

        u.assign(u0)
        p.assign(p0)
        b.assign(b0)

    def warmup(self):
        """Warm up solver by taking one time-step."""

        self._initialize()
        un, pn, bn = self._state.split()

        self._up.assign(0.0)
        with timed_stage("Warmup: Velocity-Pressure-Solve"):
            self.up_solver.solve()

        un.assign(self._up.sub(0))
        pn.assign(self._up.sub(1))

        self._btmp.assign(0.0)
        with timed_stage("Warmup: Buoyancy-Solve"):
            self.b_solver.solve()
            bn.assign(assemble(bn - self._dt_half_N2*self._btmp))

    def run_simulation(self, tmax, write=False, dumpfreq=100):
        PETSc.Sys.Print("""
        Running linear Boussinesq simulation with parameters:\n
        Hybridization: %s,\n
        Model order: %s,\n
        Refinements: %s,\n
        Layers: %s,\n
        Lid height (m): %s,\n
        Radius (m): %s,\n
        Horizontal Courant number: %s,\n
        Approx. Delta x (m): %s,\n
        Time-step size (s): %s,\n
        Stop time (s): %s.
        """ % (self._hybridization, self._order, self._refinements,
               self._nlayers, self._R, self._H, self._nu_cfl, self._dx_avg,
               self._dt, tmax))

        t = 0.0
        self._initialize()
        un, pn, bn = self._state.split()

        dumpcount = dumpfreq
        if write:
            dumpcount = self.write(dumpcount, dumpfreq)

        self.up_solver.snes.setConvergenceHistory()
        self.up_solver.snes.ksp.setConvergenceHistory()
        self.b_solver.snes.setConvergenceHistory()
        self.b_solver.snes.ksp.setConvergenceHistory()

        while t < tmax:
            t += self._dt
            self._sim_time.append(t)

            # Solve for new u and p field
            self._up.assign(0.0)
            with timed_stage("Velocity-Pressure-Solve"):
                self.up_solver.solve()

            # Update state with new u and p
            un.assign(self._up.sub(0))
            pn.assign(self._up.sub(1))

            # Reconstruct b using new u and p
            self._btmp.assign(0.0)
            with timed_stage("Buoyancy-Solve"):
                self.b_solver.solve()
                bn.assign(assemble(bn - self._dt_half_N2*self._btmp))

            # Collect residual reductions for u-p system
            r0 = self.up_solver.snes.ksp.getRhs()

            # Assemble u-p residual
            self._assemble_upr()
            rn = self._up_residual

            r0norm = r0.norm()
            rnnorm = rn.dat.norm
            reduction = rnnorm/r0norm
            self._up_residual_reductions.append(reduction)

            # Collect KSP iterations
            outer_ksp = self.up_solver.snes.ksp
            if self._hybridization:
                ctx = outer_ksp.getPC().getPythonContext()
                inner_ksp = ctx.trace_ksp
            else:
                ksps = outer_ksp.getPC().getFieldSplitSubKSP()
                _, inner_ksp = ksps

            outer_its = outer_ksp.getIterationNumber()
            inner_its = inner_ksp.getIterationNumber()
            self._outer_ksp_iterations.append(outer_its)
            self._inner_ksp_iterations.append(inner_its)

            if write:
                with timed_stage("Dump output"):
                    dumpcount = self.write(dumpcount, dumpfreq)
