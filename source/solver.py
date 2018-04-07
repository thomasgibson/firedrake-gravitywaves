from firedrake import *
from firedrake.utils import cached_property
from pyop2.profiling import timed_stage


class ExplicitSolver(object):
    """Solver for the linearized compressible Boussinesq equations
    (includes Coriolis term). This solver uses an explicit matrix solver
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

    def __init__(self, W2, W3, Wb, dt, c, N, Omega, R, rtol=1.0E-8,
                 inner_solver_type="gamg", hybridization=False, monitor=False):
        """The constructor for the ExplicitSolver.

        :arg W2: The HDiv velocity space.
        :arg W3: The L2 pressure space.
        :arg Wb: The "Charney-Phillips" space for the buoyancy field.
        :arg dt: A positive real number denoting the time-step size.
        :arg c: A positive real number denoting the speed of sound waves
            in dry air.
        :arg N: A positive real number describing the Brunt–Väisälä frequency.
        :arg Omega: A positive real number; the angular rotation rate of the
            Earth.
        :arg R: A positive real number denoting the radius of the spherical
            mesh (Earth-size).
        :arg rtol: The relative tolerance for the solver.
        :arg inner_solver_type: A string describing which inner-most solver to
            use on the pressure space (approximate Schur-complement) or the
            trace space (hybridization).
        :arg hybridization: A boolean switch between using a hybridized
            mixed method (True) on the velocity-pressure system, or GMRES
            with an approximate Schur-complement preconditioner (False).
        :arg monitor: A boolean switch with turns on/off KSP monitoring
            of the problem residuals (primarily for debugging and checking
            convergence of the solver). When profiling, keep this set
            to `False`.
        """

        self.hybridization = hybridization
        self.monitor = monitor
        self.rtol = rtol

        if inner_solver_type == "gamg":
            self.params = self.gamg_paramters
        elif inner_solver_type == "hypre":
            self.params = self.hypre_parameters
        elif inner_solver_type == "direct":
            self.params = self.direct_parameters
        else:
            raise ValueError("Unknown inner solver type")

        # Timestepping parameters and physical constants
        self._dt = dt
        self._c = c
        self._N = N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)
        self._omega_c2 = Constant((0.5*dt*c)**2)

        # Compatible finite element spaces
        self._Wmixed = W2 * W3
        self._W2 = self._Wmixed.sub(0)
        self._W3 = self._Wmixed.sub(1)
        self._Wb = Wb

        # Functions for state solutions
        self._up = Function(self._Wmixed)
        self._b = Function(self._Wb)
        self._btmp = Function(self._Wb)

        self._state = Function(self._W2 * self._W3 * self._Wb, name="State")

        # Outward normal vector
        mesh = self._W3.mesh()
        x = SpatialCoordinate(mesh)
        R = sqrt(inner(x, x))
        self._khat = interpolate(x/R, mesh.coordinates.function_space())

        # Coriolis term
        fexpr = 2*Omega*x[2]/R
        Vcg = FunctionSpace(mesh, "CG", 1)
        self._f = interpolate(fexpr, Vcg)

        # Construct linear solvers
        self._build_up_solver()
        self._build_b_solver()

        self.up_residual_reductions = []

    @cached_property
    def direct_parameters(self):
        """Solver parameters using a direct method (LU)"""

        inner_params = {'ksp_type': 'preonly',
                        'pc_type': 'lu',
                        'pc_factor_mat_solver_package': 'mumps'}

        if self.hybridization:
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
                        'ksp_rtol': self.rtol,
                        'pc_type': 'hypre',
                        'pc_hypre_type': 'boomeramg',
                        'pc_hypre_boomeramg_no_CF': False,
                        'pc_hypre_boomeramg_coarsen_type': 'HMIS',
                        'pc_hypre_boomeramg_interp_type': 'ext+i',
                        'pc_hypre_boomeramg_P_max': 0,
                        'pc_hypre_boomeramg_agg_nl': 0,
                        'pc_hypre_boomeramg_max_level': 5,
                        'pc_hypre_boomeramg_strong_threshold': 0.25}

        if self.monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self.hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': inner_params}
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': self.rtol,
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
            if self.monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    @cached_property
    def gamg_paramters(self):
        """Solver parameters for the velocity-pressure system using
        algebraic multigrid.
        """

        inner_params = {'ksp_type': 'cg',
                        'pc_type': 'gamg',
                        'ksp_rtol': self.rtol,
                        'mg_levels': {'ksp_type': 'chebyshev',
                                      'ksp_max_it': 2,
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'}}
        if self.monitor:
            inner_params['ksp_monitor_true_residual'] = True

        if self.hybridization:
            params = {'ksp_type': 'preonly',
                      'mat_type': 'matfree',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': inner_params}
        else:
            params = {'ksp_type': 'gmres',
                      'ksp_rtol': self.rtol,
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
            if self.monitor:
                params['ksp_monitor_true_residual'] = True

        return params

    @cached_property
    def _build_up_bilinear_form(self):
        """Bilinear form for the gravity wave velocity-pressure
        subsystem.
        """

        utest, ptest = TestFunctions(self._Wmixed)
        u, p = TrialFunctions(self._Wmixed)

        def outward(u):
            return cross(self._khat, u)

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
        return a_up

    def _build_up_rhs(self, u0, p0, b0):
        """Right-hand side for the gravity wave velocity-pressure
        subsystem.
        """

        def outward(u):
            return cross(self._khat, u)

        utest, ptest = TestFunctions(self._Wmixed)
        L_up = (dot(utest, u0)
                + self._dt_half*dot(utest, self._f*outward(u0))
                + self._dt_half*dot(utest, self._khat*b0)
                + ptest*p0) * dx

        return L_up

    def up_residual(self, old_state, new_up):
        """Returns the residual of the velocity-pressure system."""

        u0, p0, b0 = old_state.split()
        res = self._build_up_rhs(u0, p0, b0)
        L = self._build_up_bilinear_form
        res -= action(L, new_up)

        return res

    def _build_up_solver(self):
        """Constructs the solver for the velocity-pressure increments."""

        # strong no-slip boundary conditions on the top
        # and bottom of the atmospheric domain
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]

        un, pn, bn = self._state.split()

        a_up = self._build_up_bilinear_form
        L_up = self._build_up_rhs(un, pn, bn)

        # Set up linear solver
        up_problem = LinearVariationalProblem(a_up, L_up,
                                              self._up, bcs=bcs)
        params = self.params
        solver = LinearVariationalSolver(up_problem, solver_parameters=params,
                                         options_prefix='up-implicit-solver')
        self.linear_solver = solver

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
        if self.monitor:
            b_params['ksp_monitor_true_residual'] = True

        # Solver for buoyancy update
        b_solver = LinearVariationalSolver(b_problem,
                                           solver_parameters=b_params)
        self.b_solver = b_solver

    def initialize(self, u, p, b):
        """Initialized the solver state with initial conditions
        for the velocity, pressure, and buoyancy fields.

        :arg u: An initial condition (`firedrake.Function`)
                for the velocity field.
        :arg p: An initial condition for the pressure field.
        :arg b: And finally an function describing the initial
                state of the buoyancy field.
        """

        u0, p0, b0 = self._state.split()
        u0.assign(u)
        p0.assign(p)
        b0.assign(b)

    def solve(self):
        """Solves the linear gravity wave problem at a particular
        time-step in two-stages. First, the velocity and pressure
        solutions are computed, then buoyancy is reconstructed from
        the computed fields. The solver state is then updated.
        """

        # Previous state
        un, pn, bn = self._state.split()

        # Initial residual
        self._up.assign(0.0)
        self._b.assign(0.0)
        r0 = assemble(self.up_residual(self._state, self._up))

        # Main solver stage
        with timed_stage("Velocity-Pressure-Solve"):
            self.linear_solver.solve()

        # Residual after solving
        rn = assemble(self.up_residual(self._state, self._up))
        self.up_residual_reductions.append(rn.dat.norm/r0.dat.norm)

        # Update state
        un.assign(self._up.sub(0))
        pn.assign(self._up.sub(1))

        # Reconstruct b
        self._btmp.assign(0.0)
        with timed_stage("Buoyancy-Solve"):
            self.b_solver.solve()
            bn.assign(assemble(bn - self._dt_half_N2*self._btmp))
