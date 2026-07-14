"""
FEniCS (DOLFINx) implementation of the non-dimensionalized pheromone-enzyme system.

Solves on a 2D rectangular (r, theta) domain representing the exterior of a
unit sphere in spherical coordinates with azimuthal symmetry.

  r     in [1, r_max]
  theta in [0, pi]       (theta=0 points toward external source)

The 3D spherical Laplacian with azimuthal symmetry, written in the (r, theta)
plane, has metric factors r^2 sin(theta) that appear in the weak form.

Fields:
  Q(r) = gamma / r                    [known, harmonic]
  P(r, theta)   = total external species   [Laplace + point source, Robin BC]
  psi(r, theta) = complex concentration    [nonlinear reaction-diffusion, Neumann BC]

Recovered:
  phi = Q - psi   (enzyme)
  chi = P - psi   (free pheromone)

Equations:
  nabla^2 P + 4*pi*alpha * delta^3(r - rho0 zhat) = 0
      BC at r=1:      dP/dr = beta * (P - psi)     [Robin]
      BC at r=r_max:  P = 0                          [Dirichlet]
      BC at theta=0,pi: regularity (natural)

  nabla^2 psi + 4*pi*zeta * [(Q-psi)(P-psi) - kappa*psi] = 0
      BC at r=1:      dpsi/dr = 0                    [Neumann]
      BC at r=r_max:  psi = 0                         [Dirichlet]
      BC at theta=0,pi: regularity (natural)

SNR metric:
  SNR = int dOmega chi(1,theta) cos(theta)  /  sqrt( int dOmega chi(1,theta) )

Requires:
  pip install fenics-dolfinx gmsh meshio
  (or conda install -c conda-forge fenics-dolfinx mpich petsc4py)
  (or docker run -ti dolfinx/dolfinx:stable)
"""

import numpy as np
from dataclasses import dataclass

try:
    from dolfinx import mesh, fem, io, default_scalar_type
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    import ufl
    from mpi4py import MPI
    import gmsh

    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    print("DOLFINx not found. Install via:")
    print("  conda install -c conda-forge fenics-dolfinx mpich petsc4py")
    print("or use the FEniCS Docker image:")
    print("  docker run -ti dolfinx/dolfinx:stable")


# ================================================================
# Parameters
# ================================================================
@dataclass
class Params:
    gamma: float = 1.0
    alpha: float = 0.5
    beta:  float = 5.0
    zeta:  float = 5.0
    kappa: float = 0.2
    rho0:  float = 5.0


# ================================================================
# Mesh generation with Gmsh
# ================================================================
def create_mesh(r_max=15.0, lc_inner=0.05, lc_outer=0.3):
    """
    Create a 2D rectangular mesh for the (r, theta) domain.

      r     in [1, r_max]
      theta in [0, pi]

    Returns dolfinx mesh and facet tags:
      tag 1 = cell wall (r = 1)
      tag 2 = far field  (r = r_max)
      tag 3 = axis theta = 0
      tag 4 = axis theta = pi
    """
    gmsh.initialize()
    gmsh.model.add("spherical_domain")

    # Corners of the rectangular domain
    p1 = gmsh.model.occ.addPoint(1.0,   0.0,    0, lc_inner)    # (r=1, theta=0)
    p2 = gmsh.model.occ.addPoint(r_max,  0.0,    0, lc_outer)   # (r_max, theta=0)
    p3 = gmsh.model.occ.addPoint(r_max,  np.pi,  0, lc_outer)   # (r_max, theta=pi)
    p4 = gmsh.model.occ.addPoint(1.0,   np.pi,  0, lc_inner)    # (r=1, theta=pi)

    # Edges
    l_inner  = gmsh.model.occ.addLine(p1, p4)   # r = 1        (cell wall)
    l_outer  = gmsh.model.occ.addLine(p2, p3)   # r = r_max    (far field)
    l_axis0  = gmsh.model.occ.addLine(p1, p2)   # theta = 0    (axis toward source)
    l_axispi = gmsh.model.occ.addLine(p4, p3)   # theta = pi   (axis away from source)

    loop = gmsh.model.occ.addCurveLoop([l_axis0, l_outer, -l_axispi, -l_inner])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l_inner],  tag=1)
    gmsh.model.setPhysicalName(1, 1, "cell_wall")

    gmsh.model.addPhysicalGroup(1, [l_outer],  tag=2)
    gmsh.model.setPhysicalName(1, 2, "far_field")

    gmsh.model.addPhysicalGroup(1, [l_axis0],  tag=3)
    gmsh.model.setPhysicalName(1, 3, "axis_theta0")

    gmsh.model.addPhysicalGroup(1, [l_axispi], tag=4)
    gmsh.model.setPhysicalName(1, 4, "axis_thetapi")

    gmsh.model.addPhysicalGroup(2, [surface],  tag=1)
    gmsh.model.setPhysicalName(2, 1, "fluid")

    # Mesh with transfinite for nice structure (optional, remove for unstructured)
    gmsh.model.mesh.generate(2)

    from dolfinx.io import gmshio
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()
    return msh, cell_tags, facet_tags


# ================================================================
# Solver
# ================================================================
def solve_system(params=None, r_max=15.0, lc_inner=0.05, lc_outer=0.3):
    """
    Solve the coupled P-psi system in spherical (r, theta) coordinates.

    The 3D Laplacian with azimuthal symmetry in weak form on the (r, theta)
    rectangle uses the volume element r^2 sin(theta) dr dtheta (times 2*pi
    from phi, which cancels).  Integration by parts gives:

      int nabla^2(u) * v * r^2 sin(theta)  dr dtheta
        = -int [ du/dr * dv/dr * r^2 sin(theta)
               + (1/r^2) du/dtheta * dv/dtheta * r^2 sin(theta) ]  dr dtheta
          + boundary terms

    Simplifying:
      = -int [ du/dr * dv/dr * r^2 sin(theta)
             + du/dtheta * dv/dtheta * sin(theta) ]  dr dtheta
        + boundary terms

    So the "metric-weighted gradient" inner product is NOT the standard
    FEniCS grad.grad — we must write the two terms explicitly.
    """
    if not HAS_DOLFINX:
        raise RuntimeError("DOLFINx is required.")

    if params is None:
        params = Params()

    # --- Mesh ---
    msh, cell_tags, facet_tags = create_mesh(
        r_max=r_max, lc_inner=lc_inner, lc_outer=lc_outer
    )

    # Coordinates: x[0] = r, x[1] = theta
    x = ufl.SpatialCoordinate(msh)
    r     = x[0]
    theta = x[1]

    # --- Function space ---
    el  = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    mel = ufl.MixedElement([el, el])
    V   = fem.functionspace(msh, mel)
    V_scalar = fem.functionspace(msh, ("Lagrange", 1))

    # Solution and test functions
    U = fem.Function(V)
    P_sol, psi_sol = ufl.split(U)
    v_P, v_psi     = ufl.TestFunctions(V)

    # --- Known field ---
    Q = params.gamma / r

    # --- Metric factors ---
    sin_th = ufl.sin(theta)
    cos_th = ufl.cos(theta)

    # --- Regularised point source at (r=rho0, theta=0) ---
    sig_r  = 3 * lc_inner
    sig_th = 3 * lc_inner  # angular width in radians
    S_raw  = ufl.exp(-0.5 * ((r - params.rho0) / sig_r)**2) \
           * ufl.exp(-0.5 * (theta / sig_th)**2)

    # Approximate normalisation: total = 4*pi*alpha
    # int S_raw * r^2 sin(theta) dr dtheta * 2*pi ≈ (sqrt(2pi)*sig_r) * (sqrt(2pi)*sig_th * rho0^2) * 2*pi
    norm_approx = (2 * np.pi
                   * np.sqrt(2 * np.pi) * sig_r
                   * np.sqrt(2 * np.pi) * sig_th
                   * params.rho0**2)
    source_scale = 4 * np.pi * params.alpha / norm_approx

    # --- Weak form helper: metric-weighted bilinear form ---
    # -int [ du/dr * dv/dr * r^2 sin(theta) + du/dtheta * dv/dtheta * sin(theta) ] dr dtheta
    def spherical_stiffness(u, v):
        """Negative of the spherical Laplacian weak form (stiffness matrix)."""
        du_dr = ufl.grad(u)[0]   # d/dr  (since x[0]=r)
        dv_dr = ufl.grad(v)[0]
        du_dth = ufl.grad(u)[1]  # d/dtheta (since x[1]=theta)
        dv_dth = ufl.grad(v)[1]
        return (du_dr * dv_dr * r**2 * sin_th
                + du_dth * dv_dth * sin_th) * ufl.dx

    # --- P equation ---
    # 0 = nabla^2 P + S_P
    # Weak form: int [grad P . grad v_P (metric)] dr dtheta = int S_P * v_P * r^2 sin(theta) dr dtheta
    F_P = (
        spherical_stiffness(P_sol, v_P)
        - source_scale * S_raw * v_P * r**2 * sin_th * ufl.dx
    )

    # --- psi equation ---
    phi = Q - psi_sol
    chi = P_sol - psi_sol
    reaction = 4 * np.pi * params.zeta * (phi * chi - params.kappa * psi_sol)

    F_psi = (
        spherical_stiffness(psi_sol, v_psi)
        - reaction * v_psi * r**2 * sin_th * ufl.dx
    )

    # --- Robin BC for P at r = 1 (cell wall, tag 1) ---
    # Original: dP/dr|_{r=1} = beta * (P - psi)
    # The boundary term from integration by parts on the r=1 edge is:
    #   - int dP/dr * v_P * r^2 sin(theta) dtheta  (at r=1, r^2=1)
    # Substituting Robin: = -beta * (P - psi) * v_P * sin(theta) dtheta
    # This term must be SUBTRACTED from the stiffness form (which has + sign):
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # At r=1: the outward normal from the domain points in the -r direction
    # (since the domain is r > 1). So dP/dn = -dP/dr.
    # Integration by parts gives: + int dP/dn * v * r^2 sin(th) ds = -int dP/dr * v * sin(th) ds(1)
    # With Robin dP/dr = beta*(P-psi):
    #   = -beta*(P-psi) * v * sin(th) ds(1)
    # But the stiffness form already has the positive sign, so this boundary
    # integral is ADDED with a negative sign:
    F_P += -params.beta * (P_sol - psi_sol) * v_P * sin_th * ds(1)

    # Neumann for psi: dpsi/dr = 0 at r=1, natural BC (no term needed)

    # Note: at theta=0 and theta=pi, sin(theta)=0 kills the angular boundary
    # terms automatically — regularity is enforced by the metric.

    # --- Combined residual ---
    F = F_P + F_psi

    # --- Dirichlet BC at far field (r = r_max): P = 0, psi = 0 ---
    V0, _ = V.sub(0).collapse()
    V1, _ = V.sub(1).collapse()

    far_facets = facet_tags.find(2)
    far_dofs_P   = fem.locate_dofs_topological((V.sub(0), V0), 1, far_facets)
    far_dofs_psi = fem.locate_dofs_topological((V.sub(1), V1), 1, far_facets)

    bc_P   = fem.dirichletbc(fem.Function(V0), far_dofs_P,   V.sub(0))
    bc_psi = fem.dirichletbc(fem.Function(V1), far_dofs_psi, V.sub(1))
    bcs = [bc_P, bc_psi]

    # --- Solve ---
    problem = NonlinearProblem(F, U, bcs=bcs)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol   = 1e-8
    solver.max_it = 50
    solver.report = True

    U.x.array[:] = 0.0

    print("Solving nonlinear system with Newton's method...")
    n_iters, converged = solver.solve(U)
    print(f"  Newton converged: {converged}, iterations: {n_iters}")

    return msh, U, V, facet_tags


# ================================================================
# SNR
# ================================================================
def compute_snr(msh, U, V, params=None):
    """
    SNR = int dOmega chi(1, theta) cos(theta)  /  sqrt( int dOmega chi(1, theta) )

    where dOmega = sin(theta) dtheta dphi  and the phi integral gives 2*pi.

    We evaluate chi = P - psi at the cell wall (r = 1, i.e. the left edge
    of the (r, theta) rectangle), then integrate over theta using the
    surface element sin(theta) dtheta.
    """
    if params is None:
        params = Params()

    V_scalar = fem.functionspace(msh, ("Lagrange", 1))
    coords = msh.geometry.x[:, :2]
    r_vals     = coords[:, 0]
    theta_vals = coords[:, 1]

    P_sol   = U.sub(0).collapse()
    psi_sol = U.sub(1).collapse()

    chi_vals = P_sol.x.array - psi_sol.x.array

    # Find DOFs on the cell wall (r ≈ 1)
    tol_r = 0.05
    wall_mask = np.abs(r_vals - 1.0) < tol_r

    theta_wall = theta_vals[wall_mask]
    chi_wall   = chi_vals[wall_mask]

    # Sort by theta for clean integration
    sort_idx   = np.argsort(theta_wall)
    theta_wall = theta_wall[sort_idx]
    chi_wall   = chi_wall[sort_idx]

    sin_th = np.sin(theta_wall)
    cos_th = np.cos(theta_wall)

    # Trapezoidal integration
    numerator   = 2 * np.pi * np.trapz(chi_wall * cos_th * sin_th, theta_wall)
    total       = 2 * np.pi * np.trapz(chi_wall * sin_th, theta_wall)
    denominator = np.sqrt(total)

    snr = numerator / denominator
    print(f"  SNR = {snr:.6f}")
    print(f"    Numerator   (dipole moment): {numerator:.6f}")
    print(f"    Denominator (sqrt total):    {denominator:.6f}")
    return snr


# ================================================================
# Post-processing and plotting
# ================================================================
def postprocess(msh, U, V, params=None, savepath_prefix="fenics_result"):
    """Extract fields and create plots."""
    if params is None:
        params = Params()

    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    P_sol   = U.sub(0).collapse()
    psi_sol = U.sub(1).collapse()

    V_scalar = fem.functionspace(msh, ("Lagrange", 1))

    # Compute Q, phi, chi
    x = ufl.SpatialCoordinate(msh)
    Q_expr = params.gamma / x[0]

    Q_func   = fem.Function(V_scalar, name="Q")
    Q_func.interpolate(fem.Expression(Q_expr, V_scalar.element.interpolation_points()))

    phi_func = fem.Function(V_scalar, name="phi_enzyme")
    chi_func = fem.Function(V_scalar, name="chi_pheromone")
    phi_func.x.array[:] = Q_func.x.array - psi_sol.x.array
    chi_func.x.array[:] = P_sol.x.array  - psi_sol.x.array

    # --- Save VTK ---
    with io.VTXWriter(msh.comm, f"{savepath_prefix}.bp",
                      [P_sol, psi_sol, phi_func, chi_func]) as f:
        f.write(0.0)

    # --- Map (r, theta) -> (z, rho_cyl) for plotting ---
    coords = msh.geometry.x[:, :2]
    r_vals     = coords[:, 0]
    theta_vals = coords[:, 1]
    z_vals   = r_vals * np.cos(theta_vals)
    rho_vals = r_vals * np.sin(theta_vals)

    # Build triangulation in (z, rho) space
    topology = msh.topology
    topology.create_connectivity(2, 0)
    cells = topology.connectivity(2, 0)
    triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])

    triang        = tri.Triangulation(z_vals,  rho_vals, triangles)
    triang_mirror = tri.Triangulation(z_vals, -rho_vals, triangles)

    fields = [
        (phi_func.x.array, 'φ (enzyme)'),
        (chi_func.x.array, 'χ (free pheromone)'),
        (psi_sol.x.array,  'ψ (complex)'),
        (P_sol.x.array,    'P = χ + ψ'),
    ]

    th_circ = np.linspace(0, 2 * np.pi, 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (data, title) in zip(axes.flat, fields):
        ax.tripcolor(triang, data, shading='gouraud', cmap='YlOrBr')
        im = ax.tripcolor(triang_mirror, data, shading='gouraud', cmap='YlOrBr')
        ax.plot(np.cos(th_circ), np.sin(th_circ), 'k-', lw=1.5)
        ax.plot(params.rho0, 0, 'k+', ms=10, mew=2)
        ax.set_xlim(-5, params.rho0 + 3)
        ax.set_ylim(-4, 6)
        ax.set_xlabel('z')
        ax.set_ylabel('ρ')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_2d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {savepath_prefix}_2d.png")

    # --- On-axis profile (theta ≈ 0, toward source) ---
    axis_mask = theta_vals < 0.15
    r_axis = r_vals[axis_mask]
    sort_idx = np.argsort(r_axis)
    r_axis = r_axis[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in [
        (phi_func.x.array, 'φ (enzyme)', 'r'),
        (chi_func.x.array, 'χ (free pheromone)', 'b'),
        (psi_sol.x.array,  'ψ (complex)', 'g'),
    ]:
        ax.plot(r_axis, data[axis_mask][sort_idx], color=color, lw=2, label=label)

    ax.axvline(x=1, color='gray', ls=':', alpha=0.5, label='Cell wall')
    ax.axvline(x=params.rho0, color='gray', ls='--', alpha=0.5,
               label=f'Source (ρ₀={params.rho0})')
    ax.set_xlabel('r')
    ax.set_ylabel('Concentration')
    ax.set_title('On-axis profiles (θ=0, toward source)')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_axis.png", dpi=150)
    plt.close()
    print(f"Saved {savepath_prefix}_axis.png")

    # --- Surface profile ---
    tol_r = 0.05
    wall_mask = np.abs(r_vals - 1.0) < tol_r
    theta_wall = theta_vals[wall_mask]
    sort_idx = np.argsort(theta_wall)
    theta_wall = theta_wall[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    for data, label, color in [
        (phi_func.x.array, 'φ (enzyme)', 'r'),
        (chi_func.x.array, 'χ (free pheromone)', 'b'),
        (psi_sol.x.array,  'ψ (complex)', 'g'),
    ]:
        ax.plot(np.degrees(theta_wall), data[wall_mask][sort_idx],
                color=color, lw=2, label=label)

    ax.set_xlabel('θ (degrees from source)')
    ax.set_ylabel('Concentration at cell surface')
    ax.set_title('Angular profile on cell wall')
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_surface.png", dpi=150)
    plt.close()
    print(f"Saved {savepath_prefix}_surface.png")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    if not HAS_DOLFINX:
        raise SystemExit("DOLFINx is required. See message above.")

    params = Params(
        gamma = 1.0,
        alpha = 1.0,
        beta  = 1.0,
        zeta  = 1.0,
        kappa = 1.0,
        rho0  = 4.0,
    )

    msh, U, V, facet_tags = solve_system(
        params=params,
        r_max=15.0,
        lc_inner=0.05,
        lc_outer=0.3,
    )

    compute_snr(msh, U, V, params=params)
    postprocess(msh, U, V, params=params, savepath_prefix="fenics_result")
    print("Done.")