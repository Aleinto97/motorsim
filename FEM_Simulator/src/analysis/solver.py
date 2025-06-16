from pathlib import Path

import gmsh
import numpy as np
import scipy.constants
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
import pyvista as pv
from dolfinx import plot as dfx_plot


def run_analysis(mesh_path: Path, phase_currents: dict[str, float] | None = None):
    """Run a simple 2-D magnetostatic solve on the given mesh."""
    print("\n--- Running Magnetostatic Analysis ---")

    try:
        # ------------------------------------------------------------------
        # 1. Load mesh and retrieve physical group names
        # ------------------------------------------------------------------
        domain, cell_tags, facet_tags = gmshio.read_from_msh(
            str(mesh_path), MPI.COMM_WORLD, 0, gdim=2
        )

        gmsh.initialize()
        gmsh.open(str(mesh_path))
        phys_groups = gmsh.model.getPhysicalGroups()
        tag_to_name = {
            tag: gmsh.model.getPhysicalName(dim, tag) for dim, tag in phys_groups
        }
        gmsh.finalize()

        # ------------------------------------------------------------------
        # 2. Function spaces
        # ------------------------------------------------------------------
        V = fem.functionspace(domain, ("CG", 1))   # Magnetic vector potential A_z
        DG0 = fem.functionspace(domain, ("DG", 0))  # Cell-wise constants

        # ------------------------------------------------------------------
        # 3. Material properties (relative permeability) & source current
        # ------------------------------------------------------------------
        mu_r = fem.Function(DG0)
        mu_0 = scipy.constants.mu_0
        Jz = fem.Function(DG0)  # out-of-plane source current density

        # Defaults
        mu_r.x.array.fill(1.0)  # air everywhere
        Jz.x.array.fill(0.0)

        print("Assigning material properties, magnets and windings…")
        if phase_currents is None:
            phase_currents = {"A": 0.0, "B": 0.0, "C": 0.0}

        for tag, name in tag_to_name.items():
            cells = np.where(cell_tags.values == tag)[0]
            if name in {"stator_steel", "rotor_steel"}:
                print(f"  - High µ on '{name}' (Tag {tag})")
                mu_r.x.array[cells] = 1000.0
            elif name == "magnets":
                print(f"  - Permanent magnet region (Tag {tag}) → Jz source")
                # Simple equivalent magnet source current density (A/m^2)
                J_eq = 1e6  # placeholder value for demonstration
                Jz.x.array[cells] = J_eq
            elif name.startswith("phase_"):
                phase = name.split("_")[1]
                I = phase_currents.get(phase, 0.0)
                if abs(I) > 0:
                    print(f"  - Applying current {I} A to {name}")
                    J_phase = I * 1e6  # placeholder scaling
                    Jz.x.array[cells] = J_phase

        # ------------------------------------------------------------------
        # 4. Weak formulation  ∇·(1/µ ∇A) = 0
        # ------------------------------------------------------------------
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = (1.0 / (mu_0 * mu_r)) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = Jz * v * ufl.dx

        # ------------------------------------------------------------------
        # 5. Boundary condition  A = 0 on outer boundary (named in gmsh)
        # ------------------------------------------------------------------
        try:
            outer_bdy_tag = next(
                t for t, n in tag_to_name.items() if n == "outer_boundary"
            )
            facets_outer = np.where(facet_tags.values == outer_bdy_tag)[0]
            if facets_outer.size:
                bc_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_outer)
            else:
                raise RuntimeError("Outer boundary facets not found.")
        except StopIteration:
            # Fallback: use nodes near stator outer radius
            Rext_est = np.sqrt(np.max(np.sum(domain.geometry.x[:, :2] ** 2, axis=1)))
            bc_dofs = fem.locate_dofs_geometrical(
                V, lambda x: np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2), Rext_est, rtol=1e-2)
            )
        if bc_dofs.size == 0:
            raise RuntimeError("Failed to locate boundary DOFs for Dirichlet BC.")
        bc = fem.dirichletbc(fem.Constant(domain, 0.0), bc_dofs, V)

        # ------------------------------------------------------------------
        # 6. Solve
        # ------------------------------------------------------------------
        print("Solving the linear system…")
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        })
        Az = problem.solve()
        Az.name = "A_z"
        print("-> Solve complete.")

        # ------------------------------------------------------------------
        # 7. Post-processing  B = curl A
        # ------------------------------------------------------------------
        W = fem.VectorFunctionSpace(domain, ("DG", 0))
        B = fem.Function(W, name="B")
        expr = fem.Expression(
            ufl.as_vector((Az.dx(1), -Az.dx(0))),
            W.element.interpolation_points(),
        )
        B.interpolate(expr)

        # ------------------------------------------------------------------
        # 8. Convert to PyVista for visualization on rank 0
        # ------------------------------------------------------------------
        if MPI.COMM_WORLD.rank == 0:
            cells, cell_types, geometry = dfx_plot.create_vtk_mesh(domain, domain.topology.dim)
            grid = pv.UnstructuredGrid(cells, cell_types, geometry)
            B_vec = B.x.array.reshape(-1, 2)
            B_mag = np.linalg.norm(B_vec, axis=1)
            grid.cell_data["B_mag"] = B_mag
        else:
            grid = None

        print("--- Analysis Complete: Fields ready for visualization ---")
        return {"Az": Az, "B": B, "grid": grid}

    except Exception as exc:
        print(f"ERROR during analysis: {exc}")
        return None 