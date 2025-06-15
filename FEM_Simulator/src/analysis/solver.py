from pathlib import Path

import gmsh
import numpy as np
import scipy.constants
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI


def run_analysis(mesh_path: Path):
    """Run a simple 2-D magnetostatic solve on the given mesh."""
    print("\n--- Running Magnetostatic Analysis ---")

    try:
        # ------------------------------------------------------------------
        # 1. Load mesh and retrieve physical group names
        # ------------------------------------------------------------------
        domain, cell_tags, _ = gmshio.read_from_msh(
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
        # 3. Material properties (relative permeability)
        # ------------------------------------------------------------------
        mu_r = fem.Function(DG0)
        mu_0 = scipy.constants.mu_0

        # Default all cells to air (1.0)
        mu_r.x.array.fill(1.0)

        print("Assigning material properties…")
        for tag, name in tag_to_name.items():
            if name in {"stator_steel", "rotor_steel"}:
                print(f"  - High µ on '{name}' (Tag {tag})")
                cells = np.where(cell_tags.values == tag)[0]
                mu_r.x.array[cells] = 1000.0

        # ------------------------------------------------------------------
        # 4. Weak formulation  ∇·(1/µ ∇A) = 0
        # ------------------------------------------------------------------
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = (1.0 / (mu_0 * mu_r)) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = fem.Constant(domain, 0.0) * v * ufl.dx  # no source currents yet

        # ------------------------------------------------------------------
        # 5. Boundary condition  A = 0 on outer boundary (named in gmsh)
        # ------------------------------------------------------------------
        try:
            outer_bdy_tag = next(
                t for t, n in tag_to_name.items() if n == "outer_boundary"
            )
            facets_outer = np.where(cell_tags.indices == outer_bdy_tag)[0]
            bc_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_outer)
        except StopIteration:
            # Fallback: radial max
            bc_dofs = fem.locate_dofs_geometrical(
                V, lambda x: np.isclose(np.linalg.norm(x, axis=0), domain.geometry.x[:, 0].max())
            )
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

        print("--- Analysis Complete: Fields A_z and B computed ---")
        return {"Az": Az, "B": B}

    except Exception as exc:
        print(f"ERROR during analysis: {exc}")
        return None 