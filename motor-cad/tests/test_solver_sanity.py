import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

from src.core.machine_factory import create_machine
from src.geometry.simple_machine_mesher import build_labeled_mesh
from src.analysis.solver import run_magnetostatic_analysis


@pytest.mark.parametrize("currents, expect_torque_positive", [
    ({}, False),  # open-circuit – cogging torque ~0
    ({"A": 50.0, "B": -25.0, "C": -25.0}, True),  # loaded
])
def test_solver_basic(currents, expect_torque_positive):
    """Run the magnetostatic solver and perform basic sanity checks."""
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build machine & mesh --------------------------------------------------
        mach = create_machine("IPMSM_V_Shape")
        msh_path = build_labeled_mesh(filename=tmp / "mesh.msh", n_pole_pairs=2, element_size=mach.stator.Rext * 0.02)

        # Import mesh into FEniCSx --------------------------------------------
        domain_mesh, cell_tags, facet_tags = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, 0, gdim=2)

        gmsh.initialize()
        gmsh.open(str(msh_path))
        tag_to_name = {tag: gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups()}
        gmsh.finalize()

        # Solve ----------------------------------------------------------------
        res = run_magnetostatic_analysis(
            mach,
            domain_mesh,
            cell_tags,
            facet_tags,
            tag_to_name,
            winding_currents=currents,
            compute_torque=True,
        )

        max_B = float(np.max(res["B_mag"].x.array))
        torque = float(res.get("torque", 0.0))

        # Sanity checks --------------------------------------------------------
        assert 0.1 < max_B < 2.0, f"Unrealistic B field {max_B} T"

        if expect_torque_positive:
            assert abs(torque) > 1e-3, "Loaded case should yield non-zero torque"
        else:
            # cogging torque may not be exactly zero but should be small
            assert abs(torque) < 5.0, f"Unexpected large cogging torque {torque}" 