import tempfile
from pathlib import Path

import meshio
import numpy as np

from src.core.models import MotorParameters
from src.core.regions import Region
from src.geometry.builder import build_geometry


def _build_temp_mesh(model: MotorParameters) -> Path:
    """Builds a temporary 2-D mesh file and returns its ``Path``."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        mesh_path = Path(tmp.name)
    # Build mesh and write to path
    success = build_geometry(model, mesh_2d=True, save_path=str(mesh_path))
    assert success is not None, "Geometry builder returned None – invalid motor model?"
    return mesh_path


def test_physical_groups_presence_and_balance():
    """Ensure all expected physical groups exist and are reasonably balanced.

    The test builds a full 2-D mesh, loads it via *meshio*, and inspects the
    ``field_data`` / ``cell_data`` mappings that Gmsh writes to the ``.msh``
    file.  It asserts that:

    1. Every Region defined in ``src.core.regions`` is present in the mesh.
    2. Each group owns at least one triangular cell (guards against empty tags).
    3. The three phase slot groups contain roughly the same number of cells –
       differences greater than 1 indicate a tagging error.
    """

    model = MotorParameters()
    mesh_path = _build_temp_mesh(model)

    try:
        mesh = meshio.read(mesh_path)
    finally:
        mesh_path.unlink(missing_ok=True)

    # --- 1. Presence of groups ------------------------------------------------
    field_data = mesh.field_data  # {name: [id, topological_dim]}
    expected = [
        Region.STATOR_STEEL.value,
        Region.ROTOR_STEEL.value,
        Region.MAGNETS.value,

        Region.PHASE_A.value,
        Region.PHASE_B.value,
        Region.PHASE_C.value,
    ]

    for name in expected:
        assert name in field_data, f"Missing physical group '{name}' in mesh field_data."

    # Retrieve triangle physical IDs once to avoid repeated dict look-ups
    tri_phys = mesh.cell_data_dict["gmsh:physical"]["triangle"]

    # --- 2. Each group owns at least one triangle ----------------------------
    for name in expected:
        phys_id = field_data[name][0]
        count = int(np.count_nonzero(tri_phys == phys_id))
        assert count > 0, f"Group '{name}' has no associated triangle cells (count = 0)."

    # --- 3. Balance of phase slot groups -------------------------------------
    counts = []
    for name in (Region.PHASE_A.value, Region.PHASE_B.value, Region.PHASE_C.value):
        phys_id = field_data[name][0]
        counts.append(int(np.count_nonzero(tri_phys == phys_id)))

    diff = max(counts) - min(counts)
    assert diff <= 1, (
        "Phase slot cell counts are imbalanced (A, B, C) = "
        f"{counts}. Expected difference ≤1."
    ) 