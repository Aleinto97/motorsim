import tempfile
from pathlib import Path

from src.core.models import MotorParameters
from src.geometry.builder_sector import build_geometry_sector


def test_sector_builder_runs():
    """Prototype smoke-test: sector builder returns a non-empty mesh."""
    model = MotorParameters()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        mesh_path = Path(tmp.name)

    mesh = build_geometry_sector(model, mesh_2d=True, save_path=str(mesh_path))
    mesh_path.unlink(missing_ok=True)

    assert mesh is not None and mesh.n_points > 0, "Sector builder returned empty mesh." 