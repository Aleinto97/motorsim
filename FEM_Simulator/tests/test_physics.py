import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.models import MotorParameters
from src.geometry.builder import build_geometry
from src.analysis.solver import run_analysis


@pytest.fixture(scope="module")
def motor_mesh_path() -> Path:
    """Generate a temporary 2-D mesh once for all physics tests."""
    model = MotorParameters()
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        mesh_file = Path(tmp.name)
    # Build and save mesh
    build_geometry(model, mesh_2d=True, save_path=str(mesh_file))
    yield mesh_file
    # Cleanup
    mesh_file.unlink(missing_ok=True)


def test_physics_sanity(motor_mesh_path):
    """Run open-circuit and loaded simulations, compare B-field magnitudes."""
    # 1. Open-circuit (magnets only)
    res_open = run_analysis(motor_mesh_path, phase_currents={})
    assert res_open is not None, "Open-circuit analysis failed."
    B_open = res_open["B"].x.array.reshape(-1, 2)
    max_B_open = np.max(np.linalg.norm(B_open, axis=1))

    print(f"\nMax |B| (Open-Circuit): {max_B_open:.2f} T")
    assert 1.0 < max_B_open < 2.0, "Open-circuit B-field out of expected range."

    # 2. Loaded case with phase currents
    loaded_currents = {"A": 50.0, "B": -25.0, "C": -25.0}
    res_loaded = run_analysis(motor_mesh_path, phase_currents=loaded_currents)
    assert res_loaded is not None, "Loaded analysis failed."
    B_loaded = res_loaded["B"].x.array.reshape(-1, 2)
    max_B_loaded = np.max(np.linalg.norm(B_loaded, axis=1))

    print(f"Max |B| (Loaded): {max_B_loaded:.2f} T")
    assert max_B_loaded < 3.0, "Loaded B-field unphysically high (deep saturation)."

    # 3. Compare
    assert max_B_loaded > max_B_open, "Loaded B-field should exceed open-circuit field."

    print("✅ Physics sanity checks passed.")
