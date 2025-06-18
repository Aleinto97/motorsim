import pytest
import gmsh
from src.geometry.builder_sector import build_geometry_sector, PhysicalGroup


@pytest.fixture
def default_motor_params():
    """Provides a standard set of motor parameters for testing."""
    return {
        "name": "test_motor",
        "num_slots": 12,
        "stator_outer_radius": 100,
        "stator_inner_radius": 60,
        "rotor_outer_radius": 58,
        "rotor_inner_radius": 20,
        "slot_depth": 15,
        "slot_width": 10,
        "magnet_height": 8,
    }


@pytest.fixture
def default_mesh_params():
    """Provides standard mesh parameters."""
    return {
        "global_size": 20,
        "interactive": False,  # Ensure GUI does not pop up during tests
    }


@pytest.fixture
def gmsh_setup():
    """Initializes Gmsh before a test and finalizes it after."""
    gmsh.initialize()
    yield
    gmsh.finalize()


def test_build_succeeds_and_tags_correctly(
    gmsh_setup, default_motor_params, default_mesh_params
):
    """Tests if the builder runs successfully and creates all required Physical Groups."""
    # 1. ARRANGE & ACT
    build_geometry_sector(default_motor_params, default_mesh_params)

    # 2. ASSERT – mesh exists
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    assert len(node_tags) > 0, "Mesh should have nodes after generation"

    # Retrieve physical groups and their names
    phys_groups = gmsh.model.getPhysicalGroups()
    assert phys_groups, "At least one physical group should exist"
    created_names = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in phys_groups}

    expected_names = [pg[0] for pg in PhysicalGroup.items()]
    for exp in expected_names:
        assert exp in created_names, f"Expected Physical Group '{exp}' was not created"

    # Check critical boundaries are not empty
    master_tag = gmsh.model.getPhysicalGroupsForName(
        PhysicalGroup.BOUNDARY_PERIODIC_MASTER[0]
    )[0]
    slave_tag = gmsh.model.getPhysicalGroupsForName(
        PhysicalGroup.BOUNDARY_PERIODIC_SLAVE[0]
    )[0]
    slide_tag = gmsh.model.getPhysicalGroupsForName(
        PhysicalGroup.BOUNDARY_SLIDING_INTERFACE[0]
    )[0]

    assert gmsh.model.getEntitiesForPhysicalGroup(1, master_tag), "Master periodic boundary empty"
    assert gmsh.model.getEntitiesForPhysicalGroup(1, slave_tag), "Slave periodic boundary empty"
    assert gmsh.model.getEntitiesForPhysicalGroup(1, slide_tag), "Sliding interface empty"


def test_builder_raises_error_for_invalid_slots(
    gmsh_setup, default_motor_params, default_mesh_params
):
    """Tests that the builder correctly raises a ValueError for invalid slots."""
    invalid_params = default_motor_params.copy()
    invalid_params["num_slots"] = 0

    with pytest.raises(ValueError, match="Number of slots"):
        build_geometry_sector(invalid_params, default_mesh_params) 