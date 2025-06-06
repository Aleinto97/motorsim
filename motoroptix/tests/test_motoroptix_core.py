import pytest
import os
import shutil
import tempfile
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import classes from your new modules
from motoroptix.core.motor import MotorParameters, Motor
from motoroptix.geometry.dxf_parser import DXFParser
from motoroptix.fea.electromagnetic_solver import ElectromagneticSolver
from motoroptix.optimization.optimizer import MotorOptimizer

# --- Test Data Setup ---
# Create a dummy DXF file for testing DXFParser
@pytest.fixture(scope="module")
def dummy_dxf_file():
    """Creates a temporary dummy DXF file for testing."""
    import ezdxf

    temp_dir = tempfile.mkdtemp()
    dxf_path = os.path.join(temp_dir, "dummy_motor.dxf")

    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    # Add a rotor circle
    msp.add_circle((0, 0), radius=0.02, dxfattribs={'layer': 'ROTOR'})
    # Add a stator outer circle
    msp.add_circle((0, 0), radius=0.04, dxfattribs={'layer': 'STATOR_OUTER'})
    # Add some text parameters
    msp.add_text("STACK_LENGTH=0.05", dxfattribs={'insert': (0.1, 0.01), 'height': 0.005})
    msp.add_text("NUM_SLOTS=12", dxfattribs={'insert': (0.1, 0.015), 'height': 0.005})

    doc.saveas(dxf_path)
    logger.info(f"Created dummy DXF file at: {dxf_path}")
    yield dxf_path
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up dummy DXF directory: {temp_dir}")

# Create a dummy Gmsh MSH file for testing ElectromagneticSolver
@pytest.fixture(scope="module")
def dummy_gmsh_file():
    """
    Creates a temporary dummy Gmsh MSH file (minimal content) for testing.
    This is just a placeholder; a real MSH would come from Gmsh meshing a DXF.
    """
    temp_dir = tempfile.mkdtemp()
    msh_path = os.path.join(temp_dir, "dummy_mesh.msh")

    # A minimal valid Gmsh 2.2 mesh file with 3 nodes and 1 triangle
    msh_content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "domain"
$EndPhysicalNames
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
$EndNodes
$Elements
1
1 2 2 1 1 1 2 3
$EndElements
"""
    with open(msh_path, "w") as f:
        f.write(msh_content)
    logger.info(f"Created dummy Gmsh MSH file at: {msh_path}")
    yield msh_path
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up dummy MSH directory: {temp_dir}")

# --- Test Cases ---

def test_motor_parameters_instantiation():
    """Test instantiation and validation of MotorParameters."""
    params = MotorParameters(
        name="TestMotor",
        rotor_outer_radius=0.01,
        stator_outer_radius=0.03,
        stack_length=0.04,
        num_stator_slots=24,
        num_rotor_poles=16
    )
    assert params.name == "TestMotor"
    assert params.rotor_outer_radius == 0.01
    assert params.num_stator_slots == 24

    # Test Pydantic validation (e.g., negative radius)
    with pytest.raises(ValueError):
        MotorParameters(
            name="InvalidMotor",
            rotor_outer_radius=-0.01, # Invalid
            stator_outer_radius=0.03,
            stack_length=0.04,
            num_stator_slots=24,
            num_rotor_poles=16
        )

def test_motor_base_class():
    """Test instantiation of Motor base class and abstract methods."""
    params = MotorParameters(
        name="BaseMotor",
        rotor_outer_radius=0.01,
        stator_outer_radius=0.03,
        stack_length=0.04,
        num_stator_slots=24,
        num_rotor_poles=16
    )
    motor = Motor(params)
    assert motor.parameters.name == "BaseMotor"
    assert motor.geometry is None
    assert motor.fea_results == {}

    # Test that abstract methods raise NotImplementedError
    with pytest.raises(NotImplementedError):
        motor.generate_geometry()
    with pytest.raises(NotImplementedError):
        motor.perform_fea()
    with pytest.raises(NotImplementedError):
        motor.calculate_losses()
    with pytest.raises(NotImplementedError):
        motor.calculate_efficiency()

def test_dxf_parser_load_and_extract(dummy_dxf_file):
    """Test DXF file loading and parameter extraction."""
    parser = DXFParser(dummy_dxf_file)
    assert parser.doc is not None
    assert parser.modelspace is not None

    parameters = parser.extract_parameters()
    logger.info(f"Extracted parameters from DXF: {parameters}")
    assert "rotor_outer_radius" in parameters
    assert parameters["rotor_outer_radius"] == 0.02
    assert "stator_outer_radius" in parameters
    assert parameters["stator_outer_radius"] == 0.04
    assert parameters["stack_length"] == 0.05 # Extracted from text
    assert parameters["num_slots"] == 12 # Extracted from text

    # Test creating a new DXF
    temp_dir = tempfile.mkdtemp()
    output_dxf_path = os.path.join(temp_dir, "output_motor.dxf")
    parser.create_dxf_from_parameters(parameters, output_dxf_path)
    assert os.path.exists(output_dxf_path)
    shutil.rmtree(temp_dir)


def test_electromagnetic_solver_mesh_loading(dummy_gmsh_file):
    """Test loading a Gmsh mesh."""
    solver = ElectromagneticSolver()
    solver.load_mesh_from_gmsh(dummy_gmsh_file)
    assert solver.mesh is not None
    # Check if the number of cells/nodes makes sense for the dummy mesh
    assert solver.mesh.topology.index_map(solver.mesh.topology.dim).size_local > 0
    assert solver.mesh.topology.index_map(0).size_local > 0

def test_electromagnetic_solver_setup_problem(dummy_gmsh_file):
    """Test setting up the electromagnetic problem (without solving)."""
    solver = ElectromagneticSolver()
    solver.load_mesh_from_gmsh(dummy_gmsh_file)
    
    # Define some dummy material properties
    material_props = {
        "stator_core": {"mu_r": 1000},
        "rotor_core": {"mu_r": 500},
        "air": {"mu_r": 1},
        "copper": {"sigma": 5.8e7}
    }
    solver.setup_magnetic_problem(material_props)
    assert solver.function_space is not None
    assert solver.problem is not None
    # We can't actually solve without a real mesh and proper BCs, but setup should pass.

def test_motor_optimizer_basic_run():
    """Test basic optimization run."""
    # Create a simple optimization problem
    def objective_function(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return x**2 + y**2

    # Create optimizer with shorter timeout
    optimizer = MotorOptimizer(
        objective_function=objective_function,
        n_trials=5,
        timeout=5,
        study_name="test_optimization"
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Verify results
    assert len(study.trials) > 0
    assert study.best_value < 100  # Should be better than random
    
    # Test visualization without opening browser
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import plot_optimization_history
    
    fig = plot_optimization_history(study)
    plt.close(fig.figure)  # Close the figure to avoid displaying it
    
    # Test parameter importance without opening browser
    from optuna.visualization.matplotlib import plot_param_importances
    fig = plot_param_importances(study)
    plt.close(fig.figure)  # Close the figure to avoid displaying it 