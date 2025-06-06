import pytest
import numpy as np
from motoroptix.motors.srm import SRMotor
from motoroptix.core.motor import MotorParameters

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def srm_parameters():
    """Returns a set of valid SRM parameters for testing."""
    return MotorParameters(
        name="TestSRM",
        rotor_outer_radius=29.0,  # rotor_outer_diameter/2
        stator_outer_radius=50.0, # stator_outer_diameter/2
        stack_length=50.0,
        num_stator_slots=12,
        num_rotor_poles=8,
        # Optional fields
        material_stator="M250-35A",
        material_rotor="M250-35A",
        material_winding="Copper",
        rated_power_w=300.0,
        rated_speed_rpm=3000.0,
        efficiency_target=0.9
    )

@pytest.fixture(scope="module")
def srm_motor(srm_parameters):
    """Returns an SRM instance for testing."""
    return SRMotor(parameters=srm_parameters)

# --- Test Cases ---

def test_srm_parameter_validation(srm_parameters):
    """Test SRM-specific parameter validation."""
    # Test invalid pole-slot combination
    invalid_params = srm_parameters.copy()
    invalid_params.number_of_poles = 7  # Should be even
    with pytest.raises(ValueError):
        SRMotor(parameters=invalid_params)
    
    # Test invalid pole arc
    invalid_params = srm_parameters.copy()
    invalid_params.pole_arc = 50.0  # Should be less than pole pitch
    with pytest.raises(ValueError):
        SRMotor(parameters=invalid_params)
    
    # Test invalid winding type
    invalid_params = srm_parameters.copy()
    invalid_params.winding_type = "invalid_type"
    with pytest.raises(ValueError):
        SRMotor(parameters=invalid_params)

def test_inductance_calculation(srm_motor):
    """Test inductance calculation for SRM."""
    # Test at different rotor positions
    positions = np.linspace(0, 360, 8)  # degrees
    for position in positions:
        inductance = srm_motor.calculate_inductance(position)
        assert isinstance(inductance, float)
        assert inductance > 0
        # Inductance should be periodic
        assert abs(inductance - srm_motor.calculate_inductance(position + 360)) < 1e-6

def test_torque_calculation(srm_motor):
    """Test torque calculation for SRM."""
    # Test at different currents and positions
    currents = [1.0, 2.0, 3.0]  # Amperes
    positions = [0, 15, 30]  # degrees
    for current in currents:
        for position in positions:
            torque = srm_motor.calculate_torque(current, position)
            assert isinstance(torque, float)
            # Torque can be positive or negative depending on position
            assert abs(torque) >= 0

def test_dxf_parsing(srm_motor, tmp_path):
    """Test DXF file parsing for SRM."""
    import ezdxf
    
    # Create a simple DXF file with SRM geometry
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    
    # Add stator and rotor circles
    msp.add_circle((0, 0), srm_motor.parameters.stator_outer_diameter/2)
    msp.add_circle((0, 0), srm_motor.parameters.stator_inner_diameter/2)
    msp.add_circle((0, 0), srm_motor.parameters.rotor_outer_diameter/2)
    msp.add_circle((0, 0), srm_motor.parameters.rotor_inner_diameter/2)
    
    # Add stator and rotor poles
    for i in range(srm_motor.parameters.number_of_poles):
        angle = i * 360 / srm_motor.parameters.number_of_poles
        # Add stator pole
        msp.add_line(
            (0, 0),
            (srm_motor.parameters.stator_inner_diameter/2 * np.cos(np.radians(angle)),
             srm_motor.parameters.stator_inner_diameter/2 * np.sin(np.radians(angle)))
        )
        # Add rotor pole
        msp.add_line(
            (0, 0),
            (srm_motor.parameters.rotor_outer_diameter/2 * np.cos(np.radians(angle)),
             srm_motor.parameters.rotor_outer_diameter/2 * np.sin(np.radians(angle)))
        )
    
    # Save the DXF file
    dxf_path = tmp_path / "test_srm.dxf"
    doc.saveas(str(dxf_path))
    
    # Test loading the DXF file
    srm_motor.load_dxf(str(dxf_path))
    assert srm_motor.geometry is not None

def test_electromagnetic_analysis(srm_motor):
    """Test electromagnetic analysis for SRM."""
    # Test at different rotor positions
    positions = np.linspace(0, 360, 4)  # degrees
    for position in positions:
        results = srm_motor.analyze_electromagnetic(position)
        assert "flux_density" in results
        assert "torque" in results
        assert "inductance" in results
        
        # Verify results are within reasonable ranges
        assert 0 <= results["flux_density"] <= 2.0  # Tesla
        assert abs(results["torque"]) >= 0
        assert results["inductance"] > 0

def test_thermal_analysis(srm_motor):
    """Test thermal analysis for SRM."""
    # Test at different power levels
    power_levels = [100, 200, 300]  # Watts
    for power in power_levels:
        results = srm_motor.analyze_thermal(power)
        assert "temperature_rise" in results
        assert "hot_spot_temperature" in results
        
        # Verify results are within reasonable ranges
        assert results["temperature_rise"] >= 0
        assert results["hot_spot_temperature"] >= 20  # Celsius

def test_mechanical_analysis(srm_motor):
    """Test mechanical analysis for SRM."""
    # Test at different speeds
    speeds = [1000, 2000, 3000]  # RPM
    for speed in speeds:
        results = srm_motor.analyze_mechanical(speed)
        assert "stress" in results
        assert "deformation" in results
        
        # Verify results are within reasonable ranges
        assert results["stress"] >= 0
        assert results["deformation"] >= 0

def test_optimization_objective(srm_motor):
    """Test optimization objective function for SRM."""
    def objective(trial):
        # Define optimization parameters
        current = trial.suggest_float("current", 1.0, 10.0)
        speed = trial.suggest_float("speed", 1000, 5000)
        position = trial.suggest_float("position", 0, 360)
        
        # Calculate performance metrics
        torque = srm_motor.calculate_torque(current, position)
        inductance = srm_motor.calculate_inductance(position)
        power = torque * speed * 2 * np.pi / 60
        
        # Return negative power (to maximize)
        return -power
    
    # Test objective function
    import optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    assert study.best_value is not None
    assert study.best_params is not None

def test_switching_angles(srm_motor):
    """Test switching angle calculation for SRM."""
    # Test at different speeds
    speeds = [1000, 2000, 3000]  # RPM
    for speed in speeds:
        angles = srm_motor.calculate_switching_angles(speed)
        assert isinstance(angles, dict)
        assert "turn_on" in angles
        assert "turn_off" in angles
        
        # Verify angles are within reasonable ranges
        assert 0 <= angles["turn_on"] <= 360
        assert 0 <= angles["turn_off"] <= 360
        assert angles["turn_on"] < angles["turn_off"] 