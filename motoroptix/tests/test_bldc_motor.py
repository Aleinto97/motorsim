import pytest
import numpy as np
from motoroptix.motors.bldc import BLDCMotor
from motoroptix.core.motor import MotorParameters

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def bldc_parameters():
    """Returns a set of valid BLDC motor parameters for testing."""
    return MotorParameters(
        name="TestBLDC",
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
def bldc_motor(bldc_parameters):
    """Returns a BLDC motor instance for testing."""
    return BLDCMotor(parameters=bldc_parameters)

# --- Test Cases ---

def test_bldc_parameter_validation(bldc_parameters):
    """Test BLDC-specific parameter validation."""
    # Test invalid pole-slot combination
    invalid_params = bldc_parameters.copy()
    invalid_params.number_of_poles = 7  # Should be even
    with pytest.raises(ValueError):
        BLDCMotor(parameters=invalid_params)
    
    # Test invalid winding type
    invalid_params = bldc_parameters.copy()
    invalid_params.winding_type = "invalid_type"
    with pytest.raises(ValueError):
        BLDCMotor(parameters=invalid_params)

def test_back_emf_calculation(bldc_motor):
    """Test back-EMF calculation for BLDC motor."""
    # Test at different speeds
    speeds = [1000, 2000, 3000]  # RPM
    for speed in speeds:
        back_emf = bldc_motor.calculate_back_emf(speed)
        assert isinstance(back_emf, float)
        assert back_emf > 0
        # Back-EMF should be proportional to speed
        assert abs(back_emf / speed - back_emf / speeds[0]) < 1e-6

def test_torque_calculation(bldc_motor):
    """Test torque calculation for BLDC motor."""
    # Test at different currents
    currents = [1.0, 2.0, 3.0]  # Amperes
    for current in currents:
        torque = bldc_motor.calculate_torque(current)
        assert isinstance(torque, float)
        assert torque > 0
        # Torque should be proportional to current
        assert abs(torque / current - torque / currents[0]) < 1e-6

def test_dxf_parsing(bldc_motor, tmp_path):
    """Test DXF file parsing for BLDC motor."""
    import ezdxf
    
    # Create a simple DXF file with BLDC motor geometry
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    
    # Add stator and rotor circles
    msp.add_circle((0, 0), bldc_motor.parameters.stator_outer_diameter/2)
    msp.add_circle((0, 0), bldc_motor.parameters.stator_inner_diameter/2)
    msp.add_circle((0, 0), bldc_motor.parameters.rotor_outer_diameter/2)
    msp.add_circle((0, 0), bldc_motor.parameters.rotor_inner_diameter/2)
    
    # Save the DXF file
    dxf_path = tmp_path / "test_bldc.dxf"
    doc.saveas(str(dxf_path))
    
    # Test loading the DXF file
    bldc_motor.load_dxf(str(dxf_path))
    assert bldc_motor.geometry is not None

def test_electromagnetic_analysis(bldc_motor):
    """Test electromagnetic analysis for BLDC motor."""
    # Test at different rotor positions
    positions = np.linspace(0, 360, 4)  # degrees
    for position in positions:
        results = bldc_motor.analyze_electromagnetic(position)
        assert "flux_density" in results
        assert "torque" in results
        assert "back_emf" in results
        
        # Verify results are within reasonable ranges
        assert 0 <= results["flux_density"] <= 2.0  # Tesla
        assert results["torque"] >= 0
        assert results["back_emf"] >= 0

def test_thermal_analysis(bldc_motor):
    """Test thermal analysis for BLDC motor."""
    # Test at different power levels
    power_levels = [100, 200, 300]  # Watts
    for power in power_levels:
        results = bldc_motor.analyze_thermal(power)
        assert "temperature_rise" in results
        assert "hot_spot_temperature" in results
        
        # Verify results are within reasonable ranges
        assert results["temperature_rise"] >= 0
        assert results["hot_spot_temperature"] >= 20  # Celsius

def test_mechanical_analysis(bldc_motor):
    """Test mechanical analysis for BLDC motor."""
    # Test at different speeds
    speeds = [1000, 2000, 3000]  # RPM
    for speed in speeds:
        results = bldc_motor.analyze_mechanical(speed)
        assert "stress" in results
        assert "deformation" in results
        
        # Verify results are within reasonable ranges
        assert results["stress"] >= 0
        assert results["deformation"] >= 0

def test_optimization_objective(bldc_motor):
    """Test optimization objective function for BLDC motor."""
    def objective(trial):
        # Define optimization parameters
        current = trial.suggest_float("current", 1.0, 10.0)
        speed = trial.suggest_float("speed", 1000, 5000)
        
        # Calculate performance metrics
        torque = bldc_motor.calculate_torque(current)
        back_emf = bldc_motor.calculate_back_emf(speed)
        power = torque * speed * 2 * np.pi / 60
        
        # Return negative power (to maximize)
        return -power
    
    # Test objective function
    import optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    
    assert study.best_value is not None
    assert study.best_params is not None 