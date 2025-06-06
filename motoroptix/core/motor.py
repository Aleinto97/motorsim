from pydantic import BaseModel, Field
from typing import Optional, Dict

class MotorParameters(BaseModel):
    """
    Base class for defining common motor design parameters.
    Specific motor types (BLDC, SRM) will inherit from this.
    """
    name: str = Field(..., description="Name of the motor design.")
    rotor_outer_radius: float = Field(..., gt=0, description="Rotor outer radius in meters.")
    stator_outer_radius: float = Field(..., gt=0, description="Stator outer radius in meters.")
    stack_length: float = Field(..., gt=0, description="Axial stack length in meters.")
    num_stator_slots: int = Field(..., gt=0, description="Number of stator slots.")
    num_rotor_poles: int = Field(..., gt=0, description="Number of rotor poles.")
    material_stator: str = Field("M250-35A", description="Material of the stator core.")
    material_rotor: str = Field("M250-35A", description="Material of the rotor core.")
    material_winding: str = Field("Copper", description="Material of the windings.")
    rated_power_w: Optional[float] = Field(None, gt=0, description="Rated power in Watts.")
    rated_speed_rpm: Optional[float] = Field(None, gt=0, description="Rated speed in RPM.")
    efficiency_target: Optional[float] = Field(None, gt=0, le=1, description="Target efficiency (0-1).")

    class Config:
        schema_extra = {
            "example": {
                "name": "HighSpeedBlowerMotor",
                "rotor_outer_radius": 0.02,
                "stator_outer_radius": 0.04,
                "stack_length": 0.05,
                "num_stator_slots": 12,
                "num_rotor_poles": 8,
                "rated_power_w": 1000,
                "rated_speed_rpm": 60000,
                "efficiency_target": 0.92
            }
        }

class Motor:
    """
    Base class for a generic electric motor.
    Manages common properties and provides abstract methods for analysis.
    """
    def __init__(self, parameters: MotorParameters):
        self.parameters = parameters
        self.geometry = None # Placeholder for geometric model
        self.fea_results = {} # Placeholder for FEA results

    def generate_geometry(self):
        """
        Abstract method to generate the motor's geometric model.
        Should be implemented by specific motor types.
        """
        raise NotImplementedError

    def perform_fea(self):
        """
        Abstract method to perform Finite Element Analysis on the motor.
        Should be implemented by specific motor types.
        """
        raise NotImplementedError

    def calculate_losses(self) -> Dict[str, float]:
        """
        Abstract method to calculate various motor losses.
        Returns a dictionary of loss components.
        """
        raise NotImplementedError

    def calculate_efficiency(self) -> float:
        """
        Abstract method to calculate motor efficiency.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"Motor(name='{self.parameters.name}')"
