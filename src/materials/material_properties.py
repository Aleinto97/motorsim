from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d

@dataclass
class Material:
    name: str
    mu_r: float  # Relative permeability
    conductivity: float  # Electrical conductivity (S/m)
    density: float  # Mass density (kg/m³)
    
    def get_permeability(self, B: float) -> float:
        """Get permeability at given flux density (linear material)"""
        return self.mu_r

@dataclass
class NonLinearMaterial(Material):
    B_H_curve: List[Tuple[float, float]]  # List of (B, H) points
    
    def __post_init__(self):
        # Create interpolation function for B-H curve
        B_values, H_values = zip(*self.B_H_curve)
        self._interpolator = interp1d(B_values, H_values, 
                                    kind='linear', 
                                    bounds_error=False,
                                    fill_value=(H_values[0], H_values[-1]))
    
    def get_permeability(self, B: float) -> float:
        """Get permeability at given flux density (non-linear material)"""
        H = self._interpolator(B)
        return B / (H * 4 * np.pi * 1e-7)  # Convert to relative permeability

class MaterialLibrary:
    def __init__(self):
        self.materials = {}
        self._initialize_default_materials()
    
    def _initialize_default_materials(self):
        """Initialize default material properties"""
        # Air
        self.add_material(Material(
            name="Air",
            mu_r=1.0,
            conductivity=0.0,
            density=1.225
        ))
        
        # Electrical Steel (M19)
        self.add_material(NonLinearMaterial(
            name="M19 Steel",
            mu_r=2000.0,  # Initial permeability
            conductivity=2.17e6,
            density=7650.0,
            B_H_curve=[
                (0.0, 0.0),
                (0.5, 100),
                (1.0, 200),
                (1.5, 400),
                (2.0, 1000),
                (2.5, 5000)
            ]
        ))
        
        # Neodymium Magnet (N35)
        self.add_material(Material(
            name="N35 Magnet",
            mu_r=1.05,
            conductivity=625000,
            density=7500.0
        ))
    
    def add_material(self, material: Material):
        """Add a new material to the library"""
        self.materials[material.name] = material
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get material by name"""
        return self.materials.get(name)
    
    def list_materials(self) -> List[str]:
        """List all available materials"""
        return list(self.materials.keys()) 