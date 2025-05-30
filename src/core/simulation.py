import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Dict
from geometry.motor_components import MotorGeometry, Stator, Rotor, PermanentMagnet, Winding

class SimulationEngine:
    def __init__(self):
        self.mesh_size = 0.5  # mm
        self.mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
        self.materials = {
            'air': {'mu_r': 1.0, 'sigma': 0.0},
            'iron': {'mu_r': 1000.0, 'sigma': 1e6},
            'copper': {'mu_r': 1.0, 'sigma': 5.8e7},
            'magnet': {'mu_r': 1.05, 'sigma': 0.0, 'Br': 1.2}  # N35 magnet properties
        }
        
    def generate_mesh(self, geometry: MotorGeometry) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a triangular mesh for the motor geometry"""
        # This is a simplified mesh generation - in reality, you'd use a proper meshing library
        x = np.arange(-geometry.stator.outer_radius, geometry.stator.outer_radius, self.mesh_size)
        y = np.arange(-geometry.stator.outer_radius, geometry.stator.outer_radius, self.mesh_size)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.flatten(), Y.flatten()))
        
        # Filter points inside the stator
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        mask = r <= geometry.stator.outer_radius
        points = points[mask]
        
        # Generate simple triangular elements
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        elements = tri.simplices
        
        return points, elements
    
    def solve_magnetostatic(self, geometry: MotorGeometry, current_density: float = 0.0) -> np.ndarray:
        """Solve the magnetostatic problem using finite element method"""
        points, elements = self.generate_mesh(geometry)
        
        # Assemble stiffness matrix
        n_points = len(points)
        n_elements = len(elements)
        
        # Initialize sparse matrix
        from scipy.sparse import lil_matrix
        K = lil_matrix((n_points, n_points))
        F = np.zeros(n_points)
        
        # Element-wise assembly
        for e in range(n_elements):
            element = elements[e]
            element_points = points[element]
            
            # Calculate element matrix
            Ke = self._calculate_element_matrix(element_points)
            
            # Add to global matrix
            for i in range(3):
                for j in range(3):
                    K[element[i], element[j]] += Ke[i, j]
        
        # Convert to CSC format for efficient solving
        K = K.tocsc()
        
        # Solve system
        A = spsolve(K, F)
        
        return A
    
    def _calculate_element_matrix(self, element_points: np.ndarray) -> np.ndarray:
        """Calculate element stiffness matrix"""
        # This is a simplified version - in reality, you'd need proper FEM calculations
        area = 0.5 * np.abs(np.cross(element_points[1] - element_points[0],
                                    element_points[2] - element_points[0]))
        return np.eye(3) * area
    
    def calculate_torque(self, geometry: MotorGeometry, angle: float) -> float:
        """Calculate electromagnetic torque at a given rotor angle"""
        # This is a simplified torque calculation
        # In reality, you'd need to integrate Maxwell stress tensor
        if not geometry.rotor:
            return 0.0
            
        # Calculate magnetic field
        A = self.solve_magnetostatic(geometry)
        
        # Simplified torque calculation
        torque = 0.0
        if geometry.rotor and geometry.stator:
            # Basic torque calculation based on rotor-stator interaction
            torque = 0.5 * self.mu0 * (geometry.rotor.outer_radius**2 - 
                                      geometry.rotor.inner_radius**2) * np.sin(angle)
        
        return torque
    
    def calculate_flux_linkage(self, geometry: MotorGeometry, winding: Winding) -> float:
        """Calculate flux linkage for a given winding"""
        # This is a simplified flux linkage calculation
        A = self.solve_magnetostatic(geometry)
        
        # Calculate flux through winding area
        flux = 0.0
        if geometry.stator:
            # Basic flux calculation
            flux = self.mu0 * winding.turns * winding.current * winding.width * winding.height
        
        return flux
    
    def calculate_inductance(self, geometry: MotorGeometry, winding1: Winding, winding2: Winding) -> float:
        """Calculate mutual inductance between two windings"""
        # This is a simplified inductance calculation
        flux1 = self.calculate_flux_linkage(geometry, winding1)
        flux2 = self.calculate_flux_linkage(geometry, winding2)
        
        # Basic mutual inductance calculation
        L = flux1 * flux2 / (winding1.current * winding2.current)
        
        return L 