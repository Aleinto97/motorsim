#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.motor_components import (Stator, Rotor, PermanentMagnet, 
                                         Winding, Point2D, MotorGeometry)
from src.materials.material_properties import MaterialLibrary
from src.mesh.mesh_generator import MeshGenerator
from src.core.fem_solver import FEMSolver
from src.optimization.parameter_optimizer import MotorOptimizer

def create_simple_pmsm():
    """Create a simple PMSM geometry"""
    # Create stator
    stator = Stator(
        outer_radius=50.0,
        inner_radius=30.0,
        num_slots=12,
        slot_opening=3.0,
        tooth_width=5.0,
        back_iron_thickness=5.0
    )
    
    # Create rotor
    rotor = Rotor(
        outer_radius=29.0,
        inner_radius=10.0,
        num_poles=4,
        magnet_type='SPM',
        magnet_thickness=3.0,
        magnet_width=15.0
    )
    
    # Create magnets
    magnets = []
    for i in range(rotor.num_poles):
        angle = i * (2 * np.pi / rotor.num_poles)
        magnet = PermanentMagnet(
            position=Point2D(
                (rotor.outer_radius - rotor.magnet_thickness/2) * np.cos(angle),
                (rotor.outer_radius - rotor.magnet_thickness/2) * np.sin(angle)
            ),
            width=rotor.magnet_width,
            thickness=rotor.magnet_thickness,
            angle=angle,
            magnetization=1.2  # Tesla
        )
        magnets.append(magnet)
    
    # Create windings
    windings = []
    for i in range(stator.num_slots):
        angle = i * (2 * np.pi / stator.num_slots)
        winding = Winding(
            slot_number=i,
            turns=50,
            current=5.0,
            position=Point2D(
                (stator.inner_radius + 2.0) * np.cos(angle),
                (stator.inner_radius + 2.0) * np.sin(angle)
            ),
            width=3.0,
            height=5.0
        )
        windings.append(winding)
    
    # Create motor geometry
    motor = MotorGeometry()
    motor.add_stator(stator)
    motor.add_rotor(rotor)
    for magnet in magnets:
        motor.add_magnet(magnet)
    for winding in windings:
        motor.add_winding(winding)
    
    return motor

def main():
    # Create motor geometry
    motor = create_simple_pmsm()
    
    # Initialize material library
    materials = MaterialLibrary()
    
    # Create mesh
    mesh_gen = MeshGenerator()
    nodes, elements, material_regions = mesh_gen.generate_mesh(
        points=motor.generate_geometry()[0],
        elements=motor.generate_geometry()[1],
        material_regions={
            'stator': [0],  # Example material regions
            'rotor': [1],
            'magnet': [2],
            'winding': [3],
            'air': [4]
        },
        mesh_size=1.0
    )
    
    # Initialize FEM solver
    solver = FEMSolver()
    solver.set_mesh(nodes, elements, material_regions)
    
    # Set material properties
    solver.set_material(0, materials.get_material('M19 Steel'))
    solver.set_material(1, materials.get_material('M19 Steel'))
    solver.set_material(2, materials.get_material('N35 Magnet'))
    solver.set_material(3, materials.get_material('Air'))
    solver.set_material(4, materials.get_material('Air'))
    
    # Solve
    solution = solver.solve()
    
    # Calculate field
    B = solver.calculate_field()
    
    # Print results
    print(f"Solution shape: {solution.shape}")
    print(f"Field shape: {B.shape}")
    print(f"Max field magnitude: {np.max(np.linalg.norm(B, axis=1)):.2f} T")
    
    # Example optimization
    optimizer = MotorOptimizer(solver)
    optimizer.setup_default_optimization()
    
    # Run optimization
    results = optimizer.optimize(n_calls=20)
    
    print("\nOptimization results:")
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best objective: {-results['best_objective']:.2f} Nm")

if __name__ == "__main__":
    main() 