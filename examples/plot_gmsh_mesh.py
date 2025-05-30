import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.motor_components import Stator, MotorGeometry
from src.mesh.mesh_generator import MeshGenerator

def main():
    # Create a simple PMSM geometry as before
    stator = Stator(
        outer_radius=50.0,
        inner_radius=30.0,
        num_slots=12,
        slot_opening=3.0,
        tooth_width=5.0,
        back_iron_thickness=5.0
    )
    motor = MotorGeometry()
    motor.add_stator(stator)
    points, elements = motor.generate_geometry()
    # Dummy material regions for now
    material_regions = {'stator': [0]}
    mesh_gen = MeshGenerator()
    nodes, tris, _ = mesh_gen.generate_mesh(points, elements, material_regions, mesh_size=2.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    for tri in tris:
        polygon = nodes[tri]
        poly = plt.Polygon(polygon, edgecolor='k', facecolor='none', lw=0.7)
        ax.add_patch(poly)
    ax.plot(nodes[:,0], nodes[:,1], 'o', ms=2, color='red', label='Mesh Nodes')
    ax.set_aspect('equal')
    ax.set_title('GMSH Mesh (Test Circle)')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main() 