import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry.motor_components import (Stator, Rotor, PermanentMagnet, 
                                         Winding, Point2D, MotorGeometry)

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
    pts, elems = motor.get_numpy_geometry()

    fig, ax = plt.subplots(figsize=(6, 6))
    for tri in elems:
        polygon = pts[tri]
        poly = plt.Polygon(polygon, edgecolor='k', facecolor='none', lw=0.7)
        ax.add_patch(poly)
    ax.plot(pts[:,0], pts[:,1], 'o', ms=3, color='red', label='Nodes')
    ax.set_aspect('equal')
    ax.set_title('Motor Geometry (Test Circle)')
    ax.legend()

    # Plot stator slots
    slot_polys = stator.generate_slot_polygons()
    for slot in slot_polys:
        slot_xy = np.array([[p.x, p.y] for p in slot])
        poly = plt.Polygon(slot_xy, edgecolor='blue', facecolor='cyan', alpha=0.3, lw=1.0)
        ax.add_patch(poly)

    plt.show()

if __name__ == "__main__":
    main() 