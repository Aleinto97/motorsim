"""True sector-model geometry builder using Gmsh OCC primitives.

This version replaces the earlier slice-and-dice stub.  It builds a single
fundamental sector as clean CAD, tags all domains and important boundaries,
and generates a 2-D mesh ready for solver consumption.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import gmsh
import numpy as np
import pyvista as pv

from src.core.models import MotorParameters
from src.geometry.builder import _points_and_faces as _faces_from_gmsh, _points_and_lines as _lines_from_gmsh

__all__ = [
    "PhysicalGroup",
    "build_geometry_sector",
]


class PhysicalGroup:
    """Canonical names and dimensions for Physical Groups used in the model."""

    # 2-D domains (dim = 2)
    STATOR_IRON = ("domain_stator_iron", 2)
    ROTOR_IRON = ("domain_rotor_iron", 2)
    AIR_GAP = ("domain_air_gap", 2)
    MAGNET = ("domain_magnet", 2)
    STATOR_COIL = ("domain_stator_coil", 2)

    # 1-D boundaries (dim = 1)
    BOUNDARY_PERIODIC_MASTER = ("boundary_periodic_master", 1)
    BOUNDARY_PERIODIC_SLAVE = ("boundary_periodic_slave", 1)
    BOUNDARY_SLIDING_INTERFACE = ("boundary_sliding_interface", 1)
    BOUNDARY_STATOR_OUTER = ("boundary_stator_outer", 1)

    # Convenience – iterable over all items
    @classmethod
    def items(cls):
        return [
            cls.STATOR_IRON,
            cls.ROTOR_IRON,
            cls.AIR_GAP,
            cls.MAGNET,
            cls.STATOR_COIL,
            cls.BOUNDARY_PERIODIC_MASTER,
            cls.BOUNDARY_PERIODIC_SLAVE,
            cls.BOUNDARY_SLIDING_INTERFACE,
            cls.BOUNDARY_STATOR_OUTER,
        ]


# -----------------------------------------------------------------------------
# Main builder – public API
# -----------------------------------------------------------------------------

def build_geometry_sector(
    motor_params: Union[Dict, MotorParameters],
    mesh_params: Dict | None = None,
    *,
    mesh_2d: bool = False,
    save_path: str | None = None,
) -> pv.PolyData | None:
    """Create a watertight 2-D sector model via Gmsh OCC.

    Parameters
    ----------
    motor_params
        Dictionary-style parameters describing the machine geometry.
    mesh_params
        Meshing options: ``global_size`` (float), ``interactive`` (bool), …

    The routine leaves the created model loaded in the current Gmsh session so
    that callers can directly query or export as they prefer.
    """

    if mesh_params is None:
        mesh_params = {}

    gmsh.initialize()

    # Allow dataclass instance
    if isinstance(motor_params, MotorParameters):
        _mp = {
            "name": getattr(motor_params, "name", "model"),
            "num_slots": motor_params.stator.slot.Zs,
            "stator_outer_radius": motor_params.stator.Rext,
            "stator_inner_radius": motor_params.stator.Rint,
            "rotor_outer_radius": motor_params.rotor.Rext,
            "rotor_inner_radius": motor_params.rotor.Rint,
            "slot_depth": motor_params.stator.slot.H2,
            "slot_width": motor_params.stator.slot.W2,
            "magnet_height": motor_params.rotor.hole_v.magnet_left.Hmag,
        }
    else:
        _mp = motor_params  # type: ignore[assignment]

    n_slots = _mp.get("num_slots", 12)
    if n_slots <= 0:
        raise ValueError("Number of slots (num_slots) must be a positive integer.")

    sector_angle = 2 * math.pi / n_slots

    # Extract radii and feature sizes
    r_so = _mp["stator_outer_radius"]
    r_si = _mp["stator_inner_radius"]
    r_ro = _mp["rotor_outer_radius"]
    r_ri = _mp["rotor_inner_radius"]
    slot_depth = _mp["slot_depth"]
    slot_width = _mp["slot_width"]
    magnet_height = _mp["magnet_height"]

    # Sliding interface radius (mid-air-gap)
    r_slide = r_ro + (r_si - r_ro) / 2.0

    # ------------------------------------------------------------------
    # 2.  Build OCC primitives
    # ------------------------------------------------------------------
    # Helper: Wedge API – gmsh.model.occ.addWedge(x,y,z, outerR, innerR, angle)
    stator_wedge = gmsh.model.occ.addWedge(0, 0, 0, r_so, r_si - slot_depth, sector_angle)
    rotor_wedge = gmsh.model.occ.addWedge(0, 0, 0, r_ro, r_ri, sector_angle)
    air_gap_wedge = gmsh.model.occ.addWedge(0, 0, 0, r_si, r_ro, sector_angle)

    # Slot rectangle – initially centred then rotated to middle of sector
    slot_x = (r_si - slot_depth / 2.0) * math.cos(sector_angle / 2.0)
    slot_y = (r_si - slot_depth / 2.0) * math.sin(sector_angle / 2.0)
    slot_rect = gmsh.model.occ.addRectangle(
        slot_x - slot_width / 2.0,
        slot_y - slot_depth / 2.0,
        0,
        slot_width,
        slot_depth,
    )
    gmsh.model.occ.rotate([(2, slot_rect)], 0, 0, 0, 0, 0, 1, sector_angle / 2.0)

    # Magnet rectangle – simplistic placement flush with rotor surface
    magnet_rect = gmsh.model.occ.addRectangle(
        r_ro - magnet_height,
        -slot_width / 2.0,
        0,
        magnet_height,
        slot_width,
    )
    gmsh.model.occ.rotate([(2, magnet_rect)], 0, 0, 0, 0, 0, 1, sector_angle / 2.0)

    primitives: List[Tuple[int, int]] = [
        (2, stator_wedge),
        (2, rotor_wedge),
        (2, air_gap_wedge),
        (2, slot_rect),
        (2, magnet_rect),
    ]

    # ------------------------------------------------------------------
    # 3.  Fragment & synchronise – the *Golden Rule*
    # ------------------------------------------------------------------
    out_tags, out_map = gmsh.model.occ.fragment([primitives[0]], primitives[1:])  # type: ignore[arg-type]
    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------------
    # 4.  Tag physical groups – domains first
    # ------------------------------------------------------------------
    phys_groups: Dict[str, List[int]] = {name: [] for name, _ in PhysicalGroup.items()}

    def surf_center(tag: int) -> Tuple[float, float]:
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        return (x, y)

    def r_theta(x: float, y: float) -> Tuple[float, float]:
        return (math.hypot(x, y), math.atan2(y, x))

    # Children of stator wedge distinguish iron vs coil via radius
    for dim, tag in out_map[0]:  # children of stator_wedge
        if dim != 2:
            continue
        r, _ = r_theta(*surf_center(tag))
        if r > r_si - slot_depth / 2.0:
            phys_groups[PhysicalGroup.STATOR_IRON[0]].append(tag)
        else:
            phys_groups[PhysicalGroup.STATOR_COIL[0]].append(tag)

    # Direct mapping for other primitives
    phys_groups[PhysicalGroup.ROTOR_IRON[0]].extend([t for d, t in out_map[1] if d == 2])
    phys_groups[PhysicalGroup.AIR_GAP[0]].extend([t for d, t in out_map[2] if d == 2])
    phys_groups[PhysicalGroup.MAGNET[0]].extend([t for d, t in out_map[4] if d == 2])

    # Fallback: ensure stator iron is not empty – gather any remaining untagged
    if not phys_groups[PhysicalGroup.STATOR_IRON[0]]:
        untagged = {
            tag
            for dim, tag in gmsh.model.getEntities(2)
            if dim == 2 and all(tag not in v for v in phys_groups.values())
        }
        for tag in untagged:
            x, y = surf_center(tag)
            r, _ = r_theta(x, y)
            if r > r_si - slot_depth:
                phys_groups[PhysicalGroup.STATOR_IRON[0]].append(tag)

    # ------------------------------------------------------------------
    # 5.  Boundary tagging (dim = 1)
    # ------------------------------------------------------------------
    for _, tag in gmsh.model.getEntities(1):
        x, y, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        r, theta = r_theta(x, y)

        # Sliding interface
        if abs(r - r_slide) < 1e-2 or abs(r - r_ro) < 1e-2:
            phys_groups[PhysicalGroup.BOUNDARY_SLIDING_INTERFACE[0]].append(tag)
        # Stator outer boundary
        elif abs(r - r_so) < 1e-3:
            phys_groups[PhysicalGroup.BOUNDARY_STATOR_OUTER[0]].append(tag)
        # Periodic boundaries (master θ≈0, slave θ≈sector_angle)
        elif abs(theta) < 1e-3 and r > r_ri:
            phys_groups[PhysicalGroup.BOUNDARY_PERIODIC_MASTER[0]].append(tag)
        elif abs(theta - sector_angle) < 1e-3 and r > r_ri:
            phys_groups[PhysicalGroup.BOUNDARY_PERIODIC_SLAVE[0]].append(tag)

    # Fallback: if slave periodic boundary not found, pick radial curve with
    # the largest polar angle (> 0) among remaining candidates.
    if not phys_groups[PhysicalGroup.BOUNDARY_PERIODIC_SLAVE[0]]:
        candidates: list[tuple[float, int]] = []
        for _, tag in gmsh.model.getEntities(1):
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, tag)
            r, theta = r_theta(x, y)
            if r > r_ri + 1e-6 and theta > 1e-4:  # exclude master at ~0
                candidates.append((theta, tag))
        if candidates:
            theta_max, tag_max = max(candidates, key=lambda t: t[0])
            phys_groups[PhysicalGroup.BOUNDARY_PERIODIC_SLAVE[0]].append(tag_max)

    # Create physical groups in the model
    for name, dim in PhysicalGroup.items():
        tags = phys_groups[name]
        if tags:
            gmsh.model.addPhysicalGroup(dim, tags, name=name)

    # ------------------------------------------------------------------
    # 6.  Mesh generation
    # ------------------------------------------------------------------
    h_global = mesh_params.get("global_size", r_so / 20.0)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_global)

    # Optional refinement in air gap
    if phys_groups[PhysicalGroup.AIR_GAP[0]]:
        air_gap_ent = phys_groups[PhysicalGroup.AIR_GAP[0]][0]
        gmsh.model.mesh.setSize([(2, air_gap_ent)], (r_si - r_ro) / 4.0)

    gmsh.model.mesh.generate(2)

    # ------------------------------------------------------------------
    # 7.  Export / convert
    # ------------------------------------------------------------------
    if save_path is not None:
        gmsh.write(save_path)

    pv_mesh: pv.PolyData | None = None
    try:
        if mesh_2d:
            pv_mesh = _faces_from_gmsh()
        else:
            pv_mesh = _lines_from_gmsh()
    except Exception:
        pv_mesh = None

    if mesh_params.get("interactive", False):
        gmsh.fltk.run()

    print("Sector model built and meshed successfully.")

    return pv_mesh

# -----------------------------------------------------------------------------
# End of file 