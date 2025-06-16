"""Geometry builder capable of producing either a 1-D outline or a full 2-D
triangular mesh of the motor cross-section, including stator slots and rotor
V-shaped permanent-magnet pockets.

This version fixes two long-standing issues:
1. Boolean operations now set ``removeObject=True`` and ``removeTool=True`` so
   that only the resulting clean geometry remains, avoiding overlapping
   surfaces that confused the mesher.
2. Helper routines perform robust sanity checks and raise informative errors
   when a generated mesh contains no elements.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pyvista as pv

from src.core.models import MotorParameters

try:
    import gmsh  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Please install the `gmsh` Python module in your env.") from exc

# ----------------------------------------------------------------------------
# Helpers with improved error checking
# ----------------------------------------------------------------------------

def _points_and_lines() -> pv.PolyData:
    """Return a ``pyvista.PolyData`` built from *line* elements of current mesh."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    if not node_tags.size:
        raise RuntimeError("Mesh has no nodes.")
    points = coords.reshape(-1, 3)
    tag_to_idx = {t: i for i, t in enumerate(node_tags)}

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=1)
    if not elem_types or not elem_nodes or not elem_nodes[0].size:
        raise RuntimeError("No 1-D (line) elements were found in the mesh.")

    lines = elem_nodes[0].reshape(-1, 2)
    conn = np.vectorize(tag_to_idx.get)(lines)
    vtk_lines = np.hstack([np.full((len(conn), 1), 2), conn])
    return pv.PolyData(points, lines=vtk_lines.ravel())


def _points_and_faces() -> pv.PolyData:
    """Return a ``pyvista.PolyData`` built from *triangular face* elements."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    if not node_tags.size:
        raise RuntimeError("Mesh has no nodes.")
    points = coords.reshape(-1, 3)
    tag_to_idx = {t: i for i, t in enumerate(node_tags)}

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=2)
    if not elem_types or not elem_nodes or not elem_nodes[0].size:
        raise RuntimeError("No 2-D (face) elements were found in the mesh.")

    tris = elem_nodes[0].reshape(-1, 3)
    conn = np.vectorize(tag_to_idx.get)(tris)
    faces = np.hstack([np.full((len(conn), 1), 3), conn])
    return pv.PolyData(points, faces=faces.ravel())


# ----------------------------------------------------------------------------
# Main API
# ----------------------------------------------------------------------------

def build_geometry(model: MotorParameters, *, mesh_2d: bool = False, save_path: str | None = None) -> pv.PolyData | None:
    """Construct motor cross-section using robust Gmsh OCC *fragment* operation.

    The routine builds all primitives first, then performs a single fragment
    so every resulting surface inherits a valid physical tag.  This avoids the
    missing-tag issues seen with multiple *cut* operations.
    """

    s, r = model.stator, model.rotor
    slot, hole = s.slot, r.hole_v

    if not (0 < r.Rint < r.Rext < s.Rint < s.Rext):
        return None

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("motor")
        occ = gmsh.model.occ

        # --------------------------------------------------------------
        # 1. Create stator and rotor rings (as surfaces)
        # --------------------------------------------------------------
        stator_outer = occ.addDisk(0, 0, 0, s.Rext, s.Rext)
        stator_inner = occ.addDisk(0, 0, 0, s.Rint, s.Rint)
        stator_ring, _ = occ.cut([(2, stator_outer)], [(2, stator_inner)])

        rotor_outer = occ.addDisk(0, 0, 0, r.Rext, r.Rext)
        rotor_inner = occ.addDisk(0, 0, 0, r.Rint, r.Rint)
        rotor_ring, _ = occ.cut([(2, rotor_outer)], [(2, rotor_inner)])

        # --------------------------------------------------------------
        # 2. Slot cutters
        # --------------------------------------------------------------
        slot_cutters: list[tuple[int, int]] = []
        if slot.Zs:
            cutter_h = slot.H2 + 1e-3
            cutter_w = slot.W2
            start_r = s.Rint - 5e-4
            slot_rect = occ.addRectangle(start_r, -cutter_w / 2, 0, cutter_h, cutter_w)
            for k in range(slot.Zs):
                inst = occ.copy([(2, slot_rect)])
                occ.rotate(inst, 0, 0, 0, 0, 0, 1, 2 * math.pi * k / slot.Zs)
                slot_cutters.extend(inst)

        # --------------------------------------------------------------
        # 3. Magnet pocket cutters (V-shape simplified)
        # --------------------------------------------------------------
        magnet_cutters: list[tuple[int, int]] = []
        if hole.Zh:
            mag = hole.magnet_left
            # central rectangle pair forming V
            rect1 = occ.addRectangle(-mag.Wmag / 2, 0, 0, mag.Wmag, mag.Hmag)
            rect2 = occ.copy([(2, rect1)])
            radial_offset = r.Rext - hole.H1 - hole.H2 - mag.Hmag / 2
            angle = math.asin(hole.W1 / (2 * radial_offset))
            occ.rotate([(2, rect1)], 0, 0, 0, 0, 0, 1, -angle)
            occ.translate([(2, rect1)], 0, radial_offset, 0)
            occ.rotate(rect2, 0, 0, 0, 0, 0, 1, angle)
            occ.translate(rect2, 0, radial_offset, 0)
            v_pocket, _ = occ.fuse([(2, rect1)], rect2)
            for k in range(hole.Zh):
                inst = occ.copy(v_pocket)
                occ.rotate(inst, 0, 0, 0, 0, 0, 1, 2 * math.pi * k / hole.Zh)
                magnet_cutters.extend(inst)

        # --------------------------------------------------------------
        # 4. Fragment all surfaces
        # --------------------------------------------------------------
        base_surfaces = stator_ring + rotor_ring
        fragmented, _ = occ.fragment(base_surfaces, slot_cutters + magnet_cutters)
        occ.synchronize()

        # --------------------------------------------------------------
        # 5. Classify fragmented surfaces → physical groups
        # --------------------------------------------------------------
        pgroups: dict[str, list[int]] = {}
        for dim, tag in fragmented:
            if dim != 2:
                continue
            com = occ.getCenterOfMass(dim, tag)
            radius = math.hypot(com[0], com[1])
            # crude classification by radius
            if radius > s.Rint:
                pgroups.setdefault("stator_steel", []).append(tag)
            elif radius > r.Rint:
                # distinguish magnets by checking if tag is part of magnet cutters list
                if any(tag == mc[1] for mc in magnet_cutters):
                    pgroups.setdefault("magnets", []).append(tag)
                else:
                    pgroups.setdefault("rotor_steel", []).append(tag)

        # Slot air surfaces: those inside stator inner radius minus small tol
        for dim, tag in fragmented:
            if dim != 2:
                continue
            com = occ.getCenterOfMass(dim, tag)
            radius = math.hypot(com[0], com[1])
            if radius < s.Rint - 1e-4 and radius > r.Rext + 1e-4:
                pgroups.setdefault("slots_air", []).append(tag)

        # Create physical groups
        for name, tags in pgroups.items():
            gmsh.model.addPhysicalGroup(2, tags, name=name)

        # Phase groups for slots (assign sequentially)
        if "slots_air" in pgroups:
            for idx, tag in enumerate(pgroups["slots_air"]):
                phase = "ABC"[idx % 3]
                gmsh.model.addPhysicalGroup(2, [tag], name=f"phase_{phase}")

        # Outer boundary curve
        try:
            boundaries = gmsh.model.getBoundary([(2, pgroups["stator_steel"][0])], oriented=False, recursive=True)
            radii = [math.hypot(*occ.getCenterOfMass(dim, t)[:2]) for dim, t in boundaries]
            max_r = max(radii)
            outer = [t for (d, t), r in zip(boundaries, radii) if math.isclose(r, max_r, rel_tol=1e-3)]
            if outer:
                gmsh.model.addPhysicalGroup(1, outer, name="outer_boundary")
        except Exception:
            pass

        # --------------------------------------------------------------
        # 6. Mesh
        # --------------------------------------------------------------
        if mesh_2d:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.003)
            gmsh.model.mesh.generate(2)
            mesh = _points_and_faces()
        else:
            gmsh.model.mesh.generate(1)
            mesh = _points_and_lines()

        if save_path:
            gmsh.write(save_path)
        return mesh

    except Exception as e:
        print(f"ERROR fragment builder: {e}")
        return None

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
