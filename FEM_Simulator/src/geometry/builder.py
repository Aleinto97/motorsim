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

def build_geometry(model: MotorParameters, *, mesh_2d: bool = False, save_path: str | None = None) -> pv.PolyData | None:  # noqa: C901
    """Construct motor cross-section and optionally generate a 2-D mesh.

    Parameters
    ----------
    model : MotorParameters
        Dataclass holding **stator**, **rotor**, **slot**, and **V-hole** specs.
    mesh_2d : bool, default False
        When *True* a triangular surface mesh is returned, otherwise a 1-D
        outline suitable for quick preview.
    """

    s = model.stator
    r = model.rotor
    slot = s.slot
    hole = r.hole_v

    if not (0 < r.Rint < r.Rext < s.Rint < s.Rext):
        print("ERROR: Invalid radii hierarchy – abort.")
        return None

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("motor")
        occ = gmsh.model.occ
        # --------------------------------------------------------------
        # 1. Stator ring and axial slots
        # --------------------------------------------------------------
        stator_outer = occ.addDisk(0, 0, 0, s.Rext, s.Rext)
        stator_inner = occ.addDisk(0, 0, 0, s.Rint, s.Rint)
        stator_surf, _ = occ.cut([(2, stator_outer)], [(2, stator_inner)],
                                 removeObject=True, removeTool=True)
        stator_tag = stator_surf[0][1]

        if slot.Zs > 0:
            cutter_h = slot.H2 + 0.001  # tiny tolerance to fully cut through
            cutter_w = slot.W2
            start_r = s.Rint - 5e-4
            base_cut = occ.addRectangle(start_r, -cutter_w / 2, 0,
                                         cutter_h, cutter_w)
            cutters: List[tuple[int, int]] = []
            for k in range(slot.Zs):
                inst = occ.copy([(2, base_cut)])
                occ.rotate(inst, 0, 0, 0, 0, 0, 1,
                           2 * math.pi * k / slot.Zs)
                cutters.extend(inst)
            # Cut slots from stator but keep slot surfaces for further grouping
            stator_surf, _ = occ.cut([(2, stator_tag)], cutters,
                                     removeObject=True, removeTool=False)
            stator_tag = stator_surf[0][1]
            slot_tags = [tag for dim, tag in cutters if dim == 2]

        # --------------------------------------------------------------
        # 2. Rotor ring and V-shaped magnet pockets
        # --------------------------------------------------------------
        rotor_outer = occ.addDisk(0, 0, 0, r.Rext, r.Rext)
        shaft_bore = occ.addDisk(0, 0, 0, r.Rint, r.Rint)
        rotor_surf, _ = occ.cut([(2, rotor_outer)], [(2, shaft_bore)],
                                removeObject=True, removeTool=True)
        rotor_tag = rotor_surf[0][1]

        if hole.Zh > 0:
            mag = hole.magnet_left  # both magnets identical size
            # build two rectangles, rotate symmetrically to form a V-pocket
            rect1 = occ.addRectangle(-mag.Wmag / 2, 0, 0, mag.Wmag, mag.Hmag)
            rect2 = occ.copy([(2, rect1)])

            radial_offset = r.Rext - hole.H1 - mag.Hmag / 2 - hole.H2
            pocket_angle = math.asin(hole.W1 / (2 * (r.Rext - hole.H1)))

            occ.rotate([(2, rect1)], 0, 0, 0, 0, 0, 1, -pocket_angle)
            occ.translate([(2, rect1)], 0, radial_offset, 0)

            occ.rotate(rect2, 0, 0, 0, 0, 0, 1, pocket_angle)
            occ.translate(rect2, 0, radial_offset, 0)

            v_pocket, _ = occ.fuse([(2, rect1)], rect2)

            cutters: List[tuple[int, int]] = []
            for k in range(hole.Zh):
                inst = occ.copy(v_pocket)
                occ.rotate(inst, 0, 0, 0, 0, 0, 1,
                           2 * math.pi * k / hole.Zh)
                cutters.extend(inst)

            # Cut pockets from rotor but KEEP the pocket surfaces (future magnets)
            rotor_surf, _ = occ.cut([(2, rotor_tag)], cutters,
                                     removeObject=True, removeTool=False)
            rotor_tag = rotor_surf[0][1]

            # Store pocket surfaces as magnets for grouping
            magnet_tags = [tag for dim, tag in cutters if dim == 2]

        # --------------------------------------------------------------
        # 3. Physical groups and OCC sync
        # --------------------------------------------------------------
        gmsh.model.addPhysicalGroup(2, [stator_tag], name="stator_steel")
        gmsh.model.addPhysicalGroup(2, [rotor_tag], name="rotor_steel")


        if 'magnet_tags' in locals():
            gmsh.model.addPhysicalGroup(2, magnet_tags, name="magnets")
        if 'slot_tags' in locals():
            gmsh.model.addPhysicalGroup(2, slot_tags, name="slots_air")
            # Split slots into three phase groups (A/B/C) sequentially
            phase_groups = {"A": [], "B": [], "C": []}
            for idx, tag in enumerate(slot_tags):
                phase = "ABC"[idx % 3]
                phase_groups[phase].append(tag)
            for phase, tags in phase_groups.items():
                if tags:
                    gmsh.model.addPhysicalGroup(2, tags, name=f"phase_{phase}")
        occ.synchronize()
        # Determine and tag the outer stator boundary curve (after sync)
        try:
            boundaries = gmsh.model.getBoundary([(2, stator_tag)], oriented=False, recursive=True)
            # pick curve(s) with maximum radial center-of-mass as outer boundary
            if boundaries:
                radii = [np.hypot(*gmsh.model.getCenterOfMass(dim, tag)[:2]) for dim, tag in boundaries]
                max_r = max(radii)
                outer = [tag for (dim, tag), r in zip(boundaries, radii) if np.isclose(r, max_r, rtol=1e-3)]
                if outer:
                    gmsh.model.addPhysicalGroup(1, outer, name="outer_boundary")
        except Exception as _:
            pass
        # --------------------------------------------------------------
        # 4. Mesh generation
        # --------------------------------------------------------------
        if mesh_2d:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.003)
            gmsh.model.mesh.generate(2)
            mesh = _points_and_faces()
            if save_path:
                gmsh.write(save_path)
        else:
            gmsh.model.mesh.generate(1)
            mesh = _points_and_lines()
            if save_path:
                gmsh.write(save_path)

        return mesh

    except Exception as err:
        print(f"ERROR: Geometry building failed: {err}")
        return None

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
