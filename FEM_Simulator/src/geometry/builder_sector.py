"""Prototype sector-model geometry builder.

This file introduces *build_geometry_sector* which will eventually build a
single fundamental sector (360° / Zs) of the motor cross-section, mesh it and –
in a later iteration – optionally replicate the meshed sector to form the full
machine.

At this first incremental stage we **delegate** to the existing full-circle
builder so that the rest of the application and tests keep working while we
develop the sector workflow in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyvista as pv
import numpy as np

from src.core.models import MotorParameters
from src.geometry.builder import build_geometry  # fallback full-circle builder

__all__ = ["build_geometry_sector"]


def build_geometry_sector(
    model: MotorParameters,
    *,
    mesh_2d: bool = False,
    save_path: Optional[str] = None,
) -> Optional[pv.PolyData]:
    """Temporary stub sector builder.

    Parameters
    ----------
    model
        Motor parameter dataclass instance.
    mesh_2d
        If *True* a 2-D triangular mesh is generated; otherwise a 1-D boundary
        representation suitable for preview.
    save_path
        When given, the geometry/mesh is written to this file (usually
        ``.msh``).
    """

    # ------------------------------------------------------------------
    # 1. Determine the fundamental geometric repetition factor.
    # ------------------------------------------------------------------
    # For an IPM machine the cross-section is periodic in the *least common
    # multiple* sense between the stator-slot pitch (Zs) and the number of
    # *magnet repetitions*.  A single V-shaped pocket holds *two* magnets,
    # therefore the rotor presents ``2 * Zh`` repeating features.
    #
    # Example            Zs = 48  |  Zh = 8
    # ------------------------------|-----------
    # Rotor repetitions  = 16      |  360/16 = 22.5°
    # Fundamental sector = 360 / gcd(48, 16) = 360 / 16 = 22.5°
    #
    # Should the user supply an unconventional combination where either Zs
    # or Zh is zero/negative we fall back to the full-circle builder to
    # keep the application functional.
    from math import gcd, pi

    Zs = model.stator.slot.Zs
    Zh = model.rotor.hole_v.Zh

    if Zs <= 0 or Zh <= 0:
        # Degenerate configuration – punt to full-circle geometry.
        return build_geometry(model, mesh_2d=mesh_2d, save_path=save_path)

    rotor_repetitions = 2 * Zh
    n_sym = gcd(Zs, rotor_repetitions)
    sector_angle = 2 * pi / n_sym

    # Store for downstream steps (meshing, PyVista transforms) – even though
    # not yet used in this provisional implementation.
    _meta: dict[str, float] = {
        "n_sym": float(n_sym),
        "sector_angle_rad": sector_angle,
    }

    # ------------------------------------------------------------------
    # 2. Generate the full-circle mesh using the proven routine and then
    #    extract the 0–θ sector with a lightweight NumPy/PyVista pass.
    # ------------------------------------------------------------------

    full_mesh = build_geometry(model, mesh_2d=mesh_2d, save_path=None)
    if full_mesh is None:
        return None

    sector_mesh = _extract_sector_mesh(full_mesh, angle_rad=sector_angle)
    if sector_mesh is None or sector_mesh.n_points == 0:
        # Extraction too aggressive – fall back.
        sector_mesh = full_mesh

    # ------------------------------------------------------------------
    # 3. Optional – persist the sector mesh if the caller provided a path.
    #    PyVista cannot write ``.msh`` directly, therefore we always write a
    #    ``.vtk`` file in that case, while still honouring the original
    #    extension if it is supported.
    # ------------------------------------------------------------------

    if save_path is not None:
        out_path = Path(save_path)
        try:
            sector_mesh.save(out_path)
        except (RuntimeError, ValueError):
            # Fallback: change suffix to .vtk which PyVista always supports.
            sector_mesh.save(out_path.with_suffix(".vtk"))

    return sector_mesh


def _extract_sector_mesh(mesh: pv.PolyData, angle_rad: float) -> pv.PolyData:
    """Return a **shallow** copy of *mesh* truncated to the 0–angle sector.

    The helper takes a pragmatic approach suitable for early prototyping:
    any cell (line or triangle) that has **all** its corner nodes strictly
    inside the angular window is kept; partially-intersecting cells are
    discarded.  The result therefore contains a small gap along the two
    radial cutting planes which will be healed in a later iteration via
    proper CAD construction.  For visualisation and high-level tests this
    simplification is acceptable and guarantees fast execution without
    additional Gmsh calls.
    """

    # Normalise polar angle to the [0, 2π) range so that a single compare
    # suffices for the 0–angle wedge.
    theta = np.mod(np.arctan2(mesh.points[:, 1], mesh.points[:, 0]), 2 * np.pi)
    inside_pt = theta <= angle_rad + 1e-10  # small tolerance to include boundary

    # Mapping old → new point indices for connectivity remapping.
    idx_old_to_new = -np.ones(mesh.n_points, dtype=int)
    idx_old_to_new[inside_pt] = np.arange(np.count_nonzero(inside_pt))

    # --- Face connectivity ------------------------------------------------
    new_faces: list[int] = []
    if mesh.faces.size:
        faces = mesh.faces.reshape(-1, 4)  # [N, n+1] with n triangles → n=3
        for f in faces:
            tri = f[1:]
            if np.all(inside_pt[tri]):
                new_faces.append(3)
                new_faces.extend(idx_old_to_new[tri])

    # --- Line connectivity ------------------------------------------------
    new_lines: list[int] = []
    if mesh.lines.size:
        lines = mesh.lines.reshape(-1, 3)  # [N, 1+2]
        for l in lines:
            seg = l[1:]
            if np.all(inside_pt[seg]):
                new_lines.append(2)
                new_lines.extend(idx_old_to_new[seg])

    # Points for the new mesh -----------------------------------------------------------------
    new_pts = mesh.points[inside_pt]
    sector_mesh = pv.PolyData(new_pts)
    if new_faces:
        sector_mesh.faces = np.asarray(new_faces, dtype=int)
    if new_lines:
        sector_mesh.lines = np.asarray(new_lines, dtype=int)
    return sector_mesh 