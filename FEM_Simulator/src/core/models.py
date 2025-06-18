from dataclasses import dataclass, field
from typing import List

# --- Stator definition remains largely the same, but with realistic defaults ---
@dataclass
class SlotParams:
    """High-fidelity parameters for a stator slot (Prius 2010 defaults)."""
    Zs: int = 48         # Number of slots
    H0: float = 0.001    # Slot opening height (m)
    W0: float = 0.0015   # Slot opening width (m)
    H2: float = 0.012    # Slot body height (m)
    W2: float = 0.003    # Slot body width (m)

    # --- Derived quantities ------------------------------------------------
    @property
    def slot_area(self) -> float:  # pragma: no cover – trivial getter
        """Return an approximate geometric cross-sectional area of the slot.

        This simple rectangular approximation (``H2 × W2``) is sufficient for
        computing a homogenised current density for the winding model.  A
        future enhancement could integrate the exact CAD profile, but that is
        unnecessary for most preliminary designs.
        """
        return self.H2 * self.W2

@dataclass
class StatorParams:
    """High-fidelity parameters for the Stator (Prius 2010 defaults)."""
    Rext: float = 134.5 / 2000  # Stator outer radius (m)
    Rint: float = 80.0 / 2000   # Stator inner radius (m)
    slot: SlotParams = field(default_factory=SlotParams)

# --- Magnet definition remains the same ---
@dataclass
class MagnetParams:
    """Parameters for a single rectangular permanent magnet."""
    Hmag: float = 0.0066   # Magnet height (thickness) (m)
    Wmag: float = 0.017    # Magnet width (m)
    mat_type: str = "N38SH"

# --- Hole definition is significantly more detailed ---
@dataclass
class HoleVParams:
    """
    High-fidelity parameters for a V-shaped rotor hole (Prius 2010 defaults).
    This includes critical structural and magnetic features.
    """
    Zh: int = 8          # Number of holes (which equals the number of poles)
    
    # --- Geometry of the V-shape pocket ---
    H1: float = 0.00355  # Depth of the V-shape tip from the rotor surface (m)
    H2: float = 0.0006   # Radial distance from magnet top to V-tip (m)
    W1: float = 0.016    # Tangential distance between magnet inner corners (m)
    W2: float = 0.002    # Width of the central "post" or "rib" (m)
    W3: float = 0.013    # Tangential distance from magnet outer corner to bridge (m)
    
    # --- NEW: Critical parameters for structural integrity and flux leakage ---
    bridge_width: float = 0.0015  # Tangential bridge width (m)
    rib_width: float = 0.0025     # Radial rib width (top of hole) (m)

    # A V-shaped hole contains two magnets, which can be identical
    magnet_left: MagnetParams = field(default_factory=MagnetParams)
    magnet_right: MagnetParams = field(default_factory=MagnetParams)

# --- Rotor definition now uses these new hole parameters ---
@dataclass
class RotorParams:
    """High-fidelity parameters for the Rotor (Prius 2010 defaults)."""
    Rext: float = 79.25 / 2000  # Rotor outer radius (m) -> 0.375mm air gap
    Rint: float = 45.0 / 2000   # Rotor inner radius (shaft) (m)
    hole_v: HoleVParams = field(default_factory=HoleVParams)

# --- Top-level model ---
@dataclass
class MotorParameters:
    """The top-level container for all motor design parameters."""

    name: str = "Toyota_Prius_2010_IPM"
    L_motor: float = 0.052  # Axial length (m)

    stator: StatorParams = field(default_factory=StatorParams)
    rotor: RotorParams = field(default_factory=RotorParams)

    # --- New: homogenised winding model -----------------------------------
    winding: "WindingParams" = field(default_factory=lambda: WindingParams())

# -----------------------------------------------------------------------------
# New – homogenised multi-turn winding model parameters
# -----------------------------------------------------------------------------

@dataclass
class WindingParams:
    """Defines the lay-out and copper fill of the stator windings."""

    num_conductors_per_slot: int = 4  # N_cond: conductors per slot (≈2 turns)
    fill_factor: float = 0.55          # K_fill – typical 0.4–0.6
