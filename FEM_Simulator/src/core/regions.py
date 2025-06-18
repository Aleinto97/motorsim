from enum import Enum

class Region(str, Enum):
    """Canonical names for all physical regions shared between geometry builder
    and solver. Using a *str*-derived Enum lets us compare directly to raw
    strings coming from Gmsh/meshio without explicit ``.value`` access in most
    cases.
    """

    # 2-D domains
    STATOR_STEEL = "stator_steel"
    ROTOR_STEEL = "rotor_steel"
    MAGNETS = "magnets"
    SLOTS_AIR = "slots_air"

    PHASE_A = "phase_A"
    PHASE_B = "phase_B"
    PHASE_C = "phase_C"

    # 1-D boundaries
    OUTER_BOUNDARY = "outer_boundary"

    @classmethod
    def values(cls):
        """Return list of raw string values (convenience)."""
        return [member.value for member in cls] 