from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum

class DomainType(Enum):
    STATOR = "stator"
    ROTOR = "rotor"
    MAGNET = "magnet"
    WINDING = "winding"
    AIR = "air"
    SHAFT = "shaft"

@dataclass
class Point2D:
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

@dataclass
class Domain:
    type: DomainType
    material: str
    color: str
    is_selected: bool = False

@dataclass
class Stator:
    outer_radius: float
    inner_radius: float
    num_slots: int
    slot_opening: float
    tooth_width: float
    back_iron_thickness: float
    domain: Domain = field(default_factory=lambda: Domain(DomainType.STATOR, "iron", "gray"))
    
    def generate_points(self) -> List[Point2D]:
        """Generate points for stator geometry"""
        points = []
        angle_step = 2 * np.pi / self.num_slots
        
        for i in range(self.num_slots):
            angle = i * angle_step
            # Outer points
            points.append(Point2D(
                self.outer_radius * np.cos(angle),
                self.outer_radius * np.sin(angle)
            ))
            # Inner points
            points.append(Point2D(
                self.inner_radius * np.cos(angle),
                self.inner_radius * np.sin(angle)
            ))
            
        return points

    def generate_slot_polygons(self) -> List[List[Point2D]]:
        """Generate polygons for stator slots (rectangular, evenly spaced)."""
        slots = []
        angle_step = 2 * np.pi / self.num_slots
        slot_depth = self.outer_radius - self.inner_radius
        for i in range(self.num_slots):
            angle = i * angle_step
            # Center of slot
            cx = (self.inner_radius + slot_depth / 2) * np.cos(angle)
            cy = (self.inner_radius + slot_depth / 2) * np.sin(angle)
            # Slot orientation
            dx = np.cos(angle)
            dy = np.sin(angle)
            # Perpendicular
            px = -dy
            py = dx
            # Rectangle corners
            w = self.slot_opening / 2
            d = slot_depth / 2
            corners = [
                Point2D(cx + w * px - d * dx, cy + w * py - d * dy),
                Point2D(cx - w * px - d * dx, cy - w * py - d * dy),
                Point2D(cx - w * px + d * dx, cy - w * py + d * dy),
                Point2D(cx + w * px + d * dx, cy + w * py + d * dy),
            ]
            slots.append(corners)
        return slots

@dataclass
class Rotor:
    outer_radius: float
    inner_radius: float
    num_poles: int
    magnet_type: str  # 'SPM' or 'IPM'
    magnet_thickness: float
    magnet_width: float
    domain: Domain = field(default_factory=lambda: Domain(DomainType.ROTOR, "iron", "gray"))
    
    def generate_points(self) -> List[Point2D]:
        """Generate points for rotor geometry"""
        points = []
        angle_step = 2 * np.pi / self.num_poles
        
        for i in range(self.num_poles):
            angle = i * angle_step
            # Outer points
            points.append(Point2D(
                self.outer_radius * np.cos(angle),
                self.outer_radius * np.sin(angle)
            ))
            # Inner points
            points.append(Point2D(
                self.inner_radius * np.cos(angle),
                self.inner_radius * np.sin(angle)
            ))
            
        return points

@dataclass
class PermanentMagnet:
    position: Point2D
    width: float
    thickness: float
    angle: float
    magnetization: float  # Tesla
    domain: Domain = field(default_factory=lambda: Domain(DomainType.MAGNET, "magnet", "red"))
    
    def generate_points(self) -> List[Point2D]:
        """Generate points for magnet geometry"""
        # Calculate corners of rectangular magnet
        half_width = self.width / 2
        half_thickness = self.thickness / 2
        
        # Base points (before rotation)
        base_points = [
            Point2D(-half_width, -half_thickness),
            Point2D(half_width, -half_thickness),
            Point2D(half_width, half_thickness),
            Point2D(-half_width, half_thickness)
        ]
        
        # Rotate and translate points
        rotation_matrix = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ])
        
        points = []
        for p in base_points:
            rotated = np.dot(rotation_matrix, p.to_array())
            points.append(Point2D(
                rotated[0] + self.position.x,
                rotated[1] + self.position.y
            ))
            
        return points

@dataclass
class Winding:
    slot_number: int
    turns: int
    current: float
    position: Point2D
    width: float
    height: float
    domain: Domain = field(default_factory=lambda: Domain(DomainType.WINDING, "copper", "orange"))
    
    def generate_points(self) -> List[Point2D]:
        """Generate points for winding geometry"""
        half_width = self.width / 2
        half_height = self.height / 2
        
        return [
            Point2D(self.position.x - half_width, self.position.y - half_height),
            Point2D(self.position.x + half_width, self.position.y - half_height),
            Point2D(self.position.x + half_width, self.position.y + half_height),
            Point2D(self.position.x - half_width, self.position.y + half_height)
        ]

@dataclass
class AirRegion:
    """Represents the air region of the motor."""
    domain: Domain = field(default_factory=lambda: Domain(DomainType.AIR, "air", "cyan"))

@dataclass
class ShaftRegion:
    """Represents the shaft region of the motor."""
    domain: Domain = field(default_factory=lambda: Domain(DomainType.SHAFT, "steel", "darkgray"))

class MotorGeometry:
    def __init__(self):
        self.stator: Optional[Stator] = None
        self.rotor: Optional[Rotor] = None
        self.magnets: List[PermanentMagnet] = []
        self.windings: List[Winding] = []
        self.air_gap: float = 0.5  # mm
        self.shaft_radius: float = 5.0  # mm
        self.selected_domain: Optional[Domain] = None
        self.air_region: AirRegion = AirRegion()
        self.shaft_region: ShaftRegion = ShaftRegion()
        
    def add_stator(self, stator: Stator):
        self.stator = stator
        
    def add_rotor(self, rotor: Rotor):
        self.rotor = rotor
        
    def add_magnet(self, magnet: PermanentMagnet):
        self.magnets.append(magnet)
        
    def add_winding(self, winding: Winding):
        self.windings.append(winding)
        
    def select_domain(self, domain_type: DomainType):
        """Select a domain type for editing"""
        if domain_type == DomainType.STATOR and self.stator:
            self.selected_domain = self.stator.domain
        elif domain_type == DomainType.ROTOR and self.rotor:
            self.selected_domain = self.rotor.domain
        elif domain_type == DomainType.MAGNET:
            for magnet in self.magnets:
                magnet.domain.is_selected = (magnet.domain.type == domain_type)
        elif domain_type == DomainType.WINDING:
            for winding in self.windings:
                winding.domain.is_selected = (winding.domain.type == domain_type)
                
    def update_domain_material(self, domain_type: DomainType, material: str, color: str):
        """Update material properties for a domain type"""
        if domain_type == DomainType.STATOR and self.stator:
            self.stator.domain.material = material
            self.stator.domain.color = color
        elif domain_type == DomainType.ROTOR and self.rotor:
            self.rotor.domain.material = material
            self.rotor.domain.color = color
        elif domain_type == DomainType.MAGNET:
            for magnet in self.magnets:
                if magnet.domain.type == domain_type:
                    magnet.domain.material = material
                    magnet.domain.color = color
        elif domain_type == DomainType.WINDING:
            for winding in self.windings:
                if winding.domain.type == domain_type:
                    winding.domain.material = material
                    winding.domain.color = color
                    
    def get_domain_properties(self, domain_type: DomainType) -> Dict:
        """Get properties for a domain type"""
        if domain_type == DomainType.STATOR and self.stator:
            return {
                'material': self.stator.domain.material,
                'color': self.stator.domain.color,
                'is_selected': self.stator.domain.is_selected
            }
        elif domain_type == DomainType.ROTOR and self.rotor:
            return {
                'material': self.rotor.domain.material,
                'color': self.rotor.domain.color,
                'is_selected': self.rotor.domain.is_selected
            }
        elif domain_type == DomainType.MAGNET:
            return {
                'material': self.magnets[0].domain.material if self.magnets else None,
                'color': self.magnets[0].domain.color if self.magnets else None,
                'is_selected': any(m.domain.is_selected for m in self.magnets)
            }
        elif domain_type == DomainType.WINDING:
            return {
                'material': self.windings[0].domain.material if self.windings else None,
                'color': self.windings[0].domain.color if self.windings else None,
                'is_selected': any(w.domain.is_selected for w in self.windings)
            }
        return None
        
    def generate_geometry(self) -> Tuple[List[Point2D], List[List[int]]]:
        """Generate the complete motor geometry"""
        points = []
        elements = []
        
        # Add stator points and elements
        if self.stator:
            stator_points = self.stator.generate_points()
            points.extend(stator_points)
            
        # Add rotor points and elements
        if self.rotor:
            rotor_points = self.rotor.generate_points()
            points.extend(rotor_points)
            
        # Add magnet points
        for magnet in self.magnets:
            magnet_points = magnet.generate_points()
            points.extend(magnet_points)
            
        # Add winding points
        for winding in self.windings:
            winding_points = winding.generate_points()
            points.extend(winding_points)
            
        # Generate triangular elements
        from scipy.spatial import Delaunay
        points_array = np.array([[p.x, p.y] for p in points])
        tri = Delaunay(points_array)
        elements = tri.simplices.tolist()
        
        return points, elements

    def get_numpy_geometry(self):
        """Return (points, elements) as numpy arrays for plotting."""
        points, elements = self.generate_geometry()
        pts = np.array([[p.x, p.y] for p in points])
        elems = np.array(elements)
        return pts, elems 