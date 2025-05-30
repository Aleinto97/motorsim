import gmsh
import numpy as np
from typing import List, Tuple, Dict
from src.geometry.motor_components import Point2D

class MeshGenerator:
    def __init__(self):
        gmsh.initialize()
        self.model = gmsh.model
        self.mesh = self.model.mesh
        
    def __del__(self):
        gmsh.finalize()
    
    def generate_mesh(self, 
                     points: List[Point2D],
                     elements: List[List[int]],
                     material_regions: Dict[str, List[int]],
                     mesh_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh from geometry
        
        Args:
            points: List of 2D points
            elements: List of element connectivity
            material_regions: Dictionary mapping material names to element lists
            mesh_size: Target mesh size
            
        Returns:
            Tuple of (nodes, elements, materials)
        """
        # Clear any existing geometry
        self.model.remove()
        
        # Add points
        point_tags = []
        for p in points:
            tag = self.model.geo.addPoint(p.x, p.y, 0)
            point_tags.append(tag)
        
        # Add lines
        line_tags = []
        for i in range(len(point_tags)):
            j = (i + 1) % len(point_tags)
            tag = self.model.geo.addLine(point_tags[i], point_tags[j])
            line_tags.append(tag)
        
        # Create curve loop and surface
        curve_loop = self.model.geo.addCurveLoop(line_tags)
        surface = self.model.geo.addPlaneSurface([curve_loop])
        
        # Set mesh size
        self.mesh.setSize(self.model.getEntities(0), mesh_size)
        
        # Generate mesh
        self.model.geo.synchronize()
        self.mesh.generate(2)
        
        # Get mesh data
        node_tags, node_coords, _ = self.mesh.getNodes()
        element_types, element_tags, element_nodes = self.mesh.getElements(2)
        # Find triangle elements (type code 2)
        tri_idx = None
        for idx, etype in enumerate(element_types):
            if etype == 2:  # 2 = triangle in GMSH
                tri_idx = idx
                break
        if tri_idx is None:
            print('Element types found:', element_types)
            print('Element properties:')
            for etype in element_types:
                print(gmsh.model.mesh.getElementProperties(etype))
            raise RuntimeError('No triangle elements found in mesh.')
        elements = np.array(element_nodes[tri_idx]).reshape(-1, 3) - 1  # 0-based
        nodes = np.array(node_coords).reshape(-1, 3)[:, :2]  # Only x,y
        
        # Create material array
        materials = np.zeros(len(elements), dtype=int)
        for i, (material_name, element_list) in enumerate(material_regions.items(), 1):
            materials[element_list] = i
        
        return nodes, elements, materials
    
    def save_mesh(self, filename: str):
        """Save mesh to file"""
        gmsh.write(filename)
    
    def load_mesh(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load mesh from file
        
        Args:
            filename: Path to mesh file
            
        Returns:
            Tuple of (nodes, elements, materials)
        """
        gmsh.open(filename)
        
        # Get mesh data
        node_tags, node_coords, _ = self.mesh.getNodes()
        element_types, element_tags, element_nodes = self.mesh.getElements(2)
        # Find triangle elements (type code 2)
        tri_idx = None
        for idx, etype in enumerate(element_types):
            if etype == 2:  # 2 = triangle in GMSH
                tri_idx = idx
                break
        if tri_idx is None:
            print('Element types found:', element_types)
            print('Element properties:')
            for etype in element_types:
                print(gmsh.model.mesh.getElementProperties(etype))
            raise RuntimeError('No triangle elements found in mesh.')
        elements = np.array(element_nodes[tri_idx]).reshape(-1, 3) - 1  # 0-based
        nodes = np.array(node_coords).reshape(-1, 3)[:, :2]  # Only x,y
        
        # Create default material array (all elements assigned to material 1)
        materials = np.ones(len(elements), dtype=int)
        
        return nodes, elements, materials
    
    def refine_mesh(self, element_size: float):
        """Refine mesh to target element size
        
        Args:
            element_size: Target element size
        """
        self.mesh.setSize(self.model.getEntities(0), element_size)
        self.mesh.generate(2)
    
    def get_element_centers(self, nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Calculate element centers
        
        Args:
            nodes: Nx2 array of node coordinates
            elements: Mx3 array of element connectivity
            
        Returns:
            Mx2 array of element center coordinates
        """
        return np.mean(nodes[elements], axis=1) 