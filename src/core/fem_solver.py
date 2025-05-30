import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import meshio

class FEMSolver:
    def __init__(self):
        self.mesh = None
        self.materials = {}
        self.current_density = {}
        self.solution = None
        self.nodes = None
        self.elements = None
        self.material_regions = None
        
    def set_mesh(self, nodes, elements, material_regions):
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.material_regions = np.array(material_regions)
        
    def load_mesh(self, mesh_file):
        """Load mesh from file (supports .msh format)"""
        self.mesh = meshio.read(mesh_file)
        return self.mesh
        
    def set_material(self, region_id, material_properties):
        """Set material properties for a region
        
        Args:
            region_id: Integer identifier for the region
            material_properties: Dict with 'mu_r' (relative permeability)
                               and optionally 'J' (current density)
        """
        self.materials[region_id] = material_properties
        
    def assemble_matrix(self):
        """Assemble the FEM system matrix for A-phi formulation"""
        if self.nodes is None or self.elements is None:
            raise ValueError("Mesh not set")
            
        n_nodes = len(self.nodes)
        K = lil_matrix((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        # Assembly loop over elements
        for element in self.elements:
            # Get element nodes
            nodes = element
            coords = self.nodes[nodes]
            
            # Calculate element matrix
            Ke = self._element_matrix(coords)
            
            # Assemble into global matrix
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    K[node_i, node_j] += Ke[i, j]
                    
        return csc_matrix(K), F
        
    def _element_matrix(self, coords):
        """Calculate element matrix for a triangle
        
        Args:
            coords: 3x2 array of triangle vertex coordinates
            
        Returns:
            3x3 element matrix
        """
        # Calculate element area
        x = coords[:, 0]
        y = coords[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Calculate gradients of shape functions
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
        
        # Element matrix (simplified for demonstration)
        Ke = np.outer(b, b) + np.outer(c, c)
        return Ke * area
        
    def solve(self):
        """Solve the magnetostatic problem"""
        K, F = self.assemble_matrix()
        self.solution = spsolve(K, F)
        return self.solution
        
    def calculate_field(self):
        """Calculate magnetic field from solution"""
        if self.solution is None:
            raise ValueError("Solve the system first")
            
        # Calculate B field (simplified)
        B = np.zeros((len(self.nodes), 2))
        # TODO: Implement proper field calculation
        return B 

    def update_parameters(self, params):
        # Stub: In a real implementation, this would update geometry/materials/mesh
        self._last_params = params
    def simulate(self):
        # Stub: In a real implementation, this would run the full simulation and return results
        # Here, return a dummy torque value that depends on the parameters for demonstration
        torque = sum(self._last_params.values()) if hasattr(self, '_last_params') else 0.0
        return {'average_torque': torque} 