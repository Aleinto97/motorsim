import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional, Dict
from src.geometry.motor_components import MotorGeometry, Point2D, DomainType, Stator, Rotor, PermanentMagnet, Winding, AirRegion, ShaftRegion
from shapely.geometry import Polygon, Point # Import Shapely for point-in-polygon tests
import matplotlib.pyplot as plt # Import matplotlib for potential visualization/debugging
from scipy.spatial import Delaunay # Import Delaunay here as it's used outside generate_mesh

class MagneticPhysics:
    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
        self.mesh_size = 0.5  # Default mesh size in mm
        self.rotor_angle = 0.0  # Current rotor angle in radians
        self.current_density = 0.0  # Current density in A/mm²
        
        # Attributes to store calculation results
        self.points: Optional[np.ndarray] = None
        self.triangles: Optional[np.ndarray] = None
        self.vector_potential: Optional[np.ndarray] = None
        self.Bx: Optional[np.ndarray] = None # Nodal Bx for visualization
        self.By: Optional[np.ndarray] = None # Nodal By for visualization
        self.Bx_elements: Optional[np.ndarray] = None # Elemental Bx for torque calculation
        self.By_elements: Optional[np.ndarray] = None # Elemental By for torque calculation
        self.element_domain_types: Optional[List[DomainType]] = None # Store domain types per element
        self.domain_geometries: Dict[DomainType, Polygon] = {}
        
        # Store Delaunay triangulation object for point lookups
        self.delaunay_mesh: Optional[Delaunay] = None
        
    def generate_mesh(self, geometry: MotorGeometry) -> Tuple[np.ndarray, np.ndarray, List[DomainType]]:
        """Generate a triangular mesh for the motor geometry at the current rotor angle and assign domain types to elements"""
        # from scipy.spatial import Delaunay # Moved import to class level
        
        # Collect points from geometry components, applying rotor rotation
        points_list = []
        # Initial domain types per point (used to help identify regions, not directly for element domains)
        point_domain_hints = []
        
        # Helper to add points and domain hints
        def add_component_points(component, domain_type, rotate_with_rotor=False):
             if component:
                 component_points = component.generate_points()
                 
                 if rotate_with_rotor and self.rotor_angle != 0.0:
                     # Apply rotation to component points
                     rotated_points = []
                     rotation_matrix = np.array([
                         [np.cos(self.rotor_angle), -np.sin(self.rotor_angle)],
                         [np.sin(self.rotor_angle), np.cos(self.rotor_angle)]
                     ])
                     for p in component_points:
                         rotated_p = np.dot(rotation_matrix, p.to_array())
                         rotated_points.append(Point2D(rotated_p[0], rotated_p[1]))
                     points_list.extend(rotated_points)
                     point_domain_hints.extend([domain_type] * len(rotated_points))
                 else:
                     points_list.extend(component_points)
                     point_domain_hints.extend([domain_type] * len(component_points))
                 
        add_component_points(geometry.stator, DomainType.STATOR, rotate_with_rotor=False)
        add_component_points(geometry.rotor, DomainType.ROTOR, rotate_with_rotor=True)
        for magnet in geometry.magnets:
            # Assume magnets rotate with the rotor for now (SPM case)
            add_component_points(magnet, DomainType.MAGNET, rotate_with_rotor=True)
        for winding in geometry.windings:
            add_component_points(winding, DomainType.WINDING, rotate_with_rotor=False)
            
        # TODO: Add points defining boundaries of Air, Shaft, etc. to improve meshing of these regions.
        # For now, meshing relies on component boundaries, and domain assignment is centroid-based.
        
        # Convert points to numpy array
        points_array = np.array([[p.x, p.y] for p in points_list])
        
        # Generate Delaunay triangulation
        try:
            self.delaunay_mesh = Delaunay(points_array)
        except Exception as e:
             print(f"Delaunay triangulation failed: {e}, adding perturbation...")
             points_array += np.random.rand(*points_array.shape) * 1e-6
             try:
                 self.delaunay_mesh = Delaunay(points_array)
             except Exception as e2:
                 print(f"Delaunay triangulation failed again after perturbation: {e2}")
                 self.points = None
                 self.triangles = None
                 self.element_domain_types = None
                 self.delaunay_mesh = None
                 return np.array([]), np.array([]), []

        self.points = points_array
        self.triangles = self.delaunay_mesh.simplices
        
        # --- Assign Domain Types to Elements based on Centroid ----
        element_domain_types = []
        
        # Define domain geometries as Shapely polygons for point-in-polygon tests (needs to use rotated geometries for rotor/magnets)
        # Pass rotate_with_rotor=False here because we are rotating the centroid point instead
        self._define_domain_geometries(geometry, rotate_with_rotor=False)
        
        for element in self.triangles:
            nodes = self.points[element]
            
            # Calculate centroid of the triangle
            centroid_x = np.mean(nodes[:, 0])
            centroid_y = np.mean(nodes[:, 1])
            centroid_point = Point(centroid_x, centroid_y)
            
            # Determine domain type based on which domain the centroid falls into
            assigned_domain_type = DomainType.AIR # Default to air
            
            # Check if centroid is within any defined domain geometry
            # Iterate through defined domains and check containment
            # Note: Order of checking might matter if domain definitions overlap or have gaps
            # A more robust approach would use a dedicated meshing tool that handles domain interfaces.
            # For this simplified approach, we'll check in a predefined order:
            
            # Check Magnet domains (iterate through individual magnets with rotation applied)
            if geometry.magnets:
                 # Need to use rotated magnet geometries for point-in-polygon test
                 # This requires regenerating magnet polygons with rotation or rotating the centroid point.
                 # Let's rotate the centroid point by the *negative* rotor angle effectively transforms it back to the original geometry frame.
                 rotated_centroid_point = self._rotate_point(centroid_point.x, centroid_point.y, -self.rotor_angle)
                 
                 for magnet in geometry.magnets:
                      magnet_pts = [(p.x, p.y) for p in magnet.generate_points()] # Original points
                      if magnet_pts and Polygon(magnet_pts).contains(Point(rotated_centroid_point[0], rotated_centroid_point[1])):
                           assigned_domain_type = DomainType.MAGNET
                           break # Assume it's the first magnet it's inside

            if assigned_domain_type == DomainType.AIR: # If not in a magnet, check other domains
                 # For domains that rotate with the rotor (Rotor, Shaft), test the rotated centroid point
                 # For domains that are stationary (Stator, Windings), test the original centroid point
                 
                 if self.domain_geometries.get(DomainType.STATOR) and self.domain_geometries[DomainType.STATOR].contains(centroid_point):
                      assigned_domain_type = DomainType.STATOR
                 elif self.domain_geometries.get(DomainType.ROTOR) and self.domain_geometries[DomainType.ROTOR].contains(Point(rotated_centroid_point[0], rotated_centroid_point[1])):
                      assigned_domain_type = DomainType.ROTOR
                 elif self.domain_geometries.get(DomainType.WINDING) and self.domain_geometries[DomainType.WINDING].contains(centroid_point):
                      assigned_domain_type = DomainType.WINDING
                 # TODO: Add checks for Shaft domain using rotated_centroid_point
                 elif self.domain_geometries.get(DomainType.SHAFT) and self.domain_geometries[DomainType.SHAFT].contains(Point(rotated_centroid_point[0], rotated_centroid_point[1])):
                      assigned_domain_type = DomainType.SHAFT

            element_domain_types.append(assigned_domain_type)

        self.element_domain_types = element_domain_types
        
        # TODO: Return a tuple indicating success and potentially a message
        return self.points, self.triangles, self.element_domain_types
        
    def _define_domain_geometries(self, geometry: MotorGeometry, rotate_with_rotor=False):
        """Define domain boundaries as Shapely polygons (simplified) for point-in-polygon tests.
           Optionally rotate rotor-attached geometries.
           Note: rotate_with_rotor flag is currently not used as we are rotating centroid point instead of geometry.
           The Shapely polygons here represent the geometry at 0 rotor angle for consistency with rotated centroid testing."""
        # This is a simplified representation. Accurate domain definition requires union/difference of shapes.
        self.domain_geometries = {}
        
        # Define geometries for major domains using Shapely at 0 rotor angle
        # Stator (approximate as ring) - Stationary
        if geometry.stator:
            theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
            outer_ring_pts = [(geometry.stator.outer_radius * np.cos(theta[i]), geometry.stator.outer_radius * np.sin(theta[i])) for i in range(100)]
            inner_ring_pts = [(geometry.stator.inner_radius * np.cos(theta[i]), geometry.stator.inner_radius * np.sin(theta[i])) for i in range(100)]
            self.domain_geometries[DomainType.STATOR] = Polygon(outer_ring_pts, [inner_ring_pts])
            # TODO: Subtract stator slot polygons from the stator domain geometry for accuracy
        
        # Rotor (approximate as circle) - Rotates with rotor_angle (but polygon defined at 0 for rotated centroid test)
        if geometry.rotor:
            theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
            rotor_circle_pts = [(geometry.rotor.outer_radius * np.cos(theta[i]), geometry.rotor.outer_radius * np.sin(theta[i])) for i in range(100)]
            self.domain_geometries[DomainType.ROTOR] = Polygon(rotor_circle_pts)
            # TODO: Subtract shaft hole and magnet regions from the rotor domain geometry for accuracy
            
            # Shaft (approximate as circle) - Rotates with rotor_angle (but polygon defined at 0 for rotated centroid test)
            shaft_circle_pts = [(geometry.rotor.inner_radius * np.cos(theta[i]), geometry.rotor.inner_radius * np.sin(theta[i])) for i in range(100)]
            self.domain_geometries[DomainType.SHAFT] = Polygon(shaft_circle_pts)

        # Magnets - Handled in generate_mesh centroid check with rotated point

        # Windings - Stationary
        # Note: Assuming windings are in slots, not simple polygons at fixed positions.
        # A proper representation needs to consider winding layout within slots.
        # For now, approximate as simple stationary polygons if defined.
        if geometry.windings:
             # TODO: Handle multiple winding regions if necessary - currently overwrites
             # Approximate winding region for point-in-polygon test
             # This needs to be more accurate based on actual winding geometry in slots.
             # For now, a placeholder if winding points are available:
             winding_pts = [] # TODO: Define approximate winding region points
             if winding_pts:
                  self.domain_geometries[DomainType.WINDING] = Polygon(winding_pts)

        # TODO: Define Air domain accurately.

    def _rotate_point(self, x: float, y: float, angle: float) -> Tuple[float, float]:
        """Rotate a point (x, y) around the origin by a given angle (in radians)."""
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle
        return x_rotated, y_rotated

    def solve_magnetostatic(self, geometry: MotorGeometry) -> np.ndarray:
        """Solve the magnetostatic problem using FEM"""
        # Generate mesh and assign domain types to elements (mesh is generated for current rotor angle)
        points, elements, element_domain_types = self.generate_mesh(geometry)
        
        if points is None or elements is None or element_domain_types is None or len(points) == 0 or len(elements) == 0 or not element_domain_types:
             print("Mesh generation failed or domain types not assigned, cannot solve magnetostatic problem.")
             self.vector_potential = None
             self.Bx = None
             self.By = None
             self.Bx_elements = None
             self.By_elements = None
             return np.array([])

        # Number of nodes
        n_nodes = len(points)
        
        # Initialize stiffness matrix and force vector
        K = lil_matrix((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        # Assemble stiffness matrix and force vector
        for i, element in enumerate(elements):
            # Get element nodes
            nodes = points[element]
            
            # Calculate element area
            area = self._triangle_area(nodes)
            
            # Get material properties for the element's domain using the assigned domain type
            element_domain_type = element_domain_types[i]
            material_props = geometry.get_domain_properties(element_domain_type)
            
            # Get permeability (mu) for stiffness matrix
            mu_r = material_props.get('mu_r', 1.0) if material_props else 1.0 # Default to air if no properties found
            mu = mu_r * self.mu0

            # Calculate element stiffness matrix using permeability
            ke = self._element_stiffness(nodes, area, mu)
            
            # Add to global stiffness matrix
            for r in range(3):
                for c in range(3):
                    K[element[r], element[c]] += ke[r, c]
                    
            # Calculate element force vector using material properties (e.g., remanence for magnets)
            fe = self._element_force(nodes, area, element_domain_type, material_props) # Pass material_props
            
            # Add to global force vector
            for r in range(3):
                F[element[r]] += fe[r]
                
        # Apply boundary conditions
        # Dirichlet BC: A = 0 on outer boundary
        # TODO: Implement proper boundary condition handling, potentially different BCs
        # on different parts of the outer boundary or using a different BC type (e.g., Neumann)
        outer_nodes = self._get_outer_nodes(points, elements)
        for node in outer_nodes:
            K[node, :] = 0
            K[node, node] = 1
            F[node] = 0
            
        # Solve system
        K = K.tocsr()
        A = spsolve(K, F)
        
        # Store vector potential
        self.vector_potential = A
        
        return A
        
    def calculate_torque(self, geometry: MotorGeometry, angle: float) -> float:
        """Calculate torque at a given rotor angle"""
        # Update rotor angle (This now affects mesh generation)
        self.rotor_angle = angle # Store angle, and it will be used in generate_mesh
        
        # Solve magnetostatic problem (mesh and solve for the current rotor angle)
        A = self.solve_magnetostatic(geometry)
        
        # Calculate magnetic field (Bx, By are stored)
        if A is not None and len(A) > 0:
             self._calculate_magnetic_field(A)
        else:
             self.Bx = None
             self.By = None
             self.Bx_elements = None
             self.By_elements = None
             print("Cannot calculate magnetic field: vector potential not available.")
             return 0.0
        
        # Calculate torque using Maxwell stress tensor in the air gap
        torque = self._calculate_torque_from_fields(geometry) # Pass geometry to access dimensions
        
        return torque
        
    def _triangle_area(self, nodes: np.ndarray) -> float:
        """Calculate area of a triangle"""
        # Ensure nodes are numpy arrays for cross product
        n0 = np.array(nodes[0])
        n1 = np.array(nodes[1])
        n2 = np.array(nodes[2])
        return 0.5 * abs(np.cross(n1 - n0, n2 - n0))
        
    def _element_stiffness(self, nodes: np.ndarray, area: float, mu: float) -> np.ndarray:
        """Calculate element stiffness matrix"""
        # Calculate element gradients (b_i and c_i in FEM formulation)
        b = np.array([nodes[1, 1] - nodes[2, 1],
                     nodes[2, 1] - nodes[0, 1],
                     nodes[0, 1] - nodes[1, 1]])
        c = np.array([nodes[2, 0] - nodes[1, 0],
                     nodes[0, 0] - nodes[2, 0],
                     nodes[1, 0] - nodes[0, 0]])
        
        # Calculate stiffness matrix contribution
        ke = (1 / (4 * area * mu)) * np.outer(b, b) + (1 / (4 * area * mu)) * np.outer(c, c)
                
        return ke
        
    def _element_force(self, nodes: np.ndarray, area: float, domain_type: DomainType, material_props: Optional[Dict]) -> np.ndarray:
        """Calculate element force vector"""
        # Initialize force vector
        fe = np.zeros(3)
        
        # Add current density contribution (for windings)
        if domain_type == DomainType.WINDING:
            # Assuming current density is uniform over the winding element
            # TODO: Get current density value from geometry or simulation parameters
            fe += self.current_density * area / 3
            
        # Add permanent magnet contribution (for magnets)
        if domain_type == DomainType.MAGNET and material_props:
            Br = material_props.get('Br', 0.0)
            mu_r = material_props.get('mu_r', 1.0)
            
            if Br > 0 and mu_r > 0:
                # TODO: Implement proper FEM formulation for magnet force contribution.
                # This is a complex topic involving magnetization vector and shape functions.
                pass # Placeholder for magnet force contribution
            
        return fe
        
    def _get_outer_nodes(self, points: np.ndarray, elements: np.ndarray) -> List[int]:
        """Get nodes on the outer boundary"""
        # Find edges that appear only once (boundary edges)
        edges = {}
        for element in elements:
            for i in range(3):
                edge = tuple(sorted([element[i], element[(i+1)%3]]))
                if edge in edges:
                    edges[edge] += 1
                else:
                    edges[edge] = 1
                    
        outer_nodes = set()
        for edge, count in edges.items():
            if count == 1:
                outer_nodes.update(edge)
                    
        return list(outer_nodes)
        
    def _calculate_magnetic_field(self, A: np.ndarray):
        """Calculate magnetic field (B = curl(A)) from vector potential
        (using stored mesh data)"""
        if self.points is None or self.triangles is None:
            print("Mesh data not available for magnetic field calculation.")
            return

        n_points = len(self.points)
        # Store elemental B values for torque calculation
        self.Bx_elements = np.zeros(len(self.triangles))
        self.By_elements = np.zeros(len(self.triangles))

        Bx = np.zeros(n_points)
        By = np.zeros(n_points)

        # Calculate B field for each element (constant within linear triangle)
        # and then average/sum at nodes for visualization.
        # A more accurate nodal B requires interpolation or a different FEM formulation.

        for i, element in enumerate(self.triangles):
            nodes = self.points[element]
            area = self._triangle_area(nodes)

            # Get nodal values of vector potential A for this element
            A_element = A[element]

            # Calculate gradients (constant within element)
            b = np.array([nodes[1, 1] - nodes[2, 1],
                         nodes[2, 1] - nodes[0, 1],
                         nodes[0, 1] - nodes[1, 1]])
            c = np.array([nodes[2, 0] - nodes[1, 0],
                         nodes[0, 0] - nodes[2, 0],
                         nodes[1, 0] - nodes[0, 0]])

            # Calculate constant B field within this element
            # Bx = dAz/dy, By = -dAz/dx
            # dAz/dy = sum(Ai * dNi/dy) = sum(Ai * c_i / (2*area))
            # dAz/dx = sum(Ai * dNi/dx) = sum(Ai * b_i / (2*area))

            Bx_element = np.sum(A_element * c) / (2 * area) # dAz/dy
            By_element = -np.sum(A_element * b) / (2 * area) # -dAz/dx

            # Store elemental B values
            self.Bx_elements[i] = Bx_element
            self.By_elements[i] = By_element

            # For visualization, assign the element's B field value to its nodes.
            # If a node is shared by multiple elements, this will sum up contributions.
            # Proper nodal B would require averaging or a different method.
            for j in range(3):
                 Bx[element[j]] += Bx_element
                 By[element[j]] += By_element

        # TODO: Implement proper nodal B calculation from elemental fields (e.g., averaging or using interpolation)
        # For now, self.Bx and self.By are sums of elemental B at shared nodes (for visualization).

        # Store nodal Bx and By (for visualization)
        self.Bx = Bx
        self.By = By

    def _calculate_torque_from_fields(self, geometry: MotorGeometry) -> float:
        """Calculate torque using magnetic field data via air gap Maxwell stress tensor"""
        if self.points is None or self.triangles is None or self.Bx_elements is None or self.By_elements is None or self.delaunay_mesh is None:
             print("Field, mesh data, or Delaunay mesh not available for torque calculation.")
             return 0.0

        # Air gap Maxwell stress tensor method steps:
        # 1. Define air gap radius and integration points.
        #    Air gap radius is typically the average of rotor outer and stator inner radii.
        if geometry.stator and geometry.rotor:
             air_gap_radius = (geometry.stator.inner_radius + geometry.rotor.outer_radius) / 2.0
        else:
             print("Stator or Rotor geometry not available to define air gap.")
             return 0.0
             
        # Number of integration points along the circle (choose a sufficient number)
        n_integration_points = 360 # 1 point per degree, adjust as needed based on mesh density
        theta = np.linspace(0, 2*np.pi, n_integration_points, endpoint=False)
        
        # Create integration points (x, y coordinates) along the air gap circle
        integration_points = np.array([
            [air_gap_radius * np.cos(theta[i]), air_gap_radius * np.sin(theta[i])]
            for i in range(n_integration_points)
        ])

        # 2. For each integration point:
        #    a. Find the mesh element containing the point.
        #    b. Get the magnetic field (Bx, By) in that element (constant B per element).
        #    c. Calculate normal (nx, ny) and tangential (tx, ty) vectors at the point.
        #    d. Calculate stress tensor components: sigma_r_theta = (1/mu0) * (Bt * Bn)
        #       where Bt = B . t (tangential component), Bn = B . n (normal component)
        #       For a circular path: n = (cos(theta), sin(theta)), t = (-sin(theta), cos(theta))
        #    e. Add contribution to torque: dTorque = r * sigma_r_theta * dL (where r is air gap radius, dL is arc length element)
        # 3. Sum up contributions along the path.

        total_torque = 0.0
        dL = (2 * np.pi * air_gap_radius) / n_integration_points # Arc length of each segment

        # Iterate through integration points
        # Use Delaunay.find_simplex to find the triangle containing each point
        # find_simplex returns the index of the simplex (triangle) containing the point, or -1 if no simplex contains it.
        containing_elements_indices = self.delaunay_mesh.find_simplex(integration_points)

        for i, element_index in enumerate(containing_elements_indices):
             if element_index != -1:
                  # Get the magnetic field (Bx, By) for this element
                  # Since B is constant within each element:
                  Bx_element = self.Bx_elements[element_index]
                  By_element = self.By_elements[element_index]

                  # Get the integration point coordinates
                  px, py = integration_points[i]

                  # Calculate normal (nx, ny) and tangential (tx, ty) vectors at the point
                  # For a circle centered at (0,0), normal vector is (x/r, y/r)
                  # Tangential vector is (-y/r, x/r)
                  r_point = np.sqrt(px**2 + py**2) # Should be equal to air_gap_radius
                  if r_point == 0:
                       continue # Avoid division by zero at origin

                  # Normalize vectors
                  nx, ny = px / r_point, py / r_point
                  tx, ty = -py / r_point, px / r_point

                  # Calculate tangential and normal components of B field
                  Bt = Bx_element * tx + By_element * ty
                  Bn = Bx_element * nx + By_element * ny

                  # Calculate stress tensor component: sigma_r_theta = (1/mu0) * Bt * Bn
                  sigma_r_theta = (1 / self.mu0) * Bt * Bn

                  # Add contribution to total torque: dTorque = r * sigma_r_theta * dL
                  dTorque = air_gap_radius * sigma_r_theta * dL
                  total_torque += dTorque

        return total_torque # Return calculated torque. 