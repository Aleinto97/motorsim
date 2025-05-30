from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                              QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox, QFormLayout, QColorDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from src.geometry.motor_components import MotorGeometry, Stator, Rotor, PermanentMagnet, Winding, Point2D, DomainType, AirRegion, ShaftRegion
from PySide6.QtCore import Qt, Signal
from shapely.geometry import Polygon, Point, LinearRing # Import necessary Shapely components
from shapely.ops import unary_union # Import unary_union for combining slot polygons

class GeometryViewer(QWidget):
    geometry_updated = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.geometry = MotorGeometry()
        self.selected_domain = None
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        
        # Create main layout for the widget
        main_layout = QVBoxLayout(self)
        
        # --- Domain Selection ---
        domain_group = QGroupBox("Domain Selection")
        domain_layout = QFormLayout(domain_group) # Use QFormLayout for key-value pairs
        
        self.domain_combo = QComboBox()
        for domain in DomainType:
            self.domain_combo.addItem(domain.value)
        self.domain_combo.currentTextChanged.connect(self.on_domain_changed)
        
        domain_layout.addRow("Select Domain:", self.domain_combo)
        
        main_layout.addWidget(domain_group)
        
        # --- Material Properties ---
        material_group = QGroupBox("Material Properties")
        material_layout = QFormLayout(material_group) # Use QFormLayout
        
        self.material_combo = QComboBox()
        self.material_combo.addItems(["iron", "copper", "magnet", "air", "custom"])
        self.material_combo.currentTextChanged.connect(lambda: self.on_material_changed(None, self.material_combo.currentText(), {})) # Simple connection, will be updated by MaterialEditor signal
        
        self.color_button = QPushButton("Select Color")
        self.color_button.clicked.connect(self.on_color_clicked)
        
        material_layout.addRow("Material:", self.material_combo)
        material_layout.addRow("Color:", self.color_button)

        main_layout.addWidget(material_group)
        
        # --- Geometry Parameters ---
        params_group = QGroupBox("Geometry Parameters")
        params_layout = QVBoxLayout(params_group) # Use QVBoxLayout for stacking groups

        # General Geometry Parameters
        general_params_group = QGroupBox("General") # Group for general parameters
        general_params_layout = QFormLayout(general_params_group)

        self.air_gap_spin = QDoubleSpinBox()
        self.air_gap_spin.setRange(0.1, 5.0)
        self.air_gap_spin.setValue(0.5)
        self.air_gap_spin.setSingleStep(0.1)
        self.air_gap_spin.valueChanged.connect(self.on_air_gap_changed)
        general_params_layout.addRow("Air Gap (mm):", self.air_gap_spin)

        self.shaft_radius_spin = QDoubleSpinBox()
        self.shaft_radius_spin.setRange(1.0, 20.0)
        self.shaft_radius_spin.setValue(5.0)
        self.shaft_radius_spin.setSingleStep(0.5)
        self.shaft_radius_spin.valueChanged.connect(self.on_shaft_radius_changed)
        general_params_layout.addRow("Shaft Radius (mm):", self.shaft_radius_spin)

        params_layout.addWidget(general_params_group) # Add general parameters group to main params layout
        
        # Stator parameters
        stator_params_group = QGroupBox("Stator")
        stator_params_layout = QFormLayout(stator_params_group)
        
        self.stator_outer_radius_spin = QDoubleSpinBox()
        self.stator_outer_radius_spin.setRange(10.0, 200.0)
        self.stator_outer_radius_spin.setValue(50.0)
        self.stator_outer_radius_spin.setSingleStep(1.0)
        self.stator_outer_radius_spin.valueChanged.connect(self.on_stator_params_changed)
        stator_params_layout.addRow("Outer Radius (mm):", self.stator_outer_radius_spin)
        
        self.stator_num_slots_spin = QSpinBox()
        self.stator_num_slots_spin.setRange(4, 100)
        self.stator_num_slots_spin.setValue(12)
        self.stator_num_slots_spin.valueChanged.connect(self.on_stator_params_changed)
        stator_params_layout.addRow("Number of Slots:", self.stator_num_slots_spin)

        self.stator_slot_opening_spin = QDoubleSpinBox()
        self.stator_slot_opening_spin.setRange(0.1, 20.0)
        self.stator_slot_opening_spin.setValue(5.0)
        self.stator_slot_opening_spin.setSingleStep(0.1)
        self.stator_slot_opening_spin.valueChanged.connect(self.on_stator_params_changed)
        stator_params_layout.addRow("Slot Opening (mm):", self.stator_slot_opening_spin)

        self.stator_tooth_width_spin = QDoubleSpinBox()
        self.stator_tooth_width_spin.setRange(1.0, 30.0)
        self.stator_tooth_width_spin.setValue(8.0)
        self.stator_tooth_width_spin.setSingleStep(0.1)
        self.stator_tooth_width_spin.valueChanged.connect(self.on_stator_params_changed)
        stator_params_layout.addRow("Tooth Width (mm):", self.stator_tooth_width_spin)

        self.stator_back_iron_spin = QDoubleSpinBox()
        self.stator_back_iron_spin.setRange(1.0, 30.0)
        self.stator_back_iron_spin.setValue(10.0)
        self.stator_back_iron_spin.setSingleStep(0.1)
        self.stator_back_iron_spin.valueChanged.connect(self.on_stator_params_changed)
        stator_params_layout.addRow("Back Iron Thickness (mm):", self.stator_back_iron_spin)

        params_layout.addWidget(stator_params_group) # Add stator parameters group to main params layout
        
        # Rotor parameters
        rotor_params_group = QGroupBox("Rotor")
        rotor_params_layout = QFormLayout(rotor_params_group)

        self.rotor_outer_radius_spin = QDoubleSpinBox()
        self.rotor_outer_radius_spin.setRange(5.0, 100.0)
        self.rotor_outer_radius_spin.setValue(29.0)
        self.rotor_outer_radius_spin.setSingleStep(1.0)
        self.rotor_outer_radius_spin.valueChanged.connect(self.on_rotor_params_changed)
        rotor_params_layout.addRow("Outer Radius (mm):", self.rotor_outer_radius_spin)
        
        self.rotor_num_poles_spin = QSpinBox()
        self.rotor_num_poles_spin.setRange(2, 20)
        self.rotor_num_poles_spin.setValue(4)
        self.rotor_num_poles_spin.valueChanged.connect(self.on_rotor_params_changed)
        rotor_params_layout.addRow("Number of Poles:", self.rotor_num_poles_spin)

        self.rotor_magnet_thickness_spin = QDoubleSpinBox()
        self.rotor_magnet_thickness_spin.setRange(1.0, 20.0)
        self.rotor_magnet_thickness_spin.setValue(5.0)
        self.rotor_magnet_thickness_spin.setSingleStep(0.1)
        self.rotor_magnet_thickness_spin.valueChanged.connect(self.on_rotor_params_changed)
        rotor_params_layout.addRow("Magnet Thickness (mm):", self.rotor_magnet_thickness_spin)

        self.rotor_magnet_width_spin = QDoubleSpinBox()
        self.rotor_magnet_width_spin.setRange(1.0, 50.0)
        self.rotor_magnet_width_spin.setValue(15.0)
        self.rotor_magnet_width_spin.setSingleStep(0.1)
        self.rotor_magnet_width_spin.valueChanged.connect(self.on_rotor_params_changed)
        rotor_params_layout.addRow("Magnet Width (mm):", self.rotor_magnet_width_spin)

        params_layout.addWidget(rotor_params_group) # Add rotor parameters group to main params layout

        main_layout.addWidget(params_group) # Add geometry parameters group to the main layout

        # --- View Controls ---
        view_group = QGroupBox("View Controls")
        view_layout = QHBoxLayout(view_group)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")
        self.reset_view_btn = QPushButton("Reset")

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_view_btn.clicked.connect(self.reset_view)

        view_layout.addWidget(self.zoom_in_btn)
        view_layout.addWidget(self.zoom_out_btn)
        view_layout.addWidget(self.reset_view_btn)

        main_layout.addWidget(view_group)

        # --- Geometry Canvas ---
        main_layout.addWidget(self.canvas) # Add the canvas to the main layout

        # Initialize zoom level
        self.zoom_level = 1.0

        # Initial plot of geometry (This will be triggered by initialize_geometry in main.py)
        # self.plot_motor_geometry(self.geometry)

    def plot_motor_geometry(self, geometry: MotorGeometry):
        """Plot the complete motor geometry"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot components
        if geometry.stator:
            self._plot_stator(ax, geometry.stator)
        
        if geometry.rotor:
            self._plot_rotor(ax, geometry.rotor)

        if geometry.rotor: # Assuming shaft is part of rotor assembly
             self._plot_shaft(ax, geometry.rotor) # Pass rotor to access inner_radius for shaft
        
        # Plot magnets (if any)
        for magnet in geometry.magnets:
            self._plot_magnet(ax, magnet)
        
        # Plot windings (if any)
        for winding in geometry.windings:
            self._plot_winding(ax, winding)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Motor Geometry')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        
        # Set axis limits with some padding
        if geometry.stator:
            limit = geometry.stator.outer_radius * 1.2
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        
        self.canvas.draw()

    def _plot_stator(self, ax, stator: Stator):
        """Plot stator geometry with slots"""
        # Create outer ring polygon
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        outer_ring_pts = [(stator.outer_radius * np.cos(theta[i]), stator.outer_radius * np.sin(theta[i])) for i in range(100)]
        outer_ring = LinearRing(outer_ring_pts)

        # Create inner ring polygon
        inner_ring_pts = [(stator.inner_radius * np.cos(theta[i]), stator.inner_radius * np.sin(theta[i])) for i in range(100)]
        inner_ring = LinearRing(inner_ring_pts)

        # Create stator body polygon (outer ring with inner hole)
        stator_body = Polygon(outer_ring, [inner_ring])

        # Create slot polygons and combine them
        slot_polygons = [Polygon([(p.x, p.y) for p in slot_poly]) for slot_poly in stator.generate_slot_polygons()]
        if slot_polygons:
             all_slots_union = unary_union(slot_polygons)
             # Subtract slots from stator body
             stator_iron = stator_body.difference(all_slots_union)
        else:
             stator_iron = stator_body

        # Plot stator iron with its domain color
        color = stator.domain.color if stator.domain else 'gray'
        if not stator_iron.is_empty:
             # Check if the result of difference is a single polygon or a multipolygon
             if isinstance(stator_iron, Polygon):
                  ax.add_patch(self._mpl_polygon(np.array(stator_iron.exterior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))
                  for interior in stator_iron.interiors:
                       ax.add_patch(self._mpl_polygon(np.array(interior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))
             else: # Handle MultiPolygon case (e.g., if stator iron is in multiple pieces)
                  for poly in stator_iron.geoms:
                       ax.add_patch(self._mpl_polygon(np.array(poly.exterior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))
                       for interior in poly.interiors:
                            ax.add_patch(self._mpl_polygon(np.array(interior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))

        # Plot slots with air domain color
        air_color = self.geometry.air_region.domain.color if self.geometry.air_region.domain else 'cyan'
        for slot_poly in slot_polygons:
             if not slot_poly.is_empty:
                  ax.add_patch(self._mpl_polygon(np.array(slot_poly.exterior.coords), facecolor=air_color, edgecolor='blue', alpha=0.5, lw=1.0))

    def _plot_rotor(self, ax, rotor: Rotor):
        """Plot rotor geometry (body excluding shaft hole) and magnets (if IPM)"""
        # Create rotor outer circle polygon
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        rotor_outer_pts = [(rotor.outer_radius * np.cos(theta[i]), rotor.outer_radius * np.sin(theta[i])) for i in range(100)]
        rotor_outer_ring = LinearRing(rotor_outer_pts)

        # Create shaft hole polygon
        shaft_pts = [(rotor.inner_radius * np.cos(theta[i]), rotor.inner_radius * np.sin(theta[i])) for i in range(100)]
        shaft_hole_ring = LinearRing(shaft_pts)

        # Create rotor body polygon (outer circle with shaft hole)
        rotor_body = Polygon(rotor_outer_ring, [shaft_hole_ring])

        # TODO: If IPM rotor, subtract magnet regions from rotor body
        # For now, assuming SPM or simple rotor body without embedded magnets affecting the fill

        # Plot rotor body with its domain color
        color = rotor.domain.color if rotor.domain else 'gray'
        if not rotor_body.is_empty:
             ax.add_patch(self._mpl_polygon(np.array(rotor_body.exterior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))
             for interior in rotor_body.interiors:
                  ax.add_patch(self._mpl_polygon(np.array(interior.coords), facecolor=color, edgecolor='k', alpha=0.8, lw=1.0))

        # Magnets are plotted separately in plot_motor_geometry

    def _plot_shaft(self, ax, rotor: Rotor):
        """Plot the shaft geometry"""
        # Create shaft circle polygon
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        shaft_pts = [(rotor.inner_radius * np.cos(theta[i]), rotor.inner_radius * np.sin(theta[i])) for i in range(100)]
        shaft_polygon = Polygon(shaft_pts)

        # Plot shaft with its domain color
        # Assuming Shaft domain is associated with AirRegion or has its own definition in geometry
        # For now, use the ShaftRegion domain color from geometry
        shaft_color = self.geometry.shaft_region.domain.color if self.geometry.shaft_region and self.geometry.shaft_region.domain else 'gray'
        if not shaft_polygon.is_empty:
             ax.add_patch(self._mpl_polygon(np.array(shaft_polygon.exterior.coords), facecolor=shaft_color, edgecolor='k', alpha=0.8, lw=1.0))

    def _plot_magnet(self, ax, magnet: PermanentMagnet):
        """Plot permanent magnet"""
        # Use magnet's generate_points to get its boundary and create a Shapely polygon
        magnet_pts = [(p.x, p.y) for p in magnet.generate_points()]
        if magnet_pts:
             magnet_polygon = Polygon(magnet_pts)

             # Use the magnet domain color for filling
             color = magnet.domain.color if magnet.domain else 'red'
             if not magnet_polygon.is_empty:
                  ax.add_patch(
                      self._mpl_polygon(np.array(magnet_polygon.exterior.coords), edgecolor='k', facecolor=color, alpha=0.8, lw=1.0)
                  )

    def _plot_winding(self, ax, winding: Winding):
        """Plot winding"""
        # Use winding's generate_points to get its boundary and create a Shapely polygon
        winding_pts = [(p.x, p.y) for p in winding.generate_points()]
        if winding_pts:
             winding_polygon = Polygon(winding_pts)

             # Use the winding domain color for filling
             color = winding.domain.color if winding.domain else 'orange'
             if not winding_polygon.is_empty:
                  ax.add_patch(
                      self._mpl_polygon(np.array(winding_polygon.exterior.coords), edgecolor='k', facecolor=color, alpha=0.8, lw=1.0)
                  )

    def _mpl_polygon(self, xy, **kwargs):
        """Create a matplotlib polygon patch from coordinates"""
        from matplotlib.patches import Polygon
        # Ensure xy is a valid sequence of points
        if len(xy) < 3:
            return None # Not a valid polygon
        return Polygon(xy, **kwargs)

    def zoom_in(self):
        """Zoom in on the plot"""
        self.zoom_level *= 1.2
        self._update_zoom()

    def zoom_out(self):
        """Zoom out of the plot"""
        self.zoom_level /= 1.2
        self._update_zoom()

    def _update_zoom(self):
        """Update the plot zoom level"""
        ax = self.figure.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Apply zoom centered around the current view center
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        width = (xlim[1] - xlim[0]) / self.zoom_level
        height = (ylim[1] - ylim[0]) / self.zoom_level
        ax.set_xlim(center_x - width / 2, center_x + width / 2)
        ax.set_ylim(center_y - height / 2, center_y + height / 2)
        self.canvas.draw()

    def reset_view(self):
        """Reset the zoom and pan to the initial view"""
        self.zoom_level = 1.0
        if self.geometry.stator:
            limit = self.geometry.stator.outer_radius * 1.2
            ax = self.figure.gca()
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        self.canvas.draw()

    def on_air_gap_changed(self, value):
        """Handle air gap value change"""
        # Update air gap in geometry and replot
        if self.geometry.stator and self.geometry.rotor:
            self.geometry.stator.inner_radius = self.geometry.rotor.outer_radius + value
            # Redraw the geometry after changing parameter
            self.plot_motor_geometry(self.geometry)
            self.geometry_updated.emit() # Emit signal after updating and plotting

    def on_shaft_radius_changed(self, value):
        """Handle shaft radius value change"""
        # Update shaft radius in geometry and replot
        if self.geometry.rotor:
            self.geometry.rotor.inner_radius = value
            # Redraw the geometry after changing parameter
            self.plot_motor_geometry(self.geometry)
            self.geometry_updated.emit() # Emit signal after updating and plotting

    def on_stator_params_changed(self):
        """Handle changes in stator parameters"""
        if self.geometry.stator and self.geometry.rotor:
            self.geometry.stator.outer_radius = self.stator_outer_radius_spin.value()
            # Recalculate inner radius based on air gap and rotor outer radius
            self.geometry.stator.inner_radius = self.geometry.rotor.outer_radius + self.air_gap_spin.value()
            self.geometry.stator.num_slots = self.stator_num_slots_spin.value()
            # Note: slot_opening, tooth_width, back_iron_thickness are related and dependent on radii and num_slots.
            # Updating them independently can lead to invalid geometry. A proper geometry model would handle interdependencies.
            # For now, update the values but be aware of potential inconsistencies if values conflict.
            self.geometry.stator.slot_opening = self.stator_slot_opening_spin.value()
            self.geometry.stator.tooth_width = self.stator_tooth_width_spin.value()
            self.geometry.stator.back_iron_thickness = self.stator_back_iron_spin.value()
            self.plot_motor_geometry(self.geometry)
            self.geometry_updated.emit()

    def on_rotor_params_changed(self):
        """Handle changes in rotor parameters"""
        if self.geometry.rotor and self.geometry.stator:
            self.geometry.rotor.outer_radius = self.rotor_outer_radius_spin.value()
            # Recalculate stator inner radius based on air gap
            self.geometry.stator.inner_radius = self.geometry.rotor.outer_radius + self.air_gap_spin.value()
            self.geometry.rotor.inner_radius = self.shaft_radius_spin.value() # Shaft radius directly sets rotor inner radius
            self.geometry.rotor.num_poles = self.rotor_num_poles_spin.value()
            # Note: Changing rotor parameters might require updating magnet positions/properties if they are IPM
            # For now, we are only changing the rotor shape, not the magnets embedded in it.
            # Magnet thickness and width are properties of the PermanentMagnet instances, not the Rotor class itself.
            # To make magnet parameters editable per magnet, we'd need a way to select and edit individual magnets.
            # For now, updating these spins won't change the existing magnet objects.
            self.rotor_magnet_thickness_spin.value() # Value is read but not used to update magnets
            self.rotor_magnet_width_spin.value() # Value is read but not used to update magnets

            self.plot_motor_geometry(self.geometry)
            self.geometry_updated.emit()

    def on_domain_changed(self, domain_name):
        """Handle domain selection change"""
        # Find the selected domain type
        for domain_type in DomainType:
            if domain_type.value == domain_name:
                self.selected_domain = domain_type
                # Update material controls based on the selected domain
                self.update_material_controls()
                # You might want to visually highlight the selected domain here
                # Redrawing the geometry could be part of highlighting
                # self.plot_motor_geometry(self.geometry) # Uncomment to redraw on domain change
                break

    def on_material_changed(self, domain_type, material, properties):
        """Handle material selection change from MaterialEditor"""
        # domain_type is None from MaterialEditor, we need to get the selected domain from GeometryViewer
        selected_domain_type = self.selected_domain
        if selected_domain_type and properties.get('color') is not None: # Ensure color is available and a domain is selected
             # Update material properties in geometry
             self.geometry.update_domain_material(
                selected_domain_type, material, properties['color']
             )
             # Redraw geometry to reflect material color change
             self.plot_motor_geometry(self.geometry)
             self.geometry_updated.emit() # Notify results viewer of geometry/material change
        elif properties.get('removed', False):
             print(f"Material {material} was removed. Update geometry to reflect default material if necessary.")
             # TODO: Implement logic in MotorGeometry or here to find and update domains that used the removed material
        else:
            print("No domain selected in Geometry Viewer to apply material change or color data is missing.")
            
    def on_color_clicked(self):
        """Handle color selection button click"""
        if self.selected_domain:
            # Get the current color of the selected domain from geometry if available, otherwise default to white
            current_color = Qt.white
            domain_props = self.geometry.get_domain_properties(self.selected_domain)
            if domain_props and domain_props.get('color'):
                 current_color = QColor(domain_props['color']) # Convert color string to QColor

            color_dialog = QColorDialog.getColor(current_color, self, "Select Domain Color")
            if color_dialog.isValid():
                color_name = color_dialog.name()
                # Update color for the selected domain
                # This color needs to be stored with the domain in MotorGeometry
                # and then used in the plotting methods.
                # For now, let's emit a signal or call a method to update the domain color
                # and then redraw the geometry.
                material = self.material_combo.currentText() # Get current material name
                # We need the full properties to update the domain correctly in MotorGeometry
                # Assuming MaterialEditor's signal provides the properties, but here we only have color.
                # This highlights a potential dependency issue: GeometryViewer needs MaterialEditor's full data.
                # For now, create a partial properties dict just with color.
                partial_properties = {'color': color_name}
                # Assuming update_domain_material can handle updating color directly
                self.geometry.update_domain_material(
                   self.selected_domain, material, color_name, partial_properties # Pass partial_properties
                )
                # Redraw geometry to reflect color change
                self.plot_motor_geometry(self.geometry)
                self.geometry_updated.emit() # Notify results viewer

    def update_material_controls(self):
        """Update material controls based on selected domain"""
        if self.selected_domain and self.geometry:
            # Get current properties for the selected domain from geometry
            props = self.geometry.get_domain_properties(self.selected_domain)
            if props:
                # Update material combo box and color button
                self.material_combo.setCurrentText(props.get('material', 'custom')) # Default to custom if material not set
                # Need to handle color button text/color display
                # For now, just update material combo
                color_name = props.get('color', 'gray')
                self.color_button.setStyleSheet(f"background-color: {color_name};")
            else:
                # Reset controls if no properties found for the domain
                self.material_combo.setCurrentIndex(0) # Select first item (air)
                # Reset color button/display
                self.color_button.setStyleSheet("") # Reset stylesheet
        else:
            # Reset controls if no domain is selected or geometry is not available
            self.material_combo.setCurrentIndex(0) # Select first item (air)
            # Reset color button/display
            self.color_button.setStyleSheet("") # Reset stylesheet

    def get_element_domain(self, element):
        # Determine which domain this element belongs to
        # This is a simplified version - you'll need to implement proper domain detection
        pass # Placeholder 