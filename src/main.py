import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QDockWidget, 
                              QWidget, QVBoxLayout, QHBoxLayout, QSplitter)
from PySide6.QtCore import Qt
from src.gui.geometry_viewer import GeometryViewer
from src.gui.material_editor import MaterialEditor
from src.gui.results_viewer import ResultsViewer
from src.geometry.motor_components import MotorGeometry, Stator, Rotor, PermanentMagnet, Winding, Point2D
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MotorSim")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Create geometry viewer
        self.geometry_viewer = GeometryViewer()
        splitter.addWidget(self.geometry_viewer)
        
        # Create results viewer
        self.results_viewer = ResultsViewer()
        splitter.addWidget(self.results_viewer)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 400])
        
        # Create material editor dock
        material_dock = QDockWidget("Material Editor", self)
        material_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.material_editor = MaterialEditor()
        material_dock.setWidget(self.material_editor)
        self.addDockWidget(Qt.RightDockWidgetArea, material_dock)
        
        # Connect signals
        self.geometry_viewer.geometry_updated.connect(self.on_geometry_updated)
        self.material_editor.material_changed.connect(self.on_material_changed)
        
        # Initialize with default geometry
        self.initialize_geometry()
        
    def initialize_geometry(self):
        # Create default stator
        stator = Stator(
            outer_radius=50.0,
            inner_radius=30.0,
            num_slots=12,
            slot_opening=5.0,
            tooth_width=8.0,
            back_iron_thickness=10.0
        )
        
        # Create default rotor
        rotor = Rotor(
            outer_radius=29.0,
            inner_radius=5.0,
            num_poles=4,
            magnet_type="SPM",
            magnet_thickness=5.0,
            magnet_width=15.0
        )
        
        # Add components to geometry
        self.geometry_viewer.geometry.add_stator(stator)
        self.geometry_viewer.geometry.add_rotor(rotor)
        
        # Add some magnets
        for i in range(4):
            angle = i * (2 * 3.14159 / 4)
            magnet = PermanentMagnet(
                position=Point2D(25.0 * np.cos(angle), 25.0 * np.sin(angle)),
                width=15.0,
                thickness=5.0,
                angle=angle,
                magnetization=1.2
            )
            self.geometry_viewer.geometry.add_magnet(magnet)
            
        # Update the view (this will also trigger results viewer update)
        self.geometry_viewer.plot_motor_geometry(self.geometry_viewer.geometry)
        self.geometry_viewer.geometry_updated.emit()
        
    def on_geometry_updated(self):
        # Update results viewer with new geometry
        self.results_viewer.update_geometry(self.geometry_viewer.geometry)
        
    def on_material_changed(self, domain_type, material, properties):
        # Check if material was removed
        if properties.get('removed', False):
            print(f"Material {material} was removed.")
            # Handle material removal, e.g., revert domains using this material to a default material
            # This part requires implementing logic in MotorGeometry to find and update domains by material.
            # For now, we just prevent the error.
            pass 
        else:
            # Update material properties in geometry
            # domain_type is None from MaterialEditor, we need to get the selected domain from GeometryViewer
            selected_domain_type = self.geometry_viewer.selected_domain
            if selected_domain_type:
                 self.geometry_viewer.geometry.update_domain_material(
                    selected_domain_type, material, properties['color']
                 )
                 self.geometry_viewer.plot_motor_geometry(self.geometry_viewer.geometry) # Redraw geometry
                 self.geometry_viewer.geometry_updated.emit() # Notify results viewer
            else:
                print("No domain selected in Geometry Viewer to apply material change.")

def main():
    print("Starting main function...")
    app = QApplication(sys.argv)
    print("QApplication created.")
    
    window = MainWindow()
    print("MainWindow created.")
    
    window.show()
    print("MainWindow.show() called.")
    
    print("Entering application event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    print("--- Script starting ---")
    main() 