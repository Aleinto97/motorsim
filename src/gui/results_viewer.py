from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QDoubleSpinBox, QGroupBox, QFormLayout)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from src.core.magnetic_physics import MagneticPhysics
from src.geometry.motor_components import MotorGeometry

class ResultsViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.physics = MagneticPhysics()
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create canvas for plots
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)
        
        # Create simulation parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()
        
        # Current control
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setRange(-100, 100)
        self.current_spin.setValue(10)
        self.current_spin.setSuffix(" A")
        params_layout.addRow("Current:", self.current_spin)
        
        # Angle control
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 360)
        self.angle_spin.setValue(0)
        self.angle_spin.setSuffix("°")
        params_layout.addRow("Rotor Angle:", self.angle_spin)
        
        # Update button
        self.update_btn = QPushButton("Update Simulation")
        self.update_btn.clicked.connect(self.update_simulation)
        params_layout.addRow("", self.update_btn)

        # Plot Torque-Angle button
        self.plot_torque_angle_btn = QPushButton("Plot Torque-Angle")
        self.plot_torque_angle_btn.clicked.connect(self.plot_torque_angle_curve)
        params_layout.addRow("", self.plot_torque_angle_btn)

        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Label to display instantaneous torque
        self.torque_label = QLabel("Torque: 0.00 N⋅m")
        layout.addWidget(self.torque_label)

    def update_geometry(self, geometry: MotorGeometry):
        self.geometry = geometry
        self.update_simulation()
        
    def update_simulation(self):
        if not hasattr(self, 'geometry'):
            # print("Geometry not available for simulation.")
            return
            
        # Update physics parameters
        self.physics.current_density = self.current_spin.value() / 1e6  # Convert to A/mm²
        self.physics.rotor_angle = np.radians(self.angle_spin.value())
        
        # Calculate torque and fields for the *current* angle
        torque = self.physics.calculate_torque(self.geometry, self.physics.rotor_angle)
        
        # Update instantaneous torque label
        self.torque_label.setText(f'Torque: {torque:.2f} N⋅m')

        # Update plots (Vector Potential and Magnetic Field Magnitude)
        self._update_field_plots()

    def _update_field_plots(self):
         """Update the vector potential and magnetic field magnitude plots."""
         if self.physics.points is None or self.physics.triangles is None or \
            self.physics.vector_potential is None or self.physics.Bx is None or self.physics.By is None:
              # print("Physics results not available for field plots.")
              return

         self.canvas.figure.clear()
         
         # Plot vector potential
         ax1 = self.canvas.figure.add_subplot(121)
         ax1.tripcolor(self.physics.points[:, 0], self.physics.points[:, 1],
                      self.physics.triangles, self.physics.vector_potential,
                      shading='flat', cmap='viridis')
         ax1.set_title('Vector Potential')
         ax1.set_aspect('equal')
         ax1.set_xlabel('X [mm]')
         ax1.set_ylabel('Y [mm]')
         
         # Plot magnetic field magnitude
         ax2 = self.canvas.figure.add_subplot(122)
         B_mag = np.sqrt(self.physics.Bx**2 + self.physics.By**2)
         ax2.tripcolor(self.physics.points[:, 0], self.physics.points[:, 1],
                      self.physics.triangles, B_mag,
                      shading='flat', cmap='plasma')
         ax2.set_title('Magnetic Field Magnitude')
         ax2.set_aspect('equal')
         ax2.set_xlabel('X [mm]')
         ax2.set_ylabel('Y [mm]')
         
         # Add torque value to plot (as a label)
         # self.canvas.figure.suptitle(f'Torque: {torque:.2f} N⋅m') # Removed to use label

         self.canvas.figure.tight_layout()
         self.canvas.draw()

    def plot_torque_angle_curve(self):
        """Calculate and plot the torque-angle characteristic curve."""
        if not hasattr(self, 'geometry'):
            print("Geometry not available to plot torque-angle curve.")
            return

        # Define angle range for the sweep (e.g., 0 to 360 degrees)
        angles_deg = np.linspace(0, 360, 72, endpoint=False) # 72 points for a smooth curve
        angles_rad = np.radians(angles_deg)
        torque_values = []

        print("Calculating torque-angle curve...")
        # Iterate through angles, calculate torque, and store results
        for angle in angles_rad:
            # Update rotor angle in physics
            self.physics.rotor_angle = angle
            
            # Calculate torque for this angle
            torque = self.physics.calculate_torque(self.geometry, angle)
            torque_values.append(torque)
            # Optional: Update a progress bar or print progress
            # print(f"Calculated torque for angle {np.degrees(angle):.1f}°")

        print("Torque-angle calculation complete.")

        # Plot the torque-angle curve
        self.canvas.figure.clear() # Clear previous plots
        ax = self.canvas.figure.add_subplot(111) # Use a single subplot for the curve

        ax.plot(angles_deg, torque_values) # Plot torque vs. angle
        ax.set_title('Torque-Angle Characteristic')
        ax.set_xlabel('Rotor Angle [degrees]')
        ax.set_ylabel('Torque [N⋅m]')
        ax.grid(True)

        self.canvas.figure.tight_layout()
        self.canvas.draw() 