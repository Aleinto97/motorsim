from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QDoubleSpinBox, QComboBox, QPushButton, QGroupBox,
                              QTableWidget, QTableWidgetItem, QHeaderView, QColorDialog)
from PySide6.QtCore import Qt, Signal
import numpy as np

class MaterialEditor(QWidget):
    material_changed = Signal(object, str, dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Material selection
        selection_group = QGroupBox("Material Selection")
        selection_layout = QHBoxLayout()
        
        self.material_combo = QComboBox()
        self.material_combo.addItems(['air', 'iron', 'copper', 'magnet', 'custom'])
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        selection_layout.addWidget(QLabel("Material:"))
        selection_layout.addWidget(self.material_combo)
        
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_material)
        selection_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_material)
        selection_layout.addWidget(self.remove_btn)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Material properties
        props_group = QGroupBox("Material Properties")
        props_layout = QVBoxLayout()
        
        # Relative permeability
        mu_layout = QHBoxLayout()
        mu_layout.addWidget(QLabel("Relative Permeability (μr):"))
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(1.0, 10000.0)
        self.mu_spin.setValue(1.0)
        self.mu_spin.setSingleStep(0.1)
        mu_layout.addWidget(self.mu_spin)
        props_layout.addLayout(mu_layout)
        
        # Conductivity
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Conductivity (σ) [S/m]:"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 1e8)
        self.sigma_spin.setValue(0.0)
        self.sigma_spin.setSingleStep(1e6)
        sigma_layout.addWidget(self.sigma_spin)
        props_layout.addLayout(sigma_layout)
        
        # Remanence (for magnets)
        br_layout = QHBoxLayout()
        br_layout.addWidget(QLabel("Remanence (Br) [T]:"))
        self.br_spin = QDoubleSpinBox()
        self.br_spin.setRange(0.0, 2.0)
        self.br_spin.setValue(0.0)
        self.br_spin.setSingleStep(0.1)
        br_layout.addWidget(self.br_spin)
        props_layout.addLayout(br_layout)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Display Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'gray'])
        color_layout.addWidget(self.color_combo)
        props_layout.addLayout(color_layout)
        
        # Apply button
        self.apply_btn = QPushButton("Apply Properties")
        self.apply_btn.clicked.connect(self.apply_properties)
        props_layout.addWidget(self.apply_btn)
        
        props_group.setLayout(props_layout)
        layout.addWidget(props_group)
        
        # Material library
        library_group = QGroupBox("Material Library")
        library_layout = QVBoxLayout()
        
        self.material_table = QTableWidget()
        self.material_table.setColumnCount(4)
        self.material_table.setHorizontalHeaderLabels(['Material', 'μr', 'σ [S/m]', 'Br [T]'])
        self.material_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        library_layout.addWidget(self.material_table)
        
        library_group.setLayout(library_layout)
        layout.addWidget(library_group)
        
        self.setLayout(layout)
        
        # Initialize with default materials
        self.update_material_table()
        
    def on_material_changed(self, material_name):
        """Update property controls when material selection changes"""
        if material_name == 'air':
            self.mu_spin.setValue(1.0)
            self.sigma_spin.setValue(0.0)
            self.br_spin.setValue(0.0)
            self.color_combo.setCurrentText('cyan')
        elif material_name == 'iron':
            self.mu_spin.setValue(1000.0)
            self.sigma_spin.setValue(1e6)
            self.br_spin.setValue(0.0)
            self.color_combo.setCurrentText('gray')
        elif material_name == 'copper':
            self.mu_spin.setValue(1.0)
            self.sigma_spin.setValue(5.8e7)
            self.br_spin.setValue(0.0)
            self.color_combo.setCurrentText('orange')
        elif material_name == 'magnet':
            self.mu_spin.setValue(1.05)
            self.sigma_spin.setValue(0.0)
            self.br_spin.setValue(1.2)
            self.color_combo.setCurrentText('red')
        # For custom materials, the values in the spins and combo box already reflect the current state
            
    def add_material(self):
        """Add a new material to the library"""
        name = f"custom_{self.material_combo.count()}"
        self.material_combo.addItem(name)
        self.material_combo.setCurrentText(name)
        self.update_material_table()
        # Emit signal for new material
        # We need to emit a dictionary with expected keys, even if values are defaults
        props = {
            'mu_r': self.mu_spin.value(),
            'sigma': self.sigma_spin.value(),
            'Br': self.br_spin.value(),
            'color': self.color_combo.currentText()
        }
        self.material_changed.emit(None, name, props)
        
    def remove_material(self):
        """Remove the selected material"""
        current = self.material_combo.currentText()
        if current.startswith('custom'):
            currentIndex = self.material_combo.currentIndex()
            # Emit signal before removing the item, indicating which material is being removed
            # Pass a dictionary with keys the receiver expects, even if values signify removal
            removed_props = {'mu_r': None, 'sigma': None, 'Br': None, 'color': None, 'removed': True}
            self.material_changed.emit(None, current, removed_props)
            
            self.material_combo.removeItem(currentIndex)
            self.update_material_table()
            
    def apply_properties(self):
        """Apply the current material properties"""
        self.update_material_table()
        # Emit signal with current material and properties
        material = self.material_combo.currentText()
        props = self.get_material_properties(material)
        if props:
            self.material_changed.emit(None, material, props)
        
    def update_material_table(self):
        """Update the material library table"""
        self.material_table.setRowCount(self.material_combo.count())
        for i in range(self.material_combo.count()):
            material = self.material_combo.itemText(i)
            self.material_table.setItem(i, 0, QTableWidgetItem(material))
            
            # Set default values based on material type or use spinbox values for custom
            if material == 'air':
                self.material_table.setItem(i, 1, QTableWidgetItem('1.0'))
                self.material_table.setItem(i, 2, QTableWidgetItem('0.0'))
                self.material_table.setItem(i, 3, QTableWidgetItem('0.0'))
            elif material == 'iron':
                self.material_table.setItem(i, 1, QTableWidgetItem('1000.0'))
                self.material_table.setItem(i, 2, QTableWidgetItem('1e6'))
                self.material_table.setItem(i, 3, QTableWidgetItem('0.0'))
            elif material == 'copper':
                self.material_table.setItem(i, 1, QTableWidgetItem('1.0'))
                self.material_table.setItem(i, 2, QTableWidgetItem('5.8e7'))
                self.material_table.setItem(i, 3, QTableWidgetItem('0.0'))
            elif material == 'magnet':
                self.material_table.setItem(i, 1, QTableWidgetItem('1.05'))
                self.material_table.setItem(i, 2, QTableWidgetItem('0.0'))
                self.material_table.setItem(i, 3, QTableWidgetItem('1.2'))
            else:
                # Custom material - use current values from spin boxes
                # Find the row for this custom material to update its values if they changed
                # A more robust approach would store custom materials in a dictionary
                # For now, let's just use the current spinbox values when updating the table if it's a custom material being edited
                # This part needs refinement for proper custom material management.
                # As a temporary fix to populate the table for custom materials when update_material_table is called:
                if material == self.material_combo.currentText() and material.startswith('custom'):
                     self.material_table.setItem(i, 1, QTableWidgetItem(str(self.mu_spin.value())))
                     self.material_table.setItem(i, 2, QTableWidgetItem(str(self.sigma_spin.value())))
                     self.material_table.setItem(i, 3, QTableWidgetItem(str(self.br_spin.value())))
                else:
                     # For other custom materials in the list, we don't have their values readily available here
                     # This highlights the need for a proper internal material dictionary
                     pass # Cannot populate table for other custom materials without stored data

    def get_material_properties(self, material_name: str) -> dict:
        """Get properties for a specific material"""
        for i in range(self.material_table.rowCount()):
            if self.material_table.item(i, 0).text() == material_name:
                # Retrieve properties from the table
                try:
                    mu_r = float(self.material_table.item(i, 1).text())
                    sigma = float(self.material_table.item(i, 2).text())
                    Br = float(self.material_table.item(i, 3).text())
                    # We also need the color, which is not stored in the table
                    # For now, let's try to get the color from the color combo if the material is currently selected
                    # This is not a reliable way to get color for any material in the list.
                    # A proper material dictionary is needed.
                    color = self.color_combo.currentText() if self.material_combo.currentText() == material_name else 'gray' # Default color
                    return {
                        'mu_r': mu_r,
                        'sigma': sigma,
                        'Br': Br,
                        'color': color # Include color
                    }
                except ValueError:
                    print(f"Error converting material properties for {material_name}")
                    return None
        return None 