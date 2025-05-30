from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton
from PySide6.QtCore import Signal

class RotorParameterPanel(QWidget):
    parameters_changed = Signal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        self.outer_radius = QDoubleSpinBox()
        self.outer_radius.setRange(5, 100)
        self.outer_radius.setValue(29.0)
        layout.addRow('Rotor outer radius', self.outer_radius)
        self.inner_radius = QDoubleSpinBox()
        self.inner_radius.setRange(1, 98)
        self.inner_radius.setValue(10.0)
        layout.addRow('Rotor inner radius', self.inner_radius)
        self.num_poles = QSpinBox()
        self.num_poles.setRange(2, 20)
        self.num_poles.setValue(4)
        layout.addRow('Number of poles', self.num_poles)
        self.magnet_type = QSpinBox()
        self.magnet_type.setRange(0, 1)
        self.magnet_type.setValue(0)
        layout.addRow('Magnet type (0=SPM, 1=IPM)', self.magnet_type)
        self.magnet_thickness = QDoubleSpinBox()
        self.magnet_thickness.setRange(0.1, 10)
        self.magnet_thickness.setValue(3.0)
        layout.addRow('Magnet thickness', self.magnet_thickness)
        self.magnet_width = QDoubleSpinBox()
        self.magnet_width.setRange(0.1, 30)
        self.magnet_width.setValue(15.0)
        layout.addRow('Magnet width', self.magnet_width)
        self.update_btn = QPushButton('Update Rotor')
        layout.addRow(self.update_btn)
        self.setLayout(layout)
        self.update_btn.clicked.connect(self.emit_parameters)
    def emit_parameters(self):
        params = {
            'outer_radius': self.outer_radius.value(),
            'inner_radius': self.inner_radius.value(),
            'num_poles': self.num_poles.value(),
            'magnet_type': 'SPM' if self.magnet_type.value() == 0 else 'IPM',
            'magnet_thickness': self.magnet_thickness.value(),
            'magnet_width': self.magnet_width.value(),
        }
        self.parameters_changed.emit(params) 