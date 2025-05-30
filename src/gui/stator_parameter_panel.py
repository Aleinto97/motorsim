from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton, QVBoxLayout
from PySide6.QtCore import Signal

class StatorParameterPanel(QWidget):
    parameters_changed = Signal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout()
        self.outer_radius = QDoubleSpinBox()
        self.outer_radius.setRange(10, 200)
        self.outer_radius.setValue(50.0)
        layout.addRow('Outer radius', self.outer_radius)
        self.inner_radius = QDoubleSpinBox()
        self.inner_radius.setRange(5, 199)
        self.inner_radius.setValue(30.0)
        layout.addRow('Inner radius', self.inner_radius)
        self.num_slots = QSpinBox()
        self.num_slots.setRange(2, 96)
        self.num_slots.setValue(12)
        layout.addRow('Number of slots', self.num_slots)
        self.slot_opening = QDoubleSpinBox()
        self.slot_opening.setRange(0.1, 20)
        self.slot_opening.setValue(3.0)
        layout.addRow('Slot opening', self.slot_opening)
        self.tooth_width = QDoubleSpinBox()
        self.tooth_width.setRange(0.1, 20)
        self.tooth_width.setValue(5.0)
        layout.addRow('Tooth width', self.tooth_width)
        self.back_iron_thickness = QDoubleSpinBox()
        self.back_iron_thickness.setRange(0.1, 20)
        self.back_iron_thickness.setValue(5.0)
        layout.addRow('Back iron thickness', self.back_iron_thickness)
        self.update_btn = QPushButton('Update Geometry')
        layout.addRow(self.update_btn)
        self.setLayout(layout)
        self.update_btn.clicked.connect(self.emit_parameters)
    def emit_parameters(self):
        params = {
            'outer_radius': self.outer_radius.value(),
            'inner_radius': self.inner_radius.value(),
            'num_slots': self.num_slots.value(),
            'slot_opening': self.slot_opening.value(),
            'tooth_width': self.tooth_width.value(),
            'back_iron_thickness': self.back_iron_thickness.value(),
        }
        self.parameters_changed.emit(params) 