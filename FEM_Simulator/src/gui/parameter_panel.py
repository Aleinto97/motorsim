"""Interactive parameter panel.

Spin-boxes are tied to a `MotorParameters` dataclass instance.  Whenever the
user edits a value the panel emits a `parameters_changed` Qt signal carrying the
updated model, allowing the rest of the UI to react immediately.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QDoubleSpinBox,
)

from src.core.models import MotorParameters


class ParameterPanel(QWidget):
    """Side panel exposing the motor parameters for editing."""

    # Qt signal – emitted whenever any parameter changes
    parameters_changed: Signal = Signal(MotorParameters)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedWidth(350)
        self.motor_model: MotorParameters | None = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Create all group boxes and spin boxes & wire them up."""
        # -------------------- Machine parameters ------------------
        machine_group = QGroupBox("Machine Parameters")
        machine_layout = QFormLayout(machine_group)
        self.motor_length = self._create_spinbox(machine_layout, "Axial Length (m):")

        self.main_layout.addWidget(machine_group)

        # ------------------------- Stator -------------------------
        stator_group = QGroupBox("Stator Parameters")
        stator_layout = QFormLayout(stator_group)

        self.stator_rext = self._create_spinbox(stator_layout, "Outer Radius (m):")
        self.stator_rint = self._create_spinbox(stator_layout, "Inner Radius (m):")
        self.stator_zs = self._create_spinbox(stator_layout, "Number of Slots (Zs):")

        self.main_layout.addWidget(stator_group)

        # ------------------------- Rotor --------------------------
        rotor_group = QGroupBox("Rotor Parameters")
        rotor_layout = QFormLayout(rotor_group)

        self.rotor_rext = self._create_spinbox(rotor_layout, "Outer Radius (m):")
        self.rotor_rint = self._create_spinbox(rotor_layout, "Inner Radius (m):")

        self.main_layout.addWidget(rotor_group)

        # ---------------------- Magnet Hole ----------------------
        hole_group = QGroupBox("Magnet Hole Parameters (V-Shape)")
        hole_layout = QFormLayout(hole_group)

        self.hole_h1 = self._create_spinbox(hole_layout, "V-Tip Depth (H1):")
        self.hole_w1 = self._create_spinbox(hole_layout, "Magnet Separation (W1):")
        self.hole_bridge_width = self._create_spinbox(hole_layout, "Bridge Width (m):")
        self.hole_rib_width = self._create_spinbox(hole_layout, "Rib Width (m):")

        self.main_layout.addWidget(hole_group)

        # ------------------------- Magnet ------------------------
        magnet_group = QGroupBox("Magnet Parameters")
        magnet_layout = QFormLayout(magnet_group)

        self.magnet_hmag = self._create_spinbox(magnet_layout, "Height (Hmag):")
        self.magnet_wmag = self._create_spinbox(magnet_layout, "Width (Wmag):")

        self.main_layout.addWidget(magnet_group)

        # Push everything to the top
        self.main_layout.addStretch()

    def _create_spinbox(self, layout: QFormLayout, label: str) -> QDoubleSpinBox:
        """Helper that sets sane ranges/precision and connects valueChanged."""
        sb = QDoubleSpinBox()
        sb.setRange(0.0001, 1.0)
        sb.setSingleStep(0.0005)
        sb.setDecimals(5)
        layout.addRow(label, sb)
        sb.valueChanged.connect(self._on_parameter_changed)
        return sb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_from_model(self, model: MotorParameters) -> None:
        """Populate widgets with *model* values without emitting signals."""
        self.motor_model = model

        # Temporarily block signals while setting values
        for sb, value in [
            (self.stator_rext, model.stator.Rext),
            (self.stator_rint, model.stator.Rint),
            (self.stator_zs, model.stator.slot.Zs),
            (self.rotor_rext, model.rotor.Rext),
            (self.rotor_rint, model.rotor.Rint),
            (self.hole_h1, model.rotor.hole_v.H1),
            (self.hole_w1, model.rotor.hole_v.W1),
            (self.hole_bridge_width, model.rotor.hole_v.bridge_width),
            (self.hole_rib_width, model.rotor.hole_v.rib_width),
            (self.magnet_hmag, model.rotor.hole_v.magnet_left.Hmag),
            (self.magnet_wmag, model.rotor.hole_v.magnet_left.Wmag),
            (self.motor_length, model.L_motor),
        ]:
            sb.blockSignals(True)
            sb.setValue(value)
            sb.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------
    def _on_parameter_changed(self) -> None:
        """Called whenever any spin-box value changes."""
        if self.motor_model is None:
            return

        # Update dataclass from widgets
        self.motor_model.stator.Rext = self.stator_rext.value()
        self.motor_model.stator.Rint = self.stator_rint.value()
        self.motor_model.stator.slot.Zs = int(self.stator_zs.value())

        self.motor_model.rotor.Rext = self.rotor_rext.value()
        self.motor_model.rotor.Rint = self.rotor_rint.value()

        hole_v = self.motor_model.rotor.hole_v
        hole_v.H1 = self.hole_h1.value()
        hole_v.W1 = self.hole_w1.value()
        hole_v.bridge_width = self.hole_bridge_width.value()
        hole_v.rib_width = self.hole_rib_width.value()

        hole_v.magnet_left.Hmag = self.magnet_hmag.value()
        hole_v.magnet_left.Wmag = self.magnet_wmag.value()
        hole_v.magnet_right.Hmag = self.magnet_hmag.value()
        hole_v.magnet_right.Wmag = self.magnet_wmag.value()

        # axial length
        self.motor_model.L_motor = self.motor_length.value()

        # Emit updated model
        self.parameters_changed.emit(self.motor_model)
