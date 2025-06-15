"""Simple panel allowing the user to trigger 2-D meshing and display statistics."""

from PySide6.QtCore import Signal
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QTextEdit,
    QFormLayout,
)


class MesherPanel(QWidget):
    """Meshing tab with a button to generate a 2-D mesh."""

    generate_mesh_requested: Signal = Signal()
    run_analysis_requested: Signal = Signal(Path)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        group = QGroupBox("Meshing")
        form = QFormLayout(group)

        self.mesh_button = QPushButton("Generate 2D Mesh")
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setFixedHeight(80)

        form.addRow(self.mesh_button)
        form.addRow("Mesh Info:", self.info_box)

        # Analysis button (initially disabled)
        self.analysis_button = QPushButton("Run Analysis")
        self.analysis_button.setEnabled(False)
        form.addRow(self.analysis_button)

        layout.addWidget(group)
        layout.addStretch()

        self.mesh_button.clicked.connect(self.generate_mesh_requested)
        self.analysis_button.clicked.connect(self._on_run_analysis_clicked)

        # internal storage for mesh path
        self._mesh_file_path: Path | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_mesh_info(self, text: str, mesh_path: Path) -> None:
        """Update info box and enable analysis button."""
        self.info_box.setText(text)
        self._mesh_file_path = mesh_path
        self.analysis_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_run_analysis_clicked(self) -> None:
        """Emit run_analysis_requested with stored mesh path."""
        if self._mesh_file_path is not None:
            self.run_analysis_requested.emit(self._mesh_file_path)
