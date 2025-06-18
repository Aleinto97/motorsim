"""Main window with parameter and meshing tabs."""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
)
from PySide6.QtCore import Qt
from pathlib import Path
import tempfile

from .canvas import MainCanvas
from .parameter_panel import ParameterPanel
from .mesher_panel import MesherPanel
from src.core.models import MotorParameters
from src.geometry.builder_sector import build_geometry_sector
from src.analysis.solver import run_analysis


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motor-CAD (Meshing Enabled)")
        self.setGeometry(100, 100, 1600, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget left side
        self.tabs = QTabWidget()
        self.parameter_panel = ParameterPanel(self)
        self.mesher_panel = MesherPanel(self)
        self.tabs.addTab(self.parameter_panel, "Parameters")
        self.tabs.addTab(self.mesher_panel, "Meshing")

        # 3-D canvas
        self.canvas = MainCanvas(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([400, 1200])
        main_layout.addWidget(splitter)

        # Signals
        self.parameter_panel.parameters_changed.connect(self.on_parameters_changed)
        self.mesher_panel.generate_mesh_requested.connect(self.generate_2d_mesh)
        self.mesher_panel.run_analysis_requested.connect(self.run_solver)

        # Initial model
        self.load_model()

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self.motor_model = MotorParameters()
        self.parameter_panel.update_from_model(self.motor_model)
        self.update_geometry_preview()

    def on_parameters_changed(self, model: MotorParameters) -> None:
        self.motor_model = model
        self.update_geometry_preview()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def update_geometry_preview(self) -> None:
        mesh = build_geometry_sector(self.motor_model, mesh_params={}, mesh_2d=False)
        if mesh is None:
            return
        plotter = self.canvas.plotter
        plotter.clear()
        plotter.add_mesh(mesh, style="wireframe", color="blue")
        plotter.reset_camera()
        plotter.view_isometric()

    def generate_2d_mesh(self) -> None:
        """Generate a 2-D mesh, save it to a temporary .msh file, and preview it."""
        print("--- Generate 2D Mesh button clicked ---")
        # Temporary file kept as attribute to prolong lifetime
        self._temp_mesh_file = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
        mesh_path = Path(self._temp_mesh_file.name)

        mesh = build_geometry_sector(
            self.motor_model,
            mesh_params={"global_size": 20},
            mesh_2d=True,
            save_path=str(mesh_path),
        )
        if mesh is None:
            self.mesher_panel.info_box.setText("Meshing failed.")
            return

        self.mesher_panel.set_mesh_info(
            f"Cells: {mesh.n_cells}\nPoints: {mesh.n_points}",
            mesh_path=mesh_path,
        )
        plotter = self.canvas.plotter
        plotter.clear()
        plotter.add_mesh(mesh, show_edges=True, color="lightgrey")
        plotter.reset_camera()
        plotter.view_isometric()

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------
    def run_solver(self, mesh_path: Path) -> None:
        """Invoked when the user clicks the Run Analysis button."""
        phase_currents = self.mesher_panel.get_phase_currents()
        results = run_analysis(mesh_path, phase_currents, model=self.motor_model)
        if results and results.get("grid") is not None:
            self.canvas.display_solution(results["grid"])
