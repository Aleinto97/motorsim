from PySide6.QtWidgets import QWidget, QVBoxLayout
from pyvistaqt import QtInteractor


class MainCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a layout that will hold the plotter
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create the PyVista Qt Plotter
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Add some basic visualization aids
        self.plotter.add_axes()
        self.plotter.add_bounding_box()
        self.plotter.show()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def display_solution(self, grid):
        """Display a PyVista grid with scalar 'B_mag'."""
        if grid is None:
            return
        self.plotter.clear()
        self.plotter.add_mesh(grid, scalars="B_mag", cmap="magma", show_edges=False)
        self.plotter.add_scalar_bar(title="|B| [T]")
        self.plotter.view_xy()
        self.plotter.reset_camera()
