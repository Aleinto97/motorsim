from src.gui.main_window import MainWindow
from PySide6.QtCore import Qt


def test_initial_canvas(qtbot, image_regression_tester):
    """
    Tests the initial state of the canvas using our custom fixture.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(100)

    screenshot = window.canvas.plotter.screenshot(transparent_background=True)

    image_regression_tester(
        screenshot=screenshot,
        baseline_path_str="baseline_images/test_initial_canvas.png",
        threshold=1.5,
    )


def test_meshing_view(qtbot, image_regression_tester):
    """
    Tests the 2D mesh view using our custom fixture.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.tabs.setCurrentIndex(1)
    qtbot.mouseClick(window.mesher_panel.mesh_button, Qt.LeftButton)
    qtbot.wait(500)  # Allow time for meshing

    screenshot = window.canvas.plotter.screenshot(transparent_background=True)

    image_regression_tester(
        screenshot=screenshot,
        baseline_path_str="baseline_images/test_2d_mesh.png",
        threshold=2.0,
    )
