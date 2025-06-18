"""Generate baseline images for visual-regression tests.

Run with the target environment activated, e.g.::

    conda run -n motorcad_app python -m tests.generate_baseline
"""

import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def capture(window: MainWindow, filename: str) -> None:
    """Helper to screenshot current canvas and save under *filename*."""
    from PIL import Image
    path = BASELINE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(window.canvas.plotter.screenshot(transparent_background=True)).save(path)
    print(f"Baseline saved → {path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    BASELINE_DIR = Path(__file__).parent / "baseline_images"

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.processEvents()

    # 1. Outline baseline
    capture(win, "test_full_motor_geometry.png")

    # 2. Mesh baseline – switch tab and click mesh
    win.tabs.setCurrentIndex(1)
    win.mesher_panel.mesh_button.click()
    app.processEvents()
    capture(win, "test_2d_mesh.png")

    win.close()
