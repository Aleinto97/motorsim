"""
Fixture-based visual regression testing utilities.

Provides:
1. --generate-baselines CLI flag to update baseline images.
2. image_regression_tester fixture for comparing screenshots to baselines.
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for 'import src'
sys.path.append(str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    """Adds our custom command-line flag to pytest."""
    parser.addoption(
        "--generate-baselines",
        action="store_true",
        default=False,
        help="Generate baseline images for visual tests.",
    )


@pytest.fixture
def image_regression_tester(request):
    """
    Returns a comparison function that visual tests can call.
    """
    generate_baselines = request.config.getoption("--generate-baselines")
    def _compare(screenshot: np.ndarray, baseline_path_str: str, threshold: float):
        """Compare screenshot with baseline or generate baseline."""
        if not isinstance(screenshot, np.ndarray):
            pytest.fail("Test did not produce a valid NumPy image array for comparison.")

        baseline_path = Path(request.fspath).parent / baseline_path_str

        # Generate baseline images
        if generate_baselines:
            print(f"\n[Visual Test] Generating baseline: {baseline_path.name}")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(screenshot).save(baseline_path)
            return

        if not baseline_path.exists():
            pytest.skip(
                f"Baseline image not found: {baseline_path}. Run pytest with --generate-baselines to create it."
            )

        baseline_img = np.array(Image.open(baseline_path).convert("RGB"))
        test_img_rgb = np.array(Image.fromarray(screenshot).convert("RGB"))

        if baseline_img.shape != test_img_rgb.shape:
            pytest.fail(
                f"Image dimension mismatch. Baseline: {baseline_img.shape}, New: {test_img_rgb.shape}",
                pytrace=False,
            )

        diff_percentage = (
            np.mean(np.abs(baseline_img.astype(float) - test_img_rgb.astype(float))) / 255
        ) * 100

        print(f" [Image diff: {diff_percentage:.2f}%]")

        if diff_percentage > threshold:
            pytest.fail(
                f"Image regression failed. Diff: {diff_percentage:.2f}% > Threshold: {threshold:.2f}%",
                pytrace=False,
            )

    return _compare
