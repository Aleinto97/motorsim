import optuna
from typing import Callable, Dict, Any, Union, Optional
import logging
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

logger = logging.getLogger(__name__)

class MotorOptimizer:
    """
    Integrates Optuna for motor design optimization.
    """

    def __init__(self, objective_function: Callable, n_trials: int = 100, timeout: int = 60, study_name: str = "motor_optimization"):
        """
        Initializes the optimizer with an objective function and Optuna study settings.

        Args:
            objective_function: The objective function to optimize
            n_trials: Maximum number of trials
            timeout: Maximum time in seconds
            study_name: Name of the study
        """
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.study = None

    def create_study(self, direction: str = None, directions: list = None, load_if_exists: bool = True):
        """
        Creates or loads an Optuna study.

        Args:
            direction: "minimize" or "maximize" for the objective function (single-objective).
            directions: List of directions for multi-objective optimization (e.g., ["minimize", "minimize"]).
            load_if_exists: If True, loads an existing study if available.
        """
        try:
            if directions is not None:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    storage="sqlite:///:memory:",
                    directions=directions,
                    load_if_exists=load_if_exists
                )
                logger.info(f"Optuna study '{self.study_name}' created/loaded with directions '{directions}'.")
            else:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    storage="sqlite:///:memory:",
                    direction=direction or "minimize",
                    load_if_exists=load_if_exists
                )
                logger.info(f"Optuna study '{self.study_name}' created/loaded with direction '{direction or 'minimize'}'.")
        except Exception as e:
            logger.error(f"Error creating/loading Optuna study: {e}")
            raise

    def optimize(self):
        """Run the optimization."""
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            storage="sqlite:///:memory:"
        )
        
        self.study.optimize(
            self.objective_function,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        return self.study

    def get_best_parameters(self) -> Dict[str, Any]:
        """Returns the parameters of the best trial."""
        if self.study and self.study.best_trial:
            return self.study.best_trial.params
        return {}

    def get_best_value(self) -> float:
        """Returns the objective value of the best trial."""
        if self.study and self.study.best_trial:
            return self.study.best_trial.value
        raise RuntimeError("No best trial found. Run optimization first.")

    def plot_optimization_history(self):
        """Plots the optimization history using matplotlib."""
        if self.study is None:
            raise RuntimeError("Optuna study not created. Call create_study first.")
        try:
            fig = plot_optimization_history(self.study)
            plt.close(fig)  # Close the figure to avoid displaying it
            logger.info("Optimization history plot generated.")
            return fig
        except Exception as e:
            logger.warning(f"Could not generate optimization history plot. Ensure matplotlib is installed: {e}")

    def plot_param_importances(self):
        """Plots parameter importances using matplotlib."""
        if self.study is None:
            raise RuntimeError("Optuna study not created. Call create_study first.")
        try:
            fig = plot_param_importances(self.study)
            plt.close(fig)  # Close the figure to avoid displaying it
            logger.info("Parameter importances plot generated.")
            return fig
        except Exception as e:
            logger.warning(f"Could not generate parameter importances plot. Ensure matplotlib is installed: {e}")
