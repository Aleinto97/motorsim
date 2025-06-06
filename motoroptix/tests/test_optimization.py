import pytest
import numpy as np
import optuna
from motoroptix.optimization.optimizer import MotorOptimizer

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def simple_objective_function():
    """Returns a simple objective function for testing."""
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return (x - 2)**2 + (y + 3)**2
    return objective

@pytest.fixture(scope="module")
def constrained_objective_function():
    """Returns an objective function with constraints for testing."""
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        
        # Constraint: x + y <= 0
        if x + y > 0:
            return float('inf')
        
        return (x - 2)**2 + (y + 3)**2
    return objective

# --- Test Cases ---

def test_objective_function_evaluation(simple_objective_function):
    """Test objective function evaluation."""
    optimizer = MotorOptimizer(
        objective_function=simple_objective_function,
        study_name="test_objective_evaluation",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    
    # Test single trial
    trial = optimizer.study.ask()
    value = simple_objective_function(trial)
    optimizer.study.tell(trial, value)
    
    assert isinstance(value, float)
    assert not np.isinf(value)
    assert not np.isnan(value)

def test_constraint_handling(constrained_objective_function):
    """Test constraint handling in optimization."""
    optimizer = MotorOptimizer(
        objective_function=constrained_objective_function,
        study_name="test_constraints",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    optimizer.optimize(n_trials=10)
    
    # Verify that constraints are satisfied in best solution
    best_params = optimizer.get_best_parameters()
    assert best_params["x"] + best_params["y"] <= 0

def test_optimizer_integration():
    """Test integration with different optimization algorithms."""
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return x**2
    
    # Test different samplers
    samplers = [
        optuna.samplers.TPESampler(),
        optuna.samplers.CmaEsSampler(),
        optuna.samplers.NSGAIISampler()
    ]
    
    for sampler in samplers:
        optimizer = MotorOptimizer(
            objective_function=objective,
            study_name=f"test_sampler_{sampler.__class__.__name__}",
            storage="sqlite:///:memory:"
        )
        
        optimizer.create_study(direction="minimize", sampler=sampler)
        optimizer.optimize(n_trials=5)
        
        assert optimizer.study is not None
        assert optimizer.study.best_value is not None

def test_convergence_behavior(simple_objective_function):
    """Test convergence behavior for simple optimization problems."""
    optimizer = MotorOptimizer(
        objective_function=simple_objective_function,
        study_name="test_convergence",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    
    # Run optimization with increasing number of trials
    n_trials_list = [5, 10, 20]
    best_values = []
    
    for n_trials in n_trials_list:
        optimizer.optimize(n_trials=n_trials)
        best_values.append(optimizer.get_best_value())
    
    # Verify that the best value improves or stays the same
    for i in range(1, len(best_values)):
        assert best_values[i] <= best_values[i-1]

def test_parameter_bounds(simple_objective_function):
    """Test handling of parameters at their bounds."""
    optimizer = MotorOptimizer(
        objective_function=simple_objective_function,
        study_name="test_parameter_bounds",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    optimizer.optimize(n_trials=10)
    
    best_params = optimizer.get_best_parameters()
    
    # Verify that parameters are within bounds
    assert -10 <= best_params["x"] <= 10
    assert -10 <= best_params["y"] <= 10
    
    # Test edge cases
    def edge_objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return abs(x - 10) + abs(y + 10)  # Should push parameters to bounds
    
    optimizer = MotorOptimizer(
        objective_function=edge_objective,
        study_name="test_edge_cases",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    optimizer.optimize(n_trials=10)
    
    best_params = optimizer.get_best_parameters()
    assert abs(best_params["x"] - 10) < 1e-6
    assert abs(best_params["y"] + 10) < 1e-6

def test_optimization_history(simple_objective_function):
    """Test optimization history tracking and visualization."""
    optimizer = MotorOptimizer(
        objective_function=simple_objective_function,
        study_name="test_history",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(direction="minimize")
    optimizer.optimize(n_trials=5)
    
    # Test history access
    assert len(optimizer.study.trials) == 5
    
    # Test visualization
    try:
        optimizer.plot_optimization_history()
        optimizer.plot_param_importances()
    except Exception as e:
        pytest.fail(f"Visualization failed: {e}")

def test_multi_objective_optimization():
    """Test multi-objective optimization capabilities."""
    def multi_objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return (x - 2)**2 + (y + 3)**2, abs(x) + abs(y)
    
    optimizer = MotorOptimizer(
        objective_function=multi_objective,
        study_name="test_multi_objective",
        storage="sqlite:///:memory:"
    )
    
    optimizer.create_study(directions=["minimize", "minimize"])
    optimizer.optimize(n_trials=10)
    
    # Verify that Pareto front is generated
    assert len(optimizer.study.best_trials) > 0
    
    # Test visualization of Pareto front
    try:
        fig = optuna.visualization.plot_pareto_front(optimizer.study)
        fig.show()
    except Exception as e:
        pytest.fail(f"Pareto front visualization failed: {e}") 