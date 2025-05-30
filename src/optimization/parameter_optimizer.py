from typing import List, Dict, Any, Callable
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

class ParameterOptimizer:
    def __init__(self, objective_function: Callable[[Dict[str, float]], float]):
        """Initialize parameter optimizer
        
        Args:
            objective_function: Function that takes a dictionary of parameters
                              and returns a scalar objective value to minimize
        """
        self.objective_function = objective_function
        self.parameter_space = []
        self.parameter_names = []
        
    def add_parameter(self, name: str, min_val: float, max_val: float, 
                     parameter_type: str = 'real'):
        """Add a parameter to optimize
        
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
            parameter_type: 'real' or 'integer'
        """
        self.parameter_names.append(name)
        if parameter_type == 'real':
            self.parameter_space.append(Real(min_val, max_val, name=name))
        elif parameter_type == 'integer':
            self.parameter_space.append(Integer(int(min_val), int(max_val), name=name))
        else:
            raise ValueError("parameter_type must be 'real' or 'integer'")
    
    def _objective_wrapper(self, **params):
        return self.objective_function(params)
    
    def optimize(self, n_calls: int = 50, n_random_starts: int = 10,
                noise: float = 1e-10) -> Dict[str, Any]:
        """Run optimization
        
        Args:
            n_calls: Number of iterations
            n_random_starts: Number of random initial points
            noise: Expected noise in objective function
            
        Returns:
            Dictionary containing optimization results
        """
        # Wrap the objective function with use_named_args at runtime
        wrapped_objective = use_named_args(self.parameter_space)(self._objective_wrapper)
        
        # Run optimization
        result = gp_minimize(
            func=wrapped_objective,
            dimensions=self.parameter_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            noise=noise,
            verbose=True
        )
        
        # Convert results to dictionary
        best_params = dict(zip(self.parameter_names, result.x))
        
        return {
            'best_parameters': best_params,
            'best_objective': result.fun,
            'all_parameters': [dict(zip(self.parameter_names, x)) for x in result.x_iters],
            'all_objectives': result.func_vals
        }

class MotorOptimizer(ParameterOptimizer):
    def __init__(self, motor_simulator):
        """Initialize motor-specific optimizer
        
        Args:
            motor_simulator: Object that can simulate motor performance
        """
        super().__init__(self._motor_objective)
        self.motor_simulator = motor_simulator
        
    def _motor_objective(self, params: Dict[str, float]) -> float:
        """Objective function for motor optimization
        
        Args:
            params: Dictionary of motor parameters
            
        Returns:
            Negative torque (to maximize torque)
        """
        # Update motor parameters
        self.motor_simulator.update_parameters(params)
        
        # Run simulation
        results = self.motor_simulator.simulate()
        
        # Return negative average torque (to maximize)
        return -results['average_torque']
    
    def setup_default_optimization(self):
        """Setup default optimization parameters for a PMSM"""
        # Magnet parameters
        self.add_parameter('magnet_width', 5.0, 20.0)
        self.add_parameter('magnet_thickness', 2.0, 10.0)
        self.add_parameter('magnet_angle', 0.0, 45.0)
        
        # Stator parameters
        self.add_parameter('tooth_width', 5.0, 15.0)
        self.add_parameter('slot_opening', 2.0, 8.0)
        self.add_parameter('back_iron_thickness', 3.0, 10.0)
        
        # Winding parameters
        self.add_parameter('winding_current', 1.0, 10.0)
        self.add_parameter('winding_turns', 10, 100, parameter_type='integer') 