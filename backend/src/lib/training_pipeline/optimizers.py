"""
Hyperparameter optimization for training pipeline.
"""
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import json
from datetime import datetime, timezone

from .pipeline import HyperParameters, TrainingMetrics


class OptimizerType(str, Enum):
    """Types of hyperparameter optimizers."""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    PSO = "particle_swarm"  # Particle Swarm Optimization


class ParameterType(str, Enum):
    """Types of hyperparameters."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterRange:
    """Defines the range/options for a hyperparameter."""
    name: str
    param_type: ParameterType
    
    # For numeric parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    
    # For categorical parameters
    choices: Optional[List[Any]] = None
    
    # Optimization hints
    log_scale: bool = False  # Use log scale for sampling
    default_value: Optional[Any] = None
    
    def __post_init__(self):
        if self.param_type == ParameterType.CATEGORICAL and not self.choices:
            raise ValueError("Categorical parameters must have choices")
        
        if self.param_type in [ParameterType.FLOAT, ParameterType.INT]:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Numeric parameters must have min_value and max_value")
    
    def sample_value(self) -> Any:
        """Sample a random value from this parameter's range."""
        if self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.choices)
        
        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        
        elif self.param_type == ParameterType.FLOAT:
            if self.log_scale:
                log_min = math.log(self.min_value)
                log_max = math.log(self.max_value)
                return math.exp(random.uniform(log_min, log_max))
            else:
                return random.uniform(self.min_value, self.max_value)
        
        elif self.param_type == ParameterType.INT:
            if self.log_scale:
                log_min = math.log(self.min_value)
                log_max = math.log(self.max_value)
                return int(math.exp(random.uniform(log_min, log_max)))
            else:
                return random.randint(int(self.min_value), int(self.max_value))
        
        return self.default_value
    
    def clip_value(self, value: Any) -> Any:
        """Clip a value to valid range."""
        if self.param_type == ParameterType.CATEGORICAL:
            return value if value in self.choices else self.choices[0]
        
        elif self.param_type == ParameterType.BOOLEAN:
            return bool(value)
        
        elif self.param_type == ParameterType.FLOAT:
            return max(self.min_value, min(self.max_value, float(value)))
        
        elif self.param_type == ParameterType.INT:
            return max(int(self.min_value), min(int(self.max_value), int(value)))
        
        return value


@dataclass
class OptimizationTrial:
    """Represents a single optimization trial."""
    trial_id: str
    hyperparameters: Dict[str, Any]
    metrics: Optional[TrainingMetrics] = None
    
    # Trial status
    status: str = "pending"  # pending, running, completed, failed
    
    # Results
    objective_value: Optional[float] = None  # Primary metric to optimize
    training_time_seconds: float = 0.0
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    optimization_id: str
    objective_metric: str = "map50_95"  # Metric to optimize
    maximize_objective: bool = True  # True to maximize, False to minimize
    
    # Optimization settings
    optimizer_type: OptimizerType = OptimizerType.RANDOM_SEARCH
    max_trials: int = 50
    max_parallel_trials: int = 2
    timeout_hours: float = 24.0
    
    # Early stopping
    early_stopping_rounds: Optional[int] = None
    min_improvement_threshold: float = 0.001
    
    # Parameter ranges
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    
    # Algorithm-specific settings
    algorithm_config: Dict[str, Any] = field(default_factory=dict)
    
    def add_parameter_range(self, param_range: ParameterRange) -> None:
        """Add a parameter range to optimize."""
        self.parameter_ranges.append(param_range)
    
    def get_parameter_range(self, name: str) -> Optional[ParameterRange]:
        """Get parameter range by name."""
        for param_range in self.parameter_ranges:
            if param_range.name == name:
                return param_range
        return None


class BaseOptimizer:
    """Base class for hyperparameter optimizers."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.trials: List[OptimizationTrial] = []
        self.best_trial: Optional[OptimizationTrial] = None
        self.is_running = False
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest next set of hyperparameters to try."""
        raise NotImplementedError
    
    def update_trial(self, trial: OptimizationTrial) -> None:
        """Update trial with results."""
        # Find and update existing trial
        for i, existing_trial in enumerate(self.trials):
            if existing_trial.trial_id == trial.trial_id:
                self.trials[i] = trial
                break
        else:
            self.trials.append(trial)
        
        # Update best trial
        if (trial.objective_value is not None and 
            trial.status == "completed"):
            
            if (self.best_trial is None or 
                self._is_better_objective(trial.objective_value, self.best_trial.objective_value)):
                self.best_trial = trial
    
    def _is_better_objective(self, new_value: float, current_best: float) -> bool:
        """Check if new objective value is better than current best."""
        if self.config.maximize_objective:
            return new_value > current_best
        else:
            return new_value < current_best
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        completed_trials = [t for t in self.trials if t.status == "completed"]
        failed_trials = [t for t in self.trials if t.status == "failed"]
        
        return {
            "optimization_id": self.config.optimization_id,
            "optimizer_type": self.config.optimizer_type,
            "total_trials": len(self.trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len(failed_trials),
            "running_trials": len([t for t in self.trials if t.status == "running"]),
            "best_objective": self.best_trial.objective_value if self.best_trial else None,
            "best_hyperparameters": self.best_trial.hyperparameters if self.best_trial else None,
            "is_running": self.is_running
        }


class RandomSearchOptimizer(BaseOptimizer):
    """Random search hyperparameter optimizer."""
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest random hyperparameters."""
        hyperparams = {}
        
        for param_range in self.config.parameter_ranges:
            hyperparams[param_range.name] = param_range.sample_value()
        
        return hyperparams


class GridSearchOptimizer(BaseOptimizer):
    """Grid search hyperparameter optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self._grid_points = self._generate_grid()
        self._current_index = 0
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate all grid points."""
        if not self.config.parameter_ranges:
            return []
        
        # Create all combinations
        import itertools
        
        param_values = []
        param_names = []
        
        for param_range in self.config.parameter_ranges:
            param_names.append(param_range.name)
            
            if param_range.param_type == ParameterType.CATEGORICAL:
                param_values.append(param_range.choices)
                
            elif param_range.param_type == ParameterType.BOOLEAN:
                param_values.append([True, False])
                
            elif param_range.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                # Generate discrete values
                if param_range.step:
                    values = []
                    current = param_range.min_value
                    while current <= param_range.max_value:
                        if param_range.param_type == ParameterType.INT:
                            values.append(int(current))
                        else:
                            values.append(current)
                        current += param_range.step
                    param_values.append(values)
                else:
                    # Default to 10 values
                    n_values = min(10, self.config.max_trials // len(self.config.parameter_ranges))
                    if param_range.log_scale:
                        log_min = math.log(param_range.min_value)
                        log_max = math.log(param_range.max_value)
                        values = [math.exp(log_min + i * (log_max - log_min) / (n_values - 1)) 
                                 for i in range(n_values)]
                    else:
                        values = [param_range.min_value + i * (param_range.max_value - param_range.min_value) / (n_values - 1)
                                 for i in range(n_values)]
                    
                    if param_range.param_type == ParameterType.INT:
                        values = [int(v) for v in values]
                    
                    param_values.append(values)
        
        # Generate all combinations
        grid_points = []
        for combination in itertools.product(*param_values):
            point = dict(zip(param_names, combination))
            grid_points.append(point)
        
        return grid_points
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest next grid point."""
        if self._current_index >= len(self._grid_points):
            # Grid exhausted, return random point
            return RandomSearchOptimizer(self.config).suggest_hyperparameters()
        
        point = self._grid_points[self._current_index]
        self._current_index += 1
        return point


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization (simplified implementation)."""
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        # Simplified BO: use acquisition function based on existing trials
        
        if len(self.trials) < 5:
            # Initial random exploration
            return RandomSearchOptimizer(self.config).suggest_hyperparameters()
        
        # Simple acquisition: explore around best parameters with some randomness
        if self.best_trial:
            base_params = self.best_trial.hyperparameters.copy()
            
            # Add noise to best parameters
            for param_range in self.config.parameter_ranges:
                name = param_range.name
                if name in base_params:
                    if param_range.param_type == ParameterType.FLOAT:
                        # Add Gaussian noise
                        noise_std = (param_range.max_value - param_range.min_value) * 0.1
                        noisy_value = base_params[name] + random.gauss(0, noise_std)
                        base_params[name] = param_range.clip_value(noisy_value)
                    
                    elif param_range.param_type == ParameterType.INT:
                        # Add integer noise
                        noise_range = max(1, int((param_range.max_value - param_range.min_value) * 0.1))
                        noisy_value = base_params[name] + random.randint(-noise_range, noise_range)
                        base_params[name] = param_range.clip_value(noisy_value)
                    
                    elif param_range.param_type == ParameterType.CATEGORICAL:
                        # Sometimes change to different category
                        if random.random() < 0.3:  # 30% chance to change
                            base_params[name] = param_range.sample_value()
            
            return base_params
        
        return RandomSearchOptimizer(self.config).suggest_hyperparameters()


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.population_size = config.algorithm_config.get("population_size", 20)
        self.mutation_rate = config.algorithm_config.get("mutation_rate", 0.1)
        self.crossover_rate = config.algorithm_config.get("crossover_rate", 0.7)
        self.elite_size = config.algorithm_config.get("elite_size", 2)
        
        self._population: List[Dict[str, Any]] = []
        self._generation = 0
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters using genetic algorithm."""
        
        if len(self._population) < self.population_size:
            # Initialize population with random individuals
            return RandomSearchOptimizer(self.config).suggest_hyperparameters()
        
        # Evolve population
        completed_trials = [t for t in self.trials if t.status == "completed" and t.objective_value is not None]
        
        if len(completed_trials) >= self.population_size:
            # Perform evolution
            self._evolve_population(completed_trials)
        
        # Return member from current population
        if self._population:
            return random.choice(self._population)
        
        return RandomSearchOptimizer(self.config).suggest_hyperparameters()
    
    def _evolve_population(self, trials: List[OptimizationTrial]) -> None:
        """Evolve the population based on trial results."""
        # Sort trials by objective value
        sorted_trials = sorted(
            trials[-self.population_size:],  # Take most recent population_size trials
            key=lambda t: t.objective_value,
            reverse=self.config.maximize_objective
        )
        
        # Create new population
        new_population = []
        
        # Keep elite individuals
        for i in range(min(self.elite_size, len(sorted_trials))):
            new_population.append(sorted_trials[i].hyperparameters.copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(sorted_trials) >= 2:
                # Crossover
                parent1 = random.choice(sorted_trials[:len(sorted_trials)//2])  # Select from better half
                parent2 = random.choice(sorted_trials[:len(sorted_trials)//2])
                child = self._crossover(parent1.hyperparameters, parent2.hyperparameters)
            else:
                # Mutation of existing individual
                parent = random.choice(sorted_trials[:len(sorted_trials)//2])
                child = parent.hyperparameters.copy()
            
            # Apply mutation
            child = self._mutate(child)
            new_population.append(child)
        
        self._population = new_population
        self._generation += 1
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring through crossover."""
        child = {}
        
        for param_range in self.config.parameter_ranges:
            name = param_range.name
            
            # Randomly choose parent
            if random.random() < 0.5:
                child[name] = parent1.get(name, param_range.default_value)
            else:
                child[name] = parent2.get(name, param_range.default_value)
        
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to an individual."""
        mutated = individual.copy()
        
        for param_range in self.config.parameter_ranges:
            name = param_range.name
            
            if random.random() < self.mutation_rate:
                # Mutate this parameter
                if param_range.param_type in [ParameterType.FLOAT, ParameterType.INT]:
                    # Add noise
                    current_value = mutated.get(name, param_range.default_value)
                    noise_scale = (param_range.max_value - param_range.min_value) * 0.1
                    
                    if param_range.param_type == ParameterType.FLOAT:
                        new_value = current_value + random.gauss(0, noise_scale)
                    else:
                        new_value = current_value + random.randint(-int(noise_scale), int(noise_scale))
                    
                    mutated[name] = param_range.clip_value(new_value)
                    
                else:
                    # Random new value for categorical/boolean
                    mutated[name] = param_range.sample_value()
        
        return mutated


class HyperParameterOptimizer:
    """Main hyperparameter optimization coordinator."""
    
    def __init__(self):
        self._optimizations: Dict[str, BaseOptimizer] = {}
    
    def create_optimization(self, config: OptimizationConfig) -> BaseOptimizer:
        """Create a new optimization instance."""
        
        # Create appropriate optimizer
        if config.optimizer_type == OptimizerType.RANDOM_SEARCH:
            optimizer = RandomSearchOptimizer(config)
        elif config.optimizer_type == OptimizerType.GRID_SEARCH:
            optimizer = GridSearchOptimizer(config)
        elif config.optimizer_type == OptimizerType.BAYESIAN:
            optimizer = BayesianOptimizer(config)
        elif config.optimizer_type == OptimizerType.GENETIC:
            optimizer = GeneticOptimizer(config)
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
        
        self._optimizations[config.optimization_id] = optimizer
        return optimizer
    
    def get_optimization(self, optimization_id: str) -> Optional[BaseOptimizer]:
        """Get optimization instance by ID."""
        return self._optimizations.get(optimization_id)
    
    def suggest_trial(self, optimization_id: str) -> Optional[OptimizationTrial]:
        """Suggest a new trial for optimization."""
        optimizer = self.get_optimization(optimization_id)
        if not optimizer:
            return None
        
        # Generate trial ID
        trial_id = f"{optimization_id}_trial_{len(optimizer.trials):04d}"
        
        # Get suggested hyperparameters
        hyperparams = optimizer.suggest_hyperparameters()
        
        # Create trial
        trial = OptimizationTrial(
            trial_id=trial_id,
            hyperparameters=hyperparams,
            status="pending"
        )
        
        optimizer.trials.append(trial)
        return trial
    
    def update_trial_result(
        self,
        optimization_id: str,
        trial_id: str,
        metrics: TrainingMetrics,
        status: str = "completed"
    ) -> bool:
        """Update trial with training results."""
        optimizer = self.get_optimization(optimization_id)
        if not optimizer:
            return False
        
        # Find trial
        trial = None
        for t in optimizer.trials:
            if t.trial_id == trial_id:
                trial = t
                break
        
        if not trial:
            return False
        
        # Update trial
        trial.metrics = metrics
        trial.status = status
        trial.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Extract objective value
        objective_metric = optimizer.config.objective_metric
        if hasattr(metrics, objective_metric):
            trial.objective_value = getattr(metrics, objective_metric)
        elif objective_metric in metrics.metrics:
            trial.objective_value = metrics.metrics[objective_metric]
        
        # Update optimizer
        optimizer.update_trial(trial)
        
        return True
    
    def get_best_hyperparameters(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters found so far."""
        optimizer = self.get_optimization(optimization_id)
        if optimizer and optimizer.best_trial:
            return optimizer.best_trial.hyperparameters
        return None
    
    def export_optimization_results(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Export optimization results."""
        optimizer = self.get_optimization(optimization_id)
        if not optimizer:
            return None
        
        return {
            "optimization_id": optimization_id,
            "config": {
                "optimizer_type": optimizer.config.optimizer_type,
                "objective_metric": optimizer.config.objective_metric,
                "maximize_objective": optimizer.config.maximize_objective,
                "max_trials": optimizer.config.max_trials,
                "parameter_ranges": [
                    {
                        "name": p.name,
                        "type": p.param_type,
                        "min_value": p.min_value,
                        "max_value": p.max_value,
                        "choices": p.choices,
                        "log_scale": p.log_scale
                    }
                    for p in optimizer.config.parameter_ranges
                ]
            },
            "status": optimizer.get_optimization_status(),
            "best_trial": {
                "trial_id": optimizer.best_trial.trial_id,
                "hyperparameters": optimizer.best_trial.hyperparameters,
                "objective_value": optimizer.best_trial.objective_value,
                "metrics": optimizer.best_trial.metrics.to_dict() if optimizer.best_trial.metrics else None
            } if optimizer.best_trial else None,
            "all_trials": [
                {
                    "trial_id": trial.trial_id,
                    "hyperparameters": trial.hyperparameters,
                    "objective_value": trial.objective_value,
                    "status": trial.status,
                    "created_at": trial.created_at,
                    "completed_at": trial.completed_at
                }
                for trial in optimizer.trials
            ]
        }