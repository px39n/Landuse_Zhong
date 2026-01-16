"""
Pareto frontier optimization
Migrated from: 8.0 Multi-objective.ipynb

Multi-objective optimization for 3E dimensions:
- Environment: Environmental suitability (predicted_prob)
- Emission: Emission mitigation (net_benefit)
- Economic: Economic feasibility (npv)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ParetoOptimizer:
    """
    Multi-objective optimizer for 3E-Synergy ranking
    
    Uses efficiency kernel functions to optimize pixel rankings
    across multiple objectives simultaneously
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Pareto optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.opt_config = config.get("optimization", {})
        
        # Optimization parameters
        self.population_size = self.opt_config.get("population_size", 100)
        self.n_generations = self.opt_config.get("n_generations", 200)
        self.crossover_prob = self.opt_config.get("crossover_prob", 0.9)
        self.mutation_prob = self.opt_config.get("mutation_prob", 0.1)
    
    def optimize(
        self,
        environment: np.ndarray,
        emission: np.ndarray,
        economic: np.ndarray,
        areas: np.ndarray,
        objectives: List[str] = ["environment", "emission", "economic"]
    ) -> Dict:
        """
        Run multi-objective optimization
        
        Args:
            environment: Environmental suitability scores
            emission: Emission mitigation values
            economic: Economic NPV values
            areas: Pixel areas (for cumulative calculation)
            objectives: List of objectives to optimize
        
        Returns:
            Dictionary with Pareto frontier solutions
        """
        logger.info(f"Running multi-objective optimization with {len(objectives)} objectives")
        
        n_pixels = len(environment)
        
        # Prepare data
        data = {
            "environment": environment,
            "emission": emission,
            "economic": economic,
            "areas": areas
        }
        
        # Create efficiency functions for each objective
        efficiency_funcs = {}
        for obj in objectives:
            efficiency_funcs[obj] = self._create_efficiency_function(
                data[obj], areas, kernel_type="decreasing"
            )
        
        # Run optimization
        try:
            # Try to use pymoo if available
            pareto_solutions = self._optimize_with_pymoo(
                data, areas, efficiency_funcs, objectives
            )
        except ImportError:
            logger.warning("pymoo not available, using simple heuristic")
            pareto_solutions = self._optimize_heuristic(
                data, areas, efficiency_funcs, objectives
            )
        
        logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        return pareto_solutions
    
    def _create_efficiency_function(
        self,
        values: np.ndarray,
        areas: np.ndarray,
        kernel_type: str = "decreasing"
    ):
        """
        Create efficiency function with kernel weighting
        
        Args:
            values: Benefit values
            areas: Pixel areas
            kernel_type: Type of kernel ('decreasing', 'uniform', 'increasing')
        
        Returns:
            Function that evaluates efficiency for given ranking
        """
        n = len(values)
        total_values = values * areas
        
        # Create kernel weights
        if kernel_type == "decreasing":
            # Decreasing kernel: prioritizes high values at front
            u = np.arange(1, n + 1) / n
            weights = 1 - u
        elif kernel_type == "uniform":
            weights = np.ones(n) / n
        elif kernel_type == "increasing":
            u = np.arange(1, n + 1) / n
            weights = u
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        weights = weights / weights.sum()  # Normalize
        
        def efficiency_func(ranking: np.ndarray) -> float:
            """
            Evaluate efficiency for given ranking
            
            Args:
                ranking: Permutation of pixel indices
            
            Returns:
                Efficiency score
            """
            # Reorder values according to ranking
            ranked_values = total_values[ranking]
            
            # Calculate weighted sum
            efficiency = np.sum(ranked_values * weights)
            
            return efficiency
        
        return efficiency_func
    
    def _optimize_with_pymoo(
        self,
        data: Dict,
        areas: np.ndarray,
        efficiency_funcs: Dict,
        objectives: List[str]
    ) -> List[Dict]:
        """
        Optimize using pymoo library
        
        Requires: pip install pymoo
        """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        
        n_pixels = len(data["environment"])
        
        class RankingProblem(Problem):
            """Pymoo problem definition for ranking optimization"""
            
            def __init__(self, data, areas, efficiency_funcs, objectives):
                self.data = data
                self.areas = areas
                self.efficiency_funcs = efficiency_funcs
                self.objectives = objectives
                
                super().__init__(
                    n_var=n_pixels,
                    n_obj=len(objectives),
                    n_ieq_constr=0,
                    xl=0,
                    xu=n_pixels - 1
                )
            
            def _evaluate(self, X, out, *args, **kwargs):
                """Evaluate objectives for population"""
                obj_values = []
                
                for x in X:
                    # Convert to ranking (permutation)
                    ranking = np.argsort(x)
                    
                    # Evaluate each objective (negate for minimization)
                    obj = []
                    for obj_name in self.objectives:
                        eff = self.efficiency_funcs[obj_name](ranking)
                        obj.append(-eff)  # Negate for minimization
                    
                    obj_values.append(obj)
                
                out["F"] = np.array(obj_values)
        
        # Create problem and algorithm
        problem = RankingProblem(data, areas, efficiency_funcs, objectives)
        algorithm = NSGA2(pop_size=self.population_size)
        
        # Run optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            verbose=True
        )
        
        # Extract Pareto solutions
        pareto_solutions = []
        for i, x in enumerate(res.X):
            ranking = np.argsort(x)
            
            solution = {
                "ranking": ranking,
                "objectives": {}
            }
            
            for j, obj_name in enumerate(objectives):
                solution["objectives"][obj_name] = -res.F[i, j]  # Undo negation
            
            pareto_solutions.append(solution)
        
        return pareto_solutions
    
    def _optimize_heuristic(
        self,
        data: Dict,
        areas: np.ndarray,
        efficiency_funcs: Dict,
        objectives: List[str]
    ) -> List[Dict]:
        """
        Simple heuristic optimization (fallback when pymoo unavailable)
        
        Generates solutions by:
        1. Single-objective rankings
        2. Weighted combinations
        """
        logger.info("Using heuristic optimization")
        
        solutions = []
        
        # Single-objective solutions
        for obj_name in objectives:
            values = data[obj_name]
            ranking = np.argsort(-values)  # Descending order
            
            solution = {
                "ranking": ranking,
                "objectives": {},
                "type": f"single_{obj_name}"
            }
            
            for obj in objectives:
                solution["objectives"][obj] = efficiency_funcs[obj](ranking)
            
            solutions.append(solution)
        
        # Weighted combinations
        n_weights = 5
        for alpha in np.linspace(0, 1, n_weights):
            for beta in np.linspace(0, 1 - alpha, n_weights):
                if alpha + beta > 1:
                    continue
                
                gamma = 1 - alpha - beta
                
                # Combined scoring
                combined_score = (
                    alpha * data["environment"] +
                    beta * data["emission"] / data["emission"].std() +
                    gamma * data["economic"] / data["economic"].std()
                )
                
                ranking = np.argsort(-combined_score)
                
                solution = {
                    "ranking": ranking,
                    "objectives": {},
                    "type": "weighted",
                    "weights": {"env": alpha, "emission": beta, "economic": gamma}
                }
                
                for obj in objectives:
                    solution["objectives"][obj] = efficiency_funcs[obj](ranking)
                
                solutions.append(solution)
        
        # Filter dominated solutions (simple Pareto filtering)
        pareto_solutions = self._filter_dominated(solutions, objectives)
        
        return pareto_solutions
    
    def _filter_dominated(
        self,
        solutions: List[Dict],
        objectives: List[str]
    ) -> List[Dict]:
        """
        Filter out dominated solutions
        
        Solution A dominates B if A is better in all objectives
        """
        pareto = []
        
        for i, sol_i in enumerate(solutions):
            dominated = False
            
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                
                # Check if sol_j dominates sol_i
                better_in_all = True
                better_in_any = False
                
                for obj in objectives:
                    if sol_j["objectives"][obj] > sol_i["objectives"][obj]:
                        better_in_any = True
                    elif sol_j["objectives"][obj] < sol_i["objectives"][obj]:
                        better_in_all = False
                
                if better_in_all and better_in_any:
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(sol_i)
        
        return pareto


def calculate_pareto_frontier(
    environment: np.ndarray,
    emission: np.ndarray,
    economic: np.ndarray,
    areas: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Convenience function to calculate Pareto frontier
    
    Args:
        environment: Environmental suitability
        emission: Emission mitigation
        economic: Economic NPV
        areas: Pixel areas (defaults to uniform)
    
    Returns:
        List of Pareto-optimal solutions
    """
    if areas is None:
        areas = np.ones(len(environment))
    
    config = {
        "optimization": {
            "population_size": 100,
            "n_generations": 100
        }
    }
    
    optimizer = ParetoOptimizer(config)
    return optimizer.optimize(environment, emission, economic, areas)
