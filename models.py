"""Cấu hình và kết quả dùng chung. Kiểm tra tại API qua utils.validation."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Coordinates = NDArray[np.float64]
DistanceMatrix = NDArray[np.float64]
Permutation = list[int]  # Chỉ chứa 1..n, không chứa kho 0.
Algorithm = Literal["ga", "hybrid_ga"]


@dataclass(frozen=True)
class DataConfig:
    """n_points không bao gồm kho; tọa độ sinh trong [low, high)."""

    n_points: int = 20
    seed: int = 42
    depot: tuple[float, float] = (50.0, 50.0)
    low: float = 0.0
    high: float = 100.0


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_size: int = 2
    seed: int = 42
    two_opt_interval: int = 10
    two_opt_max_passes: int = 20
    improvement_epsilon: float = 1e-9
    trace_limit_per_operator: int = 3


@dataclass(frozen=True)
class GenerationRecord:
    generation: int  # 0 là quần thể ban đầu.
    best_distance: float  # Tốt nhất tìm được đến hết thế hệ này.


@dataclass
class OperatorTrace:
    generation: int
    operator: Literal["selection", "ox", "mutation", "two_opt"]
    before: list[Permutation]
    after: list[Permutation]
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    algorithm: Algorithm
    config: GAConfig
    best_route: list[int]  # Tuyến đầy đủ [0, *hoán_vị, 0].
    best_distance: float
    elapsed_seconds: float
    history: list[GenerationRecord]
    trace: list[OperatorTrace] = field(default_factory=list)

