from abc import ABC, abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace


class HPOAlgorithm(ABC):
    def __init__(self, cs: ConfigurationSpace, total_budget: int, min_budget: float, max_budget: float) -> None:
        self.cs = cs
        self.total_budget = total_budget
        self.min_budget = min_budget
        self.max_budget = max_budget

    @abstractmethod
    def ask(self) -> tuple[Configuration, float]:
        raise NotImplementedError

    @abstractmethod
    def tell(self, config: Configuration, result: float, budget: float) -> None:
        raise NotImplementedError
