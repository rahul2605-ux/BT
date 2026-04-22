from abc import ABC, abstractmethod
import numpy as np 

class BaseStrategy(ABC):
    """
    Interface between trained model and environement
    """

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        - local observation -> action
        - shape of observation, action vector -> defined by the env's obs/act-space (both np.ndarray)
        """
    
    def reset(self) -> None:
        """
        called at the start of every episode -> override if policy holds current state
        """
        pass


# ---------------------------------------
# Some basic strategies
# ---------------------------------------
class RandomStrategy(BaseStrategy):
    # Samples uniformly from a give action space

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

class TimedBarrageJamming(BaseStrategy):
    def __init__(
        self,
        n_subcarriers:int,
        silent_steps: int,
        max_power_dbm: float = 30.0  # (dBm = decibels per miliwatt, 30dBm = 1 Watt)
    ):
        self.n_subcarriers = n_subcarriers
        self.silent_steps = silent_steps
        self.max_power_dbm = max_power_dbm
        self._step = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        self._step += 1
        if self._step <= self.silent_steps:
            return np.zeros(self.n_subcarriers)
        return np.full(self.n_subcarriers, self.max_power_dbm)
    
    def reset(self) -> None:
        self._step = 0



# ---------------------------------------
# More complex strategies
# ---------------------------------------

# ---------------------------------------
# THE strategy
# ---------------------------------------
