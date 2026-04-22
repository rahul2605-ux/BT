from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class BaseScatter(ABC):

    @abstractmethod
    def place(
        self, 
        count: int,
        space_bounds: tuple[float, float, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...

class RandomScatter(BaseScatter):
    def place(self, count, space_bounds, rng):
        x_max, y_max, z_max = space_bounds
        positions = rng.uniform(
            low=[0.0, 0.0, 0.0],
            high=[x_max, y_max, z_max],
            size=(count, 3),
        )

        return positions.astype(np.float32)

class SymmetricScatter(BaseScatter):
    """
    Splits agents into a regular grid placed on one half of the x-axis.

    Use case: legitimate nodes in [0, x_max/2], jammers in [x_max/2, x_max].
    The `side` parameter controls which half this team occupies.

    Parameters
    ----------
    side : str
        "left"  → x in [0, x_max/2]
        "right" → x in [x_max/2, x_max]
    altitude_fraction : float
        Agents are placed at z = z_max * altitude_fraction (fixed altitude).
        Default 0.5 = mid-altitude.
    """

    def __init__(self, side: str = "left", altitude_fraction: float = 0.5):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        self.side = side
        self.altitude_fraction = altitude_fraction

    def place(self, count, space_bounds, rng):
        x_max, y_max, z_max = space_bounds
        z = z_max * self.altitude_fraction

        # Grid dimensions: as square as possible
        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))

        x_low  = 0.0         if self.side == "left" else x_max / 2
        x_high = x_max / 2   if self.side == "left" else x_max

        xs = np.linspace(x_low, x_high, cols, endpoint=False)
        ys = np.linspace(0.0, y_max, rows, endpoint=False)

        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:count]
        grid_y = grid_y.flatten()[:count]
        grid_z = np.full(count, z)

        positions = np.stack([grid_x, grid_y, grid_z], axis=1)
        return positions.astype(np.float32)


class CustomScatter(BaseScatter):
    """
    Returns a caller-supplied fixed position array.
    Validates shape but does not modify positions.
    """

    def __init__(self, positions: np.ndarray):
        self.positions = np.array(positions, dtype=np.float32)

    def place(self, count, space_bounds, rng):
        if self.positions.shape[0] < count:
            raise ValueError(
                f"CustomScatter received {self.positions.shape[0]} positions "
                f"but {count} agents were requested."
            )
        return self.positions[:count]


def make_scatter(mode: str, **kwargs) -> BaseScatter:
    """
    Convenience factory.

    Parameters
    ----------
    mode : str
        "random" | "symmetric" | "custom"
    **kwargs
        Forwarded to the scatter constructor.
        - symmetric: side="left"|"right", altitude_fraction=0.5
        - custom:    positions=np.ndarray
    """
    dispatch = {
        "random":    RandomScatter,
        "symmetric": SymmetricScatter,
        "custom":    CustomScatter,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown scatter mode '{mode}'. "
                         f"Choose from {list(dispatch)}.")
    return dispatch[mode](**kwargs)