"""pure data configs"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.strategy import BaseStrategy

class AgentRole(Enum):
    LEGITIMATE = "legitimate"
    JAMMER = "jammer"

class ScatterMode (Enum):
    RANDOM = "random"
    SYMMETRIC = "symmetric"
    CUSTOM = "custom"


# ------------------------
# OFDM waveform parameters
# ------------------------

@dataclass
class OFDMConfig:
    """
    To be extended -> TODO: confirm protocol semantics with ADM
    """
    n_subcarriers: int = 76          # FFT size (resource grid width)
    subcarrier_spacing_hz: float = 15e3   # 15 kHz — LTE/NR default
    carrier_frequency_hz: float = 3.5e9  # 3.5 GHz — common 5G band
    n_pilot_symbols: int = 2         # OFDM symbols carrying pilots
    modulation: str = "QPSK"         # "QPSK" | "16QAM" | "64QAM"


@dataclass
class AgentSpec:
    role: AgentRole
    count: int
    policy: Optional["BaseStrategy"] = None
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    scatter_mode: ScatterMode = ScatterMode.RANDOM
    positions: Optional[np.ndarray] = None
    max_power_dbm: float = 30.0
    mobile: bool = False
    waveform_type: str = "power"
    latent_dim: int = 16

    def __post_init__(self):
        if self.scatter_mode == ScatterMode.CUSTOM:
            if self.positions is None:
                raise ValueError(
                    "ScatterMode.CUSTOM requires an explicit positions array."
                )
            if self.positions.shape != (self.count, 3):
                raise ValueError(
                    f"positions must have shape ({self.count}, 3), "
                    f"got {self.positions.shape}."
                )

@dataclass
class EnvironmentConfig:
    space_bounds: tuple = (100.0, 100.0, 50.0)
    legitimate: AgentSpec = field(
        default_factory=lambda: AgentSpec(role=AgentRole.LEGITIMATE, count=1)
    )
    jammers: AgentSpec = field(
        default_factory=lambda: AgentSpec(role=AgentRole.JAMMER, count=1)
    )
    dt_s: float = 0.1
    max_steps: int = 200
    channel_mode: str = "mock"    # "mock" | "sionna"
    seed: Optional[int] = 42


def __post_init__(self):
        if len(self.space_bounds) != 3:
            raise ValueError("space_bounds must be a 3-tuple (x, y, z).")
        if self.legitimate.role != AgentRole.LEGITIMATE:
            raise ValueError("legitimate AgentSpec must have role=LEGITIMATE.")
        if self.jammers.role != AgentRole.JAMMER:
            raise ValueError("jammers AgentSpec must have role=JAMMER.")
        if self.channel_mode not in ("mock", "sionna"):
            raise ValueError("channel_mode must be 'mock' or 'sionna'.")