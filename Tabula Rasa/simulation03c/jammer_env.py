import sionna.phy as sn
import numpy as np
import gymnasium as gym
import gymnasium.spaces
import torch

N_SYMBOLS = 128
KURT_THRESH = -1.0
BETA = 2.0 # punishment for being detected
GAMMA = 0.02 # power penalty

# sionna blocks
source   = sn.mapping.BinarySource()
constel  = sn.mapping.Constellation("qam", num_bits_per_symbol=2) # QPSK
mapper   = sn.mapping.Mapper(constellation=constel)
demapper = sn.mapping.Demapper("app", constellation=constel)

# channel
def lossless_channel(tx_syms, j_syms):
    return tx_syms + j_syms

# kurtosis detector
def _excess_kurtosis(x: torch.Tensor) -> float:
    diff = x - x.mean()
    m2 = (diff ** 2).mean()
    m4 = (diff ** 4).mean()
    return float(m4 / (m2 ** 2 + 1e-12) - 3)

def detect_jamming(rx_syms):
    kurt_i = _excess_kurtosis(rx_syms.real)
    kurt_q = _excess_kurtosis(rx_syms.imag)
    kurt   = (kurt_i + kurt_q) / 2
    return kurt > KURT_THRESH, kurt


# gymnasium env
class JammerEnv (gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low = -10.0,
            high = 10.0,
            shape = (2 * N_SYMBOLS,),
            dtype = np.float32,
        )

        self.action_space = gym.spaces.Box(
            low = -10.0,
            high = 10.0,
            shape = (2 * N_SYMBOLS,),
            dtype = np.float32,
        )

    def reset(self, seed=None, options=None):
        tx_bits = source([N_SYMBOLS, 2]) # [N_SYMBOLS x 2]
        tx_syms = mapper(tx_bits) # [N_SYMBOLS x 1]
        tx_syms = tx_syms.squeeze() # [N_SYMBOLS]

        obs = torch.stack([tx_syms.real,tx_syms.imag], dim=-1).flatten().cpu().numpy().astype(np.float32)

        return obs,{} # empty dic

    def step(self, action): # action of shape (2*N_SYMBOLS, )
        tx_bits = source([N_SYMBOLS, 2])
        tx_syms = mapper(tx_bits).squeeze()

        action = torch.tensor(action, dtype=torch.float32).reshape(N_SYMBOLS, 2).to(tx_syms.device)
        jam_syms = torch.complex(action[:, 0], action[:, 1]) #action[:, 0] is all the I values, action[:, 1] is all the Q values.
        jam_power = float(jam_syms.abs().pow(2).mean())

        rx_syms = lossless_channel(tx_syms, jam_syms)

        (flagged, kurt) = detect_jamming(rx_syms)
        llr = demapper(rx_syms.unsqueeze(-1), 1e-10) #log likelyhood ratio
        rx_bits  = sn.utils.hard_decisions(llr) #Positive values are mapped to 1, Nonpositive values are mapped to 0
        ber      = float((tx_bits != rx_bits).float().mean())

        kurtosis_excess = max(0.0, kurt - KURT_THRESH)
        reward = ber - BETA * kurtosis_excess - 0.05 - GAMMA * jam_power

        obs = torch.stack([tx_syms.real, tx_syms.imag], dim=-1).flatten().cpu().numpy().astype(np.float32)

        return obs, reward, False, False, {"ber": ber, "kurtosis": kurt, "power": jam_power, "detected": flagged}
