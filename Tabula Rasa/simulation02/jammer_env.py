import sionna.phy as sn
import numpy as np
import gymnasium as gym
import gymnasium.spaces
import torch
from scipy.stats import kurtosis as sci_kurtosis

N_SYMBOLS = 128
KURT_THRESH = -1.0
BETA = 0.5 # punishment for being detected


# sionna blocks
source   = sn.mapping.BinarySource()
constel  = sn.mapping.Constellation("qam", num_bits_per_symbol=2) # QPSK
mapper   = sn.mapping.Mapper(constellation=constel)
demapper = sn.mapping.Demapper("app", constellation=constel)

# channel
def lossless_channel(tx_syms, j_syms):
    return tx_syms + j_syms

# kurtosis detector
def detect_jamming(rx_syms):
    kurt_i = sci_kurtosis(rx_syms.real.numpy(), fisher=True)
    kurt_q = sci_kurtosis(rx_syms.imag.numpy(), fisher=True)
    kurt   = float((kurt_i + kurt_q) / 2)       # average → halves variance
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

        obs = torch.stack([tx_syms.real,tx_syms.imag], dim=-1).flatten().numpy().astype(np.float32)

        return obs,{} # empty dic
    
    def step(self, action): # action of shape (2*N_SYMBOLS, )
        action = torch.tensor(action, dtype=torch.float32).reshape(N_SYMBOLS, 2) # -> (2, N_SYMBOLS)
        jam_syms = torch.complex(action[:, 0], action[:, 1]) #action[:, 0] is all the I values, action[:, 1] is all the Q values.
        jam_power = float(jam_syms.abs().pow(2).mean())

        tx_bits = source([N_SYMBOLS, 2])
        tx_syms = mapper(tx_bits).squeeze()

        rx_syms = lossless_channel(tx_syms, jam_syms)
        
        (flagged, kurt) = detect_jamming(rx_syms) 
        llr = demapper(rx_syms.unsqueeze(-1), 1e-10) #log likelyhood ratio
        rx_bits  = sn.utils.hard_decisions(llr) #Positive values are mapped to 1, Nonpositive values are mapped to 0
        ber      = float(np.mean(tx_bits.numpy() != rx_bits.numpy()))

        reward = ber - BETA * float(flagged) - 0.05  # -0.05: standing still costs you

        obs = torch.stack([tx_syms.real, tx_syms.imag], dim=-1).flatten().numpy().astype(np.float32)

        return obs, reward, False, False, {"ber": ber, "kurtosis": kurt, "power": jam_power, "detected": flagged}