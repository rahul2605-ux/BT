# Simulation 03 — PPO Jammer with Normalizing Flow Policy

## What this is
A single-agent RL jammer trained with PPO against a **kurtosis-based detector** on a lossless QPSK channel.
The jammer observes TX symbols and outputs jam symbols to maximise BER while avoiding detection.

## Files
| File | Purpose |
|---|---|
| `jammer_env.py` | Gymnasium env — sionna QPSK channel, kurtosis detector, reward |
| `train_ppo.py` | PPO training loop, logging, plot saving |
| `flow_policy.py` | Custom SB3 policy: MLP [64,64] + 3-layer NSF (zuko) action head |
| `submit.sh` | sbatch script — submit with `sbatch submit.sh` from this directory |
| `USECLUSTER.md` | Cluster how-to (submit, monitor, cancel) |
| `runs/` | Output PNGs + slurm logs saved per run |

## Architecture
- **Obs:** `[I, Q]` of TX symbols → shape `(2 * N_SYMBOLS,)`
- **Action:** `[I, Q]` of jam symbols → shape `(2 * N_SYMBOLS,)`
- **Channel:** `rx = tx + jam` (lossless)
- **Detector:** Fisher excess kurtosis of rx I and Q components, flagged if `kurt > KURT_THRESH`
- **Policy:** `FlowPolicy` — NSF replaces Gaussian head; exact log-prob, no approximation

## Current parameters
```python
# jammer_env.py
N_SYMBOLS   = 16
KURT_THRESH = -1.0
BETA        = 2.0   # kurtosis excess penalty weight
GAMMA       = 0.02  # power penalty weight

# train_ppo.py
TOTAL_STEPS   = 50_000
learning_rate = 1e-4
clip_range    = 0.1
```

## Current reward function (run003+)
```python
kurtosis_excess = max(0.0, kurt - KURT_THRESH)   # 0 when safe, >0 when detected
reward = ber - BETA * kurtosis_excess - 0.05 - GAMMA * jam_power
```
This replaced the earlier binary `BETA * float(flagged)` penalty to give the agent a **continuous gradient** toward the detection threshold.

## Optimal strategy (theoretical)
`jam = -2 * tx_syms` — flips every symbol to the opposite constellation point:
- BER = 1.0 (all bits wrong)
- rx = -tx → still QPSK distribution → kurtosis ≈ -2 → **not detected**
- Reward ≈ +0.90

The agent observes tx symbols directly, so this is a linear function of the observation. The NSF can represent it — the question is whether the reward signal is strong enough for the agent to find it.

---

## Run history

### run001 — BETA=0.5 binary, LR=3e-4, clip=0.2
- BER: 0.22 → 0.30 (slowly rising)
- Detection: ~88% flat — agent completely ignored the detector
- Kurtosis: stuck at ~-0.7, always above threshold
- Power: 2.1 → 2.5 (slowly growing, unconstrained)
- PPO KL: 0.07 → 0.46 (severe divergence)
- **Conclusion:** BETA too low, no detection gradient, LR too high

### run002 — BETA=2.0 binary, GAMMA=0.02, LR=1e-4, clip=0.1
- BER: flat ~0.25 (worse than run001 — agent more cautious but not smarter)
- Detection: ~88% flat — still no evasion
- Kurtosis: stuck at ~-0.7
- Power: stable ~2.15-2.25 (GAMMA working)
- IQ scatter: circular Gaussian blob with no structure throughout
- **Conclusion:** Binary detection signal gives no gradient toward threshold — agent can't learn direction

### run003 — continuous kurtosis penalty (CURRENT)
- Reward: `ber - 2.0 * max(0, kurt - KURT_THRESH) - 0.05 - 0.02 * power`
- Expected improvement: agent now gets gradient proportional to how far above threshold kurtosis is
- **Status: submitted, awaiting results**

---

## GPU situation
- Cluster nodes have **RTX 5060 Ti (sm_120, Blackwell)**
- Requires PyTorch nightly cu130: `pip install --pre torch --upgrade --index-url https://download.pytorch.org/whl/nightly/cu130`
- Currently using **CPU** (`CUDA_VISIBLE_DEVICES=""` in submit.sh) — GPU→CPU syncs (scipy kurtosis, numpy BER) make GPU slower than CPU for N_SYMBOLS=16
- To enable GPU properly: replace scipy kurtosis and numpy BER with pure PyTorch (see below)

## PyTorch-only kurtosis + BER (apply to use GPU efficiently)
```python
# Replace detect_jamming():
def _excess_kurtosis(x: torch.Tensor) -> float:
    diff = x - x.mean()
    m2 = (diff ** 2).mean()
    m4 = (diff ** 4).mean()
    return float(m4 / (m2 ** 2 + 1e-12) - 3)

def detect_jamming(rx_syms):
    kurt_i = _excess_kurtosis(rx_syms.real)
    kurt_q = _excess_kurtosis(rx_syms.imag)
    kurt = (kurt_i + kurt_q) / 2
    return kurt > KURT_THRESH, kurt

# Replace BER line:
ber = float((tx_bits != rx_bits).float().mean())
```

## If run003 still doesn't work → next levers
- Increase `N_SYMBOLS` to 32 — reduces kurtosis estimator variance from ±1.5 to ±0.75
- Add kurtosis to observation — agent currently blind to its own detection risk
- Increase `TOTAL_STEPS` to 200k — may just need more time
- Move to sim04 with richer channel model
