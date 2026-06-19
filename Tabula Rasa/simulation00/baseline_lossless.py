import sionna.phy as sn
import numpy as np
import matplotlib.pyplot as plt

N_TIMESTEPS = 10
JAMMER_START = 5
N_LEGIT = 2
N_JAMMER = 1
BATCH_SIZE = 512

SIGNAL_POWER = 1.0
JAMMER_POWER = 50.0
DET_THRESH = 3.0 # detection threshold

# Sionna blocks
source    = sn.mapping.BinarySource()
constel   = sn.mapping.Constellation("qam", num_bits_per_symbol=2)  # QPSK
mapper    = sn.mapping.Mapper(constellation=constel)
demapper  = sn.mapping.Demapper("app", constellation=constel)
awgn      = sn.channel.AWGN()

def detect(rx):
    power = float(rx.abs().pow(2).mean())
    return power > DET_THRESH, power

# simulation
np.random.seed(0)

log_ber = np.zeros((N_TIMESTEPS, N_LEGIT))
log_power = np.zeros((N_TIMESTEPS, N_LEGIT))
log_detected = np.zeros((N_TIMESTEPS, N_LEGIT), dtype=bool)
log_jammer = np.zeros(N_TIMESTEPS, dtype=bool)

for t in range(N_TIMESTEPS):
    j_active = t >= JAMMER_START
    log_jammer[t] = j_active

    for u in range(N_LEGIT):
        #tx
        tx_bits = source([BATCH_SIZE,2])
        tx_syms = mapper(tx_bits)

        rx_syms = tx_syms #since channel lossless

        if j_active:
            rx_syms = awgn(rx_syms, JAMMER_POWER)
            no = JAMMER_POWER
        else:
            no = 1e-10 #basically noiseless

        #detection
        flagged, power = detect(rx_syms)
        log_power[t,u] = power
        log_detected[t,u] = flagged

        llr = demapper(rx_syms, no) # log likelyhood ratio
        rx_bits = sn.utils.hard_decisions(llr)
        log_ber[t,u] = float(np.mean(tx_bits.numpy() != rx_bits.numpy()))

#_______________________________________________________________________________

# CLAUDE

# ── Console summary ───────────────────────────────────────────────────────────
hdr = f"{'t':>3} | {'Jammer':>8} | {'BER U0':>8} {'BER U1':>8} | {'Pwr U0':>8} {'Pwr U1':>8} | {'Det U0':>8} {'Det U1':>8}"
print(hdr)
print("-" * len(hdr))
for t in range(N_TIMESTEPS):
    j = "ACTIVE" if log_jammer[t] else "off"
    print(f"{t:>3} | {j:>8} | "
          f"{log_ber[t,0]:>8.4f} {log_ber[t,1]:>8.4f} | "
          f"{log_power[t,0]:>8.3f} {log_power[t,1]:>8.3f} | "
          f"{str(log_detected[t,0]):>8} {str(log_detected[t,1]):>8}")
    

# ── Plots ─────────────────────────────────────────────────────────────────────
ts  = np.arange(N_TIMESTEPS)
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax = axes[0]
for u in range(N_LEGIT):
    ax.plot(ts, log_ber[:, u], marker='o', label=f'User {u}')
ax.axvline(JAMMER_START - 0.5, color='red', ls='--', alpha=0.6, label='Jammer ON')
ax.set_ylabel('BER')
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.set_title('Simulation 00 — Lossless Channel, QPSK, Max-Power Jammer')
ax.grid(alpha=0.3)

ax = axes[1]
for u in range(N_LEGIT):
    ax.plot(ts, log_power[:, u], marker='s', label=f'User {u}')
ax.axhline(DET_THRESH, color='orange', ls=':', lw=2,
           label=f'Detection threshold ({DET_THRESH})')
ax.axvline(JAMMER_START - 0.5, color='red', ls='--', alpha=0.6)
ax.set_ylabel('Mean Rx Power')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
users_detected = log_detected.sum(axis=1)
ax.step(ts, users_detected, where='mid', color='steelblue')
ax.axvline(JAMMER_START - 0.5, color='red', ls='--', alpha=0.6, label='Jammer ON')
ax.set_ylabel('# Users Detected Jamming')
ax.set_xlabel('Timestep')
ax.set_ylim(-0.15, N_LEGIT + 0.5)
ax.set_yticks(range(N_LEGIT + 1))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_lossless.png', dpi=150, bbox_inches='tight')
plt.show()

    
