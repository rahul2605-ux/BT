# ITET / TIK cluster notes

Migration record for the move off the INFK student cluster (which was in
maintenance the week of 7 Sept 2026). Verified 2026-09-08 by
`cluster/smoke_test.sh`, job **2243203**, COMPLETED 0:0 on `tikgpu03`.

## The one thing that has to be set

The SEPP slurm-23.11 clients in `/usr/sepp/bin` ship **without** a `slurm.conf`
and fall back to a DNS SRV lookup that does not resolve in `ee.ethz.ch`, so
every slurm command dies with:

```
sinfo: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
sinfo: fatal: Could not establish a configuration source
```

Fix — already persisted in `~/.bashrc.user` (the hook ITET's stock `.bashrc`
sources at the end; do not edit `.bashrc` itself):

```bash
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
```

This is a **submit-host-only** problem. The compute nodes have their own
`/etc/slurm/slurm.conf`, so nothing inside a job needs the variable.

## What we get

| | INFK (old) | ITET (now) |
|---|---|---|
| Submit host | — | `tik42x.ee.ethz.ch` |
| Cluster / controller | — | `itet` / `slcontritet` |
| Account | `projects` | **`disco-med`** |
| Partition | `jobs` | **do not set — see below** |
| Concurrent jobs | **1** (`MaxJobsPU=1`) | **unlimited** — no MaxJobs/MaxSubmit/GrpTRES |
| Max walltime | 1 h | **2 days** |
| GPU | 1× 5060ti node | 14 nodes, see table |
| Preemption | — | none on our partitions (`PreemptMode=OFF`) |

**The lifted concurrency cap is the big win.** The old `MaxJobsPU=1` forced every
sweep to be serial; parallel array jobs are now possible.

### Partitions — let the site plugin choose

`JobSubmitPlugins=lua` **overrides** any `--partition` you set:

```
sbatch: Clearing requested partition. It is selected automatically based on
        your Slurm account membership.
sbatch: Setting partitions to : cpu.normal,disco.med,gpu.normal
```

So omit `--partition` entirely. The `disco-med` account reaches `cpu.normal`
(CPU only), `gpu.normal` and `disco.med`.

### GPUs reachable from `disco-med`

| Partition | Nodes | GPU | Per node |
|---|---|---|---|
| `gpu.normal` | `artongpu01-07` | GeForce RTX 2080 Ti | 4 |
| `disco.med` | `tikgpu02`, `tikgpu03` | TITAN Xp | 7, 6 |
| `disco.med` | `tikgpu04` | TITAN RTX | 7 |
| `disco.med` | `tikgpu05` | TITAN RTX / Tesla V100 | 5 / 2 |
| `disco.med` | `tikgpu06`, `tikgpu07`, `tikgpu09` | **GeForce RTX 3090** | 8 each |

`gpu.normal` has `PriorityTier=1000` vs `disco.med`'s `500`, so it usually
schedules sooner; the RTX 3090 nodes are the fastest available. To insist on a
type: `--gres=gpu:geforce_rtx_3090:1`. Plain `--gres=gpu:1` takes whatever is
free (the smoke test landed on a TITAN Xp after ~20 s queueing).

### Storage — and the home quota trap

`df -h /home/rrahman` reports 6.7 T free. **That is the whole NFS export, not your
quota.** The first venv build died mid-install with `Disk quota exceeded
(os error 122)`.

Measured: home held **1.8 G** (1.1 G `.vscode-server`, 754 M the repo) and the
build hit the wall after adding ~6.7 G (1.4 G venv + 5.3 G uv cache), so the
ceiling sits somewhere under ~8.5 G. Home was **not** cluttered — the venv alone
is 7.3 G and simply does not fit. There is nothing to clean up; put it elsewhere.

Note `quota -s` reports `none` and the RPC quota service is unreachable from
tik42x, so **the limit is invisible until you hit it**. Anything large goes on
net_scratch.

| Path | What | Use for |
|---|---|---|
| `/home/rrahman` | NFS home, **tight quota** | code, small artifacts — nothing bulky |
| `/itet-stor/rrahman/net_scratch` | ITET net scratch, **no quota**, 27 T free, NFS | the venv, caches, big outputs |
| `/itet-stor/rrahman` | 46 G sharelink | 28 G free |
| `/scratch` | node-local ext4, ~244 G | fast per-job temp; **not shared, wiped** |
| `/scratch_net/tik42x` | tik42x's scratch over NFS | 336 G free |

net_scratch is **not backed up** and ISG cleans up long-unused data — which is
right for a venv (rebuildable) and wrong for results.

## Python environment

Rebuilt 2026-09-08 at **`/itet-stor/rrahman/net_scratch/bt_env`** (7.3 G),
replacing the INFK `/work/scratch/rrahman/bt_env`. Activate with:

```bash
source /itet-stor/rrahman/net_scratch/bt_env/bin/activate
```

Verified by `cluster/verify_env.sh`, job **2243246**, ALL CHECKS PASSED on a
Tesla V100 — including a Sionna QPSK chain at BER 0.00000 and an EfficientNet-B0
on GPU. A real end-to-end run of `simulation08/frontier_channel.py --smoke`
(job 2243247) completed in 2.9 s with numbers matching the recorded m2 results.

| Package | Version |
|---|---|
| torch | **2.9.1+cu128** |
| torchvision | 0.24.1+cu128 |
| sionna / sionna-rt | 2.0.1 |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |
| zuko | 1.6.0 |
| numpy / scipy / matplotlib | 2.4.6 / 1.17.1 / 3.11.1 |

### Two traps that cost real time — do not undo these

**1. Pin CUDA 12.8. The default torch wheels are now CUDA 13.**
`uv pip install torch` pulls `nvidia-*-cu13`, which needs driver ≥ 580. Every
GPU node here runs **535.261.03** (CUDA 12.2). CUDA 12.x minor-version
compatibility makes cu128 work on 535; cu13 does not. Rebuild with:

```bash
uv pip install --python <venv>/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==2.9.1" torchvision
```

**2. Avoid the TITAN Xp nodes. torch 2.9 dropped Pascal.**
`tikgpu02` and `tikgpu03` are TITAN Xp = **sm_61**, and torch 2.9.1 ships
sm_70–sm_120 only. A job that lands there dies with
`CUDA error: no kernel image is available for execution on the device`. Sionna
2.0.1 requires torch ≥ 2.9.1, so downgrading is not an option. Every GPU submit
script therefore carries:

```bash
#SBATCH --constraint=geforce_rtx_2080_ti|titan_rtx|tesla_v100|geforce_rtx_3090
```

That keeps 12 of the 14 GPU nodes. Compute capabilities:
`titan_xp` sm_61 (unusable) · `tesla_v100` sm_70 · `geforce_rtx_2080_ti`,
`titan_rtx` sm_75 · `geforce_rtx_3090` sm_86.

### Rebuilding from scratch

```bash
export PATH=/usr/sepp/bin:$PATH            # uv lives here
export UV_CACHE_DIR=/itet-stor/rrahman/net_scratch/.uv-cache   # NOT in home
VENV=/itet-stor/rrahman/net_scratch/bt_env
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match "torch==2.9.1" torchvision
uv pip install --python "$VENV/bin/python" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    "sionna==2.0.1" stable-baselines3 gymnasium matplotlib scipy zuko tensorboard
```

System python has **no `pip` and no `ensurepip`** (Debian strips them), so `uv`
is the only practical route; `micromamba` is also on PATH as a fallback.

Note: `import sionna` fails **on the login node** (`libLLVM.so` missing, and the
login CPU lacks `fma`, so DrJit's LLVM fallback shuts down). It works fine on
compute nodes via the CUDA variant. Not a problem — never compute on the login
node anyway.

## Job template

```bash
#SBATCH --account=disco-med
#SBATCH --job-name=...
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1          # NOT --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G             # DefMemPerCPU=6144, so set it explicitly
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
# no --partition: the lua plugin sets it
```

## Submit-script migration — DONE (all 19)

Every `*/submit*.sh` was rewritten on 2026-09-08:

1. `--account=projects` → `--account=disco-med`
2. `--partition=jobs` → **deleted** (the lua plugin sets it)
3. `--gpus=1` → `--gres=gpu:1` + the Pascal `--constraint`
4. `/work/scratch/rrahman/bt_env` → `/itet-stor/rrahman/net_scratch/bt_env`
5. `cd "/home/rrahman/StudentClusterBT/Tabula Rasa/<dir>"` → `cd "$SLURM_SUBMIT_DIR"`,
   so the path is no longer hard-coded (keep submitting from each sim dir)
6. the per-job `pip install --quiet torchvision` — **dropped**, torchvision is
   in the venv now

`grep -rn "projects\|partition=jobs\|/work/scratch\|StudentClusterBT\|--gpus=" */submit*.sh`
returns nothing.
