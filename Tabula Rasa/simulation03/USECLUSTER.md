# Using the Student Cluster

## Submit a job
```bash
cd ~/StudentClusterBT/StudentClusterBT/simulation03
sbatch submit.sh
```

## Check status
```bash
squeue --me
```

## Watch live output
```bash
tail -f runs/slurm_<JOBID>.out
```

## Cancel a job
```bash
scancel <JOBID>
```

## Notes
- Output/plots saved to `runs/` when training completes
- Job runs independently — safe to close terminal/VS Code
- Max runtime: 7 days on `jobs` partition
- GPU (RTX 5060 Ti) requires nightly PyTorch (`cu130`) — currently using CPU via `CUDA_VISIBLE_DEVICES=""`
