# Moody second-machine replication

This directory executes the frozen Academic A4 second-machine contract on the
standalone Moody workstation. The scientific source is the clean detached rc2
tag; the controller may come from a later orchestration-only commit.

From the current GradFlow checkout on Moody:

```bash
bash experiments/academic_a4/moody/stage_moody.sh
bash /mnt/projects/gradflow-a4-moody-20260831/controller/setup_environment.sh
bash /mnt/projects/gradflow-a4-moody-20260831/controller/run_moody.sh
```

The run is single-use. Existing run directories and existing evidence are
refused. Compilation caches and environment files stay below the run
directory or `/tmp`; only the evidence directory is a scientific result.
