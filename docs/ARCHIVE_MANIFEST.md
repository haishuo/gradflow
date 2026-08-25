# GradFlow archive manifest

Created and verified on 2026-08-25 before active-tree refoundation. All
artifacts live outside the repository in
`/mnt/projects/gradflow-preservation-20260825/`; they cannot be committed by
accident.

## Artifacts

| Artifact | Source state | Contents | Bytes | SHA-256 |
|---|---|---|---:|---|
| `gradflow-fortran-port-2024-08-from-66f4133.tar.gz` | checkout `66f4133134ac0555a9681b80b4a3c305588f4cf2` before fast-forward | The complete then-present `archive/fortran-port-2024-08/` tree, including ignored caches and coverage output (54 tar entries) | 140760 | `f7991aaacdfd3aca6d28b6fd476e8c25b1a014d4a794756ce11d91426c318f33` |
| `gradflow-pre-refoundation-4c861fd.tar.gz` | updated upstream state `4c861fdf4ec31932a8dd815ae9884be8ceba3a37` | All tracked files at the exact pre-refoundation commit, under `gradflow-pre-refoundation-4c861fd/` (57 tar entries) | 308681 | `d5198b015bbd05f05451c63a46014df1c695378f1fb91ae96e87fd44fb1f7534` |
| `gradflow-complete-history-through-4c861fd.bundle` | all local refs after creating the archival tag | Complete Git history, `master`, `origin/master`, `HEAD`, the archival tag, and the Codex capture ref reported by `--all` | 976048 | `4e2adb0121c9fdadd5e15e453d5c45dc660631b92cdd540b18a9b0e4e9d6e679` |

The archival tag is `archive/pre-refoundation-2026-08-25`; its annotated tag
object is `09b5b47011c150d38618a056b56a48cd1097ee21` and it resolves to
`4c861fdf4ec31932a8dd815ae9884be8ceba3a37`. It is local only and has not been
pushed.

## Creation commands

Run from `/mnt/projects/gradflow`:

```bash
mkdir -p /mnt/projects/gradflow-preservation-20260825
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -czf /mnt/projects/gradflow-preservation-20260825/gradflow-fortran-port-2024-08-from-66f4133.tar.gz \
  archive/fortran-port-2024-08
git fetch --prune --tags origin
git merge --ff-only origin/master
git archive --format=tar --prefix=gradflow-pre-refoundation-4c861fd/ \
  4c861fdf4ec31932a8dd815ae9884be8ceba3a37 | gzip -n \
  > /mnt/projects/gradflow-preservation-20260825/gradflow-pre-refoundation-4c861fd.tar.gz
git tag -a archive/pre-refoundation-2026-08-25 \
  4c861fdf4ec31932a8dd815ae9884be8ceba3a37 \
  -m 'Archive pre-refoundation GradFlow state at 4c861fd'
git bundle create \
  /mnt/projects/gradflow-preservation-20260825/gradflow-complete-history-through-4c861fd.bundle \
  --all
```

The bundle was regenerated after tag creation so the final bundle contains
the tag.

## Verification performed

Both archives passed `gzip -t` and `tar -tzf`. The history bundle passed:

```bash
git bundle verify /mnt/projects/gradflow-preservation-20260825/gradflow-complete-history-through-4c861fd.bundle
```

Git reported five refs and “The bundle records a complete history.” Final
hashes were recorded with `sha256sum` after all artifacts reached their final
form.

## Restoration

Restore only the older port into an empty working directory:

```bash
mkdir /tmp/gradflow-old-port
tar -xzf /mnt/projects/gradflow-preservation-20260825/gradflow-fortran-port-2024-08-from-66f4133.tar.gz \
  -C /tmp/gradflow-old-port
```

Restore the exact tracked pre-refoundation source snapshot without Git:

```bash
mkdir /tmp/gradflow-pre-refoundation
tar -xzf /mnt/projects/gradflow-preservation-20260825/gradflow-pre-refoundation-4c861fd.tar.gz \
  -C /tmp/gradflow-pre-refoundation
```

Restore full history and check out the archived state:

```bash
git clone /mnt/projects/gradflow-preservation-20260825/gradflow-complete-history-through-4c861fd.bundle \
  /tmp/gradflow-restored
git -C /tmp/gradflow-restored checkout archive/pre-refoundation-2026-08-25
git -C /tmp/gradflow-restored bundle verify \
  /mnt/projects/gradflow-preservation-20260825/gradflow-complete-history-through-4c861fd.bundle
```

Use a destination other than `/tmp` for a persistent restoration.
