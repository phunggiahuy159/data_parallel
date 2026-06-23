# TEAM.md — staying in sync (reproducibility & version safety)

The #1 way an MPI cluster project breaks is **nodes running different versions
of the code** — it fails silently with wrong results. This repo has guardrails so
that can't happen unnoticed. Follow this and you're safe.

## The one rule
> **Git is the single source of truth.** Nobody edits code directly on a worker
> node. Change code → commit → push → everyone rebuilds from the same commit.

## Golden workflow

**When someone changes the code:**
```bash
# author:
git add -A && git commit -m "what changed" && git push

# everyone else, on their node:
cd ~/para && git pull && make
```

**Right before a cluster run (master = node1):** make every node identical in one
command, then run.
```bash
cd ~/para && git pull && make && make deploy   # deploy = rsync+build on workers
mpirun -np <total_cores> --hostfile cluster_setup/hosts.conf \
    ./pdsdbscan --n 20000 --eps 0.5 --min-pts 5 --verify
```

## The automatic safety net (already built in)

Every run prints a build fingerprint and **refuses to run if nodes disagree**:

- ✅ All good:
  ```
  [rank 0] build 71db8d79b033 — consistent across 6 rank(s)
  ```
- ❌ Someone is out of date — the program **aborts** instead of giving bad results:
  ```
  *** BUILD MISMATCH — ranks are running DIFFERENT binaries ***
      rank  0 : build 71db8d79b033
      rank  2 : build b0c1c82d2b13      <-- this node has old/edited code
  ```
  Fix: on the master run `make deploy` (or have that node `git pull && make`), then re-run.

The fingerprint is a SHA1 of the source, so it changes whenever *any* `.cpp/.hpp`
differs. `.gitattributes` forces LF line endings so the hash is identical on
every machine.

## Check everyone's environment matches
On each node:
```bash
make doctor
```
prints OS, g++, OpenMPI, and build hash. Compare across nodes — the **build hash
must match**, and OpenMPI major versions should match too (all Ubuntu 24.04 →
OpenMPI 4.1.x, so you're fine if everyone followed `SETUP.md`).

## Communication tips for the team
- Agree on **one commit** before a benchmark run ("everyone on `git pull`, we're
  at hash 71db8d79b033"). The startup line tells you instantly if someone isn't.
- Tag a known-good version when you have real results:
  `git tag v1-results && git push --tags`.
- Keep `mpiuser`, the `~/para` path, and hostnames (`node1/2/3`) identical on all
  nodes — the scripts assume it.
- Don't commit build artifacts or datasets (already handled by `.gitignore`).

## Pinned environment (what "reproducible" means here)
| Component | Version |
|---|---|
| OS | Ubuntu Server 24.04.x LTS |
| MPI | OpenMPI 4.1.x (`openmpi-bin libopenmpi-dev`) |
| Compiler | g++ 13 (C++17), via `mpicxx` |
| Build flags | `-O3 -std=c++17` |

If anyone's `make doctor` shows a different OpenMPI **major** version, rebuild
their VM per `SETUP.md` rather than debugging ABI mismatches.
