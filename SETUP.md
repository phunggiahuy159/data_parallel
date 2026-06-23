# Cluster SETUP — from bare laptops to a running PDSDBSCAN-D cluster

This guide takes a team of 3–4 people from **nothing** to a working 3-node MPI
cluster running this project. It is the hard-won, tested companion to
[`README.md`](README.md) (which assumes the machines already exist).

> **What you build:** an iPhone Personal Hotspot acts as the WiFi LAN. Each
> teammate runs **exactly one** Ubuntu Server VM in VirtualBox using a
> **Bridged Adapter**, so all VMs sit on the hotspot as independent machines
> (`node1`, `node2`, `node3`). MPI then runs across all three.

```
              iPhone Personal Hotspot  (172.20.10.0/28)
              /              |              \
        Laptop 1        Laptop 2        Laptop 3
       Ubuntu VM       Ubuntu VM       Ubuntu VM     ← VirtualBox, Bridged Adapter
        node1           node2           node3
     172.20.10.x     172.20.10.x     172.20.10.x
```

**Golden rules**
- One VM per physical machine (course requirement).
- Same Linux user on every node: **`mpiuser`** (MPI needs an identical user).
- Unique hostnames: **`node1` / `node2` / `node3`**.

---

## 0. Per-machine checklist (do on EVERY laptop)

| Item | Value |
|---|---|
| Hypervisor | Oracle VirtualBox 7.x |
| Guest OS | Ubuntu Server 24.04.x LTS (the *live-server* ISO) |
| vCPU / RAM / disk | 2 vCPU / 4096 MB / 20 GB |
| Username / password | `mpiuser` / `mpipass123` (keep identical across nodes) |
| Hostname | `node1` (laptop 1), `node2` (laptop 2), `node3` (laptop 3) |
| Packages | `openssh-server`, `build-essential`, `openmpi-bin`, `libopenmpi-dev` |

---

## 1. ⚠️ FIRST: the Hyper-V / VT-x trap (read before installing)

VirtualBox needs the CPU's hardware virtualization (VT-x). **If a laptop has
WSL2, Docker Desktop, Hyper-V, or Windows "Core Isolation / Memory Integrity"
enabled, Windows' Hyper-V steals VT-x** and VirtualBox silently falls back to
"NEM" mode. The symptom is brutal and confusing:

> The installed Ubuntu boots fine in the installer but then **hangs forever in
> initramfs** (`Begin: Loading essential drivers ...`) whenever the VM has more
> than 1 vCPU.

**Check (PowerShell):**
```powershell
(Get-ComputerInfo -Property HyperVisorPresent).HyperVisorPresent   # True = problem
```
Also, after the VM exists, its log `…\<vm>\Logs\VBox.log` will say `NEM` instead
of `VT-x` if affected.

**Fix (admin Command Prompt / PowerShell), then REBOOT Windows:**
```bat
bcdedit /set hypervisorlaunchtype off
```
After reboot, `HyperVisorPresent` should be `False` and the VM boots in ~15 s
with 2 vCPUs. To restore WSL2/Docker later: `bcdedit /set hypervisorlaunchtype auto`
(then reboot — but VirtualBox goes back to slow NEM while it's on).

> Laptops **without** WSL2/Docker/Hyper-V are unaffected — skip the fix.

---

## 2. Create + install the VM (each laptop)

You can click through the VirtualBox GUI, or script it with `VBoxManage`. Key
settings that matter:

1. **Network — install over NAT first.** Bridged-over-WiFi during the OS install
   can stall on package downloads. Set Adapter 1 = **NAT** for the install (add
   a port-forward host `127.0.0.1:2222` → guest `22` so you can SSH in).
2. **Do NOT install VirtualBox Guest Additions** during the unattended install —
   compiling them is the #1 thing that hangs/slows installs, and MPI doesn't need
   them. Install just `openssh-server`.
3. Boot the Ubuntu Server ISO, run the installer, and set:
   - Hostname `node1`/`node2`/`node3`, username `mpiuser`, password `mpipass123`
   - ✅ "Install OpenSSH server"
   - Use entire disk, no swap needed.

After it reboots into the installed system, install the toolchain:
```bash
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev
```

---

## 3. Switch to the iPhone hotspot + Bridged networking

1. **iPhone:** Settings → Personal Hotspot → **Allow Others to Join: ON** and
   **Maximize Compatibility: ON** (uses 2.4 GHz — far more reliable for VMs).
   Keep the iPhone **screen on / plugged in** (it drops the hotspot when idle).
2. **Every laptop:** connect its WiFi to the iPhone hotspot.
3. **VirtualBox:** VM → Settings → Network → Adapter 1 →
   **Attached to: Bridged Adapter**, Name = your **WiFi** adapter,
   Advanced → Promiscuous Mode: **Allow All**. (Power off the VM to change this.)
4. Boot the VM and confirm it got a hotspot address:
   ```bash
   ip -4 addr            # expect something like 172.20.10.x
   ping -c2 172.20.10.1  # the iPhone gateway — must succeed
   ```

> **iPhone caveat:** the hotspot subnet `172.20.10.0/28` only has ~13 usable
> addresses — plenty for 3 VMs + 3 laptops. If a VM gets **no** address, the
> phone is refusing the bridged MAC; toggle "Maximize Compatibility" and
> reconnect, or fall back to a cheap travel router as the LAN.

---

## 4. Names, hosts file, and passwordless SSH

On **each** node, note its `172.20.10.x` IP, then on **all** nodes add everyone
to `/etc/hosts` (use the real IPs):
```
172.20.10.2   node1
172.20.10.3   node2
172.20.10.4   node3
```

From **node1 (master)**, set up passwordless SSH to the workers:
```bash
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa     # if not present
ssh-copy-id node2
ssh-copy-id node3
ssh node2 hostname && ssh node3 hostname             # should print node2 / node3
```

---

## 5. Deploy + build + run the project

From **node1**, with the project in `~/para`:
```bash
cd ~/para
make                 # build pdsdbscan on node1
make deploy          # rsync + build on node2, node3 (see Makefile)
```

Edit [`cluster_setup/hosts.conf`](cluster_setup/hosts.conf) to match your cores
(2 vCPU per VM → `slots=2`):
```
node1 slots=2
node2 slots=2
node3 slots=2
```

**Smoke test across all 3 machines:**
```bash
mpirun -np 6 --hostfile cluster_setup/hosts.conf hostname     # prints all 3 names
mpirun -np 6 --hostfile cluster_setup/hosts.conf \
    ./pdsdbscan --n 20000 --eps 0.5 --min-pts 5 --verify
```
Expected: `clusters` matches the sequential count and **`ARI score ≈ 1.0`**.

Everything else — dataset generation, the full benchmark suite, CLI flags, output
format — is in [`README.md`](README.md).

---

## 6. Gotchas we already hit (so you don't have to)

| Symptom | Cause | Fix |
|---|---|---|
| Ubuntu hangs at `Loading essential drivers` with 2 vCPU | Hyper-V holds VT-x → VirtualBox in NEM mode | `bcdedit /set hypervisorlaunchtype off` + reboot (§1) |
| OS install stalls downloading packages | Bridged-over-WiFi during install | Install over **NAT**, switch to bridged afterward (§2) |
| Install takes forever / hangs late | Guest Additions compiling | Don't install Guest Additions; only `openssh-server` (§2) |
| VM gets no `172.20.10.x` IP | iPhone refusing bridged MAC / 2nd NIC not configured | Maximize Compatibility ON; ensure the bridged NIC uses DHCP |
| `mpirun` can't reach a worker | SSH / hostfile / firewall | `ssh nodeX hostname`; check `/etc/hosts`; `sudo ufw allow from <ip>` |
| `ARI` is ~0 (clusters look right by count) | label/merge bug in a stale build | rebuild **everywhere** (`make deploy`); all nodes must run the same binary |

---

## 7. Single-machine quick test (before the team is together)

You can validate everything on one laptop using NAT + oversubscription:
```bash
cd ~/para && make
mpirun --oversubscribe -np 4 ./pdsdbscan --n 5000 --eps 0.5 --min-pts 5 --verify
# expect ARI ≈ 1.0
```
