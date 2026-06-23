# Set up your cluster node with Claude — one-shot prompt

You have Claude (Claude Code). To build your Ubuntu VM, set up MPI, build this
project, and verify it — all automatically — do this:

1. **Clone this repo** on your Windows laptop and enter it:
   ```bash
   git clone https://github.com/phunggiahuy159/data_parallel.git
   cd data_parallel
   ```
2. **Open Claude Code** in that folder.
3. **Edit one line** in the prompt below — set your hostname (`node2` or `node3`).
4. **Copy the whole box below and paste it to Claude.** Then just answer Claude
   when it asks you to do a physical action (run an admin command, reboot, or
   connect the iPhone hotspot).

---

```
You are setting up MY node of a 3-machine MPI cluster for a parallel-computing
course. This repo is already cloned in the current directory — it's a parallel
DBSCAN project (PDSDBSCAN-D). FIRST read SETUP.md, README.md, and this prompt.md
in this repo: SETUP.md has the full architecture and the hard-won gotchas from my
teammate who already built node1. Then drive the ENTIRE setup on my Windows
laptop end-to-end using VBoxManage. I have Oracle VirtualBox installed.

MY NODE HOSTNAME: node2          <-- change to node3 if that's my assignment
Cluster user on every node: mpiuser  (password mpipass123, passwordless sudo)
VM target: 2 vCPU (more if my CPU allows), 4096 MB RAM, 20 GB disk,
           Ubuntu Server 24.04.x LTS.
Repo to clone INSIDE the VM: https://github.com/phunggiahuy159/data_parallel.git
           -> clone it to ~/para

Work through the phases below. Verify each step before moving on, use absolute
paths, prefer VBoxManage, and monitor long installs with screenshots
(`VBoxManage controlvm <vm> screenshotpng`) and by polling the SSH banner on
127.0.0.1:2222. STOP and tell me clearly whenever you need a physical action.

PHASE A — build my VM + verify the project (no teammates needed):
 1. Detect: VBoxManage.exe path, a drive with 30+ GB free (use <DRIVE>:\MPI), and
    my WiFi bridge name (`VBoxManage list bridgedifs`).
 2. HYPER-V / VT-X CHECK (critical — SETUP.md §1). Run in PowerShell:
    (Get-ComputerInfo -Property HyperVisorPresent).HyperVisorPresent
    If True (WSL2/Docker/Hyper-V/Core-Isolation is on), Windows is holding VT-x
    and VirtualBox will run in slow "NEM" mode → the installed Ubuntu HANGS in
    initramfs ("Begin: Loading essential drivers ...") whenever the VM has >1
    vCPU. Fix: have me run `bcdedit /set hypervisorlaunchtype off` in an ADMIN
    terminal, then REBOOT Windows, before continuing. (If False, skip.)
 3. Download the Ubuntu Server 24.04.x live-server amd64 ISO to <DRIVE>:\MPI\iso.
 4. Create the VM (hostname = my node). Set NIC1 = NAT with a port-forward
    host 127.0.0.1:2222 -> guest 22, and INSTALL OVER NAT — bridged-during-install
    stalls on package downloads. Do an unattended/minimal autoinstall that sets
    user mpiuser + passwordless sudo, hostname = my node, and installs ONLY
    openssh-server. DO NOT install VirtualBox Guest Additions (the compile
    hangs/slows the install and MPI doesn't need them). Start headless. NOTE: the
    live-installer runs its OWN sshd — mpiuser only exists AFTER the install
    finishes and the VM reboots, so wait for that reboot.
 5. SSH into the installed system (Windows has no sshpass — use
    SSH_ASKPASS_REQUIRE=force with a tiny askpass script that echoes the password,
    or install an SSH key). Then inside the VM install the toolchain:
    sudo apt-get update && sudo apt-get install -y git build-essential openmpi-bin libopenmpi-dev
 6. Clone + build the project inside the VM:
    git clone https://github.com/phunggiahuy159/data_parallel.git ~/para
    cd ~/para && make
 7. Self-test on my single VM (still on NAT, oversubscribed):
    mpirun --oversubscribe -np 4 ./pdsdbscan --n 5000 --eps 0.5 --min-pts 5 --verify
    Confirm the output shows "ARI score" ~ 1.0, and report it to me.
    (If ARI is ~0, the build is stale/broken — rebuild and re-run.)

PHASE B — join the cluster (do this when my team meets, all on the iPhone hotspot):
 8. Tell me to turn the iPhone Personal Hotspot ON (Allow Others to Join: ON,
    Maximize Compatibility: ON, screen on/plugged in) and connect my laptop WiFi
    to it.
 9. Power off the VM, switch NIC1 to Bridged Adapter on my WiFi adapter
    (Promiscuous Mode: Allow All), and boot it.
10. Configure the VM's bridged interface for DHCP (netplan), then confirm:
    `ip -4 addr` shows a 172.20.10.x address, and `ping -c2 172.20.10.1`
    (the iPhone gateway) succeeds.
11. Give me my hostname + 172.20.10.x IP so I can send it to the node1 (master)
    owner, who will add it to /etc/hosts, set up passwordless SSH, and run the
    cluster benchmark.
```

---

## For the node1 owner (master) — after everyone reports their IP

```bash
# 1) On ALL nodes, put everyone in /etc/hosts (use the real 172.20.10.x IPs):
#      172.20.10.2  node1
#      172.20.10.3  node2
#      172.20.10.4  node3
# 2) From node1, set up passwordless SSH to the workers:
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa     # if not present
ssh-copy-id node2 && ssh-copy-id node3
ssh node2 hostname && ssh node3 hostname             # should print node2 / node3
# 3) Set slots = vCPUs-per-VM in cluster_setup/hosts.conf, then build everywhere:
cd ~/para && make && make deploy
# 4) Smoke-test across all 3 machines (replace 6 with your total core count):
mpirun -np 6 --hostfile cluster_setup/hosts.conf hostname
mpirun -np 6 --hostfile cluster_setup/hosts.conf \
    ./pdsdbscan --n 20000 --eps 0.5 --min-pts 5 --verify     # expect ARI ~ 1.0
# 5) Full benchmark (charts for the report):
bash benchmark/run_benchmark.sh
```

See `SETUP.md` for cluster bring-up details and `README.md` for running, dataset
generation, and the benchmark suite.
