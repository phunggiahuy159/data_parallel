# prompt.md — one-shot setup prompt for a teammate

Copy everything in the box below and paste it to your AI coding assistant
(e.g. Claude Code) running on **your own Windows laptop**. Change the hostname
to your assigned node (`node2` or `node3`) before sending.

---

```
I'm one of 3-4 people building an MPI cluster for a parallel-computing course.
We run a parallel DBSCAN project (PDSDBSCAN-D, Patwary et al. SC'12). A teammate
already built node1 and verified it; please set up MY node the same way using
VBoxManage so it's automated. I have Oracle VirtualBox installed on Windows.

=== THE CLUSTER ===
- iPhone Personal Hotspot is the WiFi LAN. Each laptop runs exactly ONE Ubuntu
  Server VM in VirtualBox with a BRIDGED ADAPTER (one VM per machine = course rule).
- iPhone hotspot subnet is 172.20.10.0/28 (gateway .1).
- Every node uses the SAME Linux user `mpiuser` (password mpipass123) with
  passwordless sudo. MY hostname must be **node2** (or node3 — ask me which).
- Target per VM: 2 vCPU, 4096 MB RAM, 20 GB disk, Ubuntu Server 24.04.x LTS.

=== CRITICAL GOTCHAS MY TEAMMATE ALREADY HIT (do these, save ~1 hour) ===
1) HYPER-V STEALS VT-X. If my laptop has WSL2 / Docker Desktop / Hyper-V /
   "Core Isolation – Memory Integrity" on, VirtualBox drops to slow "NEM" mode
   and the installed Ubuntu HANGS in initramfs ("Begin: Loading essential
   drivers ...") when the VM has >1 vCPU.
   - Check: `(Get-ComputerInfo -Property HyperVisorPresent).HyperVisorPresent`
     and look for "NEM" vs "VT-x" in the VM's Logs\VBox.log.
   - Fix: have me run `bcdedit /set hypervisorlaunchtype off` in an ADMIN
     terminal, then REBOOT Windows. (Reverse later with `... auto`.)
2) INSTALL OVER NAT, NOT BRIDGED. Bridged-over-WiFi during the OS install stalls
   on package downloads. Use NIC = NAT for the install with a port-forward
   host 127.0.0.1:2222 -> guest 22. Switch to Bridged only AFTER install.
3) DO NOT install VirtualBox Guest Additions in the unattended install — the
   compile hangs/slows it and MPI doesn't need them. Install only openssh-server.
4) The live-installer runs its OWN sshd; `mpiuser` only exists after the install
   finishes and the VM reboots. Wait for the reboot before logging in as mpiuser.

=== STEPS ===
A. Detect: VBoxManage.exe path, a drive with 30+ GB free (put everything under
   <DRIVE>:\MPI), and my WiFi bridge name via `VBoxManage list bridgedifs`.
B. Download Ubuntu Server 24.04.x live-server amd64 ISO to <DRIVE>:\MPI\iso.
C. Create the VM (hostname node2/node3): 2 vCPU, 4096 MB, 20 GB on <DRIVE>,
   SATA controller, NIC1 = NAT + portforward 2222->22.
D. Do the Hyper-V check (gotcha #1); if needed, fix + reboot BEFORE installing.
E. `VBoxManage unattended install` with a MINIMAL autoinstall (script-template):
   ssh install-server true, packages [openssh-server], identity hostname=node2/3
   username=mpiuser, passwordless sudo. Locale en_US, tz Etc/UTC. Start headless.
F. Poll until the INSTALLED system's sshd answers as mpiuser (it self-reboots).
   Install an SSH key into ~/.ssh/authorized_keys (Windows has no sshpass; use
   SSH_ASKPASS_REQUIRE=force with a tiny askpass script that echoes the password).
G. Install the toolchain in the VM:
   `sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev`.
H. Single-node verify (still on NAT, oversubscribed):
   clone our repo or have me copy it to ~/para, then
   `cd ~/para && make && mpirun --oversubscribe -np 4 ./pdsdbscan --n 5000
    --eps 0.5 --min-pts 5 --verify`  → expect "ARI score ≈ 1.0".
I. Switch NIC to BRIDGED on my WiFi adapter (Promiscuous: Allow All). Have me
   connect the laptop to the iPhone hotspot, then configure the bridged interface
   for DHCP (netplan) and confirm `ping 172.20.10.1` works and the VM has a
   172.20.10.x IP.
J. Report back: hostname, the bridged 172.20.10.x IP, and the ARI from step H.

Use absolute paths and VBoxManage. Monitor long installs with screenshots
(`VBoxManage controlvm <vm> screenshotpng`) and by polling the SSH banner on
127.0.0.1:2222. Tell me clearly whenever you need me to do something physical
(run an admin command, reboot, or connect the iPhone hotspot).
```

---

## After everyone's node is up (run once, on node1 = master)

1. Make sure all laptops are on the **iPhone hotspot** and each VM has a
   `172.20.10.x` IP (`ip -4 addr` inside each VM).
2. On every node, put all three in `/etc/hosts`:
   ```
   172.20.10.2  node1
   172.20.10.3  node2
   172.20.10.4  node3
   ```
3. From node1, passwordless SSH to the others:
   ```bash
   ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa   # if needed
   ssh-copy-id node2 && ssh-copy-id node3
   ```
4. Set slots to your core count in `cluster_setup/hosts.conf` (2 vCPU → `slots=2`).
5. Deploy and run across all 3 machines:
   ```bash
   cd ~/para && make && make deploy
   mpirun -np 6 --hostfile cluster_setup/hosts.conf \
       ./pdsdbscan --n 20000 --eps 0.5 --min-pts 5 --verify --timing
   ```
   Expect the cluster count to match sequential and **`ARI ≈ 1.0`**.

Full details (datasets, benchmark suite, flags, output format) are in
[`README.md`](README.md); cluster bring-up details are in [`SETUP.md`](SETUP.md).
