# Firesim Getting Started Guide

reference：https://docs.fires.im/en/stable/FireSim-Basics.html

## What is Firesim?

**Introduction：**

FireSim is an open-source **FPGA-accelerated full-system hardware simulation platform** that makes it easy to validate, profile, and debug RTL hardware implementations at 10s to 100s of MHz. FireSim simplifies co-simulating ASIC RTL with cycle-accurate hardware and software models for other system components (e.g. I/Os). FireSim can productively scale from individual SoC simulations hosted on on-prem FPGAs (e.g., a single Xilinx Alveo board attached to a desktop) to massive datacenter-scale simulations harnessing hundreds of cloud FPGAs (e.g., on Amazon EC2 F1).

**Usage model：**

We choose  **"Single-Node Simulations Using One or More On-Premises FPGAs"** usage model. In this usage model, FireSim allows for simulation of targets consisting of individual SoC designs (e.g., those produced by [Chipyard](https://chipyard.readthedocs.io/)) at **150+ MHz running on on-premises FPGAs**, such as those attached to your local desktop, laptop, or cluster. Just like on the cloud, the FireSim manager can automatically distribute and manage jobs on one or more on-premises FPGAs, including running complex workloads like SPECInt2017 with full reference inputs.

There exists other usage models，such as Single-Node Simulations Using Cloud FPGAs and Datacenter/Cluster Simulations on On-Premises or Cloud FPGAs. For more details, you can check out the official documentation.

## Background

**FireSim Infrastructure Diagram :**

<img src="https://docs.fires.im/en/stable/_images/firesim_env.png" alt="FireSim Infrastructure Setup" style="zoom: 30%;" />

Machines used to build and run FireSim simulations are broadly classified into three groups:

**Manager Machine**

This is the main host machine (e.g., your local desktop or server) that you will “do work” on. This is where you’ll clone your copy of FireSim and **use the FireSim Manager to deploy builds/simulations** from.

**Build Farm Machines**

These are a collection of local machines (“build farm machines”) that are used by the FireSim manager to **run FPGA bitstream builds**. 

**Run Farm Machines**

These are a collection of local machines (“run farm machines”) **with FPGAs attached** that the manager manages and **deploys simulations onto**.

## System Setup

We run Firesim on Xilinx VC707 server.

### Set up SSH Keys
On the manager machine, generate a keypair that you will use to ssh from the manager machine into the manager machine (ssh localhost), run farm machines, and build farm machines:

```bash
cd ~/.ssh
ssh-keygen -t ed25519 -C "firesim.pem" -f firesim.pem
```

Next, add this key to the authorized_keys file on the manager machine:

```bash
cd ~/.ssh
cat firesim.pem.pub >> authorized_keys
chmod 0600 authorized_keys
```

You should also copy this public key into the ~/.ssh/authorized_keys files on all of your Run Farm and Build Farm Machines.

Returning to the Manager Machine, let’s set up an ssh-agent:

```bash
cd ~/.ssh
ssh-agent -s > AGENT_VARS
source AGENT_VARS
ssh-add firesim.pem
```

If you reboot your machine (or otherwise kill the ssh-agent), you will need to re-run the above four commands before using FireSim. If you open a new terminal (and ssh-agent is already running), you can simply run `source ~/.ssh/AGENT_VARS`.

Finally, confirm that you can now ssh localhost and ssh into your Run Farm and Build Farm Machines without being prompted for a passphrase.

## FireSim Repo Setup

**1.Fetch FireSim’s sources.**

```shell
 git clone https://github.com/firesim/firesim
 cd firesim
 # checkout latest official firesim release
 # note: this may not be the latest release if the documentation is not up to date
 git checkout 1.18.0
```

The Firesim repository contains many submodules. These submodules will be cloned in the subsequent installation script. To ensure that the subsequent steps are correct, you must first execute the following command：

```shell
# repository 'https://github.com/ucb-bar/hammer-mentor-plugins.git/' doesn't exist
cd firesim/target-design/chipyard/
git rm vlsi/hammer-mentor-plugins/
```




**2.Set up a default software environment using Conda.**  

Replace `REPLACE_ME_USER_CONDA_LOCATION` in the command below with your chosen path to install conda and run it:

```bash
./scripts/machine-launch-script.sh --prefix REPLACE_ME_USER_CONDA_LOCATION
```

If you are not sudoer, comment the following code in `./scripts/machine-launch-script.sh` :

```bash
    SUDO=""
    # prefix_parent=$(dirname "$CONDA_INSTALL_PREFIX")
    # if [[ ! -e "$prefix_parent" ]]; then
    #     mkdir -p "$prefix_parent" || SUDO=sudo
    # elif [[ ! -w "$prefix_parent" ]]; then
    #     SUDO=sudo
    # fi
```

If you have network problems, please try again！



**3.Build-setup**

When you are in conda firesim environment, run 

```bash
./build-setup.sh
```

While this command may seem simple at first glance, you may encounter various issues during its execution. In such cases, your task is to identify the commands in the script that are causing errors and resolve them one by one. Here are some common errors and their solutions.

- **ChecksumMismatchError:** Conda detected a mismatch between the expected content and downloaded content, try

  ```bash
  conda clean -all
  or 
  cd ~/conda/pkgs/
  rm -rf *
  ```

- If the command `git submodule update --init --recursive` in build-setup.sh fails to execute successfully, you can manually clone each submodule that wasn't cloned properly. Here’s how you can proceed:

  1. **Check Submodules**: First, list all the submodules in your repository using the command:

   ```
   git submodule
   ```

  2. **Manually Clone Submodules**: For each submodule listed, if it hasn't been cloned properly or is missing:

   - Navigate into the submodule directory. For example:

     ```
     cd path/to/submodule/
     ```

   - Manually clone the submodule from its source repository:

     ```
     git submodule init
     git submodule update
     or 
     git fetch
     ```


- When executing `conda-lock install --conda $(which conda) -p $RDIR/.conda-env $LOCKFILE` , using the Tsinghua conda mirror significantly improves download speeds and helps avoid most network issues. 
Modify the contents of the `/firesim/conda-reqs/conda-reqs.conda-lock.yml` file by replacing all occurrences of `https://conda.anaconda.org/` with `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/`. This change will enable the use of the Tsinghua mirror. For example, replace:

  ```
  https://conda.anaconda.org/conda-forge/linux-64/_libgcc_mutex-0.1-conda_forge.tar.bz2
  ```

  with:

  ```
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/_libgcc_mutex-0.1-conda_forge.tar.bz2
  ```

- When executing `./build-setup.sh --force $SKIP_TOOLCHAIN_ARG -s 1 -s 4 -s 5 -s 6 -s 7 -s 8 -s 9` , you should dive into `/firesim/target-design/chipyard/scripts`  for more details.

- ./build-setup.sh: line 310: /home/chenxingyan/firesim/target-design/chipyard/tools/install-circt/bin/download-release-or-nightly-circt.sh: No such file or directory

  ```bash
  cd firesim/target-design/chipyard/tools/install-circt/
  git restore --staged bin/download-release-or-nightly-circt.sh
  git restore bin/download-release-or-nightly-circt.sh
  ```


- error: the following file has local modifications:
    sims/firesim
(use --cached to keep the file, or -f to force removal)
fatal: Submodule work tree 'sims/firesim' contains local modifications; use '-f' to discard them
  ```bash
  cd firesim/target-design/chipyard/sims
  git rm --cached firesim/
  ```
  Then comment `git submodule deinit sims/firesim` in build-setup-nolog.sh. 

Once build-setup.sh completes, run:

```bash
source sourceme-manager.sh --skip-ssh-setup
```
This will perform various environment setup steps, such as adding the RISC-V tools to your path.

**4.Initializing FireSim Config Files**
The FireSim manager contains a command that will automatically provide a fresh set of configuration files for a given platform.

To run it, do the following: `firesim managerinit --platform xilinx_vcu118`

## Running a Single Node Simulation
**1.Building target software**

Build the Linux distribution like:
```bash
# assumes you already cd'd into your firesim repo
# and sourced sourceme-manager.sh
#
# then:
cd sw/firesim-software
./init-submodules.sh
./marshal -v build br-base.json
```
Before executing `./marshal -v build br-base.json` , you need to ask the system administrator to help you install the guestmount. The specific installation steps are shown in this link : https://docs.fires.im/en/stable/Getting-Started-Guides/On-Premises-FPGA-Getting-Started/Initial-Setup/RHS-Research-Nitefury-II.html?highlight=guestmount#install-guestmount

When executing `./marshal -v build br-base.json`, you may encounter such error:
`subprocess.CalledProcessError:Command 'cp -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/boards/firechip/base-workloads/br-base/overlay/usr /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount' returned non-zero exit status 1.`

This error is caused by the following commands that are automatically executed in the script:
```bash
$ guestmount --pid-file guestmount.pid -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/images/firechip/br-base/br-base.img -m /dev/sda /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
$ cp -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/boards/firechip/base-workloads/br-base/overlay/usr /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
```
One solution to this problem is to manually execute the following commands in the shell:
```bash
$ guestmount --pid-file guestmount.pid -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/images/firechip/br-base/br-base.img -m /dev/sda /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
$ cp -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/boards/firechip/base-workloads/br-base/overlay/usr /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
$ cp -a /home/xxx/firesim/target-design/chipyard/software/firemarshal/boards/firechip/base-workloads/br-base/overlay/etc /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
$ guestunmount /home/xxx/firesim/target-design/chipyard/software/firemarshal/disk-mount
```
And comment `run(sudoCmd + ['cp', '-a', str(f.src), dst])` in `/home/xxx/firesim/sw/firesim-software/wlutil`. Then run `./marshal -v build br-base.json` again.

**2.firesim enumeratefpgas**
Run the title command so that FireSim can generate a mapping from the FPGA ID used for JTAG programming to the PCIe ID used to run simulations.

Before executing this command, you must ensure that the SSH Key is set correctly. Please refer to the steps in the link : https://docs.fires.im/en/stable/Getting-Started-Guides/On-Premises-FPGA-Getting-Started/Initial-Setup/Xilinx-VCU118.html#set-up-ssh-keys.

You also need to execute the following command:
```bash
cd firesim/target-design/chipyard/generators/hwacha/
git restore --staged src/
git restore src/

cd firesim/target-design/chipyard/generators/caliptra-aes-acc/
git restore --staged src/ software/
git restore src/ software/
```

According to the FPGA board you are using, set the target_config part in `firesim/deploy/config_runtime.yaml` to the corresponding value. For example, if we are using U280, set `target_config:default_hw_config:` to `alveo_u280_firesim_rocket_singlecore_no_nic`.

If you encounter this error: `(.text+0x12): undefined reference to '__libc_csu_fini'`, please make sure that the g++ version you are using is 11. After solving this problem, you still need to ask your system administrator to help you install `libdwarf-dev`. For more details，please refer to：https://groups.google.com/g/firesim/c/URQtP3IYtlw/m/ryRnvKUvAwAJ.

**3.firesim infrasetup**