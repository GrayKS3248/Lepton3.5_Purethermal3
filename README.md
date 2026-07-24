(C) Copyright, 2026 G. Schaer.

This work is licensed under a GNU General Public License 3.0.

SPDX-License-Identifier: GPL-3.0-only

# Lepton3.5_Purethermal3

Software to run a FLIR Lepton 3.5 mounted on a Groupgets Purethermal3 board running on Windows 10.

# Installation

### From PyPi (Recommended)

It is recommended that you use either [Anaconda or Miniconda](https://www.anaconda.com/download/success).

Run the commands below to create a fresh conda environment named lepton.

```shell
conda create -n lepton -y
conda activate lepton
```

Install `pip` in the environment.

```shell
conda install pip -y
```

Install `lepton-pt`

```shell
pip install lepton-pt
```

### From Source

It is reccomended that you use either [Anaconda or Miniconda](https://www.anaconda.com/download/success).

Run the commands below to create a fresh conda environment named lepton.

```shell
conda create -n lepton -y
conda activate lepton
```

Install `pip` and `git` in the environment.

```shell
conda install pip git -y
```

Clone the Lepton3.5_Purethermal3 repository.

```git
git clone https://github.com/GrayKS3248/Lepton3.5_Purethermal3.git
```

Navigate to the repository directory and install the package.

```shell
cd Lepton3.5_Purethermal3
pip install . -e
```

# Usage

### Streaming

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, activate the Conda environment in which this package is installed and start streaming the camera using the `leprun` command.

```shell
conda activate lepton
leprun
```

When you are finshed streaming, press the `esc` while the viewer window is active to terminate the streaming.

### Recording

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, activate the Conda environment in which this package is installed and start streaming the camera using the `leprun` command and the `-r` flag.

```shell
conda activate lepton
leprun -r
```

The `-r` flag indicates that you want to record what is being streamed. All generated data is saved in a subdirectory of the directory `Lepton_Recordings`. `Lepton_Recordings` is found in the active directory. After the recording is terminated, data is rendered into a `.mp4` video.

When you are finshed recording, press the `esc` while the viewing window is active to terminate recording. Note that it will take some time after the recording is terminated to render the captured video.

### Other

You can use the `-h` flag to explore addtional flags and functionality.

```
leprun -h
```

### Lost Frames Every 3 Minutes

The FLIR Lepton camera uses automatic flat field correction (FFC) during operation to ensure image fidelity and prevent pixel drift. These automatic FFCs occur every 3 minutes and are predicated by a box reading "FFC" in the top left corner of the viewing window. They last approximately 2 seconds during which no thermal or telemetry data are transmitted by the camera resulting in dropped frames. This is unavoidable for proper Lepton function. Note the renderer automatically detects the dropped frames and locally adjusts the frame rate to maintain true playback speed.
