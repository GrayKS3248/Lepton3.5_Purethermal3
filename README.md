(C) Copyright, 2026 G. Schaer.

This work is licensed under a GNU General Public License 3.0.

SPDX-License-Identifier: GPL-3.0-only

# Lepton3.5_Purethermal3

Software to run a FLIR Lepton 3.5 mounted on a Groupgets Purethermal3 board running in Windows.

# Detailed Installation Instruction

We strongly recommend installing this package in a virtual environment:

```powershell
C:\Users\username> python -m venv .venv
C:\Users\username> .venv\Scripts\activate.bat
```

When done installing and using this package, deactivate the virtual environment with:

```console
(.venv) user@device:~$ deactivate
```

### PyPi (Recommended)
[python>=3.12](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) are required.

To install:

```powershell
(.venv) C:\Users\username> pip install lepton-pt
```

### Source
[python>=3.12](https://www.python.org/), [pip](https://pip.pypa.io/en/stable/), and [git](https://git-scm.com/) are required.
To clone the repository:

```powershell
(.venv) C:\Users\username> git clone https://github.com/GrayKS3248/Lepton3.5_Purethermal3.git
(.venv) C:\Users\username> cd Lepton3.5_Purethermal3
```

To install:

```powershell
(.venv) C:\Users\username\condynsate> pip install -e .
```

# Usage

### Streaming

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, start streaming the camera using the `leprun` command.

```powershell
(.venv) C:\Users\username> leprun
```

When you are finshed streaming, press the `esc` while the viewer window is active to terminate.

### Recording

To record a stream, use the `-r` flag.

```powershell
(.venv) C:\Users\username> leprun -r
```

All generated data is saved to the directory `Lepton_Recordings` which itself is generated in the active directory. After the recording is terminated, data is rendered into a `.mp4` video.

When you are finshed recording, press the `esc` while the viewing window is active to terminate. After termination, a background process will render the video. This may take several minutes depending on the length of the recording.

### Other

You can use the `-h` flag to explore addtional flags and functionality.

```
(.venv) C:\Users\username> leprun -h
```

### Coding Examples

Example usage is given in [examples](https://github.com/GrayKS3248/Lepton3.5_Purethermal3/tree/main/examples).

### Lost Frames Every 3 Minutes

The FLIR Lepton camera uses automatic flat [field correction (FFC)](https://en.wikipedia.org/wiki/Flat-field_correction) during operation to ensure image fidelity and prevent pixel drift. These automatic FFCs occur about every 3 minutes and are predicated by a box reading "FFC" in the top left corner of the viewing window. They last approximately 2 seconds during which time no data are transmitted by the camera. This results in unavoidable dropped frames.
