Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/

[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png

[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

# Lepton3.5_Purethermal3

Software to run a FLIR Lepton 3.5 mounted on a Groupgets Purethermal3 board running on Windows 10.

---

# Installation

You must first install a set of dependencies. It is reccomended that you use either [Anaconda or Miniconda](https://www.anaconda.com/download/success).

## Conda from .yml (RECOMMENDED)

Open a Conda prompt.

Run the command below to create a Conda environment named Lepton and automatically install all dependencies.

```shell
conda env create -f environment.yml -p PATH_TO_INSTALL
```

## Conda from Source

Open a Conda prompt.

Run the commands below to create a fresh conda environment named Lepton.

```shell
conda create -n lepton -y
conda activate lepton
```

If needed, run the following commands to add conda forge to your conda channels.

```shell
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

Finally, these commands install all external dependencies to your conda environment

```shell
conda install pip git numpy matplotlib -y
pip install opencv-python pyav
```

 ---

# Usage

## Stream

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, open a Conda prompt.

Run the following commands to activate the Lepton Conda environment and start streaming the camera.

```shell
conda activate Lepton
python lepton.py
```

When you are finshed streaming, press the `esc` while the viewing window is active to terminate the streaming.

## Record

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, open a Conda prompt.

Run the following commands to activate the Lepton Conda environment and start recording the camera.

```shell
conda activate Lepton
python lepton.py -r
```

the `-r` flag indicates that you want to record what is being streamed. All generated data is saved in a temporary folder named temp and after the recording is terminated, will be rendered into a `.avi` video.

When you are finshed recording, press the `esc` while the viewing window is active to terminate the recording.

## Other

You can use the `-h` flag to explore addtional flags and functionality.

```
python lepton.py -h

usage: lepton.py [-h] 
                 [-p PORT] 
                 [-r | --record | --no-record] 
                 [-n NAME] 
                 [-c {arctic,black_hot,ironbow,outdoor_alert,
                 rainbow,rainbow_hc,white_hot}] 
                 [-eq | --equalize | --no-equalize] 
                 [-d | --detect | --no-detect]
                 [-m | --multiframe | --no-multiframe] 
                 [-f FPS]

options:
  -h, --help            show this help message and exit
  -p PORT               Lepton camera port
  -r, --no-record       record data stream
  -n NAME               name of saved video file
  -c {arctic,black_hot,ironbow,outdoor_alert,rainbow,rainbow_hc,white_hot}
                        colormap used in viewer
  -eq, --no-equalize    apply histogram equalization to image
  -d, --no-detect       if fronts are detected
  -m, --no-multiframe   detection type
  -f FPS                target FPS of camera
```

---

# Common Errors

```shell
Error: "Captured image shape (x, y) does not equal expected image 
shape (160, 120). Are you sure the selected port is correct? 
NOTE: If captured image shape is (61, 80) the Lepton may be seated 
incorrectly and you should reseat its socket." 
while in function _stream(), 
Type of error: <class '__main__.ImageShapeException'>
```

1. ___The incorrect port is selected.___
   
   To fix, try instead running:
   
   ```shell
   python lepton.py -p A_PORT_NUMBER_THAT_IS_NOT_0
   ```
   
   Where `A_PORT_NUMBER_THAT_IS_NOT_0` is any integer that is not `0`. Each camera device has its own unique port identifier. This code defaults to using port `0` but if you have multiple cameras, the Lepton might be at a higher port number. The `-p` flag allows a users to change the selected port. 

2. ___The lepton is not seated properly in the Purethermal socket.___
   
   To fix, disconnect the Purethermal from power, completely remove the Lepton from the Purethermal socket, and reinsert it. After power is restored, try again running 
   
   ```
   python lepton.py
   ```
