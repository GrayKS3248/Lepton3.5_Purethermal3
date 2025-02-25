# Lepton3.5_Purethermal3

Software to run a FLIR Lepton 3.5 mounted on a Groupgets Purethermal3 board running on Windows 10.

# Installation

You must first install a set of dependencies. It is reccomended that you use either [Anaconda or Miniconda](https://www.anaconda.com/download/success).

### Conda from .yml (RECCOMENDED)

Open a Conda prompt.

Run the command below to create a Conda environment named Lepton and automatically install all dependencies.

```bash
conda env create -f environment.yml -p PATH_TO_INSTALL
```

### Conda from Source

Open a Conda prompt.

Run the commands below to create a fresh conda environment named Lepton.

```bash
conda create -n lepton -y
conda activate lepton
```

If needed, run the following commands to add conda forge to your conda channels.

```bash
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

Finally, these commands install all external dependencies to your conda environment

```bash
conda install pip git numpy matplotlib -y
pip install opencv-python pyav
```

# Usage

## Stream the camera

After plugging in the camera to the computer, open a Conda prompt.

Run the following commands to activate the Lepton environment and start streaming the camera.

```bash
conda activate Lepton
python lepton.py
```

NOTE: There are two causes for the following errorIf

```python
Error: "Captured image shape (x, y) does not equal expected image shape (160, 120). Are you sure the selected port is correct? NOTE: If captured image shape is (61, 80) the Lepton may be seated incorrectly and you should reseat its socket." while in function _stream(), Type of error: <class '__main__.ImageShapeException'>
```
