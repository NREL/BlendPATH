# Blending Pipeline Analysis Tool for Hydrogen
Welcome to the Blending Pipeline Analysis Tool for Hydrogen (BlendPATH) [SWR-24-10](https://www.osti.gov/doecode/biblio/117221). This is a Python-based tool. The instructions below assumes that the user possesses background knowledge in Python and has Anaconda installed. Below is a quick-start guide:

# Installation
The BlendPATH repository can by cloned to your local computer using:

`git clone https://github.nrel.gov/Hyblend/BlendPATH.git`

## Create environment
A conda environoment can useful for satisfying the Python requirement. First, open an anaconda powershell. Within the powershell, we will first create a conda environment to install BlendPATH in. This command below creates a conda environment called `BlendPATH_env`. The second argument is the Python version. We default to 3.8.19 for compatibility with Cantera for some Mac OSX users. However, more experienced users may relax this constraint in the [configuration file](pyproject.toml) Note: you may swap `BlendPATH_env` with whatever you would like to name the environment

`conda create -n BlendPATH_env python=3.8.19`

Within the same powershell, enter into the conda environment using

`conda activate BlendPATH_env`

## Installation
BlendPATH is built as a Python package that can be installed with pip. Once the conda environment is activated, navigate the directory where the BlendPATH repository was cloned. In most cases this will be in a designated GitHub folder.

`cd \Users\MY_USERNAME\Documents\GitHub\BlendPATH`

Then install BlendPATH using the command below. Be sure to include the period at the end.

`pip install .`

# VS Code setup
We recommend using VS Code for your Python IDE. Within VS Code, open your working directory for BlendPATH.

Within VS Code you will need to activate the conda environment that was created earlier. This can be achieved by pressing `Ctrl + Shift + P`
which brings up a menu window. Here type and select 
`Python: Select Interpreter`

This will bring up a list of your available environments. Select the environment that was just created. Check the file path matches what you expect.

# Quick-Start guide

BlendPATH can be run in a Python script by importing the package. There is also an example script to show usage in the [examples](examples) directory. Within this directory is a [template.py](examples/template.py) file. This uses the data files in the [wangetal2018](examples/wangetal2018) directory,

# License terms - ProFAST
By downloading and using BlendPATH, the user also agrees to the [BlendPATH license terms](LICENSE) as well as the license terms of [ProFAST](https://github.com/NREL/ProFAST/blob/main/LICENSE).

# NOTE: 
BlendPATH continues to see active development and maintenance. If you observe issues with the code, please inform the authors of said issues for continuous improvement. Lastly, please include version number when referencing or citing BlendPATH. 
