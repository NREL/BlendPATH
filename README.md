# Blending Pipeline Analysis Tool for Hydrogen
Welcome to the Blending Pipeline Analysis Tool for Hydrogen (BlendPATH) [SWR-24-10]. This is a Python-based tool. The instructions below assumes that the user possesses background knowlegde in Python and has Anaconda installed on a host Windows server. Currently, this model requires the use to have installed SAInt v3.2.10.5 and have an associated SAInt license on said server. Below is a quick-start guide:

# Installation
First you will need to log onto a server hosting the SAInt license using remote desktop. This may need to be setup with the IT department.

Now clone the BlendPATH to your local server account 

`git clone https://github.nrel.gov/Hyblend/BlendPATH.git`

## Create environment
After gaining access to the server hosting the SAInt license, open an anaconda powershell. Within the powershell, we will first create a conda environment to run BlendPATH in. This creates a conda environment called BlendPATH_env. The second argument is the python version. We recommend 3.10, which will be up to date with the necessary packages. Note: you may swap BlendPATH_env with whatever you would like to name the environment

`conda create -n BlendPATH_env python=3.10`

Within the same powershell, enter into the environment using

`conda activate BlendPATH_env`

## Installation
BlendPATH is built as a python package that can be installed with pip. Once the conda environment is activated, navigate the directory where the BlendPATH repository was cloned. In most cases this will be in a designated GitHub folder.

`cd \Users\MY_USERNAME\Documents\GitHub\BlendPATH`

Then install the BlendPATH using the command below. Be sure to include the period at the end.

`pip install .`

# VS Code setup
We recommend using VS Code for you python IDE. Within VS Code, open the directory of BlendPATH.

Within VS Code you will need to activate the conda environment that we created earlier. This can be achieved by pressing `Ctrl + Shift + P`

which brings up a menu window. Here type and select 
`Python: Select Interpreter`

This will bring up a list of your available environments. Select the environment that was just created. Check the file path matches what you expect.

# Quick-Start guide

The BlendPATH can be run in a python script by import the package. There is also an example script to show usage in the BlendPATH/examples/ directory

# License terms - ProFAST
By downloading and using BlendPATH, the user also agrees to the [license terms](https://github.com/NREL/ProFAST/blob/main/LICENSE) of [ProFAST](https://github.com/NREL/ProFAST/tree/main).

