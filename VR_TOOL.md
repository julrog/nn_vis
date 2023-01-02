<h1 align="center">VR TOOL</h2>
<p align="center">
<img align="center" src="./docs/images/vr_tool.gif" />
</p>
<p align="center">
Visualization of neural network architectures and parameters in VR.
</p>

This tool can be used to render a processed neural network in VR.

## Dependencies

* The tool uses [pyopenvr](https://github.com/cmbruns/pyopenvr) for now`and needs SteamVR installed to work
* `requirements_vr.txt`

## Installation

* `pip install -r requirements_vr.txt`

## Start

* `python start_tool_vr.py`

Or

* Run `start_tool.py --demo` to download data of an already processed model and render it.

## Controls

Using Oculus Quest 2 controller:

* left/right trigger: shrink/grow model
* left/right grip: attach model to left/right hand
* left/right joystick: rotate model when grabbed
* pressing left/right joystick: reset model position and orientation
* x/b button: rotate between some predefined rendering modes for nodes and edges
* y/a button: rotate between different class highlighting settings



### GUI
See [README.md](./README.md) for information on the desktop GUI

## Used Systems

* Windows 10
* NVIDIA GeForce RTX 3080
* AMD Ryzen 7 3700X
* Oculus Quest 2
