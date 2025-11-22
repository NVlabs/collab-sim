# collab-sim

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5-green.svg)](https://developer.nvidia.com/isaac-sim)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

**COLLAB-SIM** is a research package for GPU-accelerated robot teleoperation with MPC and collection of teleop demonstrations in Virtual Reality (VR) in simulation. It runs VR with physics simulation and high quality rendering using NVIDIA's Isaac Sim and XR, enabling rendering of the simulated environment on the VR Headset at interactive speeds (30-35 FPS), while also streaming the VR HMD and controllers states (6-DOF poses and button states) back to the robot application.

This is a research preview release to enable researchers to explore uses of VR with robotics simulation and custom VR-based workflows for robot learning and human-robot interaction.

## Table of Contents
- [Features](#main-features)
- [Examples](#examples-and-features)
- [System Requirements](#system-requirements)
- [Documentation](#documentation)
- [Publications](#related-publications)
- [Citation](#citation)
- [License](#license)

## Examples and Features

### Bimanual VR Teleoperation in Isaac Sim
The user teleops two Franka robots in Isaac Sim with left/right HTC controllers and Valve Index HMD, performing block stacking and object handover. Gif shown at 8X speed.

<p align="center">
    <img src="docs/_static/gif/collab-sim-dual-franka-8x-gif.gif" width="40%"/>
</p>

### Environment Reset
Example workflow for logging teleop demos and environment reset from VR controllers for starting a new demo. Block positions are lightly randomized at the start of each episode. Gif shown at 4X speed.

<p align="center">
    <img src="docs/_static/gif/collab-sim-franka-resetenv-4x-gif.gif" width="40%"/>
</p>

## Main Features

### VR Integration
- **Real-time VR Rendering**: Interactive physics simulation at 30-35 FPS in VR
- **Episodic Data Collection**: Record and replay VR teleoperation demonstrations
- **Custom Environments**: Load USD scenes and configure robot base poses
- **Flexible Callbacks**: Implement custom workflows through VR controller button mappings

### Robot Control
- **MPC & IK Support**: Model Predictive Control and Inverse Kinematics via [CuRobo](https://github.com/NVlabs/curobo)
- **Delta Teleoperation**: Intuitive end-effector relative motion control
- **Gripper Control**: VR controller button integration
- **Multi-Robot**: Support for single and dual-arm manipulation

### Design Philosophy
This library prioritizes **clarity and simplicity**. We minimize abstractions to keep the codebase approachable for researchers new to VR robotics and Isaac Sim. Although the layout may not be fully optimized, this straightforward approach makes learning and following the steps easier.


## System Requirements

| Component | Version/Details |
|-----------|----------------|
| **OS** | Ubuntu 22.04 (preferred) or 20.04 |
| **Isaac Sim** | [4.5](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) (Kit 106.5.0) |
| **Python** | 3.10.15 |
| **CUDA** | 11.8 |
| **VR Hardware** | Quest (ALVR), Valve Index, or HTC Vive |

### Tested VR Configurations
- **Meta Quest**: Using [ALVR](https://github.com/alvr-org/ALVR) on Ubuntu 22.04
- **Valve Index**: With HTC Vive controllers on Ubuntu 20.04/22.04
- **HTC Vive**: 2018 model on Ubuntu 20.04/22.04

> **Note**: Other VR setups may work but are untested. We develop and test on Ubuntu due to its compatibility with robotics libraries, which are often Linux-based. Our focus is on configurations commonly used in robotics research and development.



## Documentation

- **[Installation Guide](docs/install_docs.md)** - Complete setup instructions for Isaac Sim, CuRobo, and VR
- **[Usage Guide](docs/run_docs.md)** - How to run examples and use the VR controllers

### Example Scripts
- `collab_sim/demos/franka_mpc_example.py` - Single Franka robot teleoperation with MPC
- `collab_sim/demos/dual_franka_mpc_teleop.py` - Bimanual dual-arm manipulation
- `collab_sim/replay_states.py` - Replay recorded VR demonstrations



## Release History

| Version | Date | Highlights |
|---------|------|------------|
| **v1.0** | Jan 2025 | Isaac Sim 4.5 support, Ubuntu 22.04 compatibility |
| **v0.1** | Nov 2024 | Initial release with Isaac Sim 2023.1.1 |

---

## Related Publications

If you use this work, please consider citing the following publications:

**Fast Explicit-Input Assistance for Teleoperation in Clutter**
Nick Walker, Xuning Yang, Animesh Garg, Maya Cakmak, Dieter Fox, Claudia Pérez-D'Arpino
*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024*
[Paper](https://arxiv.org/abs/2402.02612)

**Inference-Time Policy Steering through Human Interactions**
Yanwei Wang, Lirui Wang, Yilun Du, Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia Pérez-D'Arpino, Dieter Fox, Julie Shah
*IEEE International Conference on Robotics and Automation (ICRA), 2025*
[Paper](http://arxiv.org/abs/2411.16627)



## Citation

If you find this work useful for your research and development, please cite as follows:

```bibtex
@misc{collab_sim_library,
    title={collab-sim library},
    author={Claudia P\'{e}rez-D'Arpino and Fabio Ramos and Dieter Fox},
    howpublished={\url{https://github.com/NVlabs/collab-sim}},
    year={2024}
}
```

## License

See License file.