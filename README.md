# collab-sim

# Introduction

**COLLAB-SIM** is a research package for GPU-accelerated robot teleoperation and collection of teleop demonstrations in Virtual Reality (VR) in simulation. It runs VR with physics simulation and high quality rendering using NVIDIA's Isaac Sim and XR, enabling rendering of the simulated environment on the VR Headset at interactive speeds (30-35FPS), while also streaming the VR HMD and controllers states (6-DOF poses and button states) back to the robot application. It contains a GPU-based MPC teleoperation implementation through [CuRobo](https://github.com/NVlabs/curobo).  

This is a research preview release to enable researchers explore uses of VR with robotics simulation and custom VR-based workflows for robot learning and human-robot interaction.

# Examples and features 

## Bimanual VR Teleoperation in Isaac Sim 
The user teleops two Franka robots in Isaac Sim with left/right HTC controllers and Valve Index HMD, performing block stacking and object handover. Gif shown at 8X speed.

<p align="center">
    <img src="docs/_static/images/vr_intro.gif" width="40%"/>
</p>


## Environment reset 
Example workflow for logging teleop demos and environment reset from VR controllers for starting a new demo. Blocks positions are lightly randomized at the start of each episode. Gif shown at 4X speed.

<p align="center">
    <img src="docs/_static/gif/collab-sim-franka-resetenv-4x-gif.gif" width="40%"/>
</p>


---

# Instructions

> [Instructions Intro](docs/readme.md)

---

# Releases 

**v0.1.0** - Nov 2024: collab-sim compatible with Isaac Sim 2023.1.1

---


# Related Publications

- Nick Walker, Xuning Yang, Animesh Garg, Maya Cakmak, Dieter Fox, and Claudia Pérez-D'Arpino. "Fast Explicit-Input Assistance for Teleoperation in Clutter." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems. IROS 2024. 
[https://arxiv.org/abs/2402.02612](https://arxiv.org/abs/2402.02612) 


- Yanwei Wang, Lirui Wang, Yilun Du, Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia Pérez-D’Arpino, Dieter Fox, Julie Shah: Inference-Time Policy Steering through Human Interactions. 
[http://arxiv.org/abs/2411.16627 ](http://arxiv.org/abs/2411.16627)

---

# License

See LICENSE file. 
