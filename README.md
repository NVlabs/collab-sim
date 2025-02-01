# collab-sim

**COLLAB-SIM** is a research package for GPU-accelerated robot teleoperation with MPC and collection of teleop demonstrations in Virtual Reality (VR) in simulation. It runs VR with physics simulation and high quality rendering using NVIDIA's Isaac Sim and XR, enabling rendering of the simulated environment on the VR Headset at interactive speeds (30-35FPS), while also streaming the VR HMD and controllers states (6-DOF poses and button states) back to the robot application. 

This is a research preview release to enable researchers to explore uses of VR with robotics simulation and custom VR-based workflows for robot learning and human-robot interaction.

# Examples and features 

## Bimanual VR Teleoperation in Isaac Sim 
The user teleops two Franka robots in Isaac Sim with left/right HTC controllers and Valve Index HMD, performing block stacking and object handover. Gif shown at 8X speed.

<p align="center">
    <img src="docs/_static/gif/collab-sim-dual-franka-8x-gif.gif" width="40%"/>
</p>


## Environment reset 
Example workflow for logging teleop demos and environment reset from VR controllers for starting a new demo. Blocks positions are lightly randomized at the start of each episode. Gif shown at 4X speed.

<p align="center">
    <img src="docs/_static/gif/collab-sim-franka-resetenv-4x-gif.gif" width="40%"/>
</p>



## Main features: 

1. **COLLAB-SIM-VR**: An implementation of a workflow between [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and VR through the XR extension, episodic VR data collection with environment reset, loading of custom USD environments, transforms for any robot base pose, pkl data saving and replay for small number of moving objects. Our workflow is primarily for teleop, but the end user can implement custom workflows through custom callbacks for the VR controller buttons. 

2. **COLLAB-SIM-Teleop**: A default implementation for teleoperating the Franka Robot, featuring both Model Predictive Control (MPC) and Inverse Kinematics (IK) using [CuRobo](https://github.com/NVlabs/curobo). It offers end-effector delta teleop for user-friendly control, gripper activation via VR controller buttons. Teleop is generalizable to other robots given a sim-ready robot model. 

This library is designed to be easy to understand. We minimize abstractions and prioritize a straightforward structure. Although the layout may not be fully optimized, this simple approach aims to make learning and following the steps easier.


# System Requirements 

1. Ubuntu 22.04 (Preferred), or 20.04. See VR HW compatibility below. 
We develop and test the system on Ubuntu due to its compatibility with robotics libraries, which are often Linux-based.

2. [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) (Kit 106.5.0, python 3.10.15, CUDA 11.8)

3. VR HW: We have tested the following setups. While other VR setups may work, our testing focuses on a subset of configurations commonly used in robotics research and development.

- Quest, using [ALVR](https://github.com/alvr-org/ALVR). Ubuntu 22.04.
- Valve Index (with HTC Vive controllers). Ubuntu 20.04 or 22.04.
- HTC Vive (2018). Ubuntu 20.04 or 22.04.




# Instructions

[Install instructions](/docs/install_docs.md)

[Run instructions](/docs/run_docs.md)



# Releases 

**v1.0** - Jan 2025: collab-sim compatible with Isaac Sim 4.5, ubuntu 20.04 and 22.04.

**v0.1** - Nov 2024: collab-sim compatible with Isaac Sim 2023.1.1, ubuntu 20.04 and 22.04.

---


# Related Publications

- Nick Walker, Xuning Yang, Animesh Garg, Maya Cakmak, Dieter Fox, and Claudia Pérez-D'Arpino. "Fast Explicit-Input Assistance for Teleoperation in Clutter." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems. IROS 2024. 
[https://arxiv.org/abs/2402.02612](https://arxiv.org/abs/2402.02612) 


- Yanwei Wang, Lirui Wang, Yilun Du, Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia Pérez-D’Arpino, Dieter Fox, Julie Shah: Inference-Time Policy Steering through Human Interactions. ICRA 2025.
[http://arxiv.org/abs/2411.16627 ](http://arxiv.org/abs/2411.16627)



# Citation
If you find this work useful for your research and development, please cite as follows:

```
@misc{collab_sim_library,
      title={collab-sim library},
      author={Claudia P\'{e}rez-D'Arpino and Fabio Ramos and Dieter Fox},
      howpublished = {\url{https://github.com/NVlabs/collab-sim}},
      year={2024},
}
```

# License

See LICENSE file. 
