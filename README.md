# collab-sim

**COLLAB-SIM** is a research package for Virtual Reality (VR) teleoperation in Isaac Sim, with physics simulation, data collection, shared control, and real-time user interaction with learned robot policies. Using GPU-accelerated MPC-based robot teleoperation in VR in Isaac Sim through the XR extension, it enables high quality 3D VR rendering of the simulated environment on the VR Headset at interactive speeds (30-35FPS), while also streaming the 6-DOF poses and button states from the VR controllers for controlling the robot application.

The package provides the following core functionalities:

### Main Classes

1. **COLLAB-SIM-VR**: An implementation of a workflow between [NVIDIA Isaac Sim 2023.1.1](https://developer.nvidia.com/isaac-sim) and VR through the XR extension, episodic VR data collection with environment reset, loading of custom USD environments, transforms for any robot base pose, pkl data saving and replay for small number of moving objects. Our workflow is primarily for teleop, but the end user can implement custom workflows through custom callbacks for the VR controller buttons. 

2. **COLLAB-SIM-Teleop**: A default implementation for teleoperating the Franka Robot, featuring both Model Predictive Control (MPC) and Inverse Kinematics (IK) using [CuRobo](https://github.com/NVlabs/curobo). It offers end-effector delta teleop for user-friendly control, gripper activation via VR controller buttons, and real-time rendering of desired vs. actual robot configurations. Teleop is generalizable to other robots given a sim-ready robot model. 


---

### Instructions

Installation and ussage instructions 
[LINK](https://docs.google.com/document/d/1L7p7x6YhpeVC25Rmw8s6nFksPhCU2zD9exAFfEuYeDk/edit?usp=sharing)


---


### Related Publications

- Nick Walker, Xuning Yang, Animesh Garg, Maya Cakmak, Dieter Fox, and Claudia Pérez-D'Arpino. "Fast Explicit-Input Assistance for Teleoperation in Clutter." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024). 
[https://arxiv.org/abs/2402.02612](https://arxiv.org/abs/2402.02612). 


- Yanwei Wang, Lirui Wang, Yilun Du, Balakumar Sundaralingam, Xuning Yang, Yu-Wei Chao, Claudia Pérez-D’Arpino, Dieter Fox, Julie Shah: Inference-Time Policy Steering through Human Interactions. (2024). Under review. 


---

#### System

- Valve Index (with HTC Vive controllers). Best Performance. 
- HTC Vive (2018)
- Oculus Quest 2 (Experimental, only Ubuntu 22.04). Not actively supported. 

Our development is mainly for ubuntu. We have tested it in Ubuntu 20.04 and 22.04. 

The package currently works with [NVIDIA Isaac Sim 2023.1.1](https://developer.nvidia.com/isaac-sim) only.

---

### License

See LICENSE file. 
