# COLLAB-SIM Docs:

This research package provides the following core functionalities:

### Main Classes

1. **COLLAB-SIM-VR**: An implementation of a workflow between [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and VR through the XR extension, episodic VR data collection with environment reset, loading of custom USD environments, transforms for any robot base pose, pkl data saving and replay for small number of moving objects. Our workflow is primarily for teleop, but the end user can implement custom workflows through custom callbacks for the VR controller buttons. 

2. **COLLAB-SIM-Teleop**: A default implementation for teleoperating the Franka Robot, featuring both Model Predictive Control (MPC) and Inverse Kinematics (IK) using [CuRobo](https://github.com/NVlabs/curobo). It offers end-effector delta teleop for user-friendly control, gripper activation via VR controller buttons. Teleop is generalizable to other robots given a sim-ready robot model. 


---

#### System

**OS:**
- Ubuntu 20.04 and 22.04.

**Isaac Sim:**
- This release is compatible with Isaac Sim 2023.1.1 only.

**HW:**
- Valve Index (with HTC Vive controllers). Best Performance. 
- HTC Vive (2018).


