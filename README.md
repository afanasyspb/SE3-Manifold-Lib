# SE(3) Manifold Optimization & Benchmarks

This repository contains a collection of algorithms and benchmarks for optimization on the SE(3) manifold, focusing on **Geometric Dual Quaternions (GeoDQ)**, and other geometric fusion methods.

Designed for Computer Vision, Robotics, and Navigation tasks.

## 📚 Algorithms Collection

| Project / Algorithm | Description | Code & Benchmarks | Documentation |
| :--- | :--- | :--- | :--- |
| **GeoDQ-Bench** | **Geometric State Fusion** using Dual Quaternion Observer. Comparative analysis with Kalman Filters (ESKF, UKFM). | [Code](/algorithms/geodq) <br> [Benchmark](/benchmarks/ronin_geodq_analysis.ipynb) | [Read details](/docs/geodq.md) |
| **ESKF / UKFM** | Error-State and Unscented Kalman Filter implementations for SE(3) navigation tasks. | [Code](/algorithms/filters) | [Read details](/docs/geodq.md) |
| **TBD** | TBD | *Coming soon* | *Coming soon* |

## 🛠️ Setup & Usage

### Installation & Environments

This repository supports two workflows requiring different environments.

### 1. Navigation Algorithms (ESKF, GeoDQ)
Standard environment for manifold optimization algorithms.
```bash
git clone https://github.com/afanasyspb/SE3-Manifold-Lib.git
cd SE3-Manifold-Lib
pip install -r requirements.txt
```

### 2. Point Cloud Processing & CGA (LiDAR, SLAM)
Requires Python 3.10 due to Open3D compatibility. Please use the provided Conda environment.
```bash
# Create the environment from file
conda env create -f environment_cga.yml

# Activate
conda activate kitti_cga_env
```
