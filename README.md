# SE(3) Manifold Optimization & Benchmarks

This repository contains a collection of algorithms and benchmarks for optimization on the SE(3) manifold, focusing on **Geometric Dual Quaternions (GeoDQ)**, **Principal Geodesic Analysis (PGA)**, and other geometric fusion methods.

Designed for Computer Vision, Robotics, and Navigation tasks.

## 📚 Algorithms Collection

| Project / Algorithm | Description | Code & Benchmarks | Documentation |
| :--- | :--- | :--- | :--- |
| **GeoDQ-Bench** | **Geometric State Fusion** using Dual Quaternion Observer. Comparative analysis with Kalman Filters (ESKF, UKFM). | [Code](/algorithms/geodq) <br> [Benchmark](/benchmarks/ronin_geodq_analysis.ipynb) | [Read details](/docs/geodq.md) |
| **ESKF / UKFM** | Error-State and Unscented Kalman Filter implementations for SE(3) navigation tasks. | [Code](/algorithms/filters) | [Read details](/docs/geodq.md) |
| **PGA-Bench** *(WIP)* | Principal Geodesic Analysis on Riemannian manifolds. | *Coming soon* | *Coming soon* |

## 🛠️ Setup & Usage

### Installation

```bash
git clone https://github.com/afanasyspb/SE3-Manifold-Lib.git
cd SE3-Manifold-Lib
pip install -r requirements.txt