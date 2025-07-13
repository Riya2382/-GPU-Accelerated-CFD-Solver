# Documentation

# Project 2: GPU-Accelerated CFD Solver

This project demonstrates a basic 2D CFD solver implemented in CUDA. It is intended to introduce GPU parallelization concepts for fluid simulations.

## Features

- 2D velocity diffusion (incompressible)
- CUDA-based finite difference stencil
- Easily extensible to convection and pressure solvers

## Build Instructions

```bash
nvcc -o gpuCFD src/main.cu
./gpuCFD

