// Main CUDA source file for the GPU CFD Solver
/* Basic 2D GPU CFD Solver using CUDA */
/* Solves 2D incompressible Navier-Stokes using finite difference */

#include <stdio.h>
#include <cuda.h>

#define NX 128
#define NY 128
#define NSTEPS 1000

__global__ void update_velocity(float *u, float *v, float *u_new, float *v_new, float dt, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < NX - 1 && j < NY - 1) {
        int idx = j * NX + i;
        u_new[idx] = u[idx] + dt * (u[idx - 1] - 2 * u[idx] + u[idx + 1]) / (dx * dx);
        v_new[idx] = v[idx] + dt * (v[idx - NX] - 2 * v[idx] + v[idx + NX]) / (dy * dy);
    }
}

int main() {
    size_t size = NX * NY * sizeof(float);
    float *u, *v, *u_new, *v_new;

    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&u_new, size);
    cudaMallocManaged(&v_new, size);

    for (int i = 0; i < NX * NY; ++i) {
        u[i] = 0.0f;
        v[i] = 0.0f;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + 15) / 16, (NY + 15) / 16);

    for (int t = 0; t < NSTEPS; ++t) {
        update_velocity<<<numBlocks, threadsPerBlock>>>(u, v, u_new, v_new, 0.01f, 0.01f, 0.01f);
        cudaDeviceSynchronize();

        float *tmp_u = u; u = u_new; u_new = tmp_u;
        float *tmp_v = v; v = v_new; v_new = tmp_v;
    }

    cudaFree(u); cudaFree(v); cudaFree(u_new); cudaFree(v_new);
    return 0;
}
