#pragma once
#include <stdio.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>  // CUDA
#include <device_launch_parameters.h>
#include "common.h"
#include "gpu_memory.h"

NWOB_NAMESPACE_BEGIN

class MemoryVisualizer
{
    public:
        GLuint image;
        cudaGraphicsResource_t CudaResource;
        cudaArray *array;
        GPUMatrix<uchar4> *gpu_matrix;
        void visualize(GPUMatrix<uchar4> *data);
};

NWOB_NAMESPACE_END