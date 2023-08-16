#pragma once

#include <cmath>
#include <iostream>
#include "common.h"
#include "helper_math.h"
#include "scene.h"

NWOB_NAMESPACE_BEGIN

/*
    Neumann boundary equation:

    u(x) = 2∫ ∂G u - 2∫ G ∂u

    where G is the Green's function of the Laplacian operator, and ∂G is the normal derivative of G.
*/

HOST_DEVICE inline float Green_func(float3 x, float3 y, float k)
{
    float r = length(x - y);
    return cos(r * k) / (4 * M_PI * r);
}

HOST_DEVICE inline float Green_func_deriv(float3 x, float3 y, float3 yn, float k)
{
    float r = length(x - y);
    return -(cos(r * k) + r * k * sin(r * k)) / (4 * M_PI * r * r * r) * dot(y - x, yn);
}

class Estimator
{

    public:
        randomState *rand_state;
        const Scene &scene;
        uint path_length;

        inline HOST_DEVICE Estimator(randomState *rand_state, const Scene &scene, uint path_length)
            : rand_state(rand_state), scene(scene), path_length(path_length){};

        inline __device__ float compute_boundary_value(const BoundaryPoint &x, float wave_number)
        {
            float sum = 0;
            float weight = 1.0f;
            BoundaryPoint pre = x, next;
            float inv_pdf;
            for (int i = 0; i < path_length; i++)
            {
                if (i == path_length - 1)
                    weight *= 0.5f;
                BoundaryPoint next;
                thrust::tie(next, inv_pdf) = scene.sphere_sample(rand_state, pre);
                sum += -weight * 2 * Green_func(pre.pos, next.pos, wave_number) * next.neumann;
                thrust::tie(next, inv_pdf) = scene.sphere_sample(rand_state, pre);
                weight *= 2 * Green_func_deriv(pre.pos, next.pos, next.normal, wave_number);
                pre = next;
            }
            return sum;
        }
};

NWOB_NAMESPACE_END