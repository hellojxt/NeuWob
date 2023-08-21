#pragma once

#include <cmath>
#include <iostream>
#include "common.h"
#include "helper_math.h"
#include "scene.h"
#include <random>
NWOB_NAMESPACE_BEGIN

/*
    Neumann boundary equation:

    u(x) = 2∫ ∂G u - 2∫ G ∂u

    where G is the Green's function of the Laplacian operator, and ∂G is the normal derivative of G.
*/

HOST_DEVICE inline complex Green_func(float3 y, float3 x, float k)
{
    float r = length(x - y);
    return exp(complex(0, k * r)) / (4 * M_PI * r);
}

HOST_DEVICE inline complex Green_func_deriv(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    return potential;
}

inline std::vector<unsigned long long> get_random_seeds(int n)
{
    std::mt19937_64 random(0);  // fixed seed (for debugging)
    std::vector<unsigned long long> seeds(n);
    for (auto &seed : seeds)
        seed = random();
    return seeds;
}

class Estimator
{

    public:
        randomState rand_state;
        const Scene &scene;
        uint path_length;

        inline __device__ Estimator(const Scene &scene, unsigned long long seed, uint path_length)
            : scene(scene), path_length(path_length)
        {
            curand_init(seed, 0, 0, &rand_state);
        };

        inline __device__ complex compute_boundary_value(const BoundaryPoint &x, float wave_number)
        {
            complex sum = 0;
            complex weight = 1.0f;
            BoundaryPoint pre = x, next;
            float inv_pdf;
            for (int i = 0; i < path_length; i++)
            {
                if (i == path_length - 1)
                    weight *= 0.5f;
                BoundaryPoint next;
                thrust::tie(next, inv_pdf) = scene.sphere_sample(&rand_state, pre);
                if (inv_pdf != 0)
                    sum += -inv_pdf * weight * 2 * Green_func(pre.pos, next.pos, wave_number) * next.neumann;
                thrust::tie(next, inv_pdf) = scene.sphere_sample(&rand_state, pre);
                if (inv_pdf != 0)
                    weight *= inv_pdf * 2 * Green_func_deriv(pre.pos, next.pos, next.normal, wave_number);
                pre = next;
            }
            return sum;
        }

        inline __device__ complex compute_domain_value(float3 x, float wave_number)
        {
            complex sum = 0;
            float inv_pdf;
            BoundaryPoint src, bp;
            src.pos = x;
            thrust::tie(bp, inv_pdf) = scene.sphere_sample(&rand_state, src);
            if (inv_pdf != 0)
                sum += -inv_pdf * Green_func(x, bp.pos, wave_number) * bp.neumann;
            thrust::tie(bp, inv_pdf) = scene.sphere_sample(&rand_state, src);
            if (inv_pdf != 0)
                sum += inv_pdf * Green_func_deriv(x, bp.pos, bp.normal, wave_number) *
                       compute_boundary_value(bp, wave_number);
            return sum;
        }

        template <class Func>
        inline __device__ complex compute_domain_value(float3 x, float wave_number, Func boundary_value_func)
        {
            complex sum = 0;
            float inv_pdf;
            BoundaryPoint src, bp;
            src.pos = x;
            thrust::tie(bp, inv_pdf) = scene.uniform_sample(&rand_state, src);

            sum += -inv_pdf * Green_func(x, bp.pos, wave_number) * bp.neumann;

            thrust::tie(bp, inv_pdf) = scene.uniform_sample(&rand_state, src);

            sum += inv_pdf * Green_func_deriv(x, bp.pos, bp.normal, wave_number) *
                   boundary_value_func(bp.pos, wave_number);

            return sum;
        }
};

NWOB_NAMESPACE_END