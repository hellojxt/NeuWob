#pragma once
#include <cmath>
#include <iostream>
#include "common.h"
#include "helper_math.h"
#include <random>

NWOB_NAMESPACE_BEGIN

/*
    Neumann boundary equation:

    u(x) = 2∫ ∂G u - 2∫ G ∂u

    where G is the Green's function of the Laplacian operator, and ∂G is the normal derivative of G.
*/

enum
{
    POSSION = 0,
    HELMHOLTZ = 1,
};

template <int type>
HOST_DEVICE inline complex Green_func(float3 y, float3 x, float k);

template <int type>
HOST_DEVICE inline complex Green_func_deriv(float3 y, float3 x, float3 xn, float k);

template <>
HOST_DEVICE inline complex Green_func<HELMHOLTZ>(float3 y, float3 x, float k)
{
    float r = length(x - y);
    if (r < 1e-6)
        return complex(0, 0);
    return exp(complex(0, k * r)) / (4 * M_PI * r);
}

template <>
HOST_DEVICE inline complex Green_func_deriv<HELMHOLTZ>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    if (r < 1e-6)
        return complex(0, 0);
    complex ikr = complex(0, 1) * r * k;
    complex potential = -exp(ikr) / (4 * M_PI * r * r * r) * (1 - ikr) * dot(x - y, xn);
    return potential;
}

template <>
HOST_DEVICE inline complex Green_func<POSSION>(float3 y, float3 x, float k)
{
    float r = length(x - y);
    return 1 / (4 * M_PI * r);
}

template <>
HOST_DEVICE inline complex Green_func_deriv<POSSION>(float3 y, float3 x, float3 xn, float k)
{
    float r = length(x - y);
    return -1 / (4 * M_PI * r * r * r) * dot(x - y, xn);
}

inline std::vector<unsigned long long> get_random_seeds(int n)
{
    std::mt19937_64 random(0);  // fixed seed (for debugging)
    std::vector<unsigned long long> seeds(n);
    for (auto &seed : seeds)
        seed = random();
    return seeds;
}

NWOB_NAMESPACE_END