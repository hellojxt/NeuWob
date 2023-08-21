#pragma once

#include <cmath>
#include <iostream>
#include "common.h"
#include "helper_math.h"

NWOB_NAMESPACE_BEGIN

// Higher-order multipole sources

template <int M, int N>
inline HOST_DEVICE complex multipole_basis(float3 x0, float3 x, float k)
{
    printf("Multipole basis not implemented for M=%d, N=%d\n", M, N);
    return complex(0, 0);
}

template <int M, int N>
inline HOST_DEVICE complex multipole_basis_deriv(float3 x0, float3 x, float k, float3 n)
{
    printf("Multipole basis not implemented for M=%d, N=%d\n", M, N);
    return complex(0, 0);
}

inline HOST_DEVICE complex spherical_hankel_2(int n, complex z)
{
    const complex i(0, 1);
    // recurrence relation
    if (n == 0)
        return i * exp(-i * z) / z;
    else if (n == 1)
        return -(z - i) / (z * z) * exp(-i * z);
    else
        return (2 * n - 1) / z * spherical_hankel_2(n - 1, z) - spherical_hankel_2(n - 2, z);
}

inline HOST_DEVICE complex spherical_hankel_2_deriv(int n, complex z)
{
    const complex i(0, 1);
    if (n == 0)
        return exp(-i * z) / z - i * exp(-i * z) / (z * z);
    else
        return 0.5f *
               (spherical_hankel_2(n - 1, z) - (spherical_hankel_2(n, z) + z * spherical_hankel_2(n + 1, z)) / z);
}

template <>
inline HOST_DEVICE complex multipole_basis<0, 0>(float3 x0, float3 x, float k)
{
    const complex i(0, 1);
    float3 r = x - x0;
    float r_norm = length(r);
    float z = k * r_norm;
    return i * 0.5f * sqrtf(1.0f / M_PI) * spherical_hankel_2(0, z);
}

template <>
inline HOST_DEVICE complex multipole_basis_deriv<0, 0>(float3 x0, float3 x, float k, float3 n)
{
    const complex i(0, 1);
    float3 r = x - x0;
    float r_norm = length(r);
    float z = k * r_norm;
    float3 hat_r = make_float3(r.x / r_norm, r.y / r_norm, r.z / r_norm);
    complex df_dr = 0.5f * sqrtf(1.0f / M_PI) * k * spherical_hankel_2_deriv(0, z);
    complex grad_x = df_dr * hat_r.x;
    complex grad_y = df_dr * hat_r.y;
    complex grad_z = df_dr * hat_r.z;
    return i * (grad_x * n.x + grad_y * n.y + grad_z * n.z);
}

template <>
inline HOST_DEVICE complex multipole_basis<1, 0>(float3 x0, float3 x, float k)
{
    const complex i(0, 1);
    float3 r = x - x0;
    float r_norm = length(r);
    float z = k * r_norm;
    return 0.5f * sqrtf(3.0f / M_PI) * spherical_hankel_2(0, z) * r.z / r_norm;
}

template <>
inline HOST_DEVICE complex multipole_basis_deriv<1, 0>(float3 x0, float3 x, float k, float3 n)
{
    const complex i(0, 1);
    float3 r = x - x0;
    float r_norm = length(r);
    float z = k * r_norm;
    float theta = acosf(r.z / r_norm);
    float phi = atan2f(r.y, r.x);
    float3 hat_r = make_float3(r.x / r_norm, r.y / r_norm, r.z / r_norm);
    float3 hat_theta = make_float3(cos(phi) * cos(theta), sin(phi) * cos(theta), -sin(theta));
    complex df_dr = (0.5f * sqrtf(3.0f / M_PI) * k * spherical_hankel_2_deriv(0, z) * cos(theta));
    complex df_dtheta = (-0.5f * sqrtf(3.0f / M_PI) * spherical_hankel_2(0, z) * sin(theta));
    complex grad_x = df_dr * hat_r.x + df_dtheta * hat_theta.x / r_norm;
    complex grad_y = df_dr * hat_r.y + df_dtheta * hat_theta.y / r_norm;
    complex grad_z = df_dr * hat_r.z + df_dtheta * hat_theta.z / r_norm;
    return grad_x * n.x + grad_y * n.y + grad_z * n.z;
}
NWOB_NAMESPACE_END