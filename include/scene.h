#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <thrust/optional.h>
#include <thrust/pair.h>
#include "common.h"
#include "helper_math.h"
#include "gpu_memory.h"
#include "estimator.h"
#include "grid.h"

NWOB_NAMESPACE_BEGIN

// triangle
struct Element
{
        int3 indices;
        float3 v0, v1, v2, n;
        complex N0, N1, N2;
        inline HOST_DEVICE float area() const noexcept { return length(cross((v1 - v0), (v2 - v0))) / 2.0f; }
        inline HOST_DEVICE complex neumann(float u, float v) const noexcept
        {
            return N0 * (1 - u - v) + N1 * u + N2 * v;
        }
        inline HOST_DEVICE float3 point(float u, float v) const noexcept { return v0 * (1 - u - v) + v1 * u + v2 * v; }
        inline HOST_DEVICE float3 center() const noexcept { return (v0 + v1 + v2) / 3.f; }
        inline HOST_DEVICE complex center_neumann() const noexcept { return (N0 + N1 + N2) / 3.f; }
};

// boundary point
struct BoundaryPoint
{
        complex neumann, dirichlet;
        float3 pos, normal;
        inline HOST_DEVICE BoundaryPoint() {}
        inline HOST_DEVICE BoundaryPoint(const Element &e, float u, float v)
            : neumann(e.neumann(u, v)), pos(e.point(u, v)), normal(e.n), dirichlet(0)
        {}
};

#define MAX_NEIGHBOR_NUM 64
using NeighborList = CellList<int, MAX_NEIGHBOR_NUM>;

class Scene
{
    public:
        size_t num_elements;
        Element *elements;
        size_t num_boundary_points;
        BoundaryPoint *boundary_points;
        Grid grid;
        NeighborList *neighbor_list;
        float total_area;
        float *area_cdf;

        inline __device__ NeighborList &get_neighbor_list(float3 pos) const noexcept
        {
            return neighbor_list[grid.get_flat_index(pos)];
        }

        inline __device__ int sample_points_index(randomState *rand_state) const noexcept
        {
            int xi = curand_uniform(rand_state) * num_boundary_points;
            if (xi == num_boundary_points)
                xi--;
            return xi;
        }

        inline __device__ BoundaryPoint sample_boundary_point(randomState *rand_state) const noexcept
        {
            float x = curand_uniform(rand_state) * total_area;
            // binary search
            uint l = 0, r = num_elements - 1;
            while (l < r)
            {
                uint mid = (l + r) / 2;
                if (area_cdf[mid] < x)
                    l = mid + 1;
                else
                    r = mid;
            }
            uint element_id = l;
            float u = curand_uniform(rand_state);
            float v = curand_uniform(rand_state);
            if (u + v > 1.f)
            {
                u = 1.f - u;
                v = 1.f - v;
            }
            return BoundaryPoint(elements[element_id], u, v);
        }
};

class SceneHost
{
    public:
        GPUMemory<Element> elements_device;
        GPUMemory<BoundaryPoint> boundary_points_device;
        GPUMemory<float> area_cdf;
        float total_area;
        Grid grid;
        GPUMemory<NeighborList> neighbor_list;

        SceneHost(const std::string config_json_file, int cut_idx = -1);

        Scene device()
        {
            return {
                .num_elements = elements_device.size(),
                .elements = elements_device.device_ptr(),
                .num_boundary_points = boundary_points_device.size(),
                .boundary_points = boundary_points_device.device_ptr(),
                .grid = grid,
                .neighbor_list = neighbor_list.device_ptr(),
                .total_area = total_area,
                .area_cdf = area_cdf.device_ptr(),
            };
        };
        void sample_boundary_points();
        void save_boundary_points(const std::string filename) const;

        void construct_neighbor_list();
};

NWOB_NAMESPACE_END