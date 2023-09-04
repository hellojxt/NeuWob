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
        complex neumann;
        float3 pos, normal;
        inline HOST_DEVICE BoundaryPoint() {}
        inline HOST_DEVICE BoundaryPoint(const Element &e, float u, float v)
            : neumann(e.neumann(u, v)), pos(e.point(u, v)), normal(e.n)
        {}
};

#define MAX_NEIGHBOR_NUM 64
using NeighborList = CellList<int, MAX_NEIGHBOR_NUM>;

class Scene
{
    public:
        size_t num_boundary_points;
        BoundaryPoint *boundary_points;
        Grid grid;
        NeighborList *neighbor_list;

        inline __device__ NeighborList &get_neighbor_list(float3 pos) const noexcept
        {
            return neighbor_list[grid.get_flat_index(pos)];
        }
};

class SceneHost
{
    public:
        std::vector<Element> elements;
        GPUMemory<Element> elements_device;
        GPUMemory<BoundaryPoint> boundary_points_device;
        GPUMemory<float> area_cdf;
        float total_area;
        Grid grid;
        GPUMemory<NeighborList> neighbor_list;

        SceneHost(const std::string config_json_file, int cut_idx = -1);

        template <class Func>
        void set_neumann(Func neumann_func)
        {
            for (int i = 0; i < elements.size(); i++)
            {
                elements[i].N0 = neumann_func(elements[i].v0, elements[i].n);
                elements[i].N1 = neumann_func(elements[i].v1, elements[i].n);
                elements[i].N2 = neumann_func(elements[i].v2, elements[i].n);
            }
            elements_device.copy_from_host(elements);
        }
        Scene device()
        {
            return {
                .num_boundary_points = boundary_points_device.size(),
                .boundary_points = boundary_points_device.device_ptr(),
                .grid = grid,
                .neighbor_list = neighbor_list.device_ptr(),
            };
        };
        void sample_boundary_points();
        void save_boundary_points(const std::string filename) const;

        void construct_neighbor_list();
};

NWOB_NAMESPACE_END