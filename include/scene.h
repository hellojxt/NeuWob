#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <thrust/optional.h>
#include <thrust/pair.h>
#include "common.h"
#include "lbvh.cuh"
#include "helper_math.h"
#include <curand_kernel.h>
#include "gpu_memory.h"

NWOB_NAMESPACE_BEGIN

// triangle
struct Element
{
        int3 indices;
        float3 v0, v1, v2, n;
        complex N0, N1, N2, NC;
        inline HOST_DEVICE float area() const noexcept { return length(cross((v1 - v0), (v2 - v0))) / 2.0f; }
        inline HOST_DEVICE complex neumann(float u, float v) const noexcept
        {
            return N0 * (1 - u - v) + N1 * u + N2 * v;
        }
        inline HOST_DEVICE float3 point(float u, float v) const noexcept { return v0 * (1 - u - v) + v1 * u + v2 * v; }
        inline HOST_DEVICE float3 center() const noexcept { return (v0 + v1 + v2) / 3.f; }
        inline HOST_DEVICE complex center_neumann() const noexcept { return NC; }
};

// boundary point
struct BoundaryPoint
{
        uint element_id;
        float u, v;
        complex neumann;
        float3 pos, normal;
        inline HOST_DEVICE BoundaryPoint() {}
        inline HOST_DEVICE BoundaryPoint(const Element &e, float u, float v)
            : element_id(0), u(u), v(v), neumann(e.neumann(u, v)), pos(e.point(u, v)), normal(e.n)
        {}
};

// line element intersection
struct LineElementIntersect
{
        thrust::pair<bool, BoundaryPoint> HOST_DEVICE operator()(const lbvh::Line<float, 3> &line,
                                                                 const Element &tri) const noexcept
        {
            BoundaryPoint bp;
            float3 e1 = tri.v1 - tri.v0;
            float3 e2 = tri.v2 - tri.v0;
            float3 line_dir = make_float3(line.dir);
            float3 line_origin = make_float3(line.origin);
            float3 p = cross(line_dir, e2);
            float det = dot(e1, p);
            if (fabs(det) < 1e-8)
                return thrust::make_pair(false, bp);
            float inv_det = 1.f / det;
            float3 t = line_origin - tri.v0;
            float u = dot(t, p) * inv_det;
            if (u < 0.f || u > 1.f)
                return thrust::make_pair(false, bp);
            float3 q = cross(t, e1);
            float v = dot(line_dir, q) * inv_det;
            if (v < 0.f || u + v > 1.f)
                return thrust::make_pair(false, bp);
            float t_ = dot(e2, q) * inv_det;
            if (t_ < 0.f)
                return thrust::make_pair(false, bp);
            bp = BoundaryPoint(tri, u, v);
            // float3 pos = line_origin + line_dir * t_;
            // assert(abs(bp.pos.x - pos.x) < 1e-3);
            // assert(abs(bp.pos.y - pos.y) < 1e-3);
            // assert(abs(bp.pos.z - pos.z) < 1e-3);
            return thrust::make_pair(true, bp);
        }
};

// bbox
struct ElementAABB
{
        HOST_DEVICE lbvh::aabb<float, 3> operator()(const Element &tri) const noexcept
        {
            lbvh::aabb<float, 3> aabb;
            aabb.lower = make_float4(fminf(fminf(tri.v0, tri.v1), tri.v2));
            aabb.upper = make_float4(fmaxf(fmaxf(tri.v0, tri.v1), tri.v2));
            return aabb;
        }
};

class Scene
{
    public:
        lbvh::bvh_device<float, 3, Element> bvh;
        float total_area;
        float *area_cdf;
        inline __device__ thrust::pair<BoundaryPoint, float> sphere_sample(randomState *rand_state,
                                                                           const BoundaryPoint &src) const
        {
            float4 origin = make_float4(src.pos);
            float phi = 2.f * M_PI * curand_uniform(rand_state);
            float4 dir;
            dir.w = 0.f;
            dir.z = 2.f * curand_uniform(rand_state) - 1.f;
            dir.x = sqrtf(1.f - dir.z * dir.z) * cosf(phi);
            dir.y = sqrtf(1.f - dir.z * dir.z) * sinf(phi);
            lbvh::Line<float, 3> line(origin, dir);
            constexpr uint buffer_size = 64;
            thrust::pair<uint, BoundaryPoint> buffer[buffer_size];
            auto num_intersections = lbvh::query_device(bvh, lbvh::query_line_intersect<float, 3>(line),
                                                        LineElementIntersect(), buffer, buffer_size);
            // printf("num_intersections: %d\n", num_intersections);
            if (num_intersections == 0)
                return {src, 0};

            uint idx = curand(rand_state) % num_intersections;
            uint element_id = buffer[idx].first;
            BoundaryPoint p_next = buffer[idx].second;
            p_next.element_id = element_id;
            float r = length(p_next.pos - src.pos);
            float inv_pdf = num_intersections * 4.f * M_PI * r * r /
                            max(abs(dot(bvh.objects[element_id].n, make_float3(dir))), 1e-5f);
            return {p_next, inv_pdf};
        }

        inline __device__ thrust::pair<BoundaryPoint, float> uniform_sample(randomState *rand_state,
                                                                            const BoundaryPoint &src) const
        {
            float x = curand_uniform(rand_state) * total_area;
            // binary search
            uint l = 0, r = bvh.num_objects - 1;
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
            float inv_pdf = total_area;
            BoundaryPoint p_next(bvh.objects[element_id], u, v);
            p_next.element_id = element_id;
            // printf("element_id: %d, u: %f, v: %f, inv_pdf: %e, total_area: %e, pos: %f %f %f\n", element_id, u, v,
            //        inv_pdf, total_area, p_next.pos.x, p_next.pos.y, p_next.pos.z);
            return {p_next, inv_pdf};
        }
};

class SceneHost

{

    public:
        lbvh::bvh<float, 3, Element, ElementAABB> bvh;
        std::vector<Element> elements;
        GPUMemory<float> area_cdf;
        float total_area;

        SceneHost(const std::string config_json_file, int cut_idx = -1);

        template <class Func>
        void set_neumann(Func neumann_func)
        {
            for (int i = 0; i < elements.size(); i++)
            {
                elements[i].N0 = neumann_func(elements[i].v0, elements[i].n);
                elements[i].N1 = neumann_func(elements[i].v1, elements[i].n);
                elements[i].N2 = neumann_func(elements[i].v2, elements[i].n);
                elements[i].NC = neumann_func(elements[i].center(), elements[i].n);
            }
            bvh = lbvh::bvh<float, 3, Element, ElementAABB>(elements.begin(), elements.end());
            std::vector<float> area_cdf_host(elements.size());
            total_area = 0.f;
            for (int i = 0; i < elements.size(); i++)
            {
                float area = elements[i].area();
                total_area += area;
                area_cdf_host[i] = total_area;
            }
            area_cdf.resize_and_copy_from_host(area_cdf_host);
        }

        Scene device()
        {
            return {lbvh::bvh_device<float, 3, Element>(bvh.get_device_repr()), total_area, area_cdf.data()};
        }
};

NWOB_NAMESPACE_END