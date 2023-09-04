#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
#include "estimator.h"
#include "scene.h"
#include "external/integrand.h"

using namespace nwob;
#define COLOR_SCALE 4

constexpr int equation_type = HELMHOLTZ;
auto G0 = Green_func<equation_type>;
auto G1 = Green_func_deriv<equation_type>;

class Reservoir
{
    public:
        uint y, M;
        float w_sum;
        HOST_DEVICE Reservoir() : y(0), M(0), w_sum(0) {}
        inline HOST_DEVICE void update(uint x, float w, float rd)
        {
            w_sum += w;
            M++;
            if (rd < w / w_sum)
                y = x;
        }
};

int main()
{
    std::string input_json_file = "../data/test.json";
    SceneHost scene_host(input_json_file);
    float3 x0 = {0.0f, 0.0f, 0.0f};
    float k = 10;
    scene_host.set_neumann([&](float3 p, float3 n) { return G1(x0, p, n, k); });
    int res = 256;
    GPUMatrix<uchar4> image(res, res);
    float3 grid_min_pos = {0, -2, -2};
    float width = 4;
    float3 dx = {0, width / res, 0};
    float3 dy = {0, 0, width / res};
    GPUMemory<unsigned long long> seeds(res * res);
    seeds.copy_from_host(get_random_seeds(res * res));

    std::vector<NeighborList> neighbor_list_host;
    std::vector<BoundaryPoint> boundary_points_host;
    neighbor_list_host.resize(scene_host.neighbor_list.size());
    scene_host.neighbor_list.copy_to_host(neighbor_list_host.data());
    boundary_points_host.resize(scene_host.boundary_points_device.size());
    scene_host.boundary_points_device.copy_to_host(boundary_points_host.data());

    int idx = 0;
    for (auto bp : boundary_points_host)
    {
        idx++;
        auto pos = bp.pos;
        auto &lst = neighbor_list_host[scene_host.grid.get_flat_index(pos)];
        std::ofstream ofs(std::string("../output/neighbor_list_") + std::to_string(idx) + ".txt");
        for (int i = 0; i < lst.size(); i++)
        {
            auto &p = boundary_points_host[lst[i]];
            ofs << p.pos.x << " " << p.pos.y << " " << p.pos.z << std::endl;
        }
        ofs << pos.x << " " << pos.y << " " << pos.z << std::endl;
    }

    // parallel_for(res * res, [seeds = seeds.device_ptr(), res, x0, grid_min_pos, dx, dy, k, scene =
    // scene_host.device(),
    //                          out = image.device_ptr()] __device__(int i) {
    //     int x = i % res;
    //     int y = i / res;
    //     float3 p = grid_min_pos + x * dx + y * dy;
    //     int spp = 100000;
    //     complex sum = 0;
    //     auto seed = seeds[i];
    //     randomState rand_state;
    //     curand_init(seed, 0, 0, &rand_state);
    //     for (int i = 0; i < spp; i++)
    //     {
    //         float inv_pdf;
    //         BoundaryPoint src, bp;
    //         thrust::tie(bp, inv_pdf) = scene.uniform_sample(&rand_state, src);
    //         sum += -inv_pdf * Green_func<HELMHOLTZ>(p, bp.pos, k) * bp.neumann;
    //         complex weight = inv_pdf * Green_func_deriv<HELMHOLTZ>(p, bp.pos, bp.normal, k);

    //         while (true)
    //         {
    //             float ksi = curand_uniform(&rand_state);
    //             float P_RR = 0.0001f;
    //             if (ksi > P_RR)
    //                 break;
    //             // if (i == path_depth - 1)
    //             //     weight *= 0.5f;
    //             //
    //             // if (i == path_depth - 1)
    //             // {
    //             //     sum += weight * Green_func<HELMHOLTZ>(x0, bp.pos, k);
    //             //     break;
    //             // }
    //             BoundaryPoint dst;
    //             thrust::tie(dst, inv_pdf) = scene.uniform_sample(&rand_state, bp);
    //             inv_pdf *= 2 / P_RR;
    //             sum += weight * (-inv_pdf * Green_func<HELMHOLTZ>(bp.pos, dst.pos, k) * dst.neumann);
    //             weight *= inv_pdf * Green_func_deriv<HELMHOLTZ>(bp.pos, dst.pos, dst.normal, k);
    //             bp = dst;
    //         }
    //     }
    //     sum /= spp;
    //     if (x == 0 && y == 0)
    //         printf("sum: %e\n", sum.real());
    //     float v = sum.real() * COLOR_SCALE + 0.5;

    //     out[x][y] = get_viridis_color(v);
    // });
    // MemoryVisualizer().write_to_png("../output/wob_real.png", &image);

    // parallel_for(res * res, [res, x0, grid_min_pos, dx, dy, k, scene = scene_host.device(),
    //                          out = image.device_ptr()] __device__(int i) {
    //     int x = i % res;
    //     int y = i / res;
    //     float3 p = grid_min_pos + x * dx + y * dy;
    //     complex sum = 0;
    //     for (int i = 0; i < scene.bvh.num_objects; i++)
    //     {
    //         auto &obj = scene.bvh.objects[i];
    //         float3 c = (obj.v0 + obj.v1 + obj.v2) / 3;
    //         complex dirichlet = Green_func<HELMHOLTZ>(x0, c, k).real();
    //         complex neumann = Green_func_deriv<HELMHOLTZ>(x0, c, obj.n, k).real();
    //         sum += face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER) * dirichlet -
    //                face2PointIntegrand(obj, p, k, nwob::SINGLE_LAYER) * neumann;
    //     }
    //     if (x == 0 && y == 0)
    //         printf("sum: %e\n", sum.real());
    //     float v = sum.real() * COLOR_SCALE + 0.5;
    //     out[x][y] = get_viridis_color(v);
    // });

    // MemoryVisualizer().write_to_png("../output/bem_real.png", &image);

    // parallel_for(res * res, [x0, grid_min_pos, dx, dy, k, res, out = image.device_ptr()] __device__(int i) {
    //     int x = i % res;
    //     int y = i / res;
    //     float3 p = grid_min_pos + x * dx + y * dy;
    //     float v = Green_func<HELMHOLTZ>(x0, p, k).real() * COLOR_SCALE + 0.5;
    //     out[x][y] = get_viridis_color(v);
    // });
    // MemoryVisualizer().write_to_png("../output/gt_real.png", &image);
}