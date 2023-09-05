#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
#include "estimator.h"
#include "scene.h"
#include "external/integrand.h"

using namespace nwob;
#define COLOR_SCALE 4

class Reservoir
{
    public:
        uint y, M;
        float w_sum;
        float W;
        HOST_DEVICE Reservoir() : y(0), M(0), w_sum(0), W(0) {}
        inline HOST_DEVICE void update(uint xi, float wi, float rd)
        {
            w_sum += wi;
            M++;
            if (rd <= wi / w_sum)
                y = xi;
        }
};

class ReservoirPair
{
    public:
        Reservoir dirichlet, neumann;
};

void set_ground_truth(SceneHost &scene_host, float3 x0, float k)
{
    parallel_for(scene_host.boundary_points_device.size(), [x0, k, scene = scene_host.device()] __device__(int i) {
        auto &bp = scene.boundary_points[i];
        bp.dirichlet = Green_func<HELMHOLTZ>(x0, bp.pos, k);
    });
}

int main()
{
    std::string input_json_file = "../data/test.json";
    SceneHost scene_host(input_json_file);
    float3 x0 = {0.0f, 0.0f, 0.0f};
    float k = 10;
    parallel_for(scene_host.boundary_points_device.size(), [x0, k, scene = scene_host.device()] __device__(int i) {
        auto &bp = scene.boundary_points[i];
        bp.neumann = Green_func_deriv<HELMHOLTZ>(x0, bp.pos, bp.normal, k);
    });
    int res = 256;
    GPUMatrix<uchar4> image(res, res);
    float3 grid_min_pos = {0, -2, -2};
    float width = 4;
    float3 dx = {0, width / res, 0};
    float3 dy = {0, 0, width / res};

    int max_iter = 1000;
    int RIS_M = 32;
    int path_depth = 1;
    // std::cin >> path_depth;
    int points_num = scene_host.boundary_points_device.size();
    GPUMemory<ReservoirPair> reservoirs(points_num);
    reservoirs.memset(0);
    GPUMemory<unsigned long long> seeds(points_num);
    seeds.copy_from_host(get_random_seeds(points_num));
    GPUMemory<randomState> rand_states(points_num);
    parallel_for(points_num, [seeds = seeds.device_ptr(), rand_states = rand_states.device_ptr()] __device__(int i) {
        curand_init(seeds[i], 0, 0, &rand_states[i]);
    });

    for (int iter_idx = 0; iter_idx < max_iter; iter_idx++)
    {
        // RIS
        // parallel_for(points_num, [RIS_M, k, scene = scene_host.device(), seeds = seeds.device_ptr(),
        //                           reservoirs = reservoirs.device_ptr()] __device__(int i) {
        //     auto &bp = scene.boundary_points[i];
        //     randomState rand_state;
        //     curand_init(seeds[i], 0, 0, &rand_state);
        //     ReservoirPair r;
        //     for (int j = 0; j < RIS_M; j++)
        //     {
        //         int xi = curand_uniform(&rand_state) * scene.num_boundary_points;
        //         auto src = scene.boundary_points[xi];
        //         float rd = curand_uniform(&rand_state);
        //         // float p_hat0 = max(abs(Green_func_deriv<HELMHOLTZ>(bp.pos, src.pos, src.normal, k).real()),
        //         1e-3f); float p_hat0 = 1.0f;
        //         // float p_hat1 = max(abs(Green_func<HELMHOLTZ>(bp.pos, src.pos, k).real()), 1e-3f);
        //         float p_hat1 = 1.0f;
        //         float p_sample = 1.0f / scene.total_area;
        //         r.dirichlet.update(xi, p_hat0 / p_sample, rd);
        //         r.neumann.update(xi, p_hat1 / p_sample, rd);
        //     }
        //     auto src = scene.boundary_points[r.dirichlet.y];
        //     // float p_hat = max(abs(Green_func_deriv<HELMHOLTZ>(bp.pos, src.pos, src.normal, k).real()), 1e-3f);
        //     float p_hat = 1.0f;
        //     r.dirichlet.W = 1.0f / p_hat * (r.dirichlet.w_sum / r.dirichlet.M);
        //     src = scene.boundary_points[r.neumann.y];
        //     // p_hat = max(abs(Green_func<HELMHOLTZ>(bp.pos, src.pos, k).real()), 1e-3f);
        //     p_hat = 1.0f;
        //     r.neumann.W = 1.0f / p_hat * (r.neumann.w_sum / r.neumann.M);
        //     reservoirs[i] = r;
        // });

        // Spatial reuse

        // Compute boundary points value
        parallel_for(points_num, [iter_idx, max_iter, x0, k, path_depth, scene = scene_host.device(),
                                  rand_states = rand_states.device_ptr(),
                                  reservoirs = reservoirs.device_ptr()] __device__(int i) {
            int trg_idx = i;
            complex sum = 0;
            complex weight = 1.0f;
            auto &trg = scene.boundary_points[trg_idx];
            // auto &src_dirichlet = scene.boundary_points[reservoirs[trg_idx].dirichlet.y];
            // auto &src_neumann = scene.boundary_points[reservoirs[trg_idx].neumann.y];
            int xi = scene.sample_points_index(&rand_states[i]);
            auto &src_dirichlet = scene.boundary_points[xi];
            xi = scene.sample_points_index(&rand_states[i]);
            auto &src_neumann = scene.boundary_points[xi];
            reservoirs[i].dirichlet.W = scene.total_area;
            reservoirs[i].neumann.W = scene.total_area;
            sum = 2 *
                  (reservoirs[i].dirichlet.W *
                       Green_func_deriv<HELMHOLTZ>(trg.pos, src_dirichlet.pos, src_dirichlet.normal, k) *
                       Green_func<HELMHOLTZ>(src_dirichlet.pos, x0, k) -
                   reservoirs[i].neumann.W * Green_func<HELMHOLTZ>(trg.pos, src_neumann.pos, k) * src_neumann.neumann);
            // for (int j = 0; j < path_depth; j++)
            // {
            //     if (j == path_depth - 1)
            //         weight *= 0.5f;  // path truncation in WOB
            //     auto &trg = scene.boundary_points[trg_idx];
            //     auto &src = scene.boundary_points[reservoirs[trg_idx].y];
            //     sum += -weight * 2 * reservoirs[i].W * Green_func<HELMHOLTZ>(trg.pos, src.pos, k) *
            //     src.neumann; weight *= 2 * reservoirs[i].W * Green_func_deriv<HELMHOLTZ>(trg.pos, src.pos,
            //     src.normal, k); trg_idx = reservoirs[trg_idx].y;
            // }
            scene.boundary_points[i].dirichlet += sum;
            if (iter_idx == max_iter - 1)
                scene.boundary_points[i].dirichlet /= max_iter;
        });
    }

    // Compute domain points value
    // set_ground_truth(scene_host, x0, k);
    parallel_for(res * res, [res, x0, grid_min_pos, dx, dy, k, scene = scene_host.device(),
                             out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        complex sum = 0;
        int sample_num = scene.num_boundary_points;
        for (int i = 0; i < sample_num; i++)
        {
            auto bp = scene.boundary_points[i];
            sum += scene.total_area * (Green_func_deriv<HELMHOLTZ>(p, bp.pos, bp.normal, k) * bp.dirichlet -
                                       0 * Green_func<HELMHOLTZ>(p, bp.pos, k) * bp.neumann);
        }
        sum /= sample_num;
        if (x == 0 && y == 0)
            printf("%f\n", sum.real());
        float v = sum.real() * COLOR_SCALE + 0.5;
        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/wob_real.png", &image);

    parallel_for(res * res, [res, x0, grid_min_pos, dx, dy, k, scene = scene_host.device(),
                             out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        complex sum = 0;
        for (int i = 0; i < scene.num_elements; i++)
        {
            auto &obj = scene.elements[i];
            float3 c = (obj.v0 + obj.v1 + obj.v2) / 3;
            complex dirichlet = Green_func<HELMHOLTZ>(x0, c, k).real();
            complex neumann = Green_func_deriv<HELMHOLTZ>(x0, c, obj.n, k).real();
            sum += face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER) * dirichlet -
                   0 * face2PointIntegrand(obj, p, k, nwob::SINGLE_LAYER) * neumann;
        }
        if (x == 0 && y == 0)
            printf("%f\n", sum.real());
        float v = sum.real() * COLOR_SCALE + 0.5;
        out[x][y] = get_viridis_color(v);
    });

    MemoryVisualizer().write_to_png("../output/bem_real.png", &image);

    parallel_for(res * res, [x0, grid_min_pos, dx, dy, k, res, out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        float v = Green_func<HELMHOLTZ>(x0, p, k).real() * COLOR_SCALE + 0.5;
        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/gt_real.png", &image);
}