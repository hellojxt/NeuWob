#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
#include "estimator.h"
#include "scene.h"
#include "external/integrand.h"

using namespace nwob;
#define MOLTIPOLR_M 0
#define COLOR_SCALE 10

int main(int argc, char *argv[])
{

    std::string input_json_file;
    if (argc != 2)
    {
        input_json_file = "../data/test.json";
    }
    else
        input_json_file = argv[1];
    SceneHost scene_host(input_json_file);
    float3 x0 = {0.0f, 0.0f, 0.0f};
    float k = 10;
    scene_host.set_neumann([&](float3 p, float3 n) { return Green_func_deriv<POSSION>(x0, p, n, k); });
    int res = 256;
    GPUMatrix<uchar4> image(res, res);
    float3 grid_min_pos = {0, -2, -2};
    float width = 4;
    float3 dx = {0, width / res, 0};
    float3 dy = {0, 0, width / res};
    GPUMemory<unsigned long long> seeds(res * res);
    seeds.copy_from_host(get_random_seeds(res * res));
    printf("seeds copied\n");
    int path_depth;
    std::cin >> path_depth;
    parallel_for(res * res, [path_depth, seeds = seeds.device_ptr(), res, x0, grid_min_pos, dx, dy, k,
                             scene = scene_host.device(), out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        int spp = 100000;
        complex sum = 0;
        auto seed = seeds[i];
        randomState rand_state;
        curand_init(seed, 0, 0, &rand_state);
        for (int i = 0; i < spp; i++)
        {
            float inv_pdf;
            BoundaryPoint src, bp;
            thrust::tie(bp, inv_pdf) = scene.uniform_sample(&rand_state, src);
            sum += -inv_pdf * Green_func<POSSION>(p, bp.pos, k) * bp.neumann;
            complex weight = inv_pdf * Green_func_deriv<POSSION>(p, bp.pos, bp.normal, k);

            for (int i = 0; i < path_depth; i++)
            {
                if (i == path_depth - 1)
                    weight *= 0.5f;
                // {
                //     sum += weight * Green_func<POSSION>(x0, bp.pos, k);
                //     break;
                // }

                BoundaryPoint dst;
                thrust::tie(dst, inv_pdf) = scene.uniform_sample(&rand_state, bp);
                inv_pdf *= 2;
                sum += weight * (-inv_pdf * Green_func<POSSION>(bp.pos, dst.pos, k) * dst.neumann);
                weight *= inv_pdf * Green_func_deriv<POSSION>(bp.pos, dst.pos, dst.normal, k);
                bp = dst;
            }
        }
        sum /= spp;
        if (x == 0 && y == 0)
            printf("sum: %e\n", sum.real());
        float v = sum.real() * COLOR_SCALE;

        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/wob_real.png", &image);

    parallel_for(res * res, [x0, grid_min_pos, dx, dy, k, res, out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        float v = Green_func<POSSION>(x0, p, k).real() * COLOR_SCALE;
        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/gt_real.png", &image);
}