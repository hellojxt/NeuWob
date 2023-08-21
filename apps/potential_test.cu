#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
#include "estimator.h"
#include "scene.h"
#include "multipole.h"
#include "external/integrand.h"

using namespace nwob;
#define MOLTIPOLR_M 1
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
    scene_host.set_neumann(
        [&](float3 p, float3 n) { return multipole_basis_deriv<MOLTIPOLR_M, 0>(x0, p, k, n).real(); });
    int res = 256;
    GPUMatrix<uchar4> image(res, res);
    float3 grid_min_pos = {0, -2, -2};
    float width = 4;
    float3 dx = {0, width / res, 0};
    float3 dy = {0, 0, width / res};
    GPUMemory<unsigned long long> seeds(res * res);
    seeds.copy_from_host(get_random_seeds(res * res));
    printf("seeds copied\n");
    parallel_for(res * res, [seeds = seeds.device_ptr(), res, x0, grid_min_pos, dx, dy, k, scene = scene_host.device(),
                             out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        // printf("x: %d, y: %d\n", x, y);
        float3 p = grid_min_pos + x * dx + y * dy;
        // printf("p: %f, %f, %f\n", p.x, p.y, p.z);
        int spp = 20000;
        int path_length = 10;
        complex sum = 0;
        auto seed = seeds[i];
        Estimator es(scene, seed, path_length);
        for (int i = 0; i < spp; i++)
        {
            sum += es.compute_domain_value(
                p, k, [x0] __device__(float3 p, float k) { return multipole_basis<MOLTIPOLR_M, 0>(x0, p, k).real(); });
        }
        sum /= spp;
        if (x == 0 && y == 0)
            printf("sum: %e\n", sum.real());
        float v = sum.real() * 4 + 0.5;

        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/wob_real.png", &image);

    parallel_for(res * res, [res, x0, grid_min_pos, dx, dy, k, scene = scene_host.device(),
                             out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        complex sum = 0;
        for (int i = 0; i < scene.bvh.num_objects; i++)
        {
            auto &obj = scene.bvh.objects[i];
            float3 c = (obj.v0 + obj.v1 + obj.v2) / 3;
            complex dirichlet = multipole_basis<MOLTIPOLR_M, 0>(x0, c, k).real();
            complex neumann = multipole_basis_deriv<MOLTIPOLR_M, 0>(x0, c, k, obj.n).real();
            sum += face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER) * dirichlet -
                   face2PointIntegrand(obj, p, k, nwob::SINGLE_LAYER) * neumann;
            // printf("face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER): %e, %e\n",
            //        face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER).real(),
            //        face2PointIntegrand(obj, p, k, nwob::DOUBLE_LAYER).imag());
            // printf("dirichlet: %e\n", dirichlet.real());
        }
        if (x == 0 && y == 0)
            printf("sum: %e\n", sum.real());
        float v = sum.real() * 4 + 0.5;
        out[x][y] = get_viridis_color(v);
    });

    MemoryVisualizer().write_to_png("../output/bem_real.png", &image);

    parallel_for(res * res, [x0, grid_min_pos, dx, dy, k, res, out = image.device_ptr()] __device__(int i) {
        int x = i % res;
        int y = i / res;
        float3 p = grid_min_pos + x * dx + y * dy;
        float v = multipole_basis<MOLTIPOLR_M, 0>(x0, p, k).real() * 4 + 0.5;
        out[x][y] = get_viridis_color(v);
    });
    MemoryVisualizer().write_to_png("../output/gt_real.png", &image);
}