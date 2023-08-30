#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
#include "estimator.h"
#include "scene.h"
#include "multipole.h"
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
    SceneHost scene_host(input_json_file, 1);
    float3 x0 = {0.0f, 0.0f, 0.0f};
    float k = 10;
    scene_host.set_neumann([&](float3 p, float3 n) { return 1.0f; });
    int res = 1;
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
        int spp = 100000;
        int path_length = 5;
        complex sum = 0;
        auto seed = seeds[i];
        randomState rand_state;
        curand_init(seed, 0, 0, &rand_state);
        auto tri = scene.bvh.objects[0];
        complex dirichlet = face2FaceIntegrand(tri, tri, k, nwob::SINGLE_LAYER) * tri.neumann(0.5f, 0.5f) /
                            (face2FaceIntegrand(tri, tri, k, nwob::DOUBLE_LAYER) - 0.5f * tri.area());

        printf("dirichlet: %e + %ei\n", dirichlet.real(), dirichlet.imag());

        BoundaryPoint bp, pre, next;
        float inv_pdf;
        for (int i = 0; i < spp; i++)
        {
            thrust::tie(bp, inv_pdf) = scene.uniform_sample(&rand_state, bp);
            pre = bp;
            thrust::tie(next, inv_pdf) = scene.uniform_sample(&rand_state, pre);
            sum += -inv_pdf * 2 * Green_func<HELMHOLTZ>(pre.pos, next.pos, k) * next.neumann;
            thrust::tie(next, inv_pdf) = scene.uniform_sample(&rand_state, pre);
            sum += inv_pdf * 2 * Green_func_deriv<HELMHOLTZ>(pre.pos, next.pos, next.normal, k) * dirichlet;
        }
        sum /= spp;
        printf("sum: %e + %ei\n", sum.real(), sum.imag());
        float v = sum.real() * COLOR_SCALE + 0.5;
        out[x][y] = get_viridis_color(v);
    });
}