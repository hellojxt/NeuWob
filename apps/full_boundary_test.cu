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
    float k = 10;
    float3 x0 = make_float3(0, 0, 0);
    scene_host.set_neumann([&](float3 p, float3 n) { return Green_func_deriv<HELMHOLTZ>(x0, p, n, k); });
    GPUMemory<unsigned long long> seeds(1);
    seeds.copy_from_host(get_random_seeds(1));
    printf("seeds copied\n");
    parallel_for(1, [seeds = seeds.device_ptr(), x0, k, scene = scene_host.device()] __device__(int i) {
        int spp1 = 5, spp2 = 100000;
        int path_length = 5;
        complex sum = 0;
        auto seed = seeds[i];
        auto &bvh = scene.bvh;
        BoundaryPoint bp, pre, next, src;
        randomState rand_state;
        curand_init(seed, 0, 0, &rand_state);
        float inv_pdf;
        auto dirichlet_func = [x0] __device__(float3 p, float k) { return Green_func<HELMHOLTZ>(x0, p, k); };
        for (int i = 0; i < spp1; i++)
        {
            int trg_id = curand(&rand_state) % bvh.num_objects;
            complex b_dirichlet = 0;
            pre.pos = bvh.objects[trg_id].center();
            for (int j = 0; j < spp2; j++)
            {
                thrust::tie(next, inv_pdf) = scene.uniform_sample(&rand_state, pre);
                b_dirichlet +=
                    inv_pdf * 2 *
                    (Green_func_deriv<HELMHOLTZ>(pre.pos, next.pos, next.normal, k) * dirichlet_func(next.pos, k) -
                     Green_func<HELMHOLTZ>(pre.pos, next.pos, k) * next.neumann);
            }
            b_dirichlet /= spp2;
            printf("b_dirichlet: %e + %ei\n", b_dirichlet.real(), b_dirichlet.imag());

            complex bem_dirichlet = 0;
            complex gt_dirichlet = dirichlet_func(bvh.objects[trg_id].center(), k);
            printf("gt_dirichlet: %e + %ei\n", gt_dirichlet.real(), gt_dirichlet.imag());

            for (int j = 0; j < bvh.num_objects; j++)
            {
                bem_dirichlet += face2FaceIntegrand(bvh.objects[j], bvh.objects[trg_id], k, nwob::DOUBLE_LAYER) *
                                     dirichlet_func(bvh.objects[j].center(), k) -
                                 face2FaceIntegrand(bvh.objects[j], bvh.objects[trg_id], k, nwob::SINGLE_LAYER) *
                                     bvh.objects[j].center_neumann();
            }
            bem_dirichlet /= (0.5f * bvh.objects[trg_id].area());
            printf("bem_dirichlet: %e + %ei\n", bem_dirichlet.real(), bem_dirichlet.imag());
        }
    });
}