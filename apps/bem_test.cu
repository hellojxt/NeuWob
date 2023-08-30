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
    float3 x0 = {0, 0, 0};
    scene_host.set_neumann([&](float3 p, float3 n) { return Green_func_deriv<HELMHOLTZ>(x0, p, n, k); });

    complex bem_dirichlet = 0;
    int trg_id = 2;
    auto &e_lst = scene_host.elements;
    auto &trg_e = e_lst[trg_id];
    printf("trg_e: (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n", trg_e.v0.x, trg_e.v0.y, trg_e.v0.z, trg_e.v1.x,
           trg_e.v1.y, trg_e.v1.z, trg_e.v2.x, trg_e.v2.y, trg_e.v2.z);
    for (int i = 0; i < e_lst.size(); i++)
    {
        auto &src_e = e_lst[i];
        complex double_layer = face2FaceIntegrand(src_e, trg_e, k, nwob::DOUBLE_LAYER);
        complex dirichlet = Green_func<HELMHOLTZ>(src_e.center(), x0, k);
        complex single_layer = face2FaceIntegrand(src_e, trg_e, k, nwob::SINGLE_LAYER);
        complex neumann = src_e.center_neumann();
        complex contrib = double_layer * dirichlet - single_layer * neumann;
        bem_dirichlet += contrib;
    }
    printf("bem_dirichlet: %e + %ei\n", bem_dirichlet.real(), bem_dirichlet.imag());
    complex gt_dirichlet = Green_func<HELMHOLTZ>(trg_e.center(), x0, k);
    gt_dirichlet *= (0.5f * trg_e.area());
    printf("gt_dirichlet: %e + %ei\n", gt_dirichlet.real(), gt_dirichlet.imag());
}
