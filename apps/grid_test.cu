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
        if (lst.size() == 0)
            printf("no neighbor for %d\n", idx);
        ofs << pos.x << " " << pos.y << " " << pos.z << std::endl;
    }
}