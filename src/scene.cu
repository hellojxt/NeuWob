#include "scene.h"
#include "nlohmann/json_fwd.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <vector>
#include <curand_kernel.h>
#include <thrust/remove.h>
NWOB_NAMESPACE_BEGIN

SceneHost::SceneHost(const std::string config_json_file, int cut_idx)
{
    using json = nlohmann::json;
    json config;
    {
        std::ifstream config_file(config_json_file);
        if (!config_file)
        {
            std::cout << "Failed to load config file: " << config_json_file << std::endl;
            return;
        }
        config_file >> config;
        config_file.close();
    }
    const std::string input_obj_file = DATA_DIR + config["input_obj_file"].get<std::string>();
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(input_obj_file))
    {
        std::cerr << "Failed to load " << input_obj_file << std::endl;
        exit(1);
    }
    if (!reader.Warning().empty())
    {
        std::cout << "WARN: " << reader.Warning() << std::endl;
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();

    std::vector<float3> vertices(attrib.vertices.size() / 3);
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++)
    {
        vertices[v] = make_float3(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
    }
    int triangle_num = 0;
    for (auto &shape : shapes)
    {
        triangle_num += shape.mesh.num_face_vertices.size();
    }

    std::vector<int3> triangles(triangle_num);

    for (auto &shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int vertice_num_per_face = shape.mesh.num_face_vertices[f];
            assert(vertice_num_per_face == 3);
            tinyobj::index_t idx0 = shape.mesh.indices[f * 3 + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[f * 3 + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[f * 3 + 2];
            triangles[f + index_offset] = make_int3(idx0.vertex_index, idx1.vertex_index, idx2.vertex_index);
        }
        index_offset += shape.mesh.num_face_vertices.size();
    }

    std::cout << "Vertices number: " << vertices.size() << "\n";
    std::cout << "Triangles number: " << triangles.size() << "\n";
    // Loaded success
    std::cout << "OBJ file:" << input_obj_file << " loaded!"
              << "\n";

    std::vector<Element> elements;
    if (cut_idx >= 1)
        elements.resize(cut_idx);
    else
        elements.resize(triangles.size());

    for (int i = 0; i < elements.size(); i++)
    {
        elements[i].v0 = vertices[triangles[i].x];
        elements[i].v1 = vertices[triangles[i].y];
        elements[i].v2 = vertices[triangles[i].z];
        elements[i].n = normalize(cross(elements[i].v1 - elements[i].v0, elements[i].v2 - elements[i].v0));
        elements[i].indices = make_int3(triangles[i].x, triangles[i].y, triangles[i].z);
    }
    elements_device.resize_and_copy_from_host(elements);
    std::vector<float> area_cdf_host(elements.size());
    total_area = 0.f;
    for (int i = 0; i < elements.size(); i++)
    {
        float area = elements[i].area();
        total_area += area;
        area_cdf_host[i] = total_area;
    }
    area_cdf.resize_and_copy_from_host(area_cdf_host);
    size_t boundary_point_num = config["boundary_point_num"].get<int>();
    boundary_points_device.resize(boundary_point_num);
    printf("Number of boundary points: %ld\n", boundary_point_num);
    sample_boundary_points();

    auto grid_min_point = config["grid_min_point"].get<std::vector<float>>();
    auto grid_max_point = config["grid_max_point"].get<std::vector<float>>();
    grid.min_pos = make_float3(grid_min_point[0], grid_min_point[1], grid_min_point[2]);
    grid.max_pos = make_float3(grid_max_point[0], grid_max_point[1], grid_max_point[2]);
    auto grid_resolution = config["grid_resolution"].get<int>();
    grid.size = make_int3(grid_resolution, grid_resolution, grid_resolution);
    grid.cell_length = ((grid.max_pos - grid.min_pos) / make_float3(grid.size)).x;
    construct_neighbor_list();
    printf("Grid size: %d %d %d\n", grid.size.x, grid.size.y, grid.size.z);
}

void SceneHost::save_boundary_points(const std::string filename) const
{
    std::vector<BoundaryPoint> boundary_points_host;
    boundary_points_host.resize(boundary_points_device.size());
    std::ofstream ofs(filename);
    boundary_points_device.copy_to_host(boundary_points_host);
    for (auto &bp : boundary_points_host)
    {
        ofs << bp.pos.x << " " << bp.pos.y << " " << bp.pos.z << "\n";
    }
}

void SceneHost::sample_boundary_points()
{
    size_t boundary_point_num = boundary_points_device.size();
    GPUMemory<unsigned long long> seeds(boundary_point_num);
    seeds.copy_from_host(get_random_seeds(boundary_point_num));
    parallel_for(boundary_point_num, [scene = device(), seeds = seeds.device_ptr()] __device__(int i) {
        auto seed = seeds[i];
        randomState rand_state;
        curand_init(seed, 0, 0, &rand_state);
        scene.boundary_points[i] = scene.sample_boundary_point(&rand_state);
    });
}

void SceneHost::construct_neighbor_list()
{
    GPUMemory<NeighborList> self_list(grid.get_cell_num());
    self_list.memset(0);
    parallel_for(boundary_points_device.size(), [grid = grid, bps = boundary_points_device.device_ptr(),
                                                 self_list = self_list.device_ptr()] __device__(int i) {
        auto bp = bps[i];
        auto grid_index = grid.get_flat_index(bp.pos);
        self_list[grid_index].atomic_append(i);
    });
    GPUMemory<int> non_empty(grid.get_cell_num());
    parallel_for(grid.get_cell_num(),
                 [self_list = self_list.device_ptr(), non_empty = non_empty.device_ptr()] __device__(int i) {
                     if (self_list[i].size() > 0)
                         non_empty[i] = i + 1;
                     else
                         non_empty[i] = 0;
                 });
    auto last_iter = thrust::remove(thrust::device, non_empty.begin(), non_empty.end(), 0);
    int non_empty_cell_num = last_iter - non_empty.begin();
    neighbor_list.resize(grid.get_cell_num());
    neighbor_list.memset(0);
    parallel_for(non_empty_cell_num, [non_empty = non_empty.device_ptr(), self_list = self_list.device_ptr(),
                                      neighbor_list = neighbor_list.device_ptr(), grid = grid] __device__(int i) {
        int flat_idx = non_empty[i] - 1;
        int3 cell_idx = grid.get_cell_index(flat_idx);
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    int3 neighbor_cell_idx = cell_idx + make_int3(x, y, z);
                    auto &src_lst = self_list[grid.get_flat_index(neighbor_cell_idx)];
                    for (int j = 0; j < src_lst.size(); j++)
                    {
                        neighbor_list[flat_idx].append(src_lst[j]);
                    }
                }
            }
        }
    });
}
NWOB_NAMESPACE_END
