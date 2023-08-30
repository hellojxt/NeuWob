#include "scene.h"
#include "nlohmann/json_fwd.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <vector>

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

    elements.clear();
    for (int i = 0; i < triangles.size(); i++)
    {
        Element e;
        e.v0 = vertices[triangles[i].x];
        e.v1 = vertices[triangles[i].y];
        e.v2 = vertices[triangles[i].z];
        e.n = normalize(cross(e.v1 - e.v0, e.v2 - e.v0));
        e.indices = make_int3(triangles[i].x, triangles[i].y, triangles[i].z);
        elements.push_back(e);
    }
    if (cut_idx >= 0)
    {
        elements.erase(elements.begin() + cut_idx, elements.end());
    }
}

NWOB_NAMESPACE_END
