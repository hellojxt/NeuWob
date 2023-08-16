#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "nlohmann/json_fwd.hpp"
#include "ui.h"
#include <nlohmann/json.hpp>
#include "estimator.h"
#include "scene.h"
using namespace nwob;

int main(int argc, char *argv[])
{
    const std::string input_json_file = argv[1];
    SceneHost scene(input_json_file);
}