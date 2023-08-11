#include <iostream>
#include "common.h"
#include "gpu_memory.h"
#include "ui.h"
using namespace nwob;

int main()
{
    GPUMatrix<uchar4> image(512, 512);
    image.memory.memset(125);
    parallel_for(image.size(), [out = image.device_ptr()] __device__(int i) {
        int y = i / out.stride();
        int x = i % out.stride();
        out[y][x] = make_uchar4(y / 2, 0, 0, 255);
    });
    MemoryVisualizer().visualize(&image);
}