#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include <stdio.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "common.h"
#include "ui.h"
#include "gpu_memory.h"

NWOB_NAMESPACE_BEGIN
static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void MemoryVisualizer::visualize(GPUMatrix<uchar4> *data)
{
    gpu_matrix = data;
    auto width = data->width();
    auto height = data->height();
    std::cout << "width: " << width << " height: " << height << std::endl;
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return;
    std::cout << "GLFW initialized" << std::endl;
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    GLFWwindow *window = glfwCreateWindow(1280, 720, "Dear ImGui Visualizer", NULL, NULL);
    if (window == NULL)
        return;

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);  // Enable vsync
    if (glewInit() != GLEW_OK)
        exit(EXIT_FAILURE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

    // Setup Dear ImGui style
    // ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    io.IniFilename = IMGUI_CONFIG_FILE;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &image);
    glBindTexture(GL_TEXTURE_2D, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    CUDA_CHECK_THROW(
        cudaGraphicsGLRegisterImage(&CudaResource, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &CudaResource, 0));
    CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&array, CudaResource, 0, 0));

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // ImGui::Text("FPS: %.4f", ImGui::GetIO().Framerate);
        ImGui::Begin("Memory Visualizer");
        CUDA_CHECK_THROW(cudaMemcpy2DToArray(array, 0, 0, gpu_matrix->data(), width * sizeof(uchar4),
                                             height * sizeof(uchar4), height, cudaMemcpyDeviceToDevice));
        ImVec2 wsize = ImGui::GetWindowContentRegionMax() - ImGui::GetWindowContentRegionMin();
        ImVec2 img_size = ImVec2(wsize.x, wsize.y - ImGui::GetFrameHeightWithSpacing() * 2);
        if (img_size.x < img_size.y)
        {
            img_size.y = img_size.x * height / width;
        }
        else
        {
            img_size.x = img_size.y * width / height;
        }
        ImGui::SetCursorPos((wsize - img_size) * 0.5f);
        ImGui::Image((ImTextureID)(uintptr_t)image, img_size, ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w,
                     clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        // glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

NWOB_NAMESPACE_END
