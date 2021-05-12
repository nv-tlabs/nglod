/******************************************************************************
 * The MIT License (MIT)

 * Copyright (c) 2021, NVIDIA CORPORATION.

 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#include <torch/torch.h>

#include <iostream> 

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <vector_types.h>
#include <spc/SPC.h>

#include "RenderCamera.h"
#include "SDF.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream> 
#include <vector>

#include <solr/util_timer.cuh>

//#define VERBOSE

#ifdef VERBOSE
#   define PROBE solr::PerfTimer probe_timer = solr::PerfTimer()
#   define PROBE_CHECK(x) probe_timer.check(x) 
#else
#   define PROBE 
#   define PROBE_CHECK(x)
#endif

inline void cudaPrintError(const char* file, const int line)
{
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("%s (%d) - %d: %s\n", file, line, status, cudaGetErrorString(status));
        int a = 0;
    }
}

#define CUDA_PRINT_ERROR() cudaPrintError(__FILE__, __LINE__)

using namespace std;

#define REFRESH_DELAY     0 //ms

unsigned int g_Width = 1280;
unsigned int g_Height = 720;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;

const char *sStartMessage = "NLOD Demo";

RenderCamera            g_Camera;

float                   g_fAspectRatio = ((float)g_Width)/((float)g_Height);
float                   g_FOV;

at::Tensor              g_tEye;
at::Tensor              g_tAt;
at::Tensor              g_tUp;
at::Tensor              g_tWorld;

SDF                     g_SDF; // The Signed Distance Function

SPC                     g_SPC; // The Structured Point Cloud 

extern uint             g_Renderer;
uint                    g_NumRenderers = 3;
uint                    g_Level;
extern uint             g_TargetLevel;

extern GLuint                       pbo;         // OpenGL pixel buffer object
extern struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
extern uint*                        d_output;
extern float                        g_milliseconds;


extern "C" uint RenderImage(uint *d_output, uint imageW, uint imageH, torch::Tensor Org, torch::Tensor Dir, torch::Tensor Nuggets, SPC* spc, SDF* sdf);

extern std::vector<at::Tensor>  spc_generate_primary_rays(
    torch::Tensor Eye, torch::Tensor At, torch::Tensor Up,
    uint imageW, uint imageH, float fov, torch::Tensor World);

extern torch::Tensor spc_raytrace(
    torch::Tensor octree,
    torch::Tensor points,
    torch::Tensor pyramid, 
    torch::Tensor Org,
    torch::Tensor Dir,
    uint targetLevel
    );

// callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void wheel(int wheel, int direction, int x, int y);
void timerEvent(int value);
void cleanup();
void reshapeWindow(int w, int h);


extern void initGLBuffers(uint windowWidth, uint windowHeight)
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, windowWidth*windowHeight * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
extern bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(g_Width, g_Height);
    glutCreateWindow("Neural SDF Renderer");
    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshapeWindow);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutCloseFunc(cleanup);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    reshapeWindow(g_Width, g_Height);

    SDK_CHECK_ERROR_GL();

    return true;
}


void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "SPC Demo: %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////

void display()
{
    sdkStartTimer(&timer);
    PROBE;

    uint num_rays = 0;

    float4x4 mWorld = g_Camera.GetWorldMatrix();

    *(reinterpret_cast<float4x4*>(g_tWorld.data_ptr<float>())) = mWorld;
    auto rays = spc_generate_primary_rays(g_tEye, g_tAt, g_tUp, g_Width, g_Height, g_FOV, g_tWorld);
    auto nuggets = spc_raytrace(g_SPC.GetOctree(), g_SPC.GetPoints(), g_SPC.GetPyramid(), rays[0], rays[1], g_TargetLevel);
    num_rays = RenderImage(d_output, g_Width, g_Height, rays[0], rays[1], nuggets, &g_SPC, &g_SDF);

    glRasterPos2i(-1, -1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(g_Width, g_Height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();

    PROBE_CHECK("frame done");
    sdkStopTimer(&timer);
    computeFPS();

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    cout << key << endl;
    switch (key)
    {
    case (27):
        glutDestroyWindow(glutGetWindow());
        return;
    case '=':
        g_Renderer = (g_Renderer + 1) % g_NumRenderers;
        return;
    case '-':
        g_Renderer = (g_NumRenderers + g_Renderer - 1) % g_NumRenderers;
        return;
    case '+':
        if (g_TargetLevel < g_SPC.GetLevel())
            g_TargetLevel++;
        return;
    case '_':
        if (g_TargetLevel > 2)
            g_TargetLevel--;
        return;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    short dir = 10;

    if (button == 0)
    {
        if (state == GLUT_DOWN)
        {
            g_Camera.OnMouseDown(x, y);
        }
        else if (state == GLUT_UP)
        {
             g_Camera.OnMouseUp();
        }
    }
    else if (button == 3)
    {
        g_Camera.OnMouseWheel(dir);
        g_FOV = g_Camera.GetFOV();    
    }
    else if (button == 4)
    {
        g_Camera.OnMouseWheel(-dir);
        g_FOV = g_Camera.GetFOV();    
    }
}


void motion(int x, int y)
{
    g_Camera.OnMouseMove(x, y);
}


void cleanup()
{
    sdkDeleteTimer(&timer);

}


void reshapeWindow(int w, int h)
{
    g_Width = w;
    g_Height = h;

    glViewport(0, 0, g_Width, g_Height);

    g_Camera.OnResize(w, h);

    g_fAspectRatio = (float)g_Width / (float)g_Height;

    float4x4 I = make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);

    g_Camera.SetWorldMatrix(I);

    float3 eye = make_float3(2.5f, 0.0f, 2.5f);
    float3 at = make_float3(0.0f, 0.0f, 0.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    g_FOV = 30.0f;

    g_Camera.SetViewMatrix(eye, at, up);
    g_Camera.SetProjMatrix(g_FOV, g_fAspectRatio, 0.1f, 10.0f);


    g_tEye = at::zeros({3}, at::device(torch::kCPU).dtype(torch::kFloat));
    g_tAt = at::zeros({3}, at::device(torch::kCPU).dtype(torch::kFloat));
    g_tUp = at::zeros({3}, at::device(torch::kCPU).dtype(torch::kFloat));
    g_tWorld = at::zeros({4,4}, at::device(torch::kCPU).dtype(torch::kFloat));

    float3* d_eye = reinterpret_cast<float3*>(g_tEye.data_ptr<float>());
    float3* d_at = reinterpret_cast<float3*>(g_tAt.data_ptr<float>());
    float3* d_up = reinterpret_cast<float3*>(g_tUp.data_ptr<float>());

    *d_eye = eye;
    *d_at = at;
    *d_up = up;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    torch::NoGradGuard no_grad;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif
    printf("%s starting...\n", sStartMessage);

    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    if (false == initGL(&argc, argv))
    {
        return -1;
    }

    std::string npz_path = argc > 1 ? argv[1] : "../data/model.npz";

    g_SPC.LoadNPZ(npz_path);
    g_TargetLevel = g_SPC.GetLevel();

    //// SDF initialization
    g_SDF.loadWeights(npz_path);
    CUDA_PRINT_ERROR();
    g_SDF.initTrinkets(g_SPC.GetPSize(), g_SPC.GetLevel(), g_SPC.GetPyramidPtr(), g_SPC.GetProotGPU());
    CUDA_PRINT_ERROR();

    //// GL initialization
    initGLBuffers(g_Width, g_Height);
    CUDA_PRINT_ERROR();
 
    //// start rendering mainloop
    glutMainLoop();

    return 0;
}

