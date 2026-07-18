#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <math_constants.h>
#include "norender.h"
#include "data.h"
#include <cuda_gl_interop.h>
#include "obstacles.h"
#include<iostream>
#include <vector>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>


cudaGraphicsResource* obstaclesRes;

extern "C" void registerObstaclesVbo() {

    

    unsigned int id = vbo_id.quad_instanced_vbo(settings.max_obstacles,2);
    if (id == 0) {
        printf("vbo id for obstacles is unintitalized \n");
        return;
    }
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&obstaclesRes, id, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("obstacle vbo registration error %s\n ", cudaGetErrorString(err));
    }
    else {
        printf("obstacle vbo registered\n");
    }

}
__global__ void filldata(int n ,quadvertex2d* obsvert,quadvertex2d* obsdata) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    quadvertex2d& o=obsvert[i];
    quadvertex2d& d=obsdata[i];

    o.x = d.x;
    o.y = d.y;
    o.width = d.width;
    o.height = d.height;
    o.rotation = d.rotation;
    o.r = 0.2f;
    o.g = 0.2f;
    o.b = 0.2f;

}

void fillobsdata() {

    cudaError_t e = cudaGraphicsMapResources(1, &obstaclesRes, 0);
    if (e != cudaSuccess) {
        printf("obstacles mapping failed %s \n", cudaGetErrorString(e));
    }
    quadvertex2d* devPtr = nullptr;
    size_t bytes = 0;
    cudaError_t er = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, obstaclesRes);
    if (er != cudaSuccess) {
        printf("obstacles pointer mapping failed %s \n", cudaGetErrorString(er));
    }
    filldata << <blocks(settings.obstacles), threads >> > (settings.obstacles, devPtr, obstacle);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGraphicsUnmapResources(1, &obstaclesRes, 0);
    if (err != cudaSuccess) {
        printf("obstacles unmapping failed %s \n", cudaGetErrorString(err));
    }
}

extern "C" void drawobstacles() {
    if (settings.obstacles != 0) {
        fillobsdata();
        render2d.drawQuadinstancedbyinterop(settings.obstacles,2);
    }
}


extern "C" void initobstacles() {
    cudaMalloc(&obstacle, settings.max_obstacles * sizeof(quadvertex2d));


}
std::vector<quadvertex2d> h_obdata;

extern "C" void copyobsdata() {
    cudaMemcpy(obstacle, h_obdata.data(), settings.obstacles * sizeof(quadvertex2d), cudaMemcpyHostToDevice);

}

extern "C" void addobstacle(float x, float y ,float rot,float width,float height) {
    quadvertex2d c;

    c.x = x;
    c.y = y;
    c.height = height;
    c.width = width;
    c.rotation = rot;
    c.r = 0.2f;
    c.g = 0.2f;
    c.b = 0.2f;

    h_obdata.push_back(c);
    settings.obstacles++;
    copyobsdata();
}

extern "C" void drawdummy() {
    if (settings.addingobstacle) {
        render2d.drawquad(settings.obx, settings.oby, 0.2f, 0.2f, 0.2f, settings.obwidth, settings.obheight, settings.obrot);
    }
}

extern "C" void resetob() {
    settings.obx = 900.0f,
    settings.oby = 450.0f;
    settings.obrot = 0.0f;
    settings.obwidth = 50.0f;
    settings.obheight = 50.0f;
}