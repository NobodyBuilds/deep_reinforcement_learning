#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "norender.h"

#include <cuda_runtime.h>
#include <iostream>
#include "data.h"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include "draw.h"


int threads = 256;
int blocks(int n) {
    return (n + threads - 1) / threads;
}


static cudaGraphicsResource *carInstRes;
void registervbo() {
    unsigned int id = vbo_id.quad_instanced_vbo(settings.cars);
    if (id == 0) {
        printf("vbo id is unintitalized \n");
        return;
    }
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&carInstRes,id , cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("vbo registration error %s\n ",cudaGetErrorString(err));
    }
    else {
        printf("vbo registered\n");
    }
}


__global__ void registercars(int n, float4* data1,ray* rays, float sx, float sy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)return;

    float4& d = data1[i];
    d.x = sx;
    d.y = sy;
    d.z = 0.0f;
    d.w = 0.0f;

    for (int k = 0; k < 6; k++) {
        rays[i].len[k] = 50.0f;
    }

}

 __constant__ float d_ray_angles[6];
void initcars(int n) {
    cudaMemcpyToSymbol(d_ray_angles, ray_angles, sizeof(ray_angles));
    registercars << <blocks(n), threads >> > (n, data1,rays, settings.spawnx, settings.spawny);
    
    printf("cars registered \n");

}




__global__ void fillcardata(int n,quadvertex2d* carvert,float4* data1,float w,float h,float r,float g,float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)return;
    quadvertex2d &c = carvert[i];
    float4& d = data1[i];
    c.x = d.x;
    c.y = d.y;
    c.width = w;
    c.height = h;
    c.r = r;
    c.g = g;
    c.b = b;
    float rot = d.z * (180.0f / CUDART_PI_F);
    if (rot < 0.0f) {
        rot += 360.0f;
    }
    if (rot > 360.0f) {
        rot = 0.0f;
    }
    c.rotation = rot;
  
}

void loadcardata() {
		
   cudaError_t e=  cudaGraphicsMapResources(1, &carInstRes, 0);
   if (e != cudaSuccess) {
       printf("car mapping failed %s \n", cudaGetErrorString(e));
   }
    quadvertex2d* devPtr=nullptr;
    size_t bytes=0;
   cudaError_t er= cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, carInstRes);
    if (er != cudaSuccess) {
        printf("car pointer mapping failed %s \n", cudaGetErrorString(er));
    }
    fillcardata << <blocks(settings.cars), threads >> > (settings.cars,devPtr, data1, settings.carwidth, settings.carheight, settings.car_r, settings.car_g, settings.car_b);
    cudaDeviceSynchronize();

  cudaError_t err=  cudaGraphicsUnmapResources(1, &carInstRes, 0);
    if (err != cudaSuccess) {
        printf("car unmapping failed %s \n", cudaGetErrorString(err));
    }
}

extern "C" void drawcars() {
    loadcardata();
    render2d.drawQuadinstancedbyinterop(settings.cars);
}






static cudaGraphicsResource* raysres;
extern "C" void registerrayvbo() {
    unsigned int id = vbo_id.line_instanced_vbo(settings.rays);
    if (id == 0) {
        printf("ray vbo id is unintitalized \n");
        return;
    }
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&raysres, id, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("ray vbo registration error %s\n ", cudaGetErrorString(err));
    }
    else {
        printf("ray vbo registered\n");
    }
}



__global__ void fillraydata(int n, linepoint2d* rayvert, ray* rays,float4* data1,int caridx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = caridx;
    if (i >= n)return;
    rayvert[i].ox = data1[c].x;
    rayvert[i].oy = data1[c].y;

    float angle = data1[c].z + d_ray_angles[i];
    float dx = cosf(angle);
    float dy = sinf(angle);

    rayvert[i].dx = data1[c].x + dx * rays[c].len[i];
    rayvert[i].dy = data1[c].y + dy * rays[c].len[i];
    rayvert[i].r = 1.0f;
    rayvert[i].g = 1.0f;
    rayvert[i].b = 1.0f;
}

void loadraydata() {

    cudaError_t e= cudaGraphicsMapResources(1, &raysres, 0);
    if (e != cudaSuccess) {
        printf("ray mapping failed %s \n", cudaGetErrorString(e));
    }
    linepoint2d* devPtr = nullptr;
    size_t bytes = 0;
   cudaError_t er= cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, raysres);
    if (er != cudaSuccess) {
        printf("ray pointer mapping failed %s \n", cudaGetErrorString(er));
    }
    fillraydata << <blocks(settings.rays), threads >> > (6,devPtr,rays,data1,1);
    cudaDeviceSynchronize();

   cudaError_t err= cudaGraphicsUnmapResources(1, &raysres, 0);
    if (err != cudaSuccess) {
        printf("ray unmapping failed %s \n", cudaGetErrorString(err));
    }
}

extern "C" void drawrays() {
    loadraydata();
    render2d.drawlineinstancedbyinterop(6);
    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("draw rays error %s \n", cudaGetErrorString(err));
    }
}