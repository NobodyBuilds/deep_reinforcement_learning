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
#include <math.h>
#include "draw.h"
#include "obstacles.h"
#include "ui.h"



static cudaGraphicsResource *carInstRes;
void registervbo() {
    unsigned int id = vbo_id.quad_instanced_vbo(settings.cars,1);
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


__global__ void registercars(int n, float4* data1,ray* rays, float sx, float sy,float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)return;

    float4& d = data1[i];
    d.x = sx;
    d.y = sy;
    d.z = 0.0f;
    d.w = 0.0f;

    for (int k = 0; k < 6; k++) {
        rays[i].len[k] = dx;
    }

}

 __constant__ float d_ray_angles[6];
 
void initcars() {
    int n = settings.cars;
    cudaMemcpyToSymbol(d_ray_angles, ray_angles, sizeof(ray_angles));
    registercars << <blocks(n), threads >> > (n, data1,rays, settings.spawnx, settings.spawny,settings.ray_max_len);
    
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
    render2d.drawQuadinstancedbyinterop(settings.cars,1);
}






static cudaGraphicsResource* raysres;
extern "C" void registerrayvbo() {
    unsigned int id = vbo_id.line_instanced_vbo(settings.rays,1);
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
    render2d.drawlineinstancedbyinterop(6,1);
    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("draw rays error %s \n", cudaGetErrorString(err));
    }
}


__global__ void moverkernel(int n,float dt, float4* car,float4* data2, float max_steer,float steer_rate,float accelrate,float brakerate,float drag,float maxspeed,float wheelbase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;
    float4 d2 = __ldg(&data2[i]);
    float4 d1 = __ldg(&car[i]);

        float target_steer = d2.y* max_steer;
    float maxDelta_steer = steer_rate * dt;
   
    float delta = fminf(fmaxf(target_steer - d2.w, -maxDelta_steer),maxDelta_steer);
    data2[i].w = delta;

    float speed = d1.w;
    float rate = d2.x > 0.0f ? accelrate : brakerate;

    speed += d2.x * rate * dt;
    speed -= copysignf((drag * speed * speed * dt),speed);

    speed = fminf(fmaxf(speed, -maxspeed * 0.5f), maxspeed);

    float aval = (speed / wheelbase) * tanf(d2.w+delta);//use data2[i].w if any issue
    float heading = d1.z + aval * dt;

    car[i].x += speed * cosf(heading) * dt;
    car[i].y += speed * sinf(heading) * dt;

    car[i].z = heading;
    car[i].w = speed;
}

__global__ void test(int n, float4* data, float dt, float t, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;

    data[i].x = t;
    data[i].y = s;
}


extern "C" void stepcars() {
    test << <blocks(settings.cars), threads >> > (settings.cars, data2, dt, settings.dumx, settings.dumy);
    moverkernel << <blocks(settings.cars), threads >> > (settings.cars,dt, data1, data2,dparam.max_steer,dparam.steer_rate,dparam.accel_rate,dparam.brake_rate,dparam.drag,dparam.max_speed,dparam.wheelbase);
}

extern "C" void draw() {

    
        drawrays();
        drawcars();
        drawobstacles();
        drawdummy();
        ui_draw();
    
}