#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "norender.h"

#include <cuda_runtime.h>
#include <iostream>
#include "data.h"
#include <cuda.h>
#include <cmath>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include <math.h>
#include "draw.h"
#include "obstacles.h"
#include "ui.h"
#include "net.h"


struct drawd {
    float maxdisttotarget;
    float maxsteer;
    float maxspeed;
    float wheelbase;
    float tx;
    float ty;
    float raymaxdist;
    float accelrate;
    float brakerate;
    float steerrate;
    float drag;

};
__constant__ drawd cd;
drawd cdh;
void sync() {
    drawd& h = cdh;
    h.accelrate = dparam.accel_rate;
    h.brakerate = dparam.brake_rate;
    h.maxdisttotarget = settings.maxdisttotarget;
    h.maxspeed = dparam.max_speed;
    h.maxsteer = dparam.max_steer;
    h.raymaxdist = dparam.max_raylen;
    h.tx = settings.targetx;
    h.ty = settings.targety;
    h.wheelbase = dparam.wheelbase;
    h.steerrate = dparam.steer_rate;
    h.drag = dparam.drag;
    cudaMemcpyToSymbol(cd, &cdh, sizeof(drawd));
}

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
__global__ void initTrackingKernel(
    int n,
    const float4* __restrict__ data1,
    float4* prevData,
    float* pathLength,
    float* headingChange
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    prevData[i] = data1[i];
    pathLength[i] = 0.0f;
    headingChange[i] = 0.0f;
}

__global__ void registercars(int n, float4* data1,ray* rays,float4* data2,float4* carcontrol,int* alive, float sx, float sy,float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)return;

    float4& d = data1[i];
    float4& d2 = data2[i];

    d.x = sx;
    d.y = sy;
    d.z = 0.0f;
    d.w = 0.0f;
    d2.x = cd.maxdisttotarget;
   
    d2.y = 0.0f;
    d2.z = 0.0f;
    d2.w = 0.0f;


    alive[i] = 1;

    for (int k = 0; k < 16; k++) {
        rays[i].len[k] = dx;
    }
    carcontrol[i] = make_float4(0.f, 0.f, 0.f, 0.f);

}

 __constant__ float d_ray_angles[16];
 
void initcars() {
    int n = settings.cars;
    cudaMemcpyToSymbol(d_ray_angles, ray_angles, sizeof(ray_angles));
    registercars << <blocks(n), threads >> > (n, data1,rays,data2,carcontrol,alive, settings.spawnx, settings.spawny,settings.ray_max_len);
    
    printf("car registered \n");
  

}

void restartgeneration() {
    int n = settings.cars;


    registercars << <blocks(n), threads >> > (n, data1, rays, data2,carcontrol, alive,  settings.spawnx, settings.spawny, settings.ray_max_len);
    

}



__global__ void fillcardata(int n,quadvertex2d* carvert,float4* data1,int* alive,float w,float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)return;
    quadvertex2d &c = carvert[i];
    float4& d = data1[i];
    c.x = d.x;
    c.y = d.y;
    c.width = w;
    c.height = h;
    int a = alive[i];
    if (a == 2) {
        c.r = 0.0f; c.g = 1.0f; c.b = 0.0f;   
    }
    else if (a == 0) {
        c.r = 1.0f; c.g = 0.0f; c.b = 0.0f;  
    }
    else {
        c.r = 0.0f; c.g = 0.0f; c.b = 1.0f; 
    }
    float rot = d.z * (180.0f / CUDART_PI_F);
    if (rot < 0.0f) {
        rot += 360.0f;
    }
 
    c.rotation = rot;
  
}

void loadcardata() {
    static bool first = true;
   cudaError_t e=  cudaGraphicsMapResources(1, &carInstRes, 0);
   if (e != cudaSuccess && first) {
       printf("car mapping failed %s \n", cudaGetErrorString(e));
   }
    quadvertex2d* devPtr=nullptr;
    size_t bytes=0;
   cudaError_t er= cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, carInstRes);
    if (er != cudaSuccess&&first) {
        printf("car pointer mapping failed %s \n", cudaGetErrorString(er));
    }
    fillcardata << <blocks(settings.cars), threads >> > (settings.cars,devPtr, data1,alive, settings.carwidth, settings.carheight);

  cudaError_t err=  cudaGraphicsUnmapResources(1, &carInstRes, 0);
    if (err != cudaSuccess&&first) {
        printf("car unmapping failed %s \n", cudaGetErrorString(err));
    }
    first = false;
}

extern "C" void drawcars() {
    loadcardata();
    render2d.drawQuadinstancedbyinterop(settings.cars,1);
}






static cudaGraphicsResource* raysres;
static cudaGraphicsResource* raysdot;

extern "C" void registerraydotvbo() {
    unsigned int id = vbo_id.circle_instanced_vbo(settings.rays, 1);
    if (id == 0) {
        printf("raydot vbo id is unintitalized \n");
        return;
    }
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&raysdot, id, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("raydot vbo registration error %s\n ", cudaGetErrorString(err));
    }
    else {
        printf("raydot vbo registered\n");
    }
}

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
    registerraydotvbo();
}



__global__ void fillraydata(int n, linepoint2d* rayvert,circlevertex2d* dot,const  ray* __restrict__ rays, const float4* __restrict__ data1,int cid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = cid;
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
    float size = 5.0f;
    dot[i].x = data1[c].x + dx * rays[c].len[i];
    dot[i].y = data1[c].y + dy * rays[c].len[i];
    dot[i].size = size;
    dot[i].r = 1.0f;
    dot[i].g = 1.0f;
    dot[i].b = 1.0f;

   
}





   


void loadraydata() {
    static bool first = true;
    cudaError_t e= cudaGraphicsMapResources(1, &raysres, 0);
    if (e != cudaSuccess&&first) {
        printf("ray mapping failed %s \n", cudaGetErrorString(e));
    }
    cudaError_t te= cudaGraphicsMapResources(1, &raysdot, 0);
    if (te != cudaSuccess && first) {
        printf("raydot mapping failed %s \n", cudaGetErrorString(te));
    }

    linepoint2d* devPtr = nullptr;
    circlevertex2d* dotdevPtr = nullptr;
    size_t bytes = 0;
    size_t dotbytes = 0;

   cudaError_t er= cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, raysres);
    if (er != cudaSuccess && first) {
        printf("ray pointer mapping failed %s \n", cudaGetErrorString(er));
    }
   cudaError_t ter= cudaGraphicsResourceGetMappedPointer((void**)&dotdevPtr, &dotbytes, raysdot);
    if (ter != cudaSuccess && first) {
        printf("raydot pointer mapping failed %s \n", cudaGetErrorString(ter));


    }

  

    fillraydata << <blocks(settings.rays), threads >> > (16,devPtr,dotdevPtr,rays,data1,settings.bestcar);

   cudaError_t err= cudaGraphicsUnmapResources(1, &raysres, 0);
    if (err != cudaSuccess && first) {
        printf("ray unmapping failed %s \n", cudaGetErrorString(err));
    }
   cudaError_t terr= cudaGraphicsUnmapResources(1, &raysdot, 0);
    if (terr != cudaSuccess && first) {
        printf("raydot unmapping failed %s \n", cudaGetErrorString(terr));
    }

    first = false;
}

extern "C" void drawrays() {
    loadraydata();
    render2d.drawlineinstancedbyinterop(16,1);
    render2d.drawcircleinstancedbyinterop(16, 1);
    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("draw rays error %s \n", cudaGetErrorString(err));
    }
}


__global__ void moverkernel(int n,float dt, float4* car,float4* data2,float4* carcontrol,int* alive,ray* rays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;

    if (alive[i] == 1) {
        float4 d2 = __ldg(&data2[i]);
        float4 d1 = __ldg(&car[i]);
        float4 cc = __ldg(&carcontrol[i]);

        float sterrdir = cc.w - cc.z;
        float target_steer = sterrdir *(CUDART_PI_F/180.0f)* cd.maxsteer;
        float maxDelta_steer = cd.steerrate * (CUDART_PI_F / 180.0f) * dt;

        float delta = fminf(fmaxf(target_steer - d2.w, -maxDelta_steer), maxDelta_steer);
        data2[i].w += delta;

        float speed = d1.w;
        float move = cc.x - cc.y;
        float rate = (move>0.0f)? cd.accelrate:cd.brakerate ;


        speed +=move * rate * dt;
        speed -= copysignf((cd.drag * speed * speed * dt), speed);

        speed = fminf(fmaxf(speed, -cd.maxspeed * 0.5f), cd.maxspeed);

        float aval = (speed / cd.wheelbase) * tanf(data2[i].w);//use data2[i].w if any issue
        float heading = d1.z + aval * dt;

        car[i].x += speed * cosf(heading) * dt;
        car[i].y += speed * sinf(heading) * dt;

        car[i].z = heading;
        car[i].w = speed;
    }
   
    

}


extern "C" void stepcars() {
    
    moverkernel << <blocks(settings.cars), threads >> > (settings.cars,settings.dt, data1,data2, carcontrol,alive,rays);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("mover kernel recieved corrupted data ->%s \n", cudaGetErrorString(err));
    }
    checkcolison();
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("check colision  recieved corrupted data ->%s \n", cudaGetErrorString(err));
    }
    checkstate();
   
  
   
}

extern "C" void draw() {

    
        drawobstacles();

       if(settings.render_rays) drawrays();
        drawcars();
        drawdummy();
        ui_draw();
    
}

void unregister() {
    if (carInstRes) {
        cudaGraphicsUnregisterResource(carInstRes);
        carInstRes = nullptr;
    }
    if (raysres) {
        cudaGraphicsUnregisterResource(raysres);
        raysres = nullptr;
    }

    if (raysdot) {
        cudaGraphicsUnregisterResource(raysdot);
        raysdot = nullptr;
    }
}