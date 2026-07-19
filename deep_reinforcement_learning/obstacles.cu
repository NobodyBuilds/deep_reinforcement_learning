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
__constant__ float d2_ray_angles[6];
//render obstacles
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
    cudaMemcpyToSymbol(d2_ray_angles, ray_angles, sizeof(ray_angles));
    cudaMalloc(&obstacle, settings.max_obstacles * sizeof(quadvertex2d));


}

//add obstacles
std::vector<quadvertex2d> h_obdata;

extern "C" void copyobsdata() {
    cudaMemcpy(obstacle, h_obdata.data(), settings.obstacles * sizeof(quadvertex2d), cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("obstacle memcpy error %s\n", cudaGetErrorString(err));
    }


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
    bake_segments();
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

//bake wall segemnts
__device__ float2 wall_locals(int k,float w,float h) {

    if (k == 0) return float2{ -w,-h };
    if (k == 1) return float2{ -w, h };
    if (k == 2) return float2{  w, h };
    if (k == 3) return float2{  w, -h };
    else {
        
        return float2{ w,h };
    }
}

__global__ void setsegments(int n, float4* segments, const quadvertex2d* __restrict__ obstacles) {
    float x = (CUDART_PI_F / 180.0f);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;
   
    quadvertex2d o = obstacles[i];

    float rot = o.rotation * x;
    float c = cosf(rot);
    float s = sinf(rot);

    float w = o.width*0.5f ;
    float h = o.height*0.5f ;

    float2 data[4];

    float2 local[4] = {
        make_float2(-w,-h),
        make_float2(w,-h),
        make_float2(w,h),
        make_float2(-w,h)
    };

    for (int k = 0; k < 4; k++) {
       

        data[k].x = o.x + (local[k].x * c - local[k].y * s);
        data[k].y = o.y + (local[k].x * s + local[k].y * c);

    }
    segments[i * 4 + 0] = make_float4(data[0].x,data[0].y, data[1].x ,data[1].y);
    segments[i * 4 + 1] = make_float4(data[1].x,data[1].y, data[2].x ,data[2].y);
    segments[i * 4 + 2] = make_float4(data[2].x,data[2].y, data[3].x ,data[3].y);
    segments[i * 4 + 3] = make_float4(data[3].x,data[3].y, data[0].x ,data[0].y);

}

extern "C" void bake_segments() {

    setsegments << <blocks(settings.obstacles), threads >> > (settings.obstacles, segments, obstacle);
    cudaError_t err = cudaGetLastError();
    if (err) {
        printf("bake segment error %s\n", cudaGetErrorString(err));
    }
    else {
        printf("obstacle segments baked \n");
    }
}

//ray colison detection
__device__ float rayvssegment(float4 pos, float2 a, float2 b, float maxd) {
    float2 e = { b.x - a.x ,b.y - a.y };
    float2 c = { a.x - pos.x,a.y - pos.y };

    float demon = pos.z * e.y - pos.w * e.x;

    if (fabsf(demon) < 1e-6f) return -1.0f;

    float t = (c.x * e.y - c.y * e.x) / demon;
    float s = (pos.w * c.x - pos.z * c.y) / demon;
    if (t >= 0.0f && t <= maxd && s >= 0.0f && s <= 1.0f) {
        return t;
    }
    return -1.0f;

}

__device__ float castray(int n, float4 pos, float maxd, const float4* __restrict__ segments) {
    float closeT = maxd;
    for (int i = 0; i < n * 4; i++) {
        float2 a = make_float2( segments[i].x,segments[i].y );
        float2 b = make_float2( segments[i].z,segments[i].w );

        float t = rayvssegment(pos, a, b, maxd);

        if (t >= 0.0f &&  t < closeT) {
            closeT = t;
        }
    }
    return closeT;
}

__device__ void writeraylen(int i,int r,float dist ,ray* rays) {
    rays[i].len[r] = dist;
}

__global__ void rayhit(int n,ray* rays, const float4* __restrict__ data1, const float4* __restrict__ segment, int obcount,float maxdist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;

    float4 car = __ldg(&data1[i]);

    for (int r = 0; r < 6; r++) {
        float angle = car.z + d2_ray_angles[r];
        float4 d = {car.x,car.y, cosf(angle),sinf(angle) };
        float dist = castray(obcount, d, maxdist, segment);

        writeraylen(i, r, dist, rays);
    }
}
extern "C" void checkcolison() {
    rayhit << <blocks(settings.cars), threads >> > (settings.cars, rays, data1, segments, settings.obstacles, settings.ray_max_len);
}


