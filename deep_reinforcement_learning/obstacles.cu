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
#include <random>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <math.h>
__constant__ float d2_ray_angles[16];
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
    static bool first = true;
    cudaError_t e = cudaGraphicsMapResources(1, &obstaclesRes, 0);
    if (e != cudaSuccess && first) {
        printf("obstacles mapping failed %s \n", cudaGetErrorString(e));
    }
    quadvertex2d* devPtr = nullptr;
    size_t bytes = 0;
    cudaError_t er = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &bytes, obstaclesRes);
    if (er != cudaSuccess && first) {
        printf("obstacles pointer mapping failed %s \n", cudaGetErrorString(er));
    }
    filldata << <blocks(settings.obstacles), threads >> > (settings.obstacles, devPtr, obstacle);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGraphicsUnmapResources(1, &obstaclesRes, 0);
    if (err != cudaSuccess && first) {
        printf("obstacles unmapping failed %s \n", cudaGetErrorString(err));
    }

    first = false;
}

extern "C" void drawobstacles() {
    if (settings.obstacles != 0) {
        fillobsdata();
        render2d.drawQuadinstancedbyinterop(settings.obstacles,2);
    }
}

std::vector<float> obx = { 900.0f, 5.0f, 900.0f, 1795.0f, 900.0f, 680.0f, 640.0f, 1150.0f, 680.0f, 833.0f, 900.0f, 995.0f, 900.0f, 1602.0f, 1321.0f, 1408.0f, 1360.0f };

std::vector<float> oby  = { 895.0f, 450.0f, 5.0f, 450.0f, 280.0f, 650.0f, 125.0f, 320.0f, 290.0f, 420.0f, 204.0f, 656.0f, 876.0f, 516.0f, 682.0f, 418.0f, 176.0f };
std::vector<float> obw  = { 1800.0f, 10.0f, 1800.0f, 10.0f, 130.0f, 50.0f, 50.0f, 50.0f, 50.0f, 147.0f, 50.0f, 148.0f, 50.0f, 380.0f, 200.0f, 50.0f, 118.0f };
std::vector<float> obh  = { 10.0f, 900.0f, 10.0f, 900.0f, 150.0f, 480.0f, 320.0f, 630.0f, 50.0f, 50.0f, 162.0f, 148.0f, 154.0f, 50.0f, 185.0f, 303.0f, 196.0f };
std::vector<float> obr  = { 0.0f, 0.0f, 0.0f, 0.0f, 45.0f, 0.0f, 43.0f, 0.0f, 37.0f, 0.0f, 0.0f, 32.5f, 0.0f, 0.0f, 36.2f, 0.0f, 82.6f };
void preaddobstacle() {
    for (int i = 0; i <4; i++) {


        quadvertex2d c;
        c.x = obx[i];
        c.y = oby[i];
        c.width = obw[i];
        c.height = obh[i];
        c.rotation = obr[i];
        c.r = 0.2f;
        c.g = 0.2f;
        c.b = 0.2f;

        h_obdata.push_back(c);
        settings.obstacles++;
    }
}

extern "C" void initobstacles() {
    cudaMemcpyToSymbol(d2_ray_angles, ray_angles, sizeof(ray_angles));
    cudaMalloc(&obstacle, settings.max_obstacles * sizeof(quadvertex2d));
    //add obstacles pre defined
    preaddobstacle();
    


    copyobsdata();
    bake_segments();
}

//add obstacles

extern "C" void copyobsdata() {
    cudaError_t err = cudaMemcpy(obstacle, h_obdata.data(), settings.obstacles * sizeof(quadvertex2d), cudaMemcpyHostToDevice);
    if (err!= cudaSuccess) {
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

    for (int r = 0; r < 16; r++) {
        float angle = car.z + d2_ray_angles[r];
        float4 d = {car.x,car.y, cosf(angle),sinf(angle) };
        float dist = castray(obcount, d, maxdist, segment);

        writeraylen(i, r, dist, rays);
    }
}
extern "C" void checkcolison() {
    rayhit << <blocks(settings.cars), threads >> > (settings.cars, rays, data1, segments, settings.obstacles, settings.ray_max_len);
}


__constant__ float d_rayexitdist[16];

float rayexitdist[16];

extern "C" void setrayexitdist() {

    float hh = settings.carheight * 0.5f;
    float hw = settings.carwidth * 0.5f;

    for (int i = 0; i < 16; i++) {
        float c = fabsf(cosf(ray_angles[i]));
        float s = fabsf(sinf(ray_angles[i]));
        float tx = (c > 1e-6f) ? hw / c : INFINITY;
        float ty = (s > 1e-6f) ? hh / s : INFINITY;
        
       
        rayexitdist[i] = fminf(tx,ty);
    }
    cudaMemcpyToSymbol(d_rayexitdist, rayexitdist,  sizeof(rayexitdist));
    cudaError_t err = cudaGetLastError();
        if (err) {
            printf(" setrayexit error %s \n", cudaGetErrorString(err));
        }
}

__device__ int Alive;
__global__ void checkstatekernel(int n,int* alive,float4* data1,float4* data2, ray* rays ,float time,float tx,float ty,float size,replaybuffer* buffer ,int s ,int bn,float maxd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)return;
    bool win = false;
    int bidx = s * n + i;
    if (alive[i] == 1) {

        float4 c = __ldg(&data1[i]);
        float dx = tx - c.x;
        float dy = ty - c.y;
        float dist = sqrtf(dx * dx + dy * dy);
        bool dead = false;
        int newState = alive[i];
        for (int k = 0; k < 16; k++) {
           // float len = rays[i].len[k];

            if (dist - d_rayexitdist[k] <= size * 0.5f) {
                newState = 2;
                dead = true;
                win = true;
               
                break;

            }
        }
            if (!win) {
                for (int k = 0; k < 16; k++) {
                    if (rays[i].len[k] <= d_rayexitdist[k]) {
                        newState = 0; dead = true;
						
                        break;
                    }
                }
            }


        

            if (dead) {
                alive[i] = newState;
                buffer[bidx].done = true;
                data2[i].x = maxd;

            }
            else {

                buffer[bidx].done = false;
            }
            data2[i].z = time;

            int lastTick = bn / n - 1;
            if (s == lastTick) {
                buffer[bidx].done = true;
            }

    }
    else {
        buffer[bidx].done = true;
    }

    if (alive[i] != 1) {
        atomicAdd(&Alive, 1);
    }

    
}

extern "C" void checkstate() {
    
   
  
 
    checkstatekernel << <blocks(settings.cars),threads >> > (settings.cars,alive, data1, data2, rays, settings.timer, settings.targetx, settings.targety, settings.targetsize, d_state, settings.step, settings.replaybuffersize, settings.maxdisttotarget);
    
    int a = 0;
  cudaError_t err=  cudaMemcpyFromSymbol(&a, Alive, sizeof(int));
  if (a == settings.cars) {
      settings.alive = false;
  }
   if (err !=cudaSuccess) {
       printf("checkstate alive memcpy error %s\n", cudaGetErrorString(err));
   }
  int zero = 0;
  err=cudaMemcpyToSymbol(Alive, &zero, sizeof(int));
  if (err != cudaSuccess) {
      printf("checkstate alive reset error %s\n", cudaGetErrorString(err));
  }
}

void unregisterobs() {
    if (obstaclesRes) {
        cudaGraphicsUnregisterResource(obstaclesRes);
        obstaclesRes = nullptr;
    }
}


void clearvectors() {
    actor_weights.clear();
    critic_weights.clear();
    actor_bias.clear();
    critic_bias.clear();
    actor_layerdata.clear();
    critic_layerdata.clear();
    h_obdata.clear();
    rewardgraph.clear();
    shuffled_indices.clear();
    h_obdata.clear();
}
std::mt19937 rng(42);
std::uniform_real_distribution<float> x(50.0f, 1750.0f);

std::uniform_real_distribution<float> sx(100.0f, 700.0f);

std::uniform_real_distribution<float> tx(900.0f, 1600.0f);

std::uniform_real_distribution<float> y(50.0f, 850.0f);

std::uniform_real_distribution<float> sy(100.0f, 800.0f);

std::uniform_real_distribution<float> ty(100.0f, 800.0f);

std::uniform_real_distribution<float> w(20.0f, 200.0f);

std::uniform_real_distribution<float> h(20.0f, 200.0f);

std::uniform_real_distribution<float> r(0.0f, 360.0f);
bool isoverlaping(float tx,float ty,int i) {

    float rad = -h_obdata[i].rotation * 3.14f/ 180.0f;
    float dx = tx - h_obdata[i].x;
    float dy = ty - h_obdata[i].y;

    float localX = dx * cos(rad) - dy * sin(rad);
    float localY = dx * sin(rad) + dy * cos(rad);

    return (
        abs(localX) <= h_obdata[i].width / 2 &&
        abs(localY) <= h_obdata[i].height / 2
        );
}
bool rectOverlap(
    float x1, float y1, float w1, float h1, float r1,
    float x2, float y2, float w2, float h2, float r2
)
{
   
    float rad = -r2 * 3.14159265f / 180.0f;

    float dx = x1 - x2;
    float dy = y1 - y2;

    float localX = dx * cos(rad) - dy * sin(rad);
    float localY = dx * sin(rad) + dy * cos(rad);

    return (
        abs(localX) <= (w1 + w2) / 2 &&
        abs(localY) <= (h1 + h2) / 2
        );
}
h_float2 getpos(bool isspawn)
{
    constexpr float EPSILON = 10.0f;

    while (true)
    {
        float px = (isspawn) ? sx(rng) : tx(rng);
        float py = (isspawn) ? sy(rng) : ty(rng);

        bool overlapping = false;

        for (int i = 0; i < settings.random_obstacles_count; i++)
        {
            if (rectOverlap(
                px, py,
                EPSILON * 2, EPSILON * 2, 0,
                h_obdata[i].x,
                h_obdata[i].y,
                h_obdata[i].width,
                h_obdata[i].height,
                h_obdata[i].rotation
            ))
            {
                overlapping = true;
                break;
            }
        }

        if (!overlapping)
            return { px, py };
    }
}
void randomobs() {
    settings.obstacles = 0;
    h_obdata.clear();
   
    for (int i = 0; i < settings.random_obstacles_count; i++) {
        quadvertex2d o;
        if (i < 4) {
            o.x = obx[i];
            o.y = oby[i];
            o.height = obh[i];
            o.width = obw[i];
            o.rotation = obr[i];
            o.r = 0.2f;
            o.g = 0.2f;
            o.b = 0.2f;
        }
        else {
            while (true)
            {
                o.x = x(rng);
                o.y = y(rng);
                o.height = h(rng);
                o.width = w(rng);
                o.rotation = r(rng);

                bool overlap = false;

                for (auto& old : h_obdata)
                {
                    if (rectOverlap(
                        o.x, o.y, o.width, o.height, o.rotation,
                        old.x, old.y, old.width, old.height, old.rotation
                    ))
                    {
                        overlap = true;
                        break;
                    }
                }

                if (!overlap)
                    break;
            }
        }
        h_obdata.push_back(o);
        settings.obstacles++;
    }


   h_float2 spawn = getpos(true);
   h_float2 target = getpos(false);
    settings.spawnx = spawn.x;
    settings.spawny = spawn.y;

    settings.targetx = target.x;
    settings.targety = target.y;
    copyobsdata();
    bake_segments();
}