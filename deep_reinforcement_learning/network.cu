#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include"data.h"
#include "net.h"
#include <vector>
#include <string>
#include <random>
#include <cstdio> 
#include <cmath>
#include <math.h>
#include <math_constants.h>
#include "obstacles.h"
#include "draw.h"

struct ddata {
	float maxspeed;
	float raymaxdist;
	float maxdisttotarget;
	float targetx;
	float taregety;
	float spawnx;
	float spawny;
	float degtorad;
	int maxsteer;
	int numcars;
	int layers;
	int inputcount;
};
__constant__ ddata d;
ddata h;
void setconst() {
	h.degtorad = 3.1415927f / 180.0f;
	h.inputcount = 13;
	h.layers = settings.layers;
	h.maxdisttotarget = settings.maxdisttotarget;
	h.maxspeed = dparam.max_speed;
	h.numcars = settings.cars;
	h.targetx = settings.targetx;
	h.taregety = settings.targety;
	h.spawnx = settings.spawnx;
	h.spawny = settings.spawny;
	h.raymaxdist = settings.ray_max_len;
	h.maxsteer = dparam.max_steer;

	cudaMemcpyToSymbol(d, &h, sizeof(ddata));
	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network constant struct cpy failed %s \n", cudaGetErrorString(err));
	}

};


extern "C" void allocate(){
	int n = settings.cars;
	cudaMalloc(&data1, n * sizeof(float4));
	cudaMalloc(&data2, n * sizeof(float4));
	cudaMalloc(&alive, n * sizeof(int));
	cudaMalloc(&rays, n * sizeof(ray));
	cudaMalloc(&segments, settings.max_obstacles * 4 * sizeof(float4));

	printf("data allocated \n");
}



void setmaxdisttotarget() {
	float dx = settings.spawnx - settings.targetx;
	float dy = settings.spawny - settings.targety;
	

	settings.maxdisttotarget = sqrtf(dx * dx + dy * dy);

	setconst();
}







//weights and netrwork





int weightbuffersize = 0;
int biassize = 0;
int nodedatasize = 0;




void addlayer(int in, int out) {
	Layer l;
	l.Nin = in;
	l.Nout = out;
	l.wIdx = 0;
	l.dIdx = 0;
	l.bIdx = 0;
	layerdata.push_back(l);
	settings.layers++;
	weightbuffersize += (in * out)*settings.cars;
	nodedatasize += out *settings.cars;
	biassize += out*settings.cars ;

}

void setoffsets() {

	for (int i = 0; i < settings.layers; i++) {
		int k = i - 1;
		layerdata[i].wIdx = i == 0 ? 0 : layerdata[k].wIdx + layerdata[k].Nin * layerdata[k].Nout;
		layerdata[i].dIdx = i == 0 ? 0 : layerdata[k].dIdx + layerdata[k].Nout;
		layerdata[i].bIdx = i == 0 ? 0 : layerdata[k].bIdx + layerdata[k].Nout;

	}
}
void initlayers() {
	//easy configraton of layers
	addlayer(13, 16);
	addlayer(16, 32);
	addlayer(32, 16);
	addlayer(16, 2);

	setoffsets();


}



__host__ __device__ __forceinline__ int idx(int offset, int k) {
	return offset + k ;
}


void initWB(float min,float max) {


	
	weights.resize(weightbuffersize);
	bias.resize(biassize);


	

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(min, max);

		for (int i = 0; i < weightbuffersize; i++) weights[i] = dist(rng);
		for (int i = 0; i < biassize; i++) bias[i] = dist(rng);

		
	
}

__device__ float leakyrelu(float x) {

	return (x > 0.f) ? x : 0.01f * x;

}

__global__ void firstlayer(int n ,const float4* __restrict__ data1,const float4* __restrict__ data2,const ray* __restrict__ rays, const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals){

	float4 c = __ldg(&data1[0]);
	float4 c2 = __ldg(&data2[0]);
	float dx = d.targetx - c.x;
	float dy = d.taregety - c.y;
	float dist = sqrtf(dx * dx + dy * dy);

	float targetForward = (dx * cosf(c.z) + dy * sinf(c.z)) / d.maxdisttotarget;
	float targetSide = (-dx * sinf(c.z) + dy * cosf(c.z)) / d.maxdisttotarget;

	float angle = c.z;


	float input[13] = { sinf(angle), cosf(angle),
									targetForward,targetSide,
									c2.w / d.maxsteer,
									c.w / d.maxspeed,
									rays[0].len[0] / d.raymaxdist,
									rays[0].len[1] / d.raymaxdist,
									rays[0].len[2] / d.raymaxdist,
									rays[0].len[3] / d.raymaxdist,
									rays[0].len[4] / d.raymaxdist,
									rays[0].len[5] / d.raymaxdist,
									fminf(dist / d.maxdisttotarget,1.0f)
	};
	int insize = 13;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

		float x = bias[idx(0,i)];
		for (int k = 0; k < insize; k++) {
			x += input[k] * weights[idx(0 + i * insize , k)];

		}

		nodevals[idx(0, i)] = leakyrelu(x);
}




__global__ void solvelayers(int n, int nin, int w, int d, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias,float* dNodeData) {

	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;


		float val = dBias[idx(b,i)];

		for (int j = 0; j < nin; j++) {
			float v = dNodeData[idx(p,j)];
			val += v * dWeights[idx(w + i * nin,j)];
		}
		
		dNodeData[idx(d , i )] = isout ? tanh(val) : leakyrelu(val);
	

}

__global__ void getoutput( int d, float* nodevals, float4* data2) {
	for (int j = 0; j < 2; j++) {

		if (j == 0) {
			data2[0].x = nodevals[idx(d, j)];
		}
		else {
			data2[0].y = nodevals[idx(d, j)];
		}

	}
}

void net(){

	for (int l = 0; l < settings.layers; l++) {

		if (l == 0) {
			int nout = layerdata[l].Nout;

			firstlayer << < blocks(nout), threads >> > (nout, data1, data2, rays, d_weights, d_bias, d_nodvals);

		}
		else
		{


			int nin = layerdata[l - 1].Nout;
			int wl = layerdata[l].wIdx;
			int di = layerdata[l].dIdx;
			int b = layerdata[l].bIdx;
			int p = layerdata[l - 1].dIdx;

			bool isout = (l == settings.layers - 1);

			solvelayers << < blocks(layerdata[l].Nout), threads >> > (layerdata[l].Nout, nin, wl, di, b, p, isout, d_weights, d_bias, d_nodvals);



		}


	}

		int outd = layerdata[settings.layers - 1].dIdx;
		getoutput<<<1,1>>>( outd, d_nodvals, data2);



	
}



void runnet() {
	net();
}


void allocatenetmem() {
	cudaMalloc(&d_weights, weightbuffersize * sizeof(float));
	cudaMalloc(&d_bias, biassize * sizeof(float));
	cudaMalloc(&d_nodvals, nodedatasize * sizeof(float));

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network malloc failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network malloc done \n");
	}
}
void copywbtogpu() {
	cudaMemcpy(d_weights, weights.data(), weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias.data(), biassize * sizeof(float), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network memcpy failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network memcpy done \n");
	}
	
}




void restart() {
	clearvectors();
	unregister();
	unregisterobs();
	cudafree();
	printf("memfree on restart \n");
	weightbuffersize = 0;
	biassize = 0;
	nodedatasize = 0;
	settings.layers = 0;
	settings.obstacles = 0;
	registervbo();
	printf("car vbo registered \n");
	registerObstaclesVbo();
	printf("obstacle vbo registered \n");
	registerrayvbo();
	printf("ray vbo registered \n");

	initnetwork();
	printf("network initialized \n");
	initcars();
	printf("car initialized \n");
	initobstacles();
	printf("obstacles initialized \n");
	setrayexitdist();
	printf("rayexit set \n");
	

}


void initnetwork() {
	initlayers();
	setoffsets();
	initWB(-0.2f, 0.2f);
	allocatenetmem();
	copywbtogpu();
	allocate();
	setmaxdisttotarget();

	setconst();


}

void cudafree() {
	cudaFree(d_nodvals);
	cudaFree(d_weights);
	cudaFree(d_bias);
	cudaFree(data1);
	cudaFree(data2);
	cudaFree(alive);
	cudaFree(rays);
	cudaFree(Rays);
	cudaFree(obstacle);
	cudaFree(segments);
	cudaFree(car);

}
