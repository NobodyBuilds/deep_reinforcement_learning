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
	float maxdist;
	float maxdisttotarget;
	float targetx;
	float taregety;
	float degtorad;
	int numcars;
	int layers;
	int inputcount;
};
__constant__ ddata d;
ddata h;
void setconst() {
	h.degtorad = 3.1415927f / 180.0f;
	h.inputcount = 11;
	h.layers = settings.layers;
	h.maxdist = settings.maxdisttotarget;
	h.maxspeed = dparam.max_speed;
	h.numcars = settings.cars;
	h.targetx = settings.targetx;
	h.taregety = settings.targety;

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
	cudaMalloc(&fitness, n * sizeof(float));
	cudaMalloc(&alive, n * sizeof(int));
	cudaMalloc(&rays, n * sizeof(ray));
	cudaMalloc(&segments, settings.max_obstacles * 4 * sizeof(float4));
	cudaMalloc(&indices, n * sizeof(int));
	

	printf("data allocated \n");
}


__global__ void fitnesskernel(int n,const float4* __restrict__ data1,const float4* __restrict__ data2 ,const int* __restrict__ alive ,float* fitness,float maxdt,float tx,float ty ) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)return;

	float4 c = __ldg(&data1[i]);
	float4 d2 = __ldg(&data2[i]);
	int a = alive[i];
	float dx = tx - c.x;
	float dy = ty - c.y;
	float cudist = sqrtf(dx * dx + dy * dy);
	float val = 0.0f;
	if (a == 2) {
		val = 100.0f;
	}
	else if (a == 0) {
		val = -150.0f;
	}

	fitness[i] = (maxdt - cudist) 
		+val
		- d2.z; //time taken



}

__global__ void sortkernel(int n, int j, int k, float* fitness, int* indices) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

	unsigned int ixj = i ^ j;

	if (ixj > i) {
		bool accending = ((i & k) == 0);
		if ((fitness[i] < fitness[ixj]) == accending) {
			float tempf = fitness[i];
			fitness[i] = fitness[ixj];
			fitness[ixj] = tempf;
			int tempi = indices[i];
			indices[i] = indices[ixj];
			indices[ixj] = tempi;
		}
	}
}
void bitonicsort() {
	int n = settings.cars;

	for (int k = 2; k <= n; k <<= 1) {
		for (int j = k >> 1; j > 0; j >>= 1) {
			sortkernel << < blocks(settings.cars), threads >> > (settings.cars, j, k, fitness, indices);
		}
	}
}
__device__ float fit;
__global__ void te(int* indices, float* fitness) {
	fit = fitness[0];
	
}
void computefitness() {

	fitnesskernel << <blocks(settings.cars), threads >> > (settings.cars, data1, data2, alive, fitness, settings.maxdisttotarget, settings.targetx, settings.targety);
	bitonicsort();

	te << <1, 1 >> > (indices, fitness);
	float hfit = 0.0f;
	cudaMemcpyFromSymbol(&hfit, fit, sizeof(float));
	settings.fitness = hfit;

}
void setmaxdisttotarget() {
	float dx = settings.spawnx - settings.targetx;
	float dy = settings.spawny - settings.targety;

	settings.maxdisttotarget = sqrtf(dx * dx + dy * dy);
}

__global__ void fillindices(int n, int* indices) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	indices[i] = i;

}


__device__ unsigned long long gidx;
__global__ void caridxkernel(int n, const float4* __restrict__ data1, const int* __restrict__ alive,float tx,float ty){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	if (alive[i] == 1) {
		float4 c = __ldg(&data1[i]);

		float dx = tx - c.x;
		float dy = ty - c.y;
		float dist = sqrtf(dx * dx + dy * dy);

		unsigned int distBits = __float_as_uint(dist);
		unsigned long long key = (((unsigned long long)distBits) << 32) | (unsigned int)i;

		atomicMin(&gidx, key);
	}
}
void getbestcaridx() {
	unsigned long long initVal = ULLONG_MAX;
	cudaMemcpyToSymbol(gidx, &initVal, sizeof(unsigned long long));

	caridxkernel << <blocks(settings.cars), threads >> > (
		settings.cars, data1,alive, settings.targetx, settings.targety
		);
	unsigned long long result;
	cudaMemcpyFromSymbol(&result, gidx, sizeof(unsigned long long));
	settings.bestcaridx = (int)(result & 0xFFFFFFFFu);

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
	addlayer(10, 16);
	addlayer(16, 32);
	addlayer(32, 64);
	addlayer(64, 32);
	addlayer(32, 16);
	addlayer(16, 2);

	setoffsets();


}



__host__ __device__ __forceinline__ int idx(int offset, int k, int caridx) {
	return (offset + k) * d.numcars + caridx;
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

__device__ __forceinline__ void firstlayer(int n ,int ci,int w,int d ,int inputs,float* input,const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals){

	for (int i = 0; i < n; i++) {

		float x = bias[idx(0,i,ci)];
		for (int k = 0; k < inputs; k++) {
			x += input[k] * weights[idx(w + i * inputs , k, ci)];

		}

		nodevals[idx(d, i, ci)] = leakyrelu(x);
	}

}


__device__ void solvelayers(int n,int ci, int nin, int w, int d, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias,float* dNodeData) {

	
	for (int i = 0; i < n; i++) {


		float val = dBias[idx(b,i,ci)];

		for (int j = 0; j < nin; j++) {
			float v = dNodeData[idx(p,j,ci)];
			val += v * dWeights[idx(w + i * nin,j,ci)];
		}
		
		dNodeData[idx(d , i, ci)] = isout ? tanh(val) : leakyrelu(val);
	}

}

__device__ void getoutput(int i, int d, float* nodevals, float4* data2) {
	for (int j = 0; j < 2; j++) {

		if (j == 0) {
			data2[i].x = nodevals[idx(d, j, i)];
		}
		else {
			data2[i].y = nodevals[idx(d, j, i)];
		}

	}
}

__global__ void netkernel(int n ,const float4* __restrict__ data1,float4* data2,const int* __restrict__ alive,
	const ray* rays,const float* __restrict__ weights,
	const float* __restrict__ bias,float* nodevals ,const Layer* layer

) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

	float4 c = __ldg(&data1[i]);
	float4 c2 = __ldg(&data2[i]);

	if (alive[i] == 1) {

		for (int l = 0; l < d.layers; l++) {

			if (l == 0) {
				float dx = d.targetx - c.x;
				float dy = d.taregety - c.y;
				float dist = sqrtf(dx * dx + dy * dy);

				float angle = c.z ;
				float input[10] = { sinf(angle), cosf(angle),
									c.w / d.maxspeed,
									rays[i].len[0],
									rays[i].len[1],
									rays[i].len[2],
									rays[i].len[3],
									rays[i].len[4],
									rays[i].len[5],
									fminf(dist / d.maxdisttotarget,1.0f)
				};
				int didx = layer[0].dIdx;
				firstlayer(layer[0].Nout, i, 0,didx, 10, input, weights, bias, nodevals);
			}
			else {
				

				int nin = layer[l - 1].Nout;
				int wl = layer[l].wIdx;
				int di = layer[l].dIdx;
				int b = layer[l].bIdx;
				int p = layer[l - 1].dIdx;

				bool isout = (l == d.layers - 1);

				solvelayers(layer[l].Nout, i, nin, wl, di, b, p, isout, weights, bias, nodevals);

			}





		}
		float outd = layer[d.layers - 1].dIdx;
		getoutput(i, outd, nodevals, data2);
	}


}


void runnet() {
	netkernel << < blocks(settings.cars),threads >> > (settings.cars,data1,data2,alive,rays,d_weights,d_bias,d_nodvals,dlayer);
	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network layers failed %s \n", cudaGetErrorString(err));
	}
}


void allocatenetmem() {
	cudaMalloc(&d_weights, weightbuffersize * sizeof(float));
	cudaMalloc(&d_bias, biassize * sizeof(float));
	cudaMalloc(&d_nodvals, nodedatasize * sizeof(float));
	cudaMalloc(&dlayer, settings.layers *sizeof(Layer));

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
	cudaMemcpy(dlayer, layerdata.data(), settings.layers * sizeof(Layer),cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network memcpy failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network memcpy done \n");
	}
	
}








void mutation() {

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
	settings.cars = settings.samplecar;
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
	initWB(-0.5f, 0.5f);
	allocatenetmem();
	copywbtogpu();
	allocate();
	setmaxdisttotarget();

	fillindices << <blocks(settings.cars), threads >> > (settings.cars, indices);
	setconst();

}

void cudafree() {
	cudaFree(d_nodvals);
	cudaFree(d_weights);
	cudaFree(d_bias);
	cudaFree(data1);
	cudaFree(data2);
	cudaFree(alive);
	cudaFree(fitness);
	cudaFree(dlayer);
	cudaFree(indices);
	cudaFree(rays);
	cudaFree(Rays);
	cudaFree(obstacle);
	cudaFree(segments);
	cudaFree(car);

}