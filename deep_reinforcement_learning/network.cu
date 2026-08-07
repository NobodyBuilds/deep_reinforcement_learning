#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include"data.h"
#include "net.h"
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdio> 
#include <cmath>
#include <math.h>
#include <math_constants.h>
#include "obstacles.h"
#include "draw.h"
#include <curand_kernel.h>


curandState* d_rngstate;
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
	sync();

};
__global__ void init_rng(curandState* state, unsigned long seed) {
	curand_init(seed, 0, 0, state);
}
extern "C" void allocate(){
	int n = settings.cars;
	cudaMalloc(&data1, n * sizeof(float4));
	cudaMalloc(&data2, n * sizeof(float4));
	cudaMalloc(&alive, n * sizeof(int));
	cudaMalloc(&rays, n * sizeof(ray));
	cudaMalloc(&segments, settings.max_obstacles * 4 * sizeof(float4));
	cudaMalloc(&d_rngstate, sizeof(curandState));
	init_rng << <1, 1 >> > (d_rngstate, 1234);
	printf("data allocated \n");
}
void setmaxdisttotarget() {
	float dx = settings.spawnx - settings.targetx;
	float dy = settings.spawny - settings.targety;
	

	settings.maxdisttotarget = sqrtf(dx * dx + dy * dy);

	setconst();
}

//weights and netrwork

int actor_weightbuffersize = 0;
int critic_weightbuffersize = 0;
int actor_biassize = 0;
int critic_biassize = 0;
int actor_nodedatasize = 0;
int critic_nodedatasize = 0;


void addlayer(int in, int out,bool isactor) {
	Layer l;
	l.Nin = in;
	l.Nout = out;
	l.wIdx = 0;
	l.dIdx = 0;
	l.bIdx = 0;
	if (isactor) {
		actor_layerdata.push_back(l);
		settings.actor_layers++;
		actor_weightbuffersize += (in * out) * settings.cars;
		actor_nodedatasize += out * settings.cars;
		actor_biassize += out * settings.cars;
	}
	else {
		critic_layerdata.push_back(l);
		settings.critic_layers++;
		critic_weightbuffersize += (in * out) * settings.cars;
		critic_nodedatasize += out * settings.cars;
		critic_biassize += out * settings.cars;
	}

}

void setoffsets(bool isactor) {

	auto& layerdata = isactor ? actor_layerdata : critic_layerdata;

	for (int i = 0; i < layerdata.size(); i++) {
		int k = i - 1;
		layerdata[i].wIdx = i == 0 ? 0 : layerdata[k].wIdx + layerdata[k].Nin * layerdata[k].Nout;
		layerdata[i].dIdx = i == 0 ? 0 : layerdata[k].dIdx + layerdata[k].Nout;
		layerdata[i].bIdx = i == 0 ? 0 : layerdata[k].bIdx + layerdata[k].Nout;

	}
}
void initlayers() {
	//easy configraton of layers
	//actor network
	addlayer(13, 16,true);
	addlayer(16, 32,true);
	addlayer(32, 16,true);
	addlayer(16, 4,true);

	setoffsets(true);
	//critic network
	addlayer(13, 16, false);
	addlayer(16, 32, false);
	addlayer(32, 16, false);
	addlayer(16, 1, false);

	setoffsets(false);


}



__host__ __device__ __forceinline__ int idx(int offset, int k) {
	return offset + k ;
}


void initWB(float min,float max) {


	
	actor_weights.resize(actor_weightbuffersize);
	actor_bias.resize(actor_biassize);

	critic_weights.resize(critic_weightbuffersize);
	critic_bias.resize(critic_biassize);
	

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(min, max);
		//actor network weights and bias
		for (int i = 0; i < actor_weightbuffersize; i++) actor_weights[i] = dist(rng);
		for (int i = 0; i < actor_biassize; i++) actor_bias[i] = dist(rng);
		//critic network weights and bias
		for (int i = 0; i < critic_weightbuffersize; i++) critic_weights[i] = dist(rng);
		for (int i = 0; i < critic_biassize; i++) critic_bias[i] = dist(rng);
		
	
}
__device__ float leakyrelu(float x) {

	return (x > 0.f) ? x : 0.01f * x;

}
__global__ void firstlayer(int n ,bool isactor,int s,const float4* __restrict__ data1,const float4* __restrict__ data2,const ray* __restrict__ rays, const float* __restrict__ weights,const float* __restrict__ bias,float* nodevals,float* d_preact,replaybuffer* buffer){

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
		if (isactor) {
		for (int b = 0; b < insize; b++) {
			buffer[s].s1[b] = input[b];

		}
	}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

		float x = bias[idx(0,i)];
		for (int k = 0; k < insize; k++) {
			x += input[k] * weights[idx(0 + i * insize , k)];

		}

		if (s >= 0) { d_preact[idx(0, i)] = x; }
		nodevals[idx(0, i)] = leakyrelu(x);
}
__global__ void solvelayers(int n, int nin, int w, int D, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias,float* dNodeData,float* preact) {

	
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;


		float val = dBias[idx(b,i)];

		for (int j = 0; j < nin; j++) {
			float v = dNodeData[idx(p,j)];
			val += v * dWeights[idx(w + i * nin,j)];
		}
		
		 preact[idx(D, i)] = val; 
		dNodeData[idx(D , i )] = isout ? val : leakyrelu(val);
	

}
__device__ float dslope(float x) {
	return x > 0.f ? 1.f : 0.01f;
}
__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin, int l1d, bool outlayer, float* nodedata, float* dDelta, float* dWeights, float* dpreact) {

	float target = d_target;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)return;

	if (outlayer) {
		dDelta[d + i] = (i == d_action) ? nodedata[d + i] - target : 0.0f;

	}
	else {
		float sum = 0.0f;
		for (int k = 0; k < l1_nout; k++) {
			sum += dWeights[l1w + k * l1_nin + i] * dDelta[l1d + k];
		}
		dDelta[d + i] = sum * dslope(dpreact[d + i]);
	}
}
__global__ void tuneweights(int n, int nin, int l, int l1d, int lw, int lsize, int D, int lb, float lr, float* dNodeData, float* dWeights, float* dDelta, float* dBias,const replaybuffer* __restrict__ buffer,int s) {
	
	

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;


	for (int k = 0; k < nin; k++) {
		float node_data = (l == 0) ? buffer[s].s1.state[k] : dNodeData[l1d + k];
		dWeights[lw + i * lsize + k] -= lr * (dDelta[D + i] * node_data);
	}
	dBias[lb + i] -= lr * dDelta[D + i];
}
__global__ void getoutput(int d, float* nodevals, float4* carcontrol,replaybuffer* buffer,int s) {
	int action = 0;
	for (int j = 0; j < 4; j++) {

		if (j == 0) {
			data2[0].x = nodevals[idx(d, j)];
			action = j;
		}
		else if(j==1) {
			data2[0].y = nodevals[idx(d, j)];
			action = j;
		}
		else if(j==2) {
			data2[0].z = nodevals[idx(d, j)];
			action = j;
		}
		else if(j==3) {
			data2[0].w = nodevals[idx(d, j)];
			action = j;
		}


	}
	buffer[s].action = action;

	
}
__global__ void reward_kernel(int s,replaybuffer* buffer,float4* data1,int* alive) {

	float4 c = __ldg(&data1[0]);
	float alivereward = alive[0] == 1 ? 0.0f : -5.0f;

	float dx = c.x - d.targetx;
	float dy = c.y - d.taregety;
	float curdist = sqrtf(dx * dx + dy * dy);

	float progressreward = 1 - ((d.maxdisttotarget - curdist) / d.maxdisttotarget);

	float reward = progressreward* 2.0f + alivereward;

	buffer[s].reward = reward;

}
__global__ void getQvalue(int s,int d, float* nodevals, replaybuffer* buffer) {
	
	buffer[s].value = nodevals[d];
}





void reward(int s) {
	reward_kernel << <1, 1 >> > (s, d_state, data1, alive);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("!ERROR! reward kernel failed %s \n", cudaGetErrorString(err));
	}
}



void backpropogation(bool isactor) {
	int threads = 256;
	for (int l = settings.layers - 1; l >= 0; l--) {


		int d = layerdata[l].dIdx;
		int n = layerdata[l].Nout;
		int l1out = (l < settings.layers - 1) ? layerdata[l + 1].Nout : 0;
		int l1w =  (l< settings.layers-1)? layerdata[l + 1].wIdx:0;
		int l1d =  (l< settings.layers-1)? layerdata[l + 1].dIdx:0;
		int l1nin = (l < settings.layers - 1) ? layerdata[l + 1].Nin : 0;



		bool outlayer = (l == settings.layers - 1) ? 1 : 0;

		int blocks = (n + threads - 1) / threads;
		compute_delta << <blocks, threads >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, d_nodvals, d_delta, d_weights, d_preact);
	cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("!ERROR! compute delta failed %s \n", cudaGetErrorString(err));
		}

	}

	for (int l = settings.layers - 1; l >= 0; l--) {
		int blocks = (layerdata[l].Nout + threads - 1) / threads;
		int nin = layerdata[l].Nin;
		int l1d = (l == 0) ? 0 : layerdata[l - 1].dIdx;
		int lw = layerdata[l].wIdx;
		int lsize = (l == 0) ? 13 : layerdata[l].Nin;
		int d = layerdata[l].dIdx;
		int lb = layerdata[l].bIdx;
		tuneweights << <blocks, threads >> > (layerdata[l].Nout, nin, l, l1d, lw, lsize, d, lb,settings.lr, d_nodvals, d_weights, d_delta, d_bias,d_state,settings.step);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("!ERROR! tune weights failed %s \n", cudaGetErrorString(err));
		}


	}
}

void net(int s,bool isactor){

	auto& layerdata = isactor ? actor_layerdata : critic_layerdata;

	for (int l = 0; l < (isactor)?settings.actor_layers:settings.critic_layers; l++) {

		if (l == 0) {


			int nout = layerdata[l].Nout;
			if (isactor) {

				firstlayer << < blocks(nout), threads >> > (nout, isactor, s, data1, data2, rays, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, d_state);
			}
			else {
				firstlayer << < blocks(nout), threads >> > (nout, isactor, s, data1, data2, rays, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, d_state);
			}
		}
		else
		{


			int nin = layerdata[l - 1].Nout;
			int wl = layerdata[l].wIdx;
			int di = layerdata[l].dIdx;
			int b = layerdata[l].bIdx;
			int p = layerdata[l - 1].dIdx;

			bool isout = (l == (isactor ? settings.actor_layers : settings.critic_layers) - 1);
			if (isactor) {
				solvelayers << < blocks(layerdata[l].Nout), threads >> > (layerdata[l].Nout, nin, wl, di, b, p, isout, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, s);
			}
			else {
				solvelayers << < blocks(layerdata[l].Nout), threads >> > (layerdata[l].Nout, nin, wl, di, b, p, isout, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, s);
			}

		}


	}
	if (isactor)
	{
		int outd = layerdata[ settings.actor_layers - 1].dIdx;
		getoutput << <1, 1 >> > (outd, d_nodvals, carcontrol, d_state, s);

	}
	else {
		int outd = layerdata[settings.critic_layers - 1].dIdx;
		getQvalue << <1, 1 >> > (s, outd, d_critic_nodvals, d_state);
	}

	
}


void run_network() {
	 
	net(settings.step,true);
	stepcars();
	reward(settings.step);

	net(settings.step, false);




	settings.step++;
	if(settings.step>=settings.replaybuffersize) {
		settings.step = 0;
	}
}




void allocatenetmem() {
	cudaMalloc(&d_actor_weights, actor_weightbuffersize * sizeof(float));
	cudaMalloc(&d_critic_weights, critic_weightbuffersize * sizeof(float));
	cudaMalloc(&d_actor_bias, actor_biassize * sizeof(float));
	cudaMalloc(&d_critic_bias, critic_biassize * sizeof(float));
	cudaMalloc(&d_actor_nodvals, actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_nodvals, critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_preact, actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_preact, critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_delta, actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_delta, critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_state, settings.replaybuffersize * sizeof(replaybuffer));
	cudaMalloc(&carcontrol, sizeof(float4));


	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network malloc failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network malloc done \n");
	}
}
void copywbtogpu() {
	cudaMemcpy(d_actor_weights, actor_weights.data(), actor_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critic_weights, critic_weights.data(), critic_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_actor_bias, actor_bias.data(), actor_biassize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critic_bias, critic_bias.data(), critic_biassize * sizeof(float), cudaMemcpyHostToDevice);

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
	actor_weightbuffersize = 0;
	critic_weightbuffersize = 0;
	actor_biassize = 0;
	critic_biassize = 0;
	actor_nodedatasize = 0;
	critic_nodedatasize = 0;
	settings.actor_layers = 0;
	settings.critic_layers = 0;
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

	initWB(-0.2f, 0.2f);
	allocatenetmem();
	copywbtogpu();
	allocate();
	setmaxdisttotarget();

	setconst();


}
void cudafree() {
	cudaFree(d_actor_nodvals);
	cudaFree(d_critic_nodvals);
	cudaFree(d_actor_weights);
	cudaFree(d_critic_weights);
	cudaFree(d_actor_bias);
	cudaFree(d_critic_bias);
	cudaFree(d_actor_preact);
	cudaFree(d_critic_preact);
	cudaFree(d_actor_delta);
	cudaFree(d_critic_delta);
	cudaFree(d_state);
	cudaFree(carcontrol);
	cudaFree(data1);
	cudaFree(data2);
	cudaFree(alive);
	cudaFree(rays);
	cudaFree(Rays);
	cudaFree(obstacle);
	cudaFree(segments);
	cudaFree(car);

}
