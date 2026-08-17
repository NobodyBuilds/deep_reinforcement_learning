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
#include <fstream>


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
	h.inputcount = settings.inputs;
	h.layers = settings.actor_layers;
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
__global__ void init_rng(curandState* state, int n,unsigned long seed) {
	for (int i = 0; i < n; i++) {
		curand_init(seed, i, 0, &state[i]);
	}
}
extern "C" void allocate() {
	int n = settings.cars;
	cudaMalloc(&data1, n * sizeof(float4));
	cudaMalloc(&data2, n * sizeof(float4));
	cudaMalloc(&alive, n * sizeof(int));
	cudaMalloc(&rays, n * sizeof(ray));
	cudaMalloc(&segments, settings.max_obstacles * 4 * sizeof(float4));
	cudaMalloc(&d_rngstate, settings.cars*sizeof(curandState));
	init_rng << <1, 1 >> > (d_rngstate,settings.cars, 3476);
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
int hbatchsize = 64;
int carsnodesize = 0;

void addlayer(int in, int out, bool isactor) {
	Layer l;
	l.Nin = in;
	l.Nout = out;
	l.wIdx = 0;
	l.dIdx = 0;
	l.bIdx = 0;
	if (isactor) {
		actor_layerdata.push_back(l);
		settings.actor_layers++;
		actor_weightbuffersize += (in * out) ;
		actor_nodedatasize += out ;
		carsnodesize += out * settings.cars;
		actor_biassize += out ;
	}
	else {
		critic_layerdata.push_back(l);
		settings.critic_layers++;
		critic_weightbuffersize += (in * out) ;
		critic_nodedatasize += out ;
		critic_biassize += out;
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
	addlayer(settings.inputs, 64, true);
	addlayer(64, 32, true);
	addlayer(32, 16, true);
	addlayer(16, 6, true);

	setoffsets(true);
	//critic network
	addlayer(settings.inputs, 64, false);
	addlayer(64, 32, false);
	addlayer(32, 16, false);
	addlayer(16, 1, false);

	setoffsets(false);


}



__host__ __device__ __forceinline__ int idx(int offset, int k) {
	return offset + k;
}
__device__  int caridx(int off,int k,int ci) {
	return(off + k) * d.numcars + ci;
}

void readfiles(const std::string& dir, std::vector<float>& container)
{
	container.clear();
	std::fstream file(dir, std::ios::in);
	if (!file.is_open())
	{
		std::cerr << "Error opening " << dir << " file" << std::endl;
		return;
	}
	
	std::string val;
	
	while (file >> val)
	{
		container.push_back(std::stof(val));
		
	}
}


void initWB(float min, float max) {



	
	bool weightsloaded = true;
	readfiles("modeldata/actorweights.txt", actor_weights);
	readfiles("modeldata/criticweights.txt", critic_weights);
	readfiles("modeldata/actorbias.txt", actor_bias);
	readfiles("modeldata/criticbias.txt", critic_bias);
	if (actor_weights.size() != actor_weightbuffersize) {
		printf("%d / %d \n", actor_weights.size(),actor_weightbuffersize);
		weightsloaded = false;
	}
	if (critic_weights.size() != critic_weightbuffersize) {
		weightsloaded = false;
	}
	if (actor_bias.size() != actor_biassize) {
		weightsloaded = false;
	}
	if (critic_bias.size() != critic_biassize) {
		weightsloaded = false;
	}
	if (!weightsloaded) {
		printf("weights loading error ,generating new weights\n");
		actor_bias.clear();
		critic_bias.clear();
		actor_weights.clear();
		critic_weights.clear();
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
	else {
		printf("weights loaded from files\n");
	}
}
void write_file(const std::string &address,std::vector<float> &arr) {


	std::ofstream file(address, std::ios::trunc);
	if (!file.is_open())
	{
		std::cerr << "Error opening" <<address<< "for writing" << std::endl;
	}
	for (float v : arr)
		file << v << " ";
	file << "\n";

	file.close();

}
void save_weights() {


	cudaMemcpy(actor_weights.data(), d_actor_weights, actor_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(critic_weights.data(), d_critic_weights, critic_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(actor_bias.data(), d_actor_bias, actor_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(critic_bias.data(), d_critic_bias, critic_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (err) {
		printf(" memcpy grom device to save weights failed %s \n", cudaGetErrorString(err));
	}
	std::remove("modeldata/actorweights.txt");
	std::remove("modeldata/criticweights.txt");
	std::remove("modeldata/actorbias.txt");
	std::remove("modeldata/criticbias.txt");

	write_file("modeldata/actorweights.txt", actor_weights);
	write_file("modeldata/criticweights.txt", critic_weights);
	write_file("modeldata/actorbias.txt", actor_bias);
	write_file("modeldata/criticbias.txt", critic_bias);
	printf("weights saved \n");
}

__device__ float MSE = 0.0f;


__device__  int batchsize = 64;


__device__ float leakyrelu(float x) {

	return (x > 0.f) ? x : 0.01f * x;

}
__device__ __forceinline__ void firstlayer(int n, int ci, int w, int D, int inputs, float* input, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals) {

	for (int i = 0; i < n; i++) {

		float x = bias[idx(0, i)];
		for (int k = 0; k < inputs; k++) {
			x += input[k] * weights[idx(w + i * inputs, k)];

		}

		nodevals[caridx(D, i, ci)] = leakyrelu(x);
	}

}
__device__ void solvelayers(int n, int ci, int nin, int w, int d, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData) {


	for (int i = 0; i < n; i++) {


		float val = dBias[idx(b, i)];

		for (int j = 0; j < nin; j++) {
			float v = dNodeData[caridx(p, j, ci)];
			val += v * dWeights[idx(w + i * nin, j)];
		}

		dNodeData[caridx(d, i, ci)] = isout ? val : leakyrelu(val);
	}

}
__global__ void firstlayer(int n, bool isactor, int s, const float4* __restrict__ data1, const float4* __restrict__ data2, const ray* __restrict__ rays, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals, float* d_preact, replaybuffer* buffer) {

	float4 c = __ldg(&data1[0]);
	float4 c2 = __ldg(&data2[0]);


	float dx = d.targetx - c.x;
	float dy = d.taregety - c.y;
	float dist = sqrtf(dx * dx + dy * dy);

	float targetinsight = (dist < d.raymaxdist) ? 1.0f : 0.0f;

	float targetForward = (dx * cosf(c.z) + dy * sinf(c.z)) / d.maxdisttotarget;
	float targetSide = (-dx * sinf(c.z) + dy * cosf(c.z)) / d.maxdisttotarget;

	float angle = c.z;


	float input[24] = { sinf(angle), cosf(angle),
									targetForward,targetSide,
									c2.w / d.maxsteer,
									c.w / d.maxspeed,
									targetinsight,
									rays[0].len[0] / d.raymaxdist,
									rays[0].len[1] / d.raymaxdist,
									rays[0].len[2] / d.raymaxdist,
									rays[0].len[3] / d.raymaxdist,
									rays[0].len[4] / d.raymaxdist,
									rays[0].len[5] / d.raymaxdist,
									rays[0].len[6] / d.raymaxdist,
									rays[0].len[7] / d.raymaxdist,
									rays[0].len[8] / d.raymaxdist,
									rays[0].len[9] / d.raymaxdist,
									rays[0].len[10] / d.raymaxdist,
									rays[0].len[11] / d.raymaxdist,
									rays[0].len[12] / d.raymaxdist,
									rays[0].len[13] / d.raymaxdist,
									rays[0].len[14] / d.raymaxdist,
									rays[0].len[15] / d.raymaxdist,
									dist / d.maxdisttotarget
	};
	int insize = 24;
	if (isactor) {
		for (int b = 0; b < insize; b++) {
			buffer[s].s1[b] = input[b];

		}
	}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

	float x = bias[idx(0, i)];
	for (int k = 0; k < insize; k++) {
		x += input[k] * weights[idx(0 + i * insize, k)];

	}

	d_preact[idx(0, i)] = x;
	nodevals[idx(0, i)] = leakyrelu(x);
}
__global__ void solvelayers(int n, int nin, int w, int D, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData, float* preact) {



	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;


	float val = dBias[idx(b, i)];

	for (int j = 0; j < nin; j++) {
		float v = dNodeData[idx(p, j)];
		val += v * dWeights[idx(w + i * nin, j)];
	}

	preact[idx(D, i)] = val;
	dNodeData[idx(D, i)] = isout ?  val : leakyrelu(val);


}
__device__ float dslope(float x) {
	return x > 0.f ? 1.f : 0.01f;
}
__device__ float clamp(float val, float min, float max) {
	return fminf(fmaxf(val, min), max);
}
__device__ float get_lClip(replaybuffer* buffer, int s) {


	float diff = clamp(buffer[s].logprob - buffer[s].old_logprob, -20, 20);
	float r = expf(diff);
	
	float ep = 0.2f;
	float clipped_out = 0.f;
	float adv = buffer[s].advantage;
	if (adv > 0) {
		if (r > 1 + ep) clipped_out = 0;
		else clipped_out = -adv * r;
	}

	if (adv < 0) {
		if (r < 1 - ep) clipped_out = 0;
		else clipped_out = -adv * r;
	}
	return clipped_out;
}
__device__ float beta = 0.02f;
__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin, int l1d, bool outlayer, bool isactor, float* nodedata, float* dDelta, float* dWeights, float* dpreact, replaybuffer* buffer, int s, int* indices, int nodedatasize,int gen) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)return;
	int bidx = indices[s * batchsize + b];
	float target =  buffer[bidx].rtg;
	if (outlayer) {
		if (isactor) {
			float coeff = get_lClip(buffer, bidx);
			float pi_i = nodedata[b * nodedatasize + d + i];      
			float y = (i == buffer[bidx].action) ? 1.0f : 0.0f;

			float H = 0.0f;
			for (int j = 0; j < n; j++) {
				float pj = nodedata[b * nodedatasize + d + j];
				H -= pj * logf(fmaxf(pj, 1e-8f));
			}
			
			float entropybonus = beta * pi_i * (logf(fmaxf(pi_i, 1e-8f)) + H);

			dDelta[b * nodedatasize + d + i] = coeff * (y - pi_i) + entropybonus;
			beta= 0.02f * (1.0f / (1.0f + 0.0001f * gen));
		}
		else {
			dDelta[b * nodedatasize + d + i] = nodedata[b * nodedatasize + d + i] - target;
		}
	}
	else {
		float sum = 0.0f;
		for (int k = 0; k < l1_nout; k++) {
			sum += dWeights[l1w + k * l1_nin + i] * dDelta[b * nodedatasize + l1d + k];
		}
		dDelta[b * nodedatasize + d + i] = sum * dslope(dpreact[b * nodedatasize + d + i]);
	}
}
__global__ void tuneweights(int n, int nin, int l, int l1d, int lw, int lsize, int d, int lb, float lr, float* dNodeData, float* dWeights, float* dDelta, float* dBias, const replaybuffer* __restrict__ buffer, int s, int curbatch, int nodedatasize, int* indices) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	float biasgrad = 0.0f;
	for (int b = 0; b < curbatch; b++)
		biasgrad += dDelta[b * nodedatasize + d + i];
	dBias[lb + i] -= lr * (biasgrad / curbatch);

	for (int k = 0; k < nin; k++)
	{
		float wgrad = 0.0f;
		for (int b = 0; b < curbatch; b++)
		{
			int off = b * nodedatasize;
			float nodeVal = (l == 0) ? buffer[indices[s * batchsize + b]].s1[k] : dNodeData[off + l1d + k];
			wgrad += dDelta[off + d + i] * nodeVal;
		}
		dWeights[lw + i * lsize + k] -= lr * (wgrad / curbatch);
	}
}
__device__ void getoutput(int D, float* nodevals, float4* carcontrol, replaybuffer* buffer, int s,int ci, curandState* d_rngstate) {
	int action = 0;
	int bidx = s * d.numcars + ci;
	float m = nodevals[caridx(D, 0, ci)];
	for (int j = 1; j < 6; j++) m = fmaxf(m, nodevals[caridx(D,j,ci)]);

	float sum = 0.f;
	float exps[6];
	for (int j = 0; j < 6; j++) {
		exps[j] = expf(nodevals[caridx(D , j,ci)] - m);
		sum += exps[j];
	}//sum all vals

	for (int j = 0; j < 6; j++) {
		nodevals[caridx(D , j,ci)] = exps[j] / sum;

	}//probs




	float r = curand_uniform(&d_rngstate[ci]);
	
	float cum = 0.f;
	for (int j = 0; j < 6; j++) {
		cum += nodevals[caridx(D , j,ci)];

		if (r <= cum) { action = j; break; }
	}
	buffer[bidx].action = action;

	float4 cc = make_float4(0.f, 0.f, 0.f, 0.f);
	if (buffer[bidx].action == 0) cc.x = 1.0f; // forward
	else if (buffer[bidx].action == 1) cc.y = 1.0f; // backward
	else if (buffer[bidx].action == 2) cc.z = 1.0f; // left
	else if (buffer[bidx].action == 3) cc.w = 1.0f; // right
	else if (buffer[bidx].action == 4) { cc.x = 1.0f; cc.w = 1.0f; } // fprward+right
	else if (buffer[bidx].action == 5) { cc.x = 1.0f; cc.z = 1.0f; }// forward+left
	carcontrol[ci] = cc;

	buffer[bidx].old_logprob = logf(fmaxf(nodevals[caridx(D, action,ci)],1e-8f));





}
__global__ void getlog(int n, int d, int* indices, replaybuffer* buffer, int s, float* nodevals, int nodedatasize) {

	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b >= n)return;
	int off = b * nodedatasize;
	float m = nodevals[off + d];
	for (int j = 1; j < 6; j++) m = fmaxf(m, nodevals[off + d + j]);
	float sum = 0.f, exps[6];
	for (int j = 0; j < 6; j++) { exps[j] = expf(nodevals[off + d + j] - m); sum += exps[j]; }
	for (int j = 0; j < 6; j++) nodevals[off + d + j] = exps[j] / sum;

	int bidx = indices[s * batchsize + b];
	int action = buffer[bidx].action;             
	buffer[bidx].logprob = logf(fmaxf(nodevals[off + d + action],1e-8f));



}
__device__ float d_reward = 0.0f;
__device__ float oldr = 0.0f;
__device__ int id = 0;
__global__ void reward_kernel(int n,int s, replaybuffer* buffer, float4* data1,float4* data2,float4* control, int* alive,ray* rays) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
		int bidx = s * d.numcars + i;
	
		float4 c = __ldg(&data1[i]);
		float alivereward = alive[i] == 1 ? 0.0f : -2.0f;
		if (alive[i] == 2) {
			alivereward = 25.0f;
		}
		float dx = c.x - d.targetx;
		float dy = c.y - d.taregety;
		float curdist = sqrtf(dx * dx + dy * dy);
		
		float forward = (control[i].x > 0.0f) ? 0.1f : -0.05f;
		float prevdist =  data2[i].x;
		float diff = prevdist - curdist;
		float progressreward = 3.0f * (diff / d.raymaxdist);
		data2[i].x = curdist;

		float coneMin = 1.0f;
		int coneIdx[7] = { 13,14,15,0,1,2,3 };
		for (int k = 0; k < 7; k++)
			coneMin = fminf(coneMin, rays[i].len[coneIdx[k]] / d.raymaxdist);
		float danger = 1.0f - coneMin;
		float speedfrac = fabsf(c.w) / d.maxspeed;
		float dangerPenalty = -2.0f * danger * danger * (0.5f + 0.5f * speedfrac); 

		float timepenalty = -0.02f;
	
		float reward = progressreward + dangerPenalty + timepenalty + alivereward + forward;
	
		buffer[bidx].reward = reward;
		if (reward > oldr) {
			if (alive[i] == 1) {
				id = i;
				oldr = reward;
			}
		}
		atomicAdd(&d_reward, reward);
	
}
__device__ void getQvalue(int s,int c, int D, float* nodevals, replaybuffer* buffer) {
	int bidx = s * d.numcars + c;
	buffer[bidx].value = nodevals[caridx(D,0,c)];
	if (!isfinite(buffer[bidx].value)) {
		buffer[bidx].value = 0.0f;
		printf("BAD PPO %d value=%f \n",
					s,
					buffer[bidx].value
				);
	}
}
__global__ void computevalskernel(int cars,int ticks, replaybuffer* buffer) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= cars) return;
	float y = 0.98f;
	for (int t = 0; t < ticks; t++) {
		int i = t * cars + c;
		float val = buffer[i].reward;
		for (int u = t + 1; u < ticks; u++) {
			int j = u * cars + c;
			val += buffer[j].reward * powf(y, u - t);
			if (buffer[j].done) break;
		}
		buffer[i].rtg = val;
		buffer[i].advantage = buffer[i].rtg - buffer[i].value;
		atomicAdd(&MSE, buffer[i].advantage * buffer[i].advantage);
	}
}
__device__ float d_adv_mean;
__device__ float d_adv_var;
__global__ void compute_adv_stats(int n, replaybuffer* buffer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float a = buffer[i].advantage;
	atomicAdd(&d_adv_mean, a);
	atomicAdd(&d_adv_var, a * a);
}
__global__ void normalize_advantages(int n, replaybuffer* buffer, float mean, float std) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	buffer[i].advantage = (buffer[i].advantage - mean) / (std + 1e-8f);
}
__global__ void firstlayerfrozen(int n, int s, int* indices, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals, float* d_preact, replaybuffer* buffer, int nodedatasize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)return;



	float x = bias[idx(0, i)];
	for (int k = 0; k < 24; k++) {
		x += buffer[indices[s * batchsize + b]].s1[k] * weights[idx(0 + i * 24, k)];

	}
	int off = b * nodedatasize + i;
	d_preact[off] = x;
	nodevals[off] = leakyrelu(x);

}
__global__ void solvefrozenlayers(int n, int nin, int w, int D, int b, int p, bool isout, int* indices, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData, float* preact, int nodedatasize) {



	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int B = blockIdx.y;
	if (i >= n)return;


	float val = dBias[idx(b, i)];

	for (int j = 0; j < nin; j++) {
		float v = dNodeData[B * nodedatasize + p + j];
		val += v * dWeights[w + i * nin + j];
	}
	int off = B * nodedatasize + D + i;
	preact[off] = val;
	dNodeData[off] = isout ? val : leakyrelu(val);


}
__global__ void netkernel(int n, const float4* __restrict__ data1, float4* data2, float4* carcontrol, const int* __restrict__ alive,
	const ray* rays, const float* __restrict__ weights,
	const float* __restrict__ bias, float* nodevals, const Layer* layer, bool isactor, replaybuffer* buffer, int s,curandState* rng

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

				float targetinsight = (dist < d.raymaxdist) ? 1.0f : 0.0f;

				float targetForward = (dx * cosf(c.z) + dy * sinf(c.z)) / d.maxdisttotarget;
				float targetSide = (-dx * sinf(c.z) + dy * cosf(c.z)) / d.maxdisttotarget;

				float angle = c.z;


				float input[24] = { sinf(angle), cosf(angle),
												targetForward,targetSide,
												c2.w / d.maxsteer,
												c.w / d.maxspeed,
												targetinsight,
												rays[i].len[0] / d.raymaxdist,
												rays[i].len[1] / d.raymaxdist,
												rays[i].len[2] / d.raymaxdist,
												rays[i].len[3] / d.raymaxdist,
												rays[i].len[4] / d.raymaxdist,
												rays[i].len[5] / d.raymaxdist,
												rays[i].len[6] / d.raymaxdist,
												rays[i].len[7] / d.raymaxdist,
												rays[i].len[8] / d.raymaxdist,
												rays[i].len[9] / d.raymaxdist,
												rays[i].len[10] / d.raymaxdist,
												rays[i].len[11] / d.raymaxdist,
												rays[i].len[12] / d.raymaxdist,
												rays[i].len[13] / d.raymaxdist,
												rays[i].len[14] / d.raymaxdist,
												rays[i].len[15] / d.raymaxdist,
												dist / d.maxdisttotarget
				};
				int insize = 24;
				if (isactor) {
					int bidx = s * d.numcars + i;
					for (int b = 0; b < insize; b++) {
						buffer[bidx].s1[b] = input[b];

					}
				}
				int didx = layer[0].dIdx;

				firstlayer(layer[0].Nout, i, 0, didx, 24, input, weights, bias, nodevals);
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
		if (isactor) {
			getoutput(outd, nodevals, carcontrol, buffer, s, i, rng);
		}
		else {
			getQvalue(s, i, outd, nodevals, buffer);
		}
	}


}




void normalize_advantages() {
	
	float zero = 0.0f;
	cudaMemcpyToSymbol(d_adv_mean, &zero, sizeof(float));
	cudaMemcpyToSymbol(d_adv_var, &zero, sizeof(float));

	int t = 256;
	int b = (settings.replaybuffersize + t - 1) / t;
	compute_adv_stats << <b, t >> > (settings.replaybuffersize, d_state);
	sync();

	float mean = 0.0f, var = 0.0f;
	cudaMemcpyFromSymbol(&mean, d_adv_mean, sizeof(float));
	cudaMemcpyFromSymbol(&var, d_adv_var, sizeof(float));
	mean /= settings.replaybuffersize;
	var = var / settings.replaybuffersize - mean * mean;
	float std = sqrtf(fmaxf(var, 1e-8f));

	normalize_advantages << <b, t >> > (settings.replaybuffersize, d_state, mean, std);
}
void runfrozennet(int s, int curbatch, bool isactor) {
	auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	int layers = (isactor) ? settings.actor_layers : settings.critic_layers;

	int l1blocks = (layerdata[0].Nout + threads - 1) / threads;
	int w = layerdata[0].wIdx;
	dim3 grid(l1blocks, curbatch);

	if (isactor) {
		firstlayerfrozen << <grid, threads >> > (layerdata[0].Nout, s, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, d_state, actor_nodedatasize);
	}
	else {
		firstlayerfrozen << <grid, threads >> > (layerdata[0].Nout, s, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, d_state, critic_nodedatasize);

	}


	for (int l = 1; l < layers; l++)
	{

		int Blocks = (layerdata[l].Nout + threads - 1) / threads;
		int shared_bytes = layerdata[l].Nin * sizeof(float);

		int nin = layerdata[l - 1].Nout;
		int wl = layerdata[l].wIdx;
		int d = layerdata[l].dIdx;
		int b = layerdata[l].bIdx;
		int p = layerdata[l - 1].dIdx;

		bool isout = (l == layers - 1);
		dim3 grid2(Blocks, curbatch);
		if (isactor) {

			solvefrozenlayers << <grid2, threads >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, actor_nodedatasize);
		}
		else {

			solvefrozenlayers << <grid2, threads >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, critic_nodedatasize);
		}



	}
	if (isactor) {
		int outd = layerdata[settings.actor_layers - 1].dIdx;

		getlog << <blocks(curbatch), threads >> > (curbatch, outd, d_indices, d_state, s, d_actor_nodvals, actor_nodedatasize);
	}

}
void reward(int s) {
	reward_kernel << <blocks(settings.cars), threads >> > (settings.cars,s, d_state, data1, data2, carcontrol, alive, rays);
	int x = 0;
	float zero = 0;
	cudaMemcpyFromSymbol(&x, id, sizeof(int));
	cudaMemcpyToSymbol(oldr, &zero, sizeof(float));
	settings.bestcar = x;
	
}
void computevals() {
	
	computevalskernel << <blocks(settings.cars), threads >> > (settings.cars, settings.replaybuffersize / settings.cars, d_state);


}
void backpropogation(int s, int curBatch, bool isactor) {

	auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	int layers = (isactor) ? settings.actor_layers : settings.critic_layers;


	for (int l = layers - 1; l >= 0; l--) {


		int d = layerdata[l].dIdx;
		int n = layerdata[l].Nout;
		int l1out = (l < layers - 1) ? layerdata[l + 1].Nout : 0;
		int l1w = (l < layers - 1) ? layerdata[l + 1].wIdx : 0;
		int l1d = (l < layers - 1) ? layerdata[l + 1].dIdx : 0;
		int l1nin = (l < layers - 1) ? layerdata[l + 1].Nin : 0;



		bool outlayer = (l == layers - 1) ? 1 : 0;

		int blocks = (n + threads - 1) / threads;
		dim3 grid(blocks, curBatch);
		if (isactor) {
			compute_delta << <grid, threads >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_actor_nodvals, d_actor_delta, d_actor_weights, d_actor_preact, d_state, s, d_indices, actor_nodedatasize,settings.rolloutstep);
		}
		else {

			compute_delta << <grid, threads >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_critic_nodvals, d_critic_delta, d_critic_weights, d_critic_preact, d_state, s, d_indices, critic_nodedatasize,settings.rolloutstep);
		}


	}

	for (int l = layers - 1; l >= 0; l--) {
		int blocks = (layerdata[l].Nout + threads - 1) / threads;
		int nin = layerdata[l].Nin;
		int l1d = (l == 0) ? 0 : layerdata[l - 1].dIdx;
		int lw = layerdata[l].wIdx;
		int lsize = (l == 0) ? settings.inputs : layerdata[l].Nin;
		int d = layerdata[l].dIdx;
		int lb = layerdata[l].bIdx;

		if (isactor) {
			tuneweights << <blocks, threads >> > (layerdata[l].Nout, nin, l, l1d, lw, lsize, d, lb, settings.lr,
				d_actor_nodvals, d_actor_weights, d_actor_delta, d_actor_bias, d_state, s, curBatch, actor_nodedatasize, d_indices);
		}
		else {
			tuneweights << <blocks, threads >> > (layerdata[l].Nout, nin, l, l1d, lw, lsize, d, lb, settings.lr,
				d_critic_nodvals, d_critic_weights, d_critic_delta, d_critic_bias, d_state, s, curBatch, critic_nodedatasize, d_indices);
		}



	}

}
void frozennet(bool isactor) {
	//auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	//int layers = (isactor) ? settings.actor_layers : settings.critic_layers;
	int numbatches = (settings.replaybuffersize + hbatchsize - 1) / hbatchsize;
	for (int s = 0; s < numbatches; s++) {
		int curBatch = std::min(hbatchsize, settings.replaybuffersize - s * hbatchsize);

		runfrozennet(s, curBatch, isactor);


		backpropogation(s, curBatch, isactor);


	}
}
void net(int s, bool isactor) {

	if (isactor) {
		netkernel << <blocks(settings.cars), threads >> > (settings.cars, data1, data2, carcontrol, alive, rays, d_actor_weights, d_actor_bias, d_cars_nodevals, d_actlayer, isactor, d_state, s, d_rngstate);
	}
	else {
		netkernel << <blocks(settings.cars), threads >> > (settings.cars, data1, data2, carcontrol, alive, rays, d_critic_weights, d_critic_bias, d_cars_nodevals, d_critlayer, isactor, d_state, s, d_rngstate);

	}

}

void run_network() {

	net(settings.step, true);//actor forward pass
	net(settings.step, false);//critic forward pass
	stepcars();
	reward(settings.step);
	float h_reward = 0.0f;
	float zero = 0.0f;
	steps++;
	if(settings.nextgen){

		cudaError_t err = cudaMemcpyFromSymbol(&h_reward, d_reward, sizeof(float));
		cudaMemcpyToSymbol(d_reward, &zero, sizeof(float));
		rewardgraph.push_back((h_reward / settings.replaybuffersize)/steps);
		if (err != cudaSuccess) {
			printf("!ERROR! reward memcpy to host failed %s \n", cudaGetErrorString(err));
		}
		settings.nextgen = false;
		steps = 0;
	}



	settings.step++;
	if (  settings.step >= settings.replaybuffersize / settings.cars) {
		settings.rolloutstep++;
		computevals();
		shuffleindices();
		frozennet(false);//critic backprop
		normalize_advantages();
		frozennet(true);//actor backprop
		settings.oldmloss = settings.mloss;
		float l = 0.0f;
		cudaMemcpyFromSymbol(&l, MSE, sizeof(float));
		settings.mloss = l/(float)settings.replaybuffersize;
		
		cudaMemcpyToSymbol(MSE, &zero, sizeof(float));

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("rollout error %s\n ", cudaGetErrorString(err));
		}

		
		settings.step = 0;
		
	}
	
}

std::mt19937 shuffleRng(1234);
void shuffleindices() {
	shuffled_indices.resize(settings.replaybuffersize);
	for (int i = 0; i < settings.replaybuffersize; i++) {
		shuffled_indices[i] = i;
	}
	std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), shuffleRng);
	cudaMemcpy(d_indices, shuffled_indices.data(), settings.replaybuffersize * sizeof(int), cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf(" indices copy to gpu failed %s \n", cudaGetErrorString(err));
	}
}

void allocatenetmem() {
	cudaMalloc(&d_actor_weights, actor_weightbuffersize * sizeof(float));
	cudaMalloc(&d_critic_weights, critic_weightbuffersize * sizeof(float));
	cudaMalloc(&d_actor_bias, actor_biassize * sizeof(float));
	cudaMalloc(&d_critic_bias, critic_biassize * sizeof(float));
	cudaMalloc(&d_actor_nodvals, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_nodvals, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_preact, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_preact, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_delta, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_delta, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_state, settings.replaybuffersize * sizeof(replaybuffer));
	cudaMalloc(&carcontrol,settings.cars* sizeof(float4));
	cudaMalloc(&d_indices, settings.replaybuffersize * sizeof(int));
	cudaMalloc(&d_cars_nodevals, carsnodesize * sizeof(float));
	cudaMalloc(&d_actlayer, settings.actor_layers * sizeof(Layer));
	cudaMalloc(&d_critlayer, settings.critic_layers * sizeof(Layer));


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
	cudaMemcpy(d_actlayer, actor_layerdata.data(), settings.actor_layers * sizeof(Layer), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critlayer, critic_layerdata.data(), settings.critic_layers * sizeof(Layer), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network memcpy failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network memcpy done \n");
	}

}
void restart() {
	std::vector<quadvertex2d> tempob;
	tempob.resize(settings.obstacles);
	cudaError_t err= cudaMemcpy(tempob.data(), obstacle, settings.obstacles * sizeof(quadvertex2d), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("obstacle data memcpy to host failed %s \n", cudaGetErrorString(err));
	}
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
	settings.rolloutstep = 0;
	settings.step = 0;
	settings.gen = 0;
	critic_nodedatasize = 0;
	settings.actor_layers = 0;
	settings.critic_layers = 0;
	
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
	err = cudaMemcpy(obstacle, tempob.data(), settings.obstacles * sizeof(quadvertex2d), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("obstacle data memcpy to host failed %s \n", cudaGetErrorString(err));
	}
	tempob.clear();
	printf("obstacles initialized \n");
	setrayexitdist();
	printf("rayexit set \n");


}
void initnetwork() {
	settings.replaybuffersize -= settings.replaybuffersize % settings.cars;
	initlayers();

	initWB(-0.05f, 0.05f);
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
	cudaFree(d_indices);
	cudaFree(car);
	cudaFree(d_cars_nodevals);
	cudaFree(d_actlayer);
	cudaFree(d_critlayer);

	

}
