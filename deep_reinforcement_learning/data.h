#pragma once
#include<iostream>
#include <vector>
#include <norender.h>
#include <cuda_runtime.h>

 struct Layer {
	int Nin;
	int Nout;
	int wIdx;
	int dIdx;
	int bIdx;

};

struct param {
	//car setup data
	
	float dt = 1.0f / 120.0f;
	float spawnx = 200.0f;
	float spawny = 450.0f;
	float carwidth = 25.0f;
	float carheight = 12.0f;
	float car_r = 1.0f, car_b = 0.0f, car_g = 0.0f;
	float targetx = 1600.0f, targety = 450.0f;
	float targetsize = 40.0f;
	float dumx = 0.0, dumy = 0.0f;
	float ray_max_len = 200.0f;
	float maxdisttotarget = 0.0f;
	float maxdisttospawn = 0.0f;
	float fps = 0.0f;
	float avgFps = 0.0f;
	float fpsTimer = 0.0f;
	float ms = 0.0f;
	float timer = 0.0f;
	float accumulator = 0.0f;
	float effectiveDt = 0.0f;
	float obx = 900.0f, oby = 450.0f;
	float obrot = 0.0f;
	float obwidth = 50.0f;
	float obheight = 50.0f;
	float trainspeed = 1.0f;
	float gentime = 60.0f;
	double mloss = 0.0f;
	double oldmloss = 0.0f;
	float rollout_time = 0.0f;

	float lr = 0.001f;
	int bestcar = 0;
	int inputs = 32;
	int step = 0;
	int rolloutstep = 0;
	//int c = 1000;
	int gen = 1;
	int fpsCount = 0;
	int cars =64 ;
	int replaybuffersize = cars * 2048;
	int samplecar = cars;
	int rays = 24;
	int actor_layers = 0;
	int critic_layers = 0;
	int obstacles = 0;
	int max_obstacles = 256;
	int random_obstacles_count = 24;
	bool addingobstacle = false;
	bool alive = true;
	bool nextgen = false;
	bool randomobstacles = true;
	bool training = true;
	bool render_rays = false;
	
};
 
inline int hbatchsize = 128;
inline int randomobshold = 0;
inline int steps = 0;
inline int adam_step = 1;
inline float reach_reward = 25.0f;
inline float crash_penality = 10.0f;
inline float diff_change_reward = 1.5f;
inline float danger_penalty = 0.50f;
inline param settings;

//constant data
struct dparams {
	float wheelbase=20.0f;
	float max_steer = 35.0f;;
	float max_speed = 400.0f;;
	float max_raylen=settings.ray_max_len;
	float accel_rate = 200.0f;;
	float steer_rate=360.0f;
	float brake_rate=200.0f;
	float drag=0.002f;
	float friction=0.03f;
	float carwidth=settings.carwidth;
	float carheight=settings.carheight;
};

inline dparams dparam;
//ray predefined angles from center
inline float ray_angles[24] = {
	 0.0f,
	 0.2617994f,   // 15°
	 0.5235988f,   // 30°
	 0.7853982f,   // 45°
	 1.0471976f,   // 60°
	 1.3089969f,   // 75°
	 1.5707963f,   // 90°
	 1.8325957f,   // 105°
	 2.0943951f,   // 120°
	 2.3561945f,   // 135°
	 2.6179939f,   // 150°
	 2.8797933f,   // 165°
	 3.1415927f,   // 180°
	-2.8797933f,   // -165°
	-2.6179939f,   // -150°
	-2.3561945f,   // -135°
	-2.0943951f,   // -120°
	-1.8325957f,   // -105°
	-1.5707963f,   // -90°
	-1.3089969f,   // -75°
	-1.0471976f,   // -60°
	-0.7853982f,   // -45°
	-0.5235988f,   // -30°
	-0.2617994f    // -15°
};
//ray struct
struct ray {
	float len[24];
};


inline quadvertex2d* car =nullptr;//allocated in draw.cu
inline linepoint2d* Rays = nullptr;
inline quadvertex2d* obstacle = nullptr;

inline float4* data1=nullptr;//posx,posy,heading,speed
inline float4* data2 = nullptr;//0,0,survival time,current steer
inline ray* rays = nullptr;//rays len[6]
inline int* alive = nullptr;//alive checker
inline float4* segments = nullptr;
inline float4* carcontrol = nullptr;//forward,backword,left,right

inline float* d_actor_weights = nullptr;
inline float* d_critic_weights = nullptr;
inline float* d_actor_bias = nullptr;
inline float* d_critic_bias = nullptr;
inline float* d_actor_nodvals = nullptr;
inline float* d_critic_nodvals = nullptr;
inline float* d_cars_nodevals = nullptr;
inline float* d_actor_preact = nullptr;
inline float* d_critic_preact = nullptr;
inline float* d_actor_delta = nullptr;
inline float* d_critic_delta = nullptr;
inline int* d_indices = nullptr;
inline Layer* d_actlayer = nullptr;
inline Layer* d_critlayer = nullptr;
inline float2* actor_adam_weights = nullptr;//x== m,y==v
inline float2* actor_adam_bias = nullptr;//x== m,y==v
inline float2* critic_adam_weights = nullptr;
inline float2* critic_adam_bias = nullptr;


struct replaybuffer {
	float s1[32];
	
	float reward;
	float logprob;
	float old_logprob;
	float value;
	float rtg;
	float advantage;
	int action;
	
	bool done;
};
struct h_float2 {
	float x;
	float y;
};


inline replaybuffer* d_state = nullptr;


inline int threads = 256;
inline int blocks(int n) {
	return (n + threads - 1) / threads;
};

inline std::vector<Layer> actor_layerdata;
inline std::vector<Layer> critic_layerdata;
inline std::vector<float>actor_weights;
inline std::vector<float>critic_weights;
inline std::vector<float>actor_bias;
inline std::vector<float>critic_bias;
inline std::vector<int> shuffled_indices;

inline std::vector<quadvertex2d> h_obdata;
inline std::vector<float>rewardgraph;

