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
	float ray_max_len = 75.0f;
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

	

	int gen = 1;
	int fpsCount = 0;
	int cars =1 ;
	int rays = 6;
	int layers = 0;
	int obstacles = 0;
	int max_obstacles = 128;
	bool addingobstacle = false;
	bool alive = true;

	
};

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
inline float ray_angles[6] = {
	0.7853982f,
	0.0f,
	-0.7853982f,
	3.1415927f,
	2.3561945f,
	-2.3561945f
};
//ray struct
struct ray {
	float len[6];
};


inline quadvertex2d* car =nullptr;//allocated in draw.cu
inline linepoint2d* Rays = nullptr;
inline quadvertex2d* obstacle = nullptr;

inline float4* data1=nullptr;//posx,posy,heading,speed
inline float4* data2 = nullptr;//throttle,steer,survival time,current steer
inline ray* rays = nullptr;//rays len[6]
inline int* alive = nullptr;//alive checker
inline float4* segments = nullptr;

inline float* d_weights = nullptr;
inline float* d_bias = nullptr;
inline float* d_nodvals = nullptr;


inline int threads = 256;
inline int blocks(int n) {
	return (n + threads - 1) / threads;
};

inline std::vector<Layer> layerdata;
inline std::vector<float>weights;
inline std::vector<float>bias;
inline std::vector<quadvertex2d> h_obdata;

