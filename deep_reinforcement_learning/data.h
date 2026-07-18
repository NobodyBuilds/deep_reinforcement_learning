#pragma once
#include<iostream>
#include <vector>
#include <norender.h>
#include <cuda_runtime.h>


struct param {
	//car setup data
	float spawnx = 200.0f;
	float spawny = 450.0f;
	float carwidth = 15.0f;
	float carheight = 7.50f;
	float car_r = 1.0f, car_b = 0.0f, car_g = 0.0f;
	float targetx = 1600.0f, targety = 450.0f;
	float targetsize = 20.0f;
	float dumx = 0.0, dumy = 0.0f;
	float ray_max_len = 100.0f;

	float obx = 900.0f, oby = 450.0f;
	float obrot = 0.0f;
	float obwidth = 50.0f;
	float obheight = 50.0f;
	int cars = 5;
	int rays = 6;
	int dupecars = 5;
	int obstacles = 0;
	int max_obstacles = 64;
	bool addingobstacle = false;
};

inline param settings;

//constant data
struct dparams {
	float wheelbase=20.0f;
	float max_steer = 90.0f;;
	float max_speed = 50.0f;;
	float max_raylen=settings.ray_max_len;
	float accel_rate = 15.0f;;
	float steer_rate=10.0f;
	float brake_rate=15.0f;
	float drag=0.01f;
	float friction=0.01f;
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
inline float dt = 1.0f / 120.0f;
inline quadvertex2d* car =nullptr;//allocated in draw.cu
inline linepoint2d* Rays = nullptr;
inline quadvertex2d* obstacle = nullptr;

inline float4* data1=nullptr;//posx,posy,heading,speed
inline float4* data2 = nullptr;//throttle,steer,fitness,current steer
inline ray* rays = nullptr;//rays len[6]
inline int* alive = nullptr;//alive checker

inline int threads = 256;
inline int blocks(int n) {
	return (n + threads - 1) / threads;
}
