#include <cuda_runtime.h>
#include <iostream>
#include"data.h"
#include "net.h"

extern "C" void allocate(){
	int n = settings.cars;
	cudaMalloc(&data1, n * sizeof(float4));
	cudaMalloc(&data2, n * sizeof(float4));
	cudaMalloc(&rays, n * sizeof(ray));
	cudaMalloc(&segments, settings.max_obstacles * 4 * sizeof(float4));

	cudaMalloc(&alive, n * sizeof(int));
	printf("data allocated \n");
}


