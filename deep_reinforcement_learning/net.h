#pragma once

extern "C" {
	void allocate();
	void initnetwork();
	void setmaxdisttotarget();
	void geterror(const std::string& label, cudaError_t err);
	void copywbtogpu();
	void shuffleindices();
	void cudafree();
	void restart();
	void reward(int s);
	void run_network();
	void save_weights();
}