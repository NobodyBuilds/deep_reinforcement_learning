#pragma once

extern "C" {
	void allocate();
	void initnetwork();
	void setmaxdisttotarget();
	
	void copywbtogpu();
	void shuffleindices();
	void cudafree();
	void restart();
	void reward(int s);
	void run_network();
}