#pragma once

extern "C" {
	void allocate();
	void initnetwork();
	void setmaxdisttotarget();
	
	void copywbtogpu();
	void runnet();
	void fnet();
	void cudafree();
	void restart();
	void reward(int s);
}