#pragma once

extern "C" {
	void allocate();
	void initnetwork();
	void setmaxdisttotarget();
	void computefitness();
	void getbestcaridx();
	void copywbtogpu();
	void runnet();
	void cudafree();
	void restart();
}