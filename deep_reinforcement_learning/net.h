#pragma once

extern "C" {
	void allocate();
	void initnetwork();
	void setmaxdisttotarget();
	
	void copywbtogpu();
	void runnet();
	void cudafree();
	void restart();
}