#pragma once


extern "C" {
	void registerObstaclesVbo();
	void drawdummy();
	void initobstacles();
	void drawobstacles();
	void addobstacle(float x, float y, float rot, float width, float height);
	void resetob();
	void bake_segments();
	void checkcolison();
	void copyobsdata();
	void checkstate();
	void setrayexitdist();
	void unregisterobs();
	void clearvectors();
	void randomobs();
}