#include <iostream>
#include "norender.h"
#include<GLFW/glfw3.h>
#include "data.h"
#include "draw.h"
#include "net.h"
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "ui.h"
#include "imgui_impl_opengl3.h"
#include "obstacles.h"
#include <chrono>


	

int main() {
	noRender.createWindow(1800, 900, "car learns to drive", 0);
	noRender.setup2d();

	
	//imgui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(noRender.getwindowid(), true);
	const char* glsl_version = "#version 330";
	ImGui_ImplOpenGL3_Init(glsl_version);
	
	registervbo();
	registerObstaclesVbo();
	registerrayvbo();
	initnetwork();
	initcars();
	initobstacles();
	setrayexitdist();

	double lastTime = glfwGetTime();
	double fpsClock = lastTime;
	while (noRender.WindowOpen()) {
		
		double now = glfwGetTime();
		double frameTime = now - lastTime;
		lastTime = now;
		
		//settings.timer += frameTime;
		
		noRender.processinputs();
		noRender.clearscreen(0.1f, 0.1f, 0.1f);
		render2d.drawcircle(settings.targetx, settings.targety, 1.f, 1.f, 0.f, settings.targetsize);
		

		settings.accumulator += frameTime * settings.trainspeed;
		
		while (settings.accumulator >= settings.dt)
		{
			stepcars();
			settings.step++;
			settings.stepstotal++;

			settings.timer += settings.dt;
			//checkcolison();
			//checkstate();

			settings.accumulator -= settings.dt;
		}

		draw();

		if (settings.timer >= settings.gentime || !settings.alive ) {
			settings.timer = 0.0f;
			settings.gen++;
			settings.alive = true;
			
			restartgeneration();
		}

		noRender.swapbuffers();

		double elapsed = now - fpsClock;
		fpsClock = now;
		settings.fps = (elapsed > 0.0) ? 1.0 / elapsed : settings.fps;
		settings.fpsTimer += (float)elapsed;
		settings.fpsCount++;
		
		if (settings.fpsTimer >= 0.5f) {
			settings.avgFps = settings.fpsCount / settings.fpsTimer;
			settings.fpsTimer = 0.f;
			settings.fpsCount = 0;
		}



	}
	unregister();
	unregisterobs();
	clearvectors();
	cudafree();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	noRender.closeWindow();
	//std::cin.get();
	return 0;

}