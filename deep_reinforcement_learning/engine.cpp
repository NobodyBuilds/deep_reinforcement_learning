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
	allocate();
	initcars();
	initobstacles();

	while (noRender.WindowOpen()) {

		noRender.processinputs();
		noRender.clearscreen(0.1f, 0.1f, 0.1f);
		render2d.drawcircle(settings.targetx, settings.targety, 1.f, 1.f, 0.f, settings.targetsize);
		stepcars();
		
		draw();

		noRender.swapbuffers();

	}


	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	noRender.closeWindow();

	return 0;

}