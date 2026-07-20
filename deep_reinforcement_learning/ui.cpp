#include <iostream>
#include<imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ui.h"
#include "data.h"
#include "obstacles.h"
extern "C" void ui_draw() {
  
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("settings");
    ImGui::Text("fps: %3f  time: %3f", settings.avgFps, settings.timer);
    ImGui::Spacing();
    ImGui::Text("Gen: %d ", settings.gen);
    ImGui::Text("target");
    ImGui::DragFloat("pos x %f",&settings.targetx, 1.f, 0.0f, 5000.0f);
    ImGui::DragFloat("pos y %f",&settings.targety, 1.f, 0.0f, 5000.0f);
    ImGui::DragFloat("size %f",&settings.targetsize, 0.1f, 0.0f, 5000.0f);
    ImGui::Spacing();
    ImGui::DragFloat("car x %f",&settings.dumx, 0.01f, -1.0f, 1.0f);
    ImGui::DragFloat("car y %f",&settings.dumy, 0.01f, -1.0f, 1.0f);
    ImGui::DragFloat("steer %f ", &dparam.steer_rate, 0.10f, 0.0f, 360.0f);
    ImGui::Spacing();
    ImGui::Checkbox("add obstacle", &settings.addingobstacle);
    if (settings.addingobstacle) {
        ImGui::DragFloat("x", &settings.obx,1.0f,0.0f,2000.0f);
        ImGui::DragFloat("y", &settings.oby,1.0f,0.0f,2000.0f);
        ImGui::DragFloat("width", &settings.obwidth,1.f,0.0f,2000.0f);
        ImGui::DragFloat("height", &settings.obheight,1.f,0.0f,2000.0f);
        ImGui::DragFloat("rotation", &settings.obrot,.1f,-361.0f,361.0f);
        if (settings.obrot > 360.0f) settings.obrot = 0.0f;
        if (settings.obrot < -360.0f) settings.obrot = 0.0f;
        ImGui::Spacing();
        if (ImGui::Button("add")) {
            addobstacle(settings.obx, settings.oby, settings.obrot, settings.obwidth, settings.obheight);
            resetob();
            
            settings.addingobstacle = false;
        }

        }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}