cd "D:\visual_studio\deep_reinforcement_learning"
nvcc -std=c++17 -Xcompiler "/std:c++17 /MD" -o carLearns.exe deep_reinforcement_learning\engine.cpp deep_reinforcement_learning\draw.cu deep_reinforcement_learning\network.cu deep_reinforcement_learning\ui.cpp deep_reinforcement_learning\obstacles.cu D:\glad\src\glad.c ^
  "D:\visual_studio\imgui-1.91.1\imgui.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_draw.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_tables.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_widgets.cpp" ^
  "D:\visual_studio\imgui-1.91.1\backends\imgui_impl_glfw.cpp" ^
  "D:\visual_studio\imgui-1.91.1\backends\imgui_impl_opengl3.cpp" ^
  -I"D:\visual_studio\noRender\noRender" ^
  -I"D:\glad\include" ^
  -I"D:\visual_studio\glfw-3.4.bin.WIN64\include" ^
  -I"D:\visual_studio\imgui-1.91.1" ^
  -I"D:\visual_studio\imgui-1.91.1\backends" ^
  -L"D:\visual_studio\noRender\x64\Release" ^
  -L"D:\visual_studio\glfw-3.4.bin.WIN64\lib-vc2022" ^
  -lnoRender -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 ^
  -Xlinker "/LTCG"
echo done
pause