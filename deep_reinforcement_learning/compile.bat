cd "D:\visual_studio\deep_reinforcement_learning"

nvcc -std=c++17 -Xcompiler "/std:c++17 /MD" -o carLearns.exe deep_reinforcement_learning\engine.cpp D:\glad\src\glad.c ^
  -I"D:\visual_studio\noRender\noRender" ^
  -I"D:\glad\include" ^
  -I"D:\visual_studio\glfw-3.4.bin.WIN64\include" ^
  -L"D:\visual_studio\noRender\x64\Release" ^
  -L"D:\visual_studio\glfw-3.4.bin.WIN64\lib-vc2022" ^
  -lnoRender -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 ^
  -Xlinker "/LTCG"
echo done
pause