#include <iostream>
#include "norender.h"
#include<GLFW/glfw3.h>


int main() {
	noRender.createWindow(1800, 900, "car learns to drive", 0);
	noRender.setup2d();

	while (noRender.WindowOpen()) {
		noRender.processinputs();
		noRender.clearscreen(0.1f, 0.1f, 0.1f);



		noRender.swapbuffers();

	}
	noRender.closeWindow();

	return 0;

}