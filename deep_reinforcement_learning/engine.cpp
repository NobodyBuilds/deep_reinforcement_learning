#include <iostream>
#include "norender.h"
#include<GLFW/glfw3.h>
#include "data.h"
#include "draw.h"
#include "net.h"

int main() {
	noRender.createWindow(1800, 900, "car learns to drive", 0);
	noRender.setup2d();
	
	
	registervbo();
	registerrayvbo();
	allocate(5);
	initcars(5);
	

	while (noRender.WindowOpen()) {
		noRender.processinputs();
		noRender.clearscreen(0.1f, 0.1f, 0.1f);
		render2d.drawcircle(1600, 450, 1.f, 1.f, 0.f, 35);
		
		drawcars();
		drawrays();
		noRender.swapbuffers();

	}
	noRender.closeWindow();

	return 0;

}