test: opencl_test_minimal.cpp
	-rm opencl_test_minimal
	g++ -I/opt/amdgpu-pro/include opencl_test_minimal.cpp -o opencl_test_minimal -L/opt/amdgpu-pro/lib/x86_64-linux-gnu/ -lOpenCL -g3
	./opencl_test_minimal
