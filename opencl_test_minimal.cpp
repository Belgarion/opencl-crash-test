#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_device_id device_id[10];

typedef struct __trigg_opencl_ctx {
	uint8_t curr_seed[16], next_seed[16];
	char cp[256], *next_cp;
	int *found;
	cl_mem d_map;
	cl_mem d_phash;
	cl_mem d_found;
	uint8_t *seed;
	cl_mem d_seed;
	uint32_t *midstate, *input;
	cl_mem d_midstate256, d_input32, d_blockNumber8;
	cl_context context;
	cl_command_queue cq;
	cl_kernel k_peach;
	cl_kernel k_test;
	cl_event trigg_event;
} TriggCLCTX;

/* Max 64 GPUs Supported */
static TriggCLCTX ctx[64] = {};
//static int thrds = 3;
//static size_t threads = thrds * 1024 * 1024;
static size_t threads = 512*256;
static size_t block = 256;
//static size_t threads = 1;
//static size_t block = 1;

cl_program opencl_compile_source(cl_context context, uint8_t num_devices, cl_device_id *devices, const char *filename, const char *options) {
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	char *src = (char*)malloc(102400);
	size_t dwSize = fread( src, 1, 102400, fp);
	fclose( fp );

	const char *srcptr = src;
	size_t srcsize = dwSize;

	cl_int err;
	cl_program prog = clCreateProgramWithSource(context, 1, &srcptr, &srcsize, &err);
	if (CL_SUCCESS != err) {
		printf("clCreateProgramWithSource failed. Error: %d\n", err);
		exit(1);
	}
	err = clCompileProgram(prog, num_devices, devices, options, 0, NULL, NULL, NULL, NULL);
	if (CL_SUCCESS != err) {
		printf("clCompileProgram failed. Error: %d\n", err);
		if (err == CL_COMPILE_PROGRAM_FAILURE) {
			// Determine the size of the log
			size_t log_size;
			clGetProgramBuildInfo(prog, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			// Allocate memory for the log
			char *log = (char *)malloc(log_size);

			// Get the log
			clGetProgramBuildInfo(prog, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

			// Print the log
			printf("%s\n", log);
			free(log);
		}
		exit(1);
	}

	free(src);

	return prog;
}

int count_devices_cl() {
	cl_int err;
	cl_uint num_platforms;

	err = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if (CL_SUCCESS != err) {
		printf("clGetPlatformIDs failed. Error: %d\n", err);
		return 0;
	}
	if (num_platforms == 0) {
		printf("No OpenCL platforms detected.\n");
		return 0;
	}
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 10, device_id, &num_devices);
	if (CL_SUCCESS != err) {
		printf("clGetDeviceIDs failed. Error: %d\n", err);
		return 0;
	}
	if (num_devices == 0) {
		printf("No OpenCL devices detected.\n");
		return 0;
	}

	printf("OpenCL: Found %d platforms and %d devices\n", num_platforms, num_devices);

	for (uint32_t i = 0; i < num_devices; i++) {
		size_t len = 0;
		char name[128], vendor[128];
		cl_ulong mem_size = 0;
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, 128, name, &len);
		if (CL_SUCCESS != err) {
			printf("clGetDeviceInfo failed. Error: %d\n", err);
			continue;
		}
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, 128, vendor, &len);
		if (CL_SUCCESS != err) {
			printf("clGetDeviceInfo failed. Error: %d\n", err);
			continue;
		}
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		if (CL_SUCCESS != err) {
			printf("clGetDeviceInfo failed. Error: %d\n", err);
			continue;
		}
		name[127] = '\0';
		vendor[127] = '\0';
		printf("Device %d: %s %s %u MB\n", i, vendor, name, (unsigned int)(mem_size / 1024 / 1024));
	}

	return num_devices;
}

int peach_init_cl(uint8_t  difficulty, uint8_t *blockNumber) {
	count_devices_cl();
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };

	for (cl_uint i = 0; i < num_devices; i++) {
		cl_int err;
		ctx[i].context = clCreateContext(properties, 1, &(device_id[i]), NULL, NULL, &err);
		if (CL_SUCCESS != err) {
			printf("clCreateContext failed. Error: %d\n", err);
			exit(1);
		}
		ctx[i].cq = clCreateCommandQueue(ctx[i].context, device_id[i], 0, &err);
		if (CL_SUCCESS != err) {
			printf("clCreateCommandQueue failed. Error: %d\n", err);
			exit(1);
		}

		printf("Building with -O0\n");
		cl_program prog_minimal = opencl_compile_source(ctx[i].context, 1, &device_id[i], "cl_minimal.cl", "-cl-fp32-correctly-rounded-divide-sqrt -O0");
		cl_program prog_parts[] = {prog_minimal};
		cl_program prog = clLinkProgram(ctx[i].context, 1, &device_id[i], NULL, 1, prog_parts, NULL, NULL, &err);
		if (CL_SUCCESS != err) {
			printf("clLinkProgram failed. Error: %d\n", err);
			exit(1);
		}
		printf("Build with -O0 successful\n");

		printf("Building with -O1\n");
		prog_minimal = opencl_compile_source(ctx[i].context, 1, &device_id[i], "cl_minimal.cl", "-cl-fp32-correctly-rounded-divide-sqrt -O1");
		prog_parts[0] = prog_minimal;
		prog = clLinkProgram(ctx[i].context, 1, &device_id[i], NULL, 1, prog_parts, NULL, NULL, &err);
		if (CL_SUCCESS != err) {
			printf("clLinkProgram failed. Error: %d\n", err);
			exit(1);
		}
		printf("Build with -O1 successful\n");

		ctx[i].k_test = clCreateKernel(prog, "test", &err);
		if (CL_SUCCESS != err) {
			printf("clCreateKernel failed. Error: %d\n", err);
			exit(1);
		}

		printf("Running test\n");
		//size_t build_map_work_size = 4096*256;
		//size_t build_map_local_size = 256;
		size_t build_map_work_size = 1;
		size_t build_map_local_size = 1;
		err = clEnqueueNDRangeKernel(ctx[i].cq, ctx[i].k_test, 1, NULL, &build_map_work_size, &build_map_local_size, 0, NULL, &ctx[i].trigg_event);
		if (CL_SUCCESS != err) {
			printf("%s:%d: clEnqueueNDRangeKernel failed. Error: %d\n", __FILE__, __LINE__, err);
		}
		err = clFinish(ctx[i].cq);
		if (CL_SUCCESS != err) {
			printf("clFinish failed. Error: %d\n", err);
		}
		printf("Test complete\n");
	}

	return num_devices;
}

int main() {
	uint64_t blockNumber = 1;
	uint8_t *bnum = (uint8_t*)&blockNumber;
	uint8_t diff = 18;
	peach_init_cl(diff, bnum);
}

