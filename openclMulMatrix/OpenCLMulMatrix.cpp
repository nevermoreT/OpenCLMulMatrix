#include <iostream>
#include <CL/cl.h>
#include <cassert>
#include <windows.h>
#include <ctime>
#pragma warning( disable : 4996 )
using namespace std;


#define M 2048	
#define P 2048
#define N 2048

void RunAsCpu(
	const float *A,
	const float *B,
	float* C)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i*N + j] = 0.0;
			for (int k = 0; k < P; k++)
			{
				C[i*N + j] += A[i*P + k] * B[k*N + j];
			}
		}
	}
}

//��ʱ����
double time_stamp()
{
	LARGE_INTEGER curclock;
	LARGE_INTEGER freq;
	if (
		!QueryPerformanceCounter(&curclock) ||
		!QueryPerformanceFrequency(&freq)
		)
	{
		return -1;
	}

	return double(curclock.QuadPart) / freq.QuadPart;
}
#define OPENCL_CHECK_ERRORS(ERR)        \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    exit(1);                               \
    }
int main(int argc, const char** argv)
{
	cl_int error = 0;   // Used to handle error codes
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;

	// ����ϵͳ������OpenCLƽ̨
	cl_uint num_of_platforms = 0;
	// �õ�ƽ̨��Ŀ
	error = clGetPlatformIDs(0, 0, &num_of_platforms);
	OPENCL_CHECK_ERRORS(error);
	cout << "����ƽ̨��: " << num_of_platforms << endl;

	cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
	// �õ�����ƽ̨��ID
	error = clGetPlatformIDs(num_of_platforms, platforms, 0);
	OPENCL_CHECK_ERRORS(error);
	//����ƽ̨��ѡ��һ��Intelƽ̨��
	cl_uint selected_platform_index = num_of_platforms;
	for (cl_uint i = 0; i < num_of_platforms; ++i)
	{
		size_t platform_name_length = 0;
		error = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
		);
		OPENCL_CHECK_ERRORS(error);

		// �������Σ���һ���ǵõ����Ƶĳ���
		char* platform_name = new char[platform_name_length];
		error = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			platform_name_length,
			platform_name,
			0
		);
		OPENCL_CHECK_ERRORS(error);

		cout << "    [" << i << "] " << platform_name;

		if (
			strstr(platform_name, "Intel") &&
			selected_platform_index == num_of_platforms // have not selected yet
			)
		{
			cout << " [Selected]";
			selected_platform_index = i;
		}

		cout << endl;
		delete[] platform_name;
	}
	if (selected_platform_index == num_of_platforms)
	{
		cerr
			<< "û���ҵ�Intelƽ̨\n";
		return 1;
	}
	// Device
	cl_platform_id platform = platforms[1];
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	OPENCL_CHECK_ERRORS(error)

		// Context
		context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	OPENCL_CHECK_ERRORS(error)

		// Command-queue CL_QUEUE_PROFILING_ENABLE�������ܼ�ʱ
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	OPENCL_CHECK_ERRORS(error)

		//�����ʼ����������(��������)
	float* A_h = new float[M*P];
	float* B_h = new float[P*N];
	float* C_h = new float[M*N];
	//srand((unsigned)time(NULL));
	srand(100);
	for (int i = 0; i < M*P; i++)
		A_h[i] = rand() % 50;

	for (int i = 0; i < P*N; i++)
		B_h[i] = rand() % 50;
	//��ʼ���豸����
	// ��־λ��ʾ����ֻ�������Ҵ�nums1_h��nums2_h��������
	cl_mem A_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*M*P, A_h, &error);
	OPENCL_CHECK_ERRORS(error)
		cl_mem B_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*P*N, B_h, &error);
	OPENCL_CHECK_ERRORS(error)
		cl_mem C_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*M*N, NULL, &error);
	OPENCL_CHECK_ERRORS(error)

	cout << "CPU ���п�ʼ:" << time_stamp() << endl;
	//RunAsCpu(A_h, B_h, C_h);
	cout << "CPU ���н���:" << time_stamp() << endl;

		//��ȡOpenCLSum.cl�ļ�����

	FILE* fp = fopen("OpenCLMulMatrix.cl", "rb");
	fseek(fp, 0, SEEK_END);
	size_t src_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	const char* source = new char[src_size];
	fread((void*)source, 1, src_size, fp);
	fclose(fp);

	//������������kernel����
	cl_program program = clCreateProgramWithSource(context, 1, &source, &src_size, &error);
	OPENCL_CHECK_ERRORS(error)
		delete[] source;

	// Builds the program
	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	//OPENCL_CHECK_ERRORS(error)

		// Shows the log
		char* build_log;
	size_t log_size;
	// First call to know the proper size
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size + 1];
	// Second call to get the log
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	cout << build_log << endl;
	delete[] build_log;
	OPENCL_CHECK_ERRORS(error)
	// Extracting the kernel
	cl_kernel run_as_gpu_1 = clCreateKernel(program, "RunAsGpu_1", &error);
	OPENCL_CHECK_ERRORS(error)
	//����kernel����
	cl_int M_d = M;
	cl_int P_d = P;
	cl_int N_d = N;
	error = clSetKernelArg(run_as_gpu_1, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_1, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_1, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_1, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_1, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_1, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

		// ����kernel
	size_t globalws_1[2] = { M,N };
	cl_event ev;
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_1, 2, NULL, globalws_1, NULL, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
		//����kerenlִ��ʱ�� 
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	cl_ulong kernelExecTimeNs = endTime - startTime;
	printf("Gpu_1����ʱ�� :%8.6f ms\n", kernelExecTimeNs*1e-6);

		//ȡ��kernel����ֵ
	float* gpu_C_1 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N*sizeof(float), gpu_C_1, 0, NULL, NULL);
	//assert(memcmp(C_h, gpu_C_1, M*N * sizeof(float)) == 0);


	// Extracting the kernel
	cl_kernel run_as_gpu_2 = clCreateKernel(program, "RunAsGpu_2", &error);
	OPENCL_CHECK_ERRORS(error)
		//����kernel����
	error = clSetKernelArg(run_as_gpu_2, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_2, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_2, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_2, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_2, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_2, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

		// ����kernel
		size_t globalws_2[2] = { N,M };
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_2, 2, NULL, globalws_2, NULL, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
		//����kerenlִ��ʱ�� 
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	kernelExecTimeNs = endTime - startTime;
	printf("Gpu_2����ʱ�� :%8.6f ms\n", kernelExecTimeNs*1e-6);
		//ȡ��kernel����ֵ
	float* gpu_C_2 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N * sizeof(float), gpu_C_2, 0, NULL, NULL);

	//assert(memcmp(C_h, gpu_C_2, M*N * sizeof(float)) == 0);
	
	// Extracting the kernel
	cl_kernel run_as_gpu_3 = clCreateKernel(program, "RunAsGpu_3", &error);
	OPENCL_CHECK_ERRORS(error)
		//����kernel����
	error = clSetKernelArg(run_as_gpu_3, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_3, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_3, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_3, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_3, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_3, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

		// ����kernel
		size_t globalws_3[2] = { M,N };
		size_t localsz_3[2] = { 16,16 };
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_3, 2, NULL, globalws_3, localsz_3, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
		//����kerenlִ��ʱ�� 
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	kernelExecTimeNs = endTime - startTime;
	printf("Gpu_3����ʱ�� :%8.6f ms\n", kernelExecTimeNs*1e-6);
		//ȡ��kernel����ֵ
	float* gpu_C_3 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N * sizeof(float), gpu_C_3, 0, NULL, NULL);
	//assert(memcmp(C_h, gpu_C_3, M*N * sizeof(float)) == 0);

	// Extracting the kernel
	cl_kernel run_as_gpu_4 = clCreateKernel(program, "RunAsGpu_4", &error);
	OPENCL_CHECK_ERRORS(error)
		//����kernel����
		error = clSetKernelArg(run_as_gpu_4, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_4, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_4, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_4, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_4, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_4, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

		// ����kernel
		size_t globalws_4[2] = { M/4,N };
	
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_4, 2, NULL, globalws_4, NULL, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
		//����kerenlִ��ʱ�� 
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
			sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	kernelExecTimeNs = endTime - startTime;
	printf("Gpu_4����ʱ�� :%8.6f ms\n", kernelExecTimeNs*1e-6);
	//ȡ��kernel����ֵ
	float* gpu_C_4 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N * sizeof(float), gpu_C_4, 0, NULL, NULL);
	//assert(memcmp(C_h, gpu_C_4, M*N * sizeof(float)) == 0);


	//---------------------------------------------------------
	// Extracting the kernel 5
	cl_kernel run_as_gpu_5 = clCreateKernel(program, "RunAsGpu_5", &error);
	OPENCL_CHECK_ERRORS(error)
		//����kernel����
		error = clSetKernelArg(run_as_gpu_5, 0, sizeof(cl_mem), &A_d);
	error |= clSetKernelArg(run_as_gpu_5, 1, sizeof(cl_mem), &B_d);
	error |= clSetKernelArg(run_as_gpu_5, 2, sizeof(int), &M_d);
	error |= clSetKernelArg(run_as_gpu_5, 3, sizeof(int), &N_d);
	error |= clSetKernelArg(run_as_gpu_5, 4, sizeof(int), &P_d);
	error |= clSetKernelArg(run_as_gpu_5, 5, sizeof(cl_mem), &C_d);
	OPENCL_CHECK_ERRORS(error)

		// ����kernel
		size_t globalws_5[2] = { M/4,N/4 };
	size_t localsz_5[2] = { 16,16 };
	error = clEnqueueNDRangeKernel(queue, run_as_gpu_5, 2, NULL, globalws_5, localsz_5, 0, NULL, &ev);
	clFinish(queue);
	OPENCL_CHECK_ERRORS(error)
		//����kerenlִ��ʱ�� 
		clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
			sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &endTime, NULL);
	kernelExecTimeNs = endTime - startTime;
	printf("Gpu_5����ʱ�� :%8.6f ms\n", kernelExecTimeNs*1e-6);
	//ȡ��kernel����ֵ
	float* gpu_C_5 = new float[M*N];
	clEnqueueReadBuffer(queue, C_d, CL_TRUE, 0, M*N * sizeof(float), gpu_C_5, 0, NULL, NULL);
	assert(memcmp(C_h, gpu_C_3, M*N * sizeof(float)) == 0);

	delete[] A_h;
	delete[] B_h;
	delete[] C_h;
	delete[] gpu_C_1;
	delete[] gpu_C_2;
	delete[] platforms;
	clReleaseKernel(run_as_gpu_1);
	clReleaseKernel(run_as_gpu_2);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(C_d);
	system("pause");
	return 0;
}