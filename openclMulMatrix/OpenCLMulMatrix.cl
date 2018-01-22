
__kernel void RunAsGpu_1(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float* C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float sum = 0;
	for(int i = 0;i<P;i++)
	{
		sum += A[x*P + i]*B[i*N + y];//对于同一个i A访问同一个地址（广播）B访问按顺序
	}
	C[x*N + y] = sum;//[0,0],[1,0],[2,0]  ndrange 从第一维开始变化，因此对C的访问跨度非常大
}

__kernel void RunAsGpu_2(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float* C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float sum = 0;
	for(int i = 0;i<P;i++)
	{
		sum += A[y*P + i]*B[i*N + x];
	}
	C[y*N + x] = sum;
}

__kernel void RunAsGpu_3(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float* C)
{
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	__local float ta[16][16];
	__local float tb[16][16];

	int ab = P * 16 * by;
	int ae = ab + P;
	int bb = 16 * bx;

	float sum = 0.0f;
	for (int i = ab,j=bb; i<ae; i+=16,j+=16*N)
	{
		ta[ty][tx] = A[i + ty * P + tx];
		tb[ty][tx] = B[j + ty * N + tx];  //采用local memory 加矩阵分块思想优化
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < 16; k++)
		{
			sum += ta[ty][k] * tb[k][tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[16*N*by+bx*16+ty*N+tx] = sum;
}

__kernel void RunAsGpu_4(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float* C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float4 sum = 0.0f;
	int xb = x * 4;
	for (int i = 0; i<P; i++)
	{
		sum.x += A[y*P + i] * B[i*N + xb];
		sum.y += A[y*P + i] * B[i*N + xb+1];
		sum.z += A[y*P + i] * B[i*N + xb+2];
		sum.w += A[y*P + i] * B[i*N + xb+3];
	}
	C[y*N + xb] = sum.x;
	C[y*N + xb+1] = sum.y;
	C[y*N + xb+2] = sum.z;
	C[y*N + xb+3] = sum.w;

}



__kernel void RunAsGpu_5(
	__global  float *A,
	__global  float4 *B,
	int M,
	int N,
	int P,
	__global float4* C)
{
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	const int BS = get_local_size(0);
	__local float4 ta[16][16];
	__local float4 tb[16][16];

	int ab = 4*P * BS * by;
	int ae = ab + P;
	int bb = BS * bx;

	float4 v[4];
	for (int ii = 0; ii < 4; ii++)
		v[ii] = 0.0f;
	const int N_float4 = N / 4;
	for (int i = ab, j = bb; i<ae; i += BS, j += BS * N_float4)
	{
		float4 temp;
		temp.x = A[0 * BS*P + i + ty * P + tx];
		temp.y = A[1 * BS*P + i + ty * P + tx];
		temp.z = A[2 * BS*P + i + ty * P + tx];
		temp.w = A[3 * BS*P + i + ty * P + tx];
		ta[ty][tx] = temp;
		tb[ty][tx] = B[j + ty * N_float4+ tx];  //采用local memory 加矩阵分块思想优化
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < BS; k++)
		{
			v[0] += ta[ty][k].x * tb[k][tx];
			v[1] += ta[ty][k].y * tb[k][tx];
			v[2] += ta[ty][k].z * tb[k][tx];
			v[3] += ta[ty][k].w * tb[k][tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (int ii=0;ii<4;ii++)
	C[N_float4 *(BS*(ii+by*4)+ty) + bx * BS + tx] = v[ii];
}



__kernel void RunAsGpu_6(
	__global  float *A,
	__global  float *B,
	int M,
	int N,
	int P,
	__global float* C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	C[x*N + y] = 0;
	for(int i = 0;i<P;i++)
	{
		C[x*N + y] += A[x*P + i]*B[i*N + y];
	}
}