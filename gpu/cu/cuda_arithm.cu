#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "gpumat.h"
#include "cuda_common.h"

#include "common_devices.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{
namespace internal{

////////////////////////////////

/**
 * @brief memset
 * @param A = val
 * @param val

 */
template< class T >
__global__ void memset(Mtx A, T val)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = val;
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * A.cols + col] = dA[row * A.cols + col] + dB[row * B.cols + col];
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = val1 * A .+ val2 * B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, T valA, T valB, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * A.cols + col] = valA * dA[row * A.cols + col] + valB * dB[row * B.cols + col];
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = valA * dA[row * A.cols + col] + valB * dB[row * B.cols + col];
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C -> C = valA * A - valB * B
 * @param valA
 * @param valB
 */template< class T >
__global__ void sub(Mtx A, Mtx B, Mtx C, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = valA * dA[row * A.cols + col] - valB * dB[row * B.cols + col];
}

template< class T >
__global__ void sub(Mtx A, Mtx B, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = valA * dA[row * A.cols + col] - valB * dB[row * B.cols + col];
}

template< class T >
__global__ void subWithColumn(Mtx A, Mtx B, Mtx mul, T valA, T valB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T *dM = (T*)mul.data;

	if(row < A.rows && col < A.cols){
		T m = dM[row];
		dA[row * A.cols + col] = (valA * dA[row * A.cols + col] - valB * dB[row * B.cols + col]) * m;
	}
}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmul(Mtx A, Mtx B, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	T sC = 0;

	if(row < A.rows && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
		}
		DC[row * B.cols + col] = alpha * sC;
	}
}

/**
 * @brief add2matmul
 * @param A
 * @param B
 * @param C - out C += A * B
 */
template< class T >
__global__ void add2matmul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	T sC = 0;

	if(row < A.rows && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
		}
		DC[row * B.cols + col] += sC;
	}
}


/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
template< class T >
__global__ void matmulT1(Mtx At, Mtx B, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)At.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	T sC = 0;

//	s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k];
	if(row < At.cols && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[i * At.cols + row] * DB[i * B.cols + col];
		}
		DC[row * C.cols + col] = alpha * sC;
	}

}

/**
 * @brief add2matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C += A' * B
 */
template< class T >
__global__ void add2matmulT1(Mtx At, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)At.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	T sC = 0;

//	s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k];
	if(row < At.cols && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[i * At.cols + row] * DB[i * B.cols + col];
		}
		DC[row * C.cols + col] += sC;
	}

}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
template< class T >
__global__ void matmulT2(Mtx A, Mtx Bt, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)Bt.data;
	T* DC = (T*)C.data;

	T sC = 0;

//	s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
	if(row < A.rows && col < C.cols){
		for(int i = 0; i < A.cols; i++){
//			sC += DA[row * B.rows + i] * DB[i * B.cols + col];
			sC += DA[row * A.cols + i] * DB[col * Bt.cols + i];
		}
		DC[row * C.cols + col] = alpha * sC;
	}
}

/**
 * @brief add2matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C += A * B'
 */
template< class T >
__global__ void add2matmulT2(Mtx A, Mtx Bt, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)Bt.data;
	T* DC = (T*)C.data;

	T sC = 0;

//	s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
	if(row < A.rows && col < C.cols){
		for(int i = 0; i < A.cols; i++){
//			sC += DA[row * B.rows + i] * DB[i * B.cols + col];
			sC += DA[row * A.cols + i] * DB[col * Bt.cols + i];
		}
		DC[row * C.cols + col] += sC;
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
template< class T >
__global__ void mulval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * value;
}

/**
 * @brief mulval
 * @param A -> A = A * value
 * @param value - mat 1x1
 */
template< class T >
__global__ void mulval(Mtx A, double value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] *= value;
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
template< class T >
__global__ void addval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] + value;
}

/**
 * @brief addval
 * @param A -> A += value
 * @param value - mat 1x1
 */
template< class T >
__global__ void addval(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] += value;
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
template< class T >
__global__ void subval(Mtx A, T value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] - value;
}

/**
 * @brief subval
 * @param A -> A - val
 * @param value - mat 1x1
 */
template< class T >
__global__ void subval(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] -= value;
}

/**
 * @brief subval
 * @param A -> A - val
 * @param value - mat 1x1
 */
template< class T >
__global__ void subval_inv(Mtx A, T value)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = value - dA[row * A.cols + col];
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
template< class T >
__global__ void subval(T value, Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = value - dA[row * A.cols + col];

}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
template< class T >
__global__ void biasPlus(Mtx A, const Mtx bias)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dBias = (T*)bias.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] += dBias[col];

}

/**
 * @brief scale_and_shift
 * @param A
 * @param scales
 * @param biases
 * @param C[i, j] = A[i,j] * scales[j] + biases[j]
 */
template< class T >
__global__ void scale_and_shift(const Mtx A, const Mtx scales, const Mtx biases, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;
	T* dBias = (T*)biases.data;
	T* dScales = (T*)scales.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * dScales[col] + dBias[col];
}

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
template< class T >
__global__ void elemwiseMul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * dB[row * B.cols + col];
}

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 */
template< class T >
__global__ void elemwiseMul(Mtx A, Mtx B)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] *= dB[row * A.cols + col];
}

/**
 * @brief elemwiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
template< class T >
__global__ void elemwiseDiv(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] / dB[row * B.cols + col];
}

/**
 * @brief transpose
 * @param A
 * @param C = A'
 */
template< class T >
__global__ void transpose(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[col * C.cols + row] = dA[row * A.cols + col];
}

/**
 * @brief sqrt
 * @param A
 * @param C = sqrt(A)
 */
template< class T >
__global__ void sqrt(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = std::sqrt(dA[row * A.cols + col]);
}

/**
 * @brief sqr
 * @param A
 * @param C = A .* a
 */
template< class T >
__global__ void sqr(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dC[row * C.cols + col] = val * val;
	}
}

/**
 * @brief reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
        dC[row * C.cols + col] = val * (T)(val > 0);
	}
}

/**
 * @brief reLu
 * @param A = reLu(A)
 */
template< class T >
__global__ void reLu(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
        dA[row * A.cols + col] = val * (T)(val > 0);
	}
}

/**
 * @brief deriv_reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void deriv_reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
        dC[row * C.cols + col] = (T)(dA[row * A.cols + col] > 0);
}

template< class T >
__global__ void deriv_reLu(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
        dA[row * A.cols + col] = (T)(dA[row * A.cols + col] > 0);
}

//////////leakyRelu

/**
 * @brief leakyReLu
 * @param A
 * @param x
 * @param C = leakyReLu(A)
 */
template< class T >
__global__ void leakyReLu(Mtx A, T x, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
        dC[row * C.cols + col] =  val > 0 ? val : val * x;
	}
}

/**
 * @brief leakyReLu
 * @param A = leakyReLu(A)
 * @param x
 */
template< class T >
__global__ void leakyReLu(Mtx A, T x)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
        dA[row * A.cols + col] = val > 0 ? val : val * x;
	}
}

/**
 * @brief deriv_leakyReLu
 * @param A
 * @param C = deriv_leakyReLu(A)
 */
template< class T >
__global__ void deriv_leakyReLu(Mtx A, T x, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
        dC[row * C.cols + col] = (val > 0)? 1 : x;
	}
}

template< class T >
__global__ void deriv_leakyReLu(Mtx A, T x)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dA[row * A.cols + col] = (val >= 0)? 1 : x;
	}
}

//////////

/**
 * @brief sigmoid
 * @param A
 * @param C = sigmoid(A)
 */
template< class T >
__global__ void sigmoid(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = 1 / (1 + exp(-dA[row * A.cols + col]));
}

/**
 * @brief sigmoid
 * @param A
 * @param C = sigmoid(A)
 */
template< class T >
__global__ void sigmoid(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = 1 / (1 + exp(-dA[row * A.cols + col]));
}

/**
 * @brief deriv_sigmoid
 * @param A
 * @param C = deriv_sigmoid(A)
 */
template< class T >
__global__ void deriv_sigmoid(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dC[row * C.cols + col] = val * (1 - val);
	}
}

template< class T >
__global__ void deriv_sigmoid(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dA[row * A.cols + col] = val * (1 - val);
	}
}

template< class T >
__global__ void back_delta_sigmoid(Mtx sigmoid, Mtx target)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)sigmoid.data;
	T* dC = (T*)target.data;

	if(row < sigmoid.rows && col < sigmoid.cols){
		T val = dA[row * sigmoid.cols + col];
		T target = dC[row * sigmoid.cols + col];
		T delta = val - target;
		dA[row * sigmoid.cols + col] = delta * val * (1 - val);
	}
}

template< class T >
__global__ void back_delta_sigmoid(Mtx sigmoid, Mtx target, Mtx mC)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)sigmoid.data;
	T* dC = (T*)target.data;
	T* dM = (T*)mC.data;

	if(row < sigmoid.rows && col < sigmoid.cols){
		T val = dA[row * sigmoid.cols + col];
		T target = dC[row * sigmoid.cols + col];
		T delta = val - target;
		T m = dM[row];
		dA[row * sigmoid.cols + col] = (delta * val * (1 - val)) * m;
	}
}

/**
 * @brief tanh
 * @param A
 * @param C = tanh(A)
 */
template< class T >
__global__ void tanh(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = exp(2 * dA[row * A.cols + col]);
		dC[row * C.cols + col] = (val - 1.) / (val + 1.);
	}
}

/**
 * @brief tanh
 * @param A
 * @param C = tanh(A)
 */
template< class T >
__global__ void tanh(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = exp(2 * dA[row * A.cols + col]);
		dA[row * A.cols + col] = (val - 1.) / (val + 1.);
	}
}

/**
 * @brief deriv_tanh
 * @param A
 * @param C = deriv_tanh(A)
 */
template< class T >
__global__ void deriv_tanh(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dC[row * C.cols + col] = (1. - val * val);
	}
}

template< class T >
__global__ void deriv_tanh(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col];
		dA[row * A.cols + col] = (1. - val * val);
	}
}

/**
 * @brief _exp
 * @param A
 * @param C = exp(A)
 */
template< class T >
__global__ void _exp(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = exp(dA[row * A.cols + col]);
		dC[row * C.cols + col] = val;
	}
}


/**
 * @brief _exp
 * @param A = exp(A)
 */
template< class T >
__global__ void _exp(Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols){
		T val = exp(dA[row * A.cols + col]);
		dA[row * A.cols + col] = val;
	}
}

/**
 * @brief max_rows
 * @param A
 * @param Max = max(A[..., j]
 */
template< class T >
__global__ void max_rows(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = dA[0 * A.cols + col];
		for(int i = 1; i < A.rows; i++){
			val = max(val, dA[i * A.cols + col]);
		}
		dmax[col] = val;
	}
}

/**
 * @brief max_cols
 * @param A
 * @param Max = max(A[i, ...]
 */
template< class T >
__global__ void max_cols(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + 0];
		for(int i = 1; i < A.cols; i++){
			val = max(val, dA[row * A.cols + i]);
		}
		dmax[row] = val;
	}
}

/**
 * @brief exp_rows
 * @param A
 * @param Max
 * @param C = exp(A[i, j] - Max[j])
 */
template< class T >
__global__ void exp_rows(Mtx A, Mtx Max, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col] - dmax[col];
		val = exp(val);
		dC[row * C.cols + col] = val;
	}
}

/**
 * @brief exp_rows
 * @param A
 * @param Max
 * @param C = exp(A[i, j] - Max[j])
 */
template< class T >
__global__ void exp_rows(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col] - dmax[col];
		val = exp(val);
		dA[row * A.cols + col] = val;
	}
}

/**
 * @brief exp_cols
 * @param A
 * @param Max
 * @param C = exp(A[i, j] - Max[i])
 */
template< class T >
__global__ void exp_cols(Mtx A, Mtx Max, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col] - dmax[row];
		val = exp(val);
		dC[row * C.cols + col] = val;
	}

}

/**
 * @brief exp_cols
 * @param A
 * @param Max
 * @param C = exp(A[i, j] - Max[i])
 */
template< class T >
__global__ void exp_cols(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = dA[row * A.cols + col] - dmax[row];
		val = exp(val);
		dA[row * A.cols + col] = val;
	}

}

////

/**
 * @brief sub_ln_rows
 * @param A = ln(A[i, j]) - ln(Max[j])
 * @param Max
 */
template< class T >
__global__ void sub_ln_rows(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = log(dA[row * A.cols + col]) - log(dmax[col]);
		dA[row * A.cols + col] = val;
	}
}

/**
 * @brief sub_ln_cols
 * @param A = ln(A[i, j]) - ln(Max[i])
 * @param Max
 */
template< class T >
__global__ void sub_ln_cols(Mtx A, Mtx Max)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dmax = (T*)Max.data;

	if(row < A.rows && col < A.cols){
		T val = log(dA[row * A.cols + col]) - log(dmax[row]);
		dA[row * A.cols + col] = val;
	}

}

////

/**
 * @brief sum_rows
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_rows(Mtx C, Mtx cols, T val = (T)1.)
{
	//int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(col < C.cols){
		dZ[col] = 0;
		for(int i = 0; i < C.rows; i++){
			dZ[col] += dC[i * C.cols + col];
		}
		dZ[col] *= val;
	}
}

/**
 * @brief add2sum_rows
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void add2sum_rows(Mtx C, Mtx cols, T val = (T)1.)
{
	//int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(col < C.cols){
		T s = 0;
		for(int i = 0; i < C.rows; i++){
			s += dC[i * C.cols + col];
		}
		dZ[col] += s * val;
	}
}

/**
 * @brief sum_cols
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_cols(Mtx C, Mtx rows, T val = (T)1.)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows){
		dZ[row] = 0;
		for(int i = 0; i < C.cols; i++){
			dZ[row] += dC[row * C.cols + i];
		}
		dZ[row] *= val;
	}
}

/**
 * @brief div_col
 * @param C -> in/out C[i, j] /=  cols[j]
 * @param cols -> in
 */
template< class T >
__global__ void div_col(Mtx C, Mtx cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(row < C.rows && col < C.cols){
		dC[row * C.cols + col] = abs(dZ[col]) > 1e-12? dC[row * C.cols + col] / dZ[col] : 0;
	}
}

/**
 * @brief div_row
 * @param C -> in/out C[i, j] /=  rows[i]
 * @param rows -> in
 */
template< class T >
__global__ void div_row(Mtx C, Mtx rows)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows && col < C.cols){
		dC[row * C.cols + col] = abs(dZ[row]) > 1e-12? dC[row * C.cols + col] / dZ[row] : 0;
	}
}

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
template< class T >
__global__ void adamgrad(Mtx A, Mtx gA, const Mtx mA, const Mtx vA, T alpha, T sb1, T sb2, T betha1, T betha2)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	const T eps = 10e-8;

	T* dA = (T*)A.data;
	T* dgA = (T*)gA.data;
	T* dmA = (T*)mA.data;
	T* dvA = (T*)vA.data;
	if(row < A.rows && col < A.cols){
		T g = dgA[row * A.cols + col];
		T g2 = g * g;
		T m0 = dmA[row * A.cols + col];
		T v0 = dvA[row * A.cols + col];

		m0 = betha1 * m0 + (1 - betha1) * g;
		v0 = betha2 * v0 + (1 - betha2) * g2;

		dmA[row * A.cols + col] = m0;
		dvA[row * A.cols + col] = v0;

		T m = sb1 * m0;
		T v = sb2 * v0;
		T val = alpha * m / (::sqrt(v) + eps);
		dA[row * A.cols + col] -= val;
	}
}

template< typename T >
__global__ void adagrad(Mtx A, Mtx hist_gA, const Mtx gA, T alpha, T betha)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    const T eps = 10e-6;

    T* dA = (T*)A.data;
    T* dgA = (T*)gA.data;
    T* dhgA = (T*)hist_gA.data;
    if(row < A.rows && col < A.cols){
        T g = dgA[row * A.cols + col];
        T g2 = g * g;
        T hA = dhgA[row * A.cols + col];
        hA = betha * hA + (1 - betha) * g2;
        dhgA[row * A.cols + col] = hA;
        dA[row * A.cols + col] -= alpha * g / (eps + ::sqrt(hA));
    }
}

///*******************

/**
 * @brief matmul_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmul_shared(Mtx A, Mtx B, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int _row = threadIdx.y;
	int _col = threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	DMtx CSub = getSubMatrix<T>(C, blockRow, blockCol);

	float sC = 0;

	for(int m = 0; m < A.cols / BLOCKSIZE + 1; ++m){
		DMtx ASub = getSubMatrix<T>(A, blockRow, m);
		DMtx BSub = getSubMatrix<T>(B, m, blockCol);

		__shared__ T As[BLOCKSIZE][BLOCKSIZE];
		__shared__ T Bs[BLOCKSIZE][BLOCKSIZE];

		if(row < A.rows && m * BLOCKSIZE + _col < A.cols)
			As[_row][_col] = getEl<T>(ASub, _row, _col);
//		else
//			As[_row][_col] = 0;
		if(m * BLOCKSIZE + _row < B.rows && col < B.cols)
			Bs[_row][_col] = getEl<T>(BSub, _row, _col);
//		else
//			Bs[_row][_col] = 0;

		__syncthreads();

		for(int e = 0; e < BLOCKSIZE; ++e){
			if(m * BLOCKSIZE + e < A.cols)
				sC += As[_row][e] * Bs[e][_col];
		}
		__syncthreads();
	}

	if(row < C.rows && col < C.cols){
		setEl<T>(CSub, _row, _col, alpha * sC);
	}
}

/**
 * @brief matmulT1_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmulT1_shared(Mtx At, Mtx B, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int _row = threadIdx.y;
	int _col = threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	DMtx CSub = getSubMatrix<T>(C, blockRow, blockCol);

	float sC = 0;

	for(int m = 0; m < At.rows / BLOCKSIZE + 1; ++m){
		DMtx ASub = getSubMatrix<T>(At, m, blockRow);
		DMtx BSub = getSubMatrix<T>(B, m, blockCol);

		__shared__ T As[BLOCKSIZE][BLOCKSIZE];
		__shared__ T Bs[BLOCKSIZE][BLOCKSIZE];

		if(m * BLOCKSIZE + _row < At.rows && blockRow * BLOCKSIZE + _col < At.cols)
			As[_row][_col] = getEl<T>(ASub, _row, _col);
//		else
//			As[_row][_col] = 0;
		if(m * BLOCKSIZE + _row < B.rows && blockCol * BLOCKSIZE + _col < B.cols)
			Bs[_row][_col] = getEl<T>(BSub, _row, _col);
//		else
//			Bs[_row][_col] = 0;

		__syncthreads();

		for(int e = 0; e < BLOCKSIZE; ++e){
			if(m * BLOCKSIZE + e < At.rows)
				sC += As[e][_row] * Bs[e][_col];
		}
		__syncthreads();
	}

	if(row < C.rows && col < C.cols){
		setEl<T>(CSub, _row, _col, alpha * sC);
	}
}

/**
 * @brief matmulT2_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmulT2_shared(Mtx A, Mtx Bt, Mtx C, T alpha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int _row = threadIdx.y;
	int _col = threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	DMtx CSub = getSubMatrix<T>(C, blockRow, blockCol);

	float sC = 0;

	for(int m = 0; m < A.cols / BLOCKSIZE + 1; ++m){
		DMtx ASub = getSubMatrix<T>(A, blockRow, m);
		DMtx BSub = getSubMatrix<T>(Bt, blockCol, m);

		__shared__ T As[BLOCKSIZE][BLOCKSIZE];
		__shared__ T Bs[BLOCKSIZE][BLOCKSIZE];

		if(row < A.rows && m * BLOCKSIZE + _col < A.cols)
			As[_row][_col] = getEl<T>(ASub, _row, _col);
//		else
//			As[_row][_col] = 0;
		if(blockCol * BLOCKSIZE + _row < Bt.rows && m * BLOCKSIZE + _col < Bt.cols)
			Bs[_row][_col] = getEl<T>(BSub, _row, _col);
//		else
//			Bs[_row][_col] = 0;

		__syncthreads();

		for(int e = 0; e < BLOCKSIZE; ++e){
			if(m * BLOCKSIZE + e < A.cols)
				sC += As[_row][e] * Bs[_col][e];
		}
		__syncthreads();
	}

	if(row < C.rows && col < C.cols){
		setEl<T>(CSub, _row, _col, alpha * sC);
	}
}

/**
 * @brief sum_rows_shared
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_rows_shared(Mtx C, Mtx cols, T val = (T)1.)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int _row = threadIdx.y;
	int _col = threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	DMtx CSub = getSubMatrix<T>(cols, blockRow, blockCol);

	float sC = 0;

	for(int m = 0; m < C.rows / BLOCKSIZE + 1; ++m){
		DMtx ASub = getSubMatrix<T>(C, m, blockRow);
//		DMtx BSub = getSubMatrix<T>(B, m, blockCol);

		__shared__ T As[BLOCKSIZE][BLOCKSIZE];

		if(m * BLOCKSIZE + _row < C.rows && blockRow * BLOCKSIZE + _col < C.cols)
			As[_row][_col] = getEl<T>(ASub, _row, _col);
//		else
//			As[_row][_col] = 0;

		__syncthreads();

		for(int e = 0; e < BLOCKSIZE; ++e){
			if(m * BLOCKSIZE + e < C.rows)
				sC += As[e][_row];
		}
		__syncthreads();
	}

//	if(row < C.rows && col < C.cols){
//		for(int i = 0; i < B.rows; i++){
//			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
//		}
		//DC[row * B.cols + col] = sC;
//		setEl<T>(CSub, _row, _col, sC);
//	}
	if(row < C.rows && col < C.cols){
		setEl<T>(CSub, _col, _row, sC);
	}
}

template< typename T >
__global__ void subIndOne(Mtx A, Mtx Ind, Mtx B)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < A.rows && col < A.cols){
		T *dA = (T*)A.data;
		T *dI = (T*)Ind.data;
		T *dB = (T*)B.data;

		if((int)dI[row * Ind.cols] == col)
			dB[row * A.cols + col] = dA[row * A.cols + col] - 1.;
		else
			dB[row * A.cols + col] = dA[row * A.cols + col];
	}
}

template< typename T >
__global__ void subIndOne(SmallMtxArray vecA, Mtx Ind, SmallMtxArray vecB)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < vecA.count && col < vecA.mtx[0].cols){
		Mtx &A = vecA.mtx[row];
		Mtx &B = vecB.mtx[row];
		T *dA = (T*)A.data;
		T *dI = (T*)Ind.data;
		T *dB = (T*)B.data;

		if((int)dI[row * Ind.cols] == col)
            dB[0 * A.cols + col] = dA[0 * A.cols + col] - 1.;
		else
            dB[0 * A.cols + col] = dA[0 * A.cols + col];
	}
}

template<typename T>
__global__ void hconcat2(const SmallMtxArray list, Mtx A)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T *dR = (T*)A.data;

	if(row < A.rows && col < A.cols){

		int cumoff = 0;
		int id = 0;

//		for(int i = 0; i < list.count; ++i){
//			int cols = list.mtx[i].cols;
//			if(cumoff + cols >= col){
//				id = i;
//				break;
//			}
//			cumoff += cols;
//		}

		while(cumoff + list.mtx[id].cols < col + 1 && id < list.count){
			cumoff += (list.mtx[id].cols);
			id++;
		}

		Mtx L = list.mtx[id];

		T *dL = (T*)L.data;
		dR[row * A.cols + col] = dL[row * L.cols + col - cumoff];
	}
}

template<typename T>
__global__ void hsplit2(Mtx A, SmallMtxArray list)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T *dR = (T*)A.data;

	if(row < A.rows && col < A.cols){

		int cumoff = 0;
		int id = 0;

//		for(int i = 0; i < list.count; ++i){
//			int cols = list.mtx[i].cols;
//			if(cumoff + cols >= col){
//				id = i;
//				break;
//			}
//			cumoff += cols;
//		}

		while(cumoff + list.mtx[id].cols < col + 1 && id < list.count){
			cumoff += (list.mtx[id].cols);
			id++;
		}

		Mtx L = list.mtx[id];

		T *dL = (T*)L.data;
		dL[row * L.cols + col - cumoff] = dR[row * A.cols + col];
	}
}

template< typename T >
__global__ void mul2deriv(Mtx D, Mtx A, gpumat::etypefunction func, Mtx DA, T param1 = 0, T param2 = 0, T param3 = 0)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < D.rows && col < D.cols){
		T *dD = (T*)D.data;
		T *dA = (T*)A.data;
		T *dDA = (T*)DA.data;

		int off = row * DA.cols + col;
		switch (func) {
			default:
			case LINEAR:
				dDA[off] = dD[off];
				break;
			case RELU:{
				T val = dA[off];
                dDA[off] = (T)(val > 0) * dD[off];
				break;
			}
			case LEAKYRELU:{
                T val = dA[off] > 0.? 1. : param1;
				dDA[off] = val * dD[off];
				break;
			}
			case SIGMOID:{
				T val = dA[off];
				val = val * (1 - val);
				dDA[off] = val * dD[off];
				break;
			}
			case TANH:{
				T val = dA[off];
				val = (1. - val * val);
				dDA[off] = val * dD[off];
				break;
			}
		}
	}
}

template<typename T>
__global__ void m2mpbaf(Mtx A, Mtx B, Mtx C, etypefunction func, Mtx D, T param1, T param2, T param3)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;
	T* DD = (T*)D.data;

	T sC = 0;

	if(row < A.rows && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
		}
		sC += DC[col];

		switch (func) {
			case RELU:
                sC = sC * (T)(sC > 0);
				break;
			case LEAKYRELU:
                sC = sC > 0? sC : param1 * sC;
				break;
			case SIGMOID:
				sC = exp(-sC);
				sC = 1. / (1. + sC);
				break;
			case TANH:
				sC = exp(2. * sC);
				sC = (sC - 1.) / (sC + 1.);
				break;
			default:
				break;
		}

		DD[row * B.cols + col] = sC;
	}
}

template<typename T>
__global__ void momentum_optimizer(Mtx W, Mtx M, Mtx G, T alpha, T betha)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dW = (T*)W.data;
	T* dM = (T*)M.data;
	T* dG = (T*)G.data;

	if(row < W.rows && col < W.cols){
		T g = dG[row * G.cols + col];
		T m = dM[row * M.cols + col];
		m = betha * m + (1 - betha) * g;
		dM[row * M.cols + col] = m;
		dW[row * W.cols + col] -= alpha * m;
	}
}

}

}

//////// end namespace /////////////////

/**
 * @brief cuda_memset
 * @param A = val
 * @param val
 */
extern "C"
void cuda_memset(GpuMat& A, double val)
{
	if(A.empty())
		return;

	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::memset<double> <<<dimGrid, dimBlock>>>(A, (double)val);
		break;
	case GPU_FLOAT:
		internal::memset<float> <<<dimGrid, dimBlock>>>(A, (float)val);
		break;
	}
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief cuda_add_params
 * @param A
 * @param val1
 * @param B
 * @param val2
 * @param C = val1 * A + val2 * B
 */
extern "C"
void cuda_add_params(const GpuMat& A, const GpuMat& B, double val1, double val2, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, (double)val1, (double)val2, C);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, (float)val1, (float)val2, C);
		break;
	}
}

/**
 * @brief cuda_add_paramsA
 * @param A -> A = val1 * A + val2 * B
 * @param val
 * @param B
 */
extern "C"
void cuda_add_paramsA(GpuMat& A, const GpuMat& B, double valA, double valB)
{
	int x1 = B.cols / BLOCKSIZE + 1;
	int x2 = B.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (B.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA, double valB)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sub<double> <<<dimGrid, dimBlock>>>(A, B, C, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::sub<float> <<<dimGrid, dimBlock>>>(A, B, C, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief cuda_subA
 * @param A = A * valA - B * valB
 * @param B
 * @param valA
 * @param valB
 */
extern "C"
void cuda_subA(GpuMat& A, const GpuMat& B, double valA, double valB)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sub<double> <<<dimGrid, dimBlock>>>(A, B, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::sub<float> <<<dimGrid, dimBlock>>>(A, B, (float)valA, (float)valB);
		break;
	}
}

/**
 * @brief cuda_subWithColumn
 * @param A = A * valA - B * valB
 * @param B
 * @param valA
 * @param valB
 */
extern "C"
void cuda_subWithColumn(GpuMat& A, const GpuMat& B, const GpuMat& mulColumn, double valA, double valB)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subWithColumn<double> <<<dimGrid, dimBlock>>>(A, B, mulColumn, (double)valA, (double)valB);
		break;
	case GPU_FLOAT:
		internal::subWithColumn<float> <<<dimGrid, dimBlock>>>(A, B, mulColumn, (float)valA, (float)valB);
		break;
	}
}


/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmul<double> <<<dimGrid, dimBlock>>>(A, B, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmul<float> <<<dimGrid, dimBlock>>>(A, B, C, alpha);
		break;
	}
}

/**
 * @brief add2matmul
 * @param A
 * @param B
 * @param C - out C += A * B
 */
extern "C"
void cuda_add2matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add2matmul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::add2matmul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief matmul_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul_shared(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmul_shared<double> <<<dimGrid, dimBlock>>>(A, B, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmul_shared<float> <<<dimGrid, dimBlock>>>(A, B, C, alpha);
		break;
	}
}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha)
{
	//	int r = At.cols;
	//	int c = B.cols;

	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (At.type) {
	case GPU_DOUBLE:
		internal::matmulT1<double> <<<dimGrid, dimBlock>>>(At, B, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmulT1<float> <<<dimGrid, dimBlock>>>(At, B, C, alpha);
		break;
	}
}

/**
 * @brief add2matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C += A' * B
 */
extern "C"
void cuda_add2matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{
	//	int r = At.cols;
	//	int c = B.cols;

	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (At.type) {
	case GPU_DOUBLE:
		internal::add2matmulT1<double> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	case GPU_FLOAT:
		internal::add2matmulT1<float> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	}
}

/**
 * @brief matmulT1_shared
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1_shared(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha)
{
	//	int r = At.cols;
	//	int c = B.cols;

	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (At.type) {
	case GPU_DOUBLE:
		internal::matmulT1_shared<double> <<<dimGrid, dimBlock>>>(At, B, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmulT1_shared<float> <<<dimGrid, dimBlock>>>(At, B, C, alpha);
		break;
	}
}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmulT2<double> <<<dimGrid, dimBlock>>>(A, Bt, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmulT2<float> <<<dimGrid, dimBlock>>>(A, Bt, C, alpha);
		break;
	}
}

/**
 * @brief add2matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C += A * B'
 */
extern "C"
void cuda_add2matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add2matmulT2<double> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	case GPU_FLOAT:
		internal::add2matmulT2<float> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	}
}

/**
 * @brief matmulT2_shared
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2_shared(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha)
{
	int x1 = C.cols / BLOCKSIZE + 1;
	int x2 = C.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmulT2_shared<double> <<<dimGrid, dimBlock>>>(A, Bt, C, alpha);
		break;
	case GPU_FLOAT:
		internal::matmulT2_shared<float> <<<dimGrid, dimBlock>>>(A, Bt, C, alpha);
		break;
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
	}
}

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
extern "C"
void cuda_mulvalA(const GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval_in(GpuMat& A, const double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addval(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::addval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::addval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
		break;
	}
}

/**
 * @brief addval
 * @param A -> A += val
 * @param value - mat 1x1
 */
extern "C"
void cuda_addvalA(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::addval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::addval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_AvaltoC(const GpuMat& A, double value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(A, (double)value, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(A, (float)value, C);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valAtoC(double value, const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>((double)value, A, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>((float)value, A, C);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 */
extern "C"
void cuda_subval_Aval(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 */
extern "C"
void cuda_subval_valA(GpuMat& A, double value)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval_inv<double> <<<dimGrid, dimBlock>>>(A, (double)value);
		break;
	case GPU_FLOAT:
		internal::subval_inv<float> <<<dimGrid, dimBlock>>>(A, (float)value);
		break;
	}
}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
extern "C"
void cuda_biasPlus(GpuMat& A, const GpuMat& bias)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::biasPlus<double> <<<dimGrid, dimBlock>>>(A, bias);
		break;
	case GPU_FLOAT:
		internal::biasPlus<float> <<<dimGrid, dimBlock>>>( A, bias);
		break;
	}
}

/**
 * @brief scale_and_shift
 * @param A
 * @param scales
 * @param biases
 * @param C[i, j] = A[i,j] * scales[j] + biases[j]
 */
extern "C"
void cuda_scale_and_shift(const GpuMat& A, const GpuMat& scales, const GpuMat& biases, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::scale_and_shift<double> <<<dimGrid, dimBlock>>>(A, scales, biases, C);
		break;
	case GPU_FLOAT:
		internal::scale_and_shift<float> <<<dimGrid, dimBlock>>>(A, scales, biases, C);
		break;
	}
}


/**
 * @brief elemwiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
extern "C"
void cuda_elemwiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseMul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemwiseMul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief elemwiseMul
 * @param A = A .* B
 * @param B
 */
extern "C"
void cuda_elemwiseMulA(GpuMat& A, const GpuMat& B)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseMul<double> <<<dimGrid, dimBlock>>>(A, B);
		break;
	case GPU_FLOAT:
		internal::elemwiseMul<float> <<<dimGrid, dimBlock>>>(A, B);
		break;
	}
}


/**
 * @brief elemwiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
extern "C"
void cuda_elemwiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemwiseDiv<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemwiseDiv<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief cuda_transpose
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_transpose(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::transpose<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::transpose<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_elemwiseSqrt
 * @param A
 * @param C = sqrt(A)
 */
extern "C"
void cuda_elemwiseSqrt(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sqrt<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sqrt<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_elemwiseSqr
 * @param A
 * @param C =  A .* a
 */
extern "C"
void cuda_elemwiseSqr(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sqr<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sqr<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_sumrows
 * @param A
 * @param C - out C[j] = sum(A[i, j])(i = [1..rows])
 */
extern "C"
void cuda_sumrows(const GpuMat& A, GpuMat& sums, double val)
{
	int x1 = A.cols / BLOCKSIZE + 1;
//	int x2 = A.rows / BLOCKSIZE + 1;

	switch (A.type) {
	case GPU_DOUBLE:
			internal::sum_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (double)val);
		break;
	case GPU_FLOAT:
			internal::sum_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (float)val);
		break;
	}
}

/**
 * @brief cuda_sumcols
 * @param A
 * @param C - out C[i] = sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumcols(const GpuMat& A, GpuMat& sums, double val)
{
//	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	switch (A.type) {
	case GPU_DOUBLE:
			internal::sum_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, sums, (double)val);
		break;
	case GPU_FLOAT:
			internal::sum_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, sums, (float)val);
		break;
	}
}

/**
 * @brief cuda_add2sumrows
 * @param A
 * @param C - out C[i] += sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_add2sumrows(const GpuMat& A, GpuMat& sums, double val)
{
	int x1 = A.cols / BLOCKSIZE + 1;
//	int x2 = A.rows / BLOCKSIZE + 1;

	switch (A.type) {
	case GPU_DOUBLE:
			internal::add2sum_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (double)val);
		break;
	case GPU_FLOAT:
			internal::add2sum_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, sums, (float)val);
		break;
	}
}

/**
 * @brief cuda_sumrows_shared
 * @param A
 * @param C - out C[i] = sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumrows_shared(const GpuMat& A, GpuMat& sums, double val)
{
	int x1 = A.rows / BLOCKSIZE + 1;
//	int x2 = A.rows / BLOCKSIZE + 1;

	switch (A.type) {
	case GPU_DOUBLE:
			internal::sum_rows_shared<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, BLOCKSIZE)>>>(A, sums, (double)val);
		break;
	case GPU_FLOAT:
			internal::sum_rows_shared<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, BLOCKSIZE)>>>(A, sums, (float)val);
		break;
	}
}

////////////////

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::reLu<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::reLu<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

/**
 * @brief cuda_derivReLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_derivReLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::deriv_reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

extern "C"
void cuda_derivReLu2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_reLu<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::deriv_reLu<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

////////////////////

/**
 * @brief cuda_leakyReLu
 * @param A
 * @param C = leakyReLu(A)
 */
extern "C"
void cuda_leakyReLu(const GpuMat& A, double x, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::leakyReLu<double> <<<dimGrid, dimBlock>>>(A, x, C);
		break;
	case GPU_FLOAT:
		internal::leakyReLu<float> <<<dimGrid, dimBlock>>>(A, x, C);
		break;
	}
}

/**
 * @brief cuda_reLu
 * @param A
 * @param C = leakyReLu(A)
 */
extern "C"
void cuda_leakyReLu2(GpuMat& A, double x)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::leakyReLu<double> <<<dimGrid, dimBlock>>>(A, x);
		break;
	case GPU_FLOAT:
		internal::leakyReLu<float> <<<dimGrid, dimBlock>>>(A, x);
		break;
	}
}

/**
 * @brief cuda_derivLeakyReLu
 * @param A
 * @param C = derivLeakyRelu(A)
 */
extern "C"
void cuda_derivLeakyReLu(const GpuMat& A, double x, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_leakyReLu<double> <<<dimGrid, dimBlock>>>(A, x, C);
		break;
	case GPU_FLOAT:
		internal::deriv_leakyReLu<float> <<<dimGrid, dimBlock>>>(A, x, C);
		break;
	}
}

extern "C"
void cuda_derivLeakyReLu2(GpuMat& A, double x)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_leakyReLu<double> <<<dimGrid, dimBlock>>>(A, x);
		break;
	case GPU_FLOAT:
		internal::deriv_leakyReLu<float> <<<dimGrid, dimBlock>>>(A, x);
		break;
	}
}

////////////////////

/**
 * @brief cuda_sigmoid
 * @param A
 * @param C = sigmoid(A)
 */
extern "C"
void cuda_sigmoid(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sigmoid<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sigmoid<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_sigmoid2
 * @param A
 */
extern "C"
void cuda_sigmoid2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sigmoid<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::sigmoid<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

/**
 * @brief cuda_deriv_sigmoid
 * @param A
 * @param C = derivative sigmoid(A)
 */
extern "C"
void cuda_deriv_sigmoid(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_sigmoid<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::deriv_sigmoid<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

extern "C"
void cuda_deriv_sigmoid2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_sigmoid<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::deriv_sigmoid<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

/**
 * @brief cuda_back_delta_sigmoid
 * @param sigmoid
 * @param delta
 */
extern "C"
void cuda_back_delta_sigmoid(GpuMat &sigmoid, const GpuMat &target)
{
	int x1 = sigmoid.cols / BLOCKSIZE + 1;
	int x2 = sigmoid.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (sigmoid.type) {
	case GPU_DOUBLE:
		internal::back_delta_sigmoid<double> <<<dimGrid, dimBlock>>>(sigmoid, target);
		break;
	case GPU_FLOAT:
		internal::back_delta_sigmoid<float> <<<dimGrid, dimBlock>>>(sigmoid, target);
		break;
	}
}

/**
 * @brief cuda_back_delta_sigmoid2
 * @param sigmoid
 * @param delta
 */
extern "C"
void cuda_back_delta_sigmoid2(GpuMat &sigmoid, const GpuMat &target, const GpuMat& mulColumn)
{
	int x1 = sigmoid.cols / BLOCKSIZE + 1;
	int x2 = sigmoid.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (sigmoid.type) {
	case GPU_DOUBLE:
		internal::back_delta_sigmoid<double> <<<dimGrid, dimBlock>>>(sigmoid, target, mulColumn);
		break;
	case GPU_FLOAT:
		internal::back_delta_sigmoid<float> <<<dimGrid, dimBlock>>>(sigmoid, target, mulColumn);
		break;
	}
}

///////////////

/**
 * @brief cuda_tanh
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_tanh(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::tanh<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::tanh<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_tanh
 * @param A
 */
extern "C"
void cuda_tanh2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::tanh<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::tanh<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

/**
 * @brief cuda_deriv_tanh
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_deriv_tanh(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_tanh<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::deriv_tanh<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

extern "C"
void cuda_deriv_tanh2(GpuMat& A)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_tanh<double> <<<dimGrid, dimBlock>>>(A);
		break;
	case GPU_FLOAT:
		internal::deriv_tanh<float> <<<dimGrid, dimBlock>>>(A);
		break;
	}
}

/////////////////

/**
 * @brief cuda_softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 * axis = 0: exp(x[i, j]) / S(exp(x[k, j]) = exp(  ln(exp(x[i, j] - max(x[..., j])) - ln(S(exp(x[k, j] - max(x[..., j]))))  )
 * axis = 1: exp(x[i, j]) / S(exp(x[i, k]) = exp(  ln(exp(x[i, j] - max(x[i, ...])) - ln(S(exp(x[i, k] - max(x[i, ...]))))  )
 */
extern "C"
void cuda_softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
			if(axis == 0){
				internal::max_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::exp_rows<double> <<<dimGrid, dimBlock>>>(A, partZ, C);
				internal::sum_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
				internal::sub_ln_rows<double> <<<dimGrid, dimBlock>>>(C, partZ);
				internal::_exp<double> <<<dimGrid, dimBlock>>>(C);
			}else{
				internal::max_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::exp_cols<double> <<<dimGrid, dimBlock>>>(A, partZ, C);
				internal::sum_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
				internal::sub_ln_cols<double> <<<dimGrid, dimBlock>>>(C, partZ);
				internal::_exp<double> <<<dimGrid, dimBlock>>>(C);
			}

//		internal::_exp<double> <<<dimGrid, dimBlock>>>(A, C);
//		if(axis == 0){
//			internal::sum_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
//			internal::div_col<double> <<<dimGrid, dimBlock>>>(C, partZ);
//		}else{
//			internal::sum_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
//			internal::div_row<double> <<<dimGrid, dimBlock>>>(C, partZ);
//		}
		break;
	case GPU_FLOAT:
			if(axis == 0){
				internal::max_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::exp_rows<float> <<<dimGrid, dimBlock>>>(A, partZ, C);
				internal::sum_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
				internal::sub_ln_rows<float> <<<dimGrid, dimBlock>>>(C, partZ);
				internal::_exp<float> <<<dimGrid, dimBlock>>>(C);
			}else{
				internal::max_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::exp_cols<float> <<<dimGrid, dimBlock>>>(A, partZ, C);
				internal::sum_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
				internal::sub_ln_cols<float> <<<dimGrid, dimBlock>>>(C, partZ);
				internal::_exp<float> <<<dimGrid, dimBlock>>>(C);
			}
//		internal::_exp<float> <<<dimGrid, dimBlock>>>(A, C);
//		if(axis == 0){
//			internal::sum_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
//			internal::div_col<float> <<<dimGrid, dimBlock>>>(C, partZ);
//		}else{
//			internal::sum_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
//			internal::div_row<float> <<<dimGrid, dimBlock>>>(C, partZ);
//		}
		break;
	}
}

/**
 * @brief cuda_softmax2
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 * axis = 0: exp(x[i, j]) / S(exp(x[k, j]) = exp(  ln(exp(x[i, j] - max(x[..., j])) - ln(S(exp(x[k, j] - max(x[..., j]))))  )
 * axis = 1: exp(x[i, j]) / S(exp(x[i, k]) = exp(  ln(exp(x[i, j] - max(x[i, ...])) - ln(S(exp(x[i, k] - max(x[i, ...]))))  )
 */
extern "C"
void cuda_softmax2(GpuMat& A, int axis, GpuMat& partZ)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
			if(axis == 0){
				internal::max_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::exp_rows<double> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::sum_rows<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::sub_ln_rows<double> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::_exp<double> <<<dimGrid, dimBlock>>>(A);
			}else{
				internal::max_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::exp_cols<double> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::sum_cols<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::sub_ln_cols<double> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::_exp<double> <<<dimGrid, dimBlock>>>(A);
			}
		break;
	case GPU_FLOAT:
			if(axis == 0){
				internal::max_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::exp_rows<float> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::sum_rows<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(A, partZ);
				internal::sub_ln_rows<float> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::_exp<float> <<<dimGrid, dimBlock>>>(A);
			}else{
				internal::max_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::exp_cols<float> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::sum_cols<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(A, partZ);
				internal::sub_ln_cols<float> <<<dimGrid, dimBlock>>>(A, partZ);
				internal::_exp<float> <<<dimGrid, dimBlock>>>(A);
			}
		break;
	}
}

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
extern "C"
void cuda_adamgrad(GpuMat& A, const GpuMat &gA, GpuMat& mA, GpuMat& vA,
				   double alpha, double sb1, double sb2, double betha1, double betha2)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::adamgrad<double> <<<dimGrid, dimBlock>>>(A, gA, mA, vA, (double)alpha, (double)sb1, (double)sb2, betha1, betha2);
		break;
	case GPU_FLOAT:
		internal::adamgrad<float> <<<dimGrid, dimBlock>>>(A, gA, mA, vA, (float)alpha, (float)sb1, (float)sb2, betha1, betha2);
		break;
	}
}

/**
 * @brief cuda_adagrad
 * @param A
 * @param gA
 * @param hist_gA
 * @param alpha
 * @param betha
 */
extern "C"
void cuda_adagrad(GpuMat& A, GpuMat& hist_gA, const GpuMat& gA, double alpha, double betha)
{
    int x1 = A.cols / BLOCKSIZE + 1;
    int x2 = A.rows / BLOCKSIZE + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

    switch (A.type) {
    case GPU_DOUBLE:
        internal::adagrad<double> <<<dimGrid, dimBlock>>>(A, hist_gA, gA, alpha, betha);
        break;
    case GPU_FLOAT:
        internal::adagrad<float> <<<dimGrid, dimBlock>>>(A, hist_gA, gA, alpha, betha);
        break;
    }
}

/**
 * @brief cuda_subIndOne
 * @param A
 * @param Ind
 * @param B = A : A[row, col == Ind[row]] - 1
 */
extern "C"
void cuda_subIndOne(const GpuMat& A, const GpuMat& Ind, GpuMat& B)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subIndOne<double> <<<dimGrid, dimBlock>>>(A, Ind, B);
		break;
	case GPU_FLOAT:
		internal::subIndOne<float> <<<dimGrid, dimBlock>>>(A, Ind, B);
		break;
	}
}

/**
 * @brief cuda_vecSubIndOne
 * @param vecA
 * @param Ind
 * @param B = A : A[row, col == Ind[row]] - 1
 */
extern "C"
void cuda_vecSubIndOne(const std::vector< GpuMat >& vecA, const GpuMat& Ind, std::vector< GpuMat >& B)
{
	int x1 = vecA[0].cols / BLOCKSIZE + 1;
	int x2 = vecA.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vecA[0].type) {
	case GPU_DOUBLE:
		internal::subIndOne<double> <<<dimGrid, dimBlock>>>(vecA, Ind, B);
		break;
	case GPU_FLOAT:
		internal::subIndOne<float> <<<dimGrid, dimBlock>>>(vecA, Ind, B);
		break;
	}
}


/**
 * @brief cuda_hconcat2
 * @param list
 * @param res
 */
extern "C"
void cuda_hconcat2(const std::vector< GpuMat > &list, GpuMat& res)
{
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

//	internal::SmallMtxArray mlist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hconcat2<double> <<<dimGrid, dimBlock>>>(list, res);
		break;
	case GPU_FLOAT:
		internal::hconcat2<float> <<<dimGrid, dimBlock>>>(list, res);
		break;
	}
}

/**
 * @brief cuda_hsplit2
 * @param list
 * @param res
 */
extern "C"
void cuda_hsplit2(const GpuMat& res, std::vector< GpuMat > &list)
{
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray mlist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hsplit2<double> <<<dimGrid, dimBlock>>>(res, mlist);
		break;
	case GPU_FLOAT:
		internal::hsplit2<float> <<<dimGrid, dimBlock>>>(res, mlist);
		break;
	}
}

/**
 * @brief cuda_mul2deriv
 * @param D
 * @param A
 * @param func
 * @param DA
 */
extern "C"
void cuda_mul2deriv(const GpuMat &D, const GpuMat &A, etypefunction func, GpuMat &DA, double param1, double param2, double param3)
{
	int x1 = D.cols / BLOCKSIZE + 1;
	int x2 = D.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (D.type) {
	case GPU_DOUBLE:
		internal::mul2deriv<double> <<<dimGrid, dimBlock>>>(D, A, func, DA, param1, param2, param3);
		break;
	case GPU_FLOAT:
		internal::mul2deriv<float> <<<dimGrid, dimBlock>>>(D, A, func, DA, param1, param2, param3);
		break;
	}
}

/**
 * @brief m2mpbaf
 * @param A
 * @param B
 * @param C
 * @param func
 * @param D
 */
extern "C"
void cuda_m2mpbaf(const GpuMat &A, const GpuMat &B, const GpuMat &C, etypefunction func, GpuMat &D, double param1, double param2, double param3)
{
	int x1 = D.cols / BLOCKSIZE + 1;
	int x2 = D.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (D.type) {
	case GPU_DOUBLE:
		internal::m2mpbaf<double> <<<dimGrid, dimBlock>>>(A, B, C, func, D, param1, param2, param3);
		break;
	case GPU_FLOAT:
		internal::m2mpbaf<float> <<<dimGrid, dimBlock>>>(A, B, C, func, D, param1, param2, param3);
		break;
	}
}

/**
 * @brief cuda_momentum_optimizer
 * @param W
 * @param M
 * @param G
 * @param alpha
 * @param betha
 */
extern "C"
void cuda_momentum_optimizer(GpuMat &W, GpuMat &M, const GpuMat &G, double alpha, double betha)
{
	int x1 = W.cols / BLOCKSIZE + 1;
	int x2 = W.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (W.type) {
	case GPU_DOUBLE:
		internal::momentum_optimizer<double> <<<dimGrid, dimBlock>>>(W, M, G, alpha, betha);
		break;
	case GPU_FLOAT:
		internal::momentum_optimizer<float> <<<dimGrid, dimBlock>>>(W, M, G, alpha, betha);
		break;
	}

}
