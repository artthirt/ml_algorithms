#ifndef GPU_H
#define GPU_H

#include <iostream>
#include <vector>

#include "common_types.h"
#include "cuda_types.h"

#ifdef _MSC_VER
	typedef unsigned char u_char;
#endif

namespace gpumat{

enum{
	GPU_FLOAT = 1,
	GPU_DOUBLE = 2
};

const int sizeof_enum[] = {
	0,
	sizeof(float),		/// GPU_FLOAT
	sizeof(double)		/// GPU_DOUBLE
};

#define SIZEOF_TYPE(type) (sizeof_enum[type])

namespace internal{
	struct SmallMtxArray;
}

class GpuMat{
public:
	int type;
	int rows;
	int cols;
	uint8_t* data;

	GpuMat();
	GpuMat(int rows, int cols, int type);
	GpuMat(int rows, int cols, int type, void* data);
	GpuMat(const GpuMat& mat);
	~GpuMat();

	GpuMat &operator =(const GpuMat& mat);

	////////////

	GpuMat& ones();
	GpuMat& zeros();

	////////////

	int depth() const;
	int size() const;
	int total() const;
	bool empty() const;

	ct::Size sz() const;

	void resize(int rows, int cols, int type);
	void resize(const ct::Size& sz, int type);
	void resize(const GpuMat& mat);

	void copyTo(GpuMat& mat) const;

	void setData(void* data);
	void getData(void *data) const;

	void free();

	void swap_dims();

	std::string operator()() const;

	std::string print(int _rows = -1) const;

	void save(const std::string filename) const;

	void release();

	///** internal **///
	internal::SmallMtxArray sderiv;
	internal::SmallMtxArray sW;


private:
};

/**
 * @brief memset
 * @param A
 * @param val
 */
void memset(GpuMat& A, double val);
/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
void add(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief add
 * @param A
 * @param B
 * @param C
 * @param valA
 * @param valB
 */
void add(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA = 1., double valB = 1.);
/**
 * @brief add
 * @param A -> A = valA * A + valB * B
 * @param valA
 * @param B
 * @param valB
 */
void add(GpuMat& A, const GpuMat& B, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A = A * valA - B * valB
 * @param B
 */
void sub(GpuMat& A, const GpuMat& B, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A - = (A .- B) * mulColumn
 * @param B
 * @param mulColumn
 */
void subWithColumn(GpuMat& A, const GpuMat& B, const GpuMat& mulColumn, double valA = 1., double valB = 1.);

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void matmul(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmul
 * @param A
 * @param B
 * @param C - out C += A * B
 */
void add2matmul(const GpuMat &A, const GpuMat &B, GpuMat &C);

/**
 * @brief matmul_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void matmul_shared(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C += A' * B
 */
void add2matmulT1(const GpuMat &At, const GpuMat &B, GpuMat &C);

/**
 * @brief matmulT1_shared
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void matmulT1_shared(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmulT2
 * @param A
 * @param Bt - B
 * @param C -out C += A * B'
 */
void add2matmulT2(const GpuMat &A, const GpuMat &Bt, GpuMat &C);

/**
 * @brief matmulT2_shared
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void matmulT2_shared(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha = 1.);

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
void mulval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
void mulval(GpuMat& A, double value);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
void addval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 */
void addval(GpuMat &A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
void subval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
void subval(double value, const GpuMat& A, GpuMat& C);

/**
 * @brief subval
 * @param A - > A - value
 * @param value - mat 1x1
 */
void subval(GpuMat& A, double value);

/**
 * @brief subval
 * @param A -> value - A
 * @param value - mat 1x1
 */
void subval(double value, GpuMat& A);

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
void biasPlus(GpuMat& A, const GpuMat& bias);

/**
 * @brief scale_and_shift
 * @param A
 * @param scales
 * @param biases
 * @param C[i, j] = A[i,j] * scales[j] + biases[j]
 */
void scale_and_shift(const GpuMat& A, const GpuMat& scales, const GpuMat& biases, GpuMat& C);

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
void elemwiseMult(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseMul
 * @param A = A.* B
 * @param B
 */
void elemwiseMult(GpuMat& A, const GpuMat& B);

/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
void elemwiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseSqrt
 * @param A
 * @param B
 * @param C - out C = sqrt(A)
 */
void elemwiseSqrt(const GpuMat& A, GpuMat& C);

/**
 * @brief elemiseSqr
 * @param A
 * @param B
 * @param C - out C = sqrt(A)
 */
void elemwiseSqr(const GpuMat& A, GpuMat& C);

/**
 * @brief sumRows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j]) (j = [1, cols])
 */
void sumRows(const GpuMat& A, GpuMat& C, double val = 1.);

/**
 * @brief add2sumRows
 * @param A
 * @param C
 * @param val - out C[i] += val * sum(A[i, j]) (j = [1, cols])
 */
void add2sumRows(const GpuMat &A, GpuMat &C, double val);

/**
 * @brief sumRows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j]) (j = [1, cols])
 */
void sumRows_shared(const GpuMat& A, GpuMat& C, double val = 1.);

/**
 * @brief transpose
 * @param A
 * @param C - out C = A'
 */
void transpose(const GpuMat& A, GpuMat& C);

/**
 * @brief reLu
 * @param A
 * @param C - out C = reLu(A)
 */
void reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief reLu
 * @param A
 */
void reLu(GpuMat& A);

/**
 * @brief deriv_reLu
 * @param A
 * @param C - out C = deriv_reLu(A)
 */
void deriv_reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief deriv_reLu
 * @param A
 */
void deriv_reLu(GpuMat& A);

////

/**
 * @brief leakyReLu
 * @param A
 * @param C - out C = reLu(A)
 */
void leakyReLu(const GpuMat& A, double x, GpuMat& C);

/**
 * @brief leakyReLu
 * @param A
 */
void leakyReLu(GpuMat& A, double x);

/**
 * @brief deriv_leakyReLu
 * @param A
 * @param C - out C = deriv_reLu(A)
 */
void deriv_leakyReLu(const GpuMat& A, double x, GpuMat& C);

/**
 * @brief deriv_leakyReLu
 * @param A
 */
void deriv_leakyReLu(GpuMat& A, double x);

////

/**
 * @brief sigmoid
 * @param A
 * @param C - out C = sigmoid(A)
 */
void sigmoid(const GpuMat& A, GpuMat& C);

/**
 * @brief sigmoid
 * @param A
 */
void sigmoid(GpuMat& A);

/**
 * @brief deriv_sigmoid
 * @param A
 * @param C - out C = deriv_sigmoid(A)
 */
void deriv_sigmoid(const GpuMat& A, GpuMat& C);

/**
 * @brief back_delta_sigmoid
 * @param sigmoid = (target - sigmoid) * sigmoid * (1 - sigmoid)
 * @param target
 */
void back_delta_sigmoid(GpuMat &sigmoid, const GpuMat &target);

/**
 * @brief back_delta_sigmoid
 * @param sigmoid = (target - sigmoid) * sigmoid * (1 - sigmoid)
 * @param target
 */
void back_delta_sigmoid(GpuMat &sigmoid, const GpuMat &target, const GpuMat& mulColumn);

/**
 * @brief deriv_sigmoid
 * @param A
 */
void deriv_sigmoid(GpuMat& A);

/**
 * @brief tanh
 * @param A
 * @param C - out C = tanh(A)
 */
void tanh(const GpuMat& A, GpuMat& C);

/**
 * @brief tanh
 * @param A = tanh(A)
 */
void tanh(GpuMat& A);

/**
 * @brief deriv_tanh
 * @param A
 * @param C - out C = deriv_tanh(A)
 */
void deriv_tanh(const GpuMat& A, GpuMat& C);

/**
 * @brief deriv_tanh
 * @param A
 */
void deriv_tanh(GpuMat& A);

/**
 * @brief softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 * @param partZ = sum(exp(A), axis)
 */
void softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ);

/**
 * @brief softmax
 * @param A = softmax(A)
 * @param axis -> 0 - in row, 1 - in col
 * @param partZ = sum(exp(A), axis)
 */
void softmax(GpuMat& A, int axis, GpuMat& partZ);

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void sub_adamGrad(GpuMat& A, const GpuMat& gA, GpuMat &mA, GpuMat &vA,
				  double alpha, double sb1, double sb2, double betha1, double betha2);

/**
 * @brief subIndOne
 * @param A
 * @param Ind
 * @param B
 */
void subIndOne(const GpuMat& A, const GpuMat& Ind, GpuMat& B);

/**
 * @brief subIndOne
 * @param vA
 * @param Ind
 * @param B
 */
void subIndOne(const std::vector< GpuMat >& vA, const GpuMat& Ind, std::vector<GpuMat> &B);

/**
 * @brief hconcat
 * @param list
 * @param res
 */
void hconcat2(const std::vector<GpuMat> &list, gpumat::GpuMat& res);

/**
 * @brief hsplit
 * @param res
 * @param cols
 * @param list
 */
void hsplit2(const GpuMat& res, std::vector< int > cols, std::vector< GpuMat >& list);

/**
 * @brief mul2deriv
 * @param D
 * @param A
 * @param func
 * @param DA
 * @param param1
 * @param param2
 * @param param3
 */
void mul2deriv(const GpuMat& D, const gpumat::GpuMat& A, gpumat::etypefunction func, gpumat::GpuMat& DA,
			   double param1 = 0, double param2 = 0, double param3 = 0);

/**
 * @brief m2mpbaf
 * matmul A on B plus bias C and apply function func
 * @param A
 * @param B
 * @param C	- bias
 * @param func - not SOFTMAX
 * @param D - result
 * @param param1 - for LEAKYRELU parameter of multiplication
 * @param param2 - for future
 * @param param3 - for future
 */
void m2mpbaf(const GpuMat& A, const GpuMat& B, const GpuMat& C, etypefunction func, GpuMat& D,
			 double param1 = 0, double param2 = 0, double param3 = 0);

/**
 * @brief momentum_optimizer
 * @param W
 * @param M
 * @param G
 * @param alpha
 * @param betha
 */
void momentum_optimizer(GpuMat& W, GpuMat& M, const GpuMat& G, double alpha, double betha);

}

#endif // GPU_H
