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

class GPU_EXPORTS GpuMat{
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

    void reshape(int new_rows, int new_cols);

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

/////////////////////////

class GPU_EXPORTS _BN{
public:
	_BN(){
		X = Y = D = 0;
		channels = 1;
	}
	int channels;

	/// inputs and output;
	std::vector< GpuMat > *X;
	std::vector< GpuMat > *Y;
	std::vector< GpuMat > *D;

	/// internal variables
	std::vector< GpuMat > Xu;
	GpuMat Mean;
	GpuMat Var;
	GpuMat gamma;
	GpuMat betha;

//	GpuMat dMean;
	GpuMat dVar;
	GpuMat dgamma;
	GpuMat dbetha;
	std::vector< GpuMat > Dout;
};

/////////////////////////

/**
 * @brief memset
 * @param A
 * @param val
 */
void GPU_EXPORTS memset(GpuMat& A, double val);
/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
void GPU_EXPORTS add(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief add
 * @param A
 * @param B
 * @param C
 * @param valA
 * @param valB
 */
void GPU_EXPORTS add(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA = 1., double valB = 1.);
/**
 * @brief add
 * @param A -> A = valA * A + valB * B
 * @param valA
 * @param B
 * @param valB
 */
void GPU_EXPORTS add(GpuMat& A, const GpuMat& B, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void GPU_EXPORTS sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A = A * valA - B * valB
 * @param B
 */
void GPU_EXPORTS sub(GpuMat& A, const GpuMat& B, double valA = 1., double valB = 1.);
/**
 * @brief sub
 * @param A - = (A .- B) * mulColumn
 * @param B
 * @param mulColumn
 */
void GPU_EXPORTS subWithColumn(GpuMat& A, const GpuMat& B, const GpuMat& mulColumn, double valA = 1., double valB = 1.);

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void GPU_EXPORTS matmul(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmul
 * @param A
 * @param B
 * @param C - out C += A * B
 */
void GPU_EXPORTS add2matmul(const GpuMat &A, const GpuMat &B, GpuMat &C);

/**
 * @brief matmul_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void GPU_EXPORTS matmul_shared(const GpuMat& A, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void GPU_EXPORTS matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C += A' * B
 */
void GPU_EXPORTS add2matmulT1(const GpuMat &At, const GpuMat &B, GpuMat &C);

/**
 * @brief matmulT1_shared
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void GPU_EXPORTS matmulT1_shared(const GpuMat& At, const GpuMat& B, GpuMat& C, double alpha = 1.);

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void GPU_EXPORTS matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha = 1.);

/**
 * @brief add2matmulT2
 * @param A
 * @param Bt - B
 * @param C -out C += A * B'
 */
void GPU_EXPORTS add2matmulT2(const GpuMat &A, const GpuMat &Bt, GpuMat &C);

/**
 * @brief matmulT2_shared
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void GPU_EXPORTS matmulT2_shared(const GpuMat& A, const GpuMat& Bt, GpuMat& C, double alpha = 1.);

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
void GPU_EXPORTS mulval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
void GPU_EXPORTS mulval(GpuMat& A, double value);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
void GPU_EXPORTS addval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 */
void GPU_EXPORTS addval(GpuMat &A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
void GPU_EXPORTS subval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
void GPU_EXPORTS subval(double value, const GpuMat& A, GpuMat& C);

/**
 * @brief subval
 * @param A - > A - value
 * @param value - mat 1x1
 */
void GPU_EXPORTS subval(GpuMat& A, double value);

/**
 * @brief subval
 * @param A -> value - A
 * @param value - mat 1x1
 */
void GPU_EXPORTS subval(double value, GpuMat& A);

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
void GPU_EXPORTS biasPlus(GpuMat& A, const GpuMat& bias);

/**
 * @brief scale_and_shift
 * @param A
 * @param scales
 * @param biases
 * @param C[i, j] = A[i,j] * scales[j] + biases[j]
 */
void GPU_EXPORTS scale_and_shift(const GpuMat& A, const GpuMat& scales, const GpuMat& biases, GpuMat& C);

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
void GPU_EXPORTS elemwiseMult(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseMul
 * @param A = A.* B
 * @param B
 */
void GPU_EXPORTS elemwiseMult(GpuMat& A, const GpuMat& B);

/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
void GPU_EXPORTS elemwiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseSqrt
 * @param A
 * @param B
 * @param C - out C = sqrt(A)
 */
void GPU_EXPORTS elemwiseSqrt(const GpuMat& A, GpuMat& C);

/**
 * @brief elemiseSqr
 * @param A
 * @param B
 * @param C - out C = sqrt(A)
 */
void GPU_EXPORTS elemwiseSqr(const GpuMat& A, GpuMat& C);

/**
 * @brief sumRows
 * @param A
 * @param C - out C[j] = val * sum(A[i, j]) (i = [1..rows])
 */
void GPU_EXPORTS sumRows(const GpuMat& A, GpuMat& C, double val = 1.);

/**
 * @brief sumCols
 * @param A
 * @param C - out C[i] = val * sum(A[i, j]) (j = [1..cols])
 */
void GPU_EXPORTS sumCols(const GpuMat& A, GpuMat& C, double val = 1.);

/**
 * @brief add2sumRows
 * @param A
 * @param C
 * @param val - out C[i] += val * sum(A[i, j]) (j = [1, cols])
 */
void GPU_EXPORTS add2sumRows(const GpuMat &A, GpuMat &C, double val);

/**
 * @brief sumRows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j]) (j = [1, cols])
 */
void GPU_EXPORTS sumRows_shared(const GpuMat& A, GpuMat& C, double val = 1.);

/**
 * @brief transpose
 * @param A
 * @param C - out C = A'
 */
void GPU_EXPORTS transpose(const GpuMat& A, GpuMat& C);

/**
 * @brief reLu
 * @param A
 * @param C - out C = reLu(A)
 */
void GPU_EXPORTS reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief reLu
 * @param A
 */
void GPU_EXPORTS reLu(GpuMat& A);

/**
 * @brief deriv_reLu
 * @param A
 * @param C - out C = deriv_reLu(A)
 */
void GPU_EXPORTS deriv_reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief deriv_reLu
 * @param A
 */
void GPU_EXPORTS deriv_reLu(GpuMat& A);

////

/**
 * @brief leakyReLu
 * @param A
 * @param C - out C = reLu(A)
 */
void GPU_EXPORTS leakyReLu(const GpuMat& A, double x, GpuMat& C);

/**
 * @brief leakyReLu
 * @param A
 */
void GPU_EXPORTS leakyReLu(GpuMat& A, double x);

/**
 * @brief deriv_leakyReLu
 * @param A
 * @param C - out C = deriv_reLu(A)
 */
void GPU_EXPORTS deriv_leakyReLu(const GpuMat& A, double x, GpuMat& C);

/**
 * @brief deriv_leakyReLu
 * @param A
 */
void GPU_EXPORTS deriv_leakyReLu(GpuMat& A, double x);

////

/**
 * @brief sigmoid
 * @param A
 * @param C - out C = sigmoid(A)
 */
void GPU_EXPORTS sigmoid(const GpuMat& A, GpuMat& C);

/**
 * @brief sigmoid
 * @param A
 */
void GPU_EXPORTS sigmoid(GpuMat& A);

/**
 * @brief deriv_sigmoid
 * @param A
 * @param C - out C = deriv_sigmoid(A)
 */
void GPU_EXPORTS deriv_sigmoid(const GpuMat& A, GpuMat& C);

/**
 * @brief back_delta_sigmoid
 * @param sigmoid = (target - sigmoid) * sigmoid * (1 - sigmoid)
 * @param target
 */
void GPU_EXPORTS back_delta_sigmoid(GpuMat &sigmoid, const GpuMat &target);

/**
 * @brief back_delta_sigmoid
 * @param sigmoid = (target - sigmoid) * sigmoid * (1 - sigmoid)
 * @param target
 */
void GPU_EXPORTS back_delta_sigmoid(GpuMat &sigmoid, const GpuMat &target, const GpuMat& mulColumn);

/**
 * @brief deriv_sigmoid
 * @param A
 */
void GPU_EXPORTS deriv_sigmoid(GpuMat& A);

/**
 * @brief tanh
 * @param A
 * @param C - out C = tanh(A)
 */
void GPU_EXPORTS tanh(const GpuMat& A, GpuMat& C);

/**
 * @brief tanh
 * @param A = tanh(A)
 */
void GPU_EXPORTS tanh(GpuMat& A);

/**
 * @brief deriv_tanh
 * @param A
 * @param C - out C = deriv_tanh(A)
 */
void GPU_EXPORTS deriv_tanh(const GpuMat& A, GpuMat& C);

/**
 * @brief deriv_tanh
 * @param A
 */
void GPU_EXPORTS deriv_tanh(GpuMat& A);

/**
 * @brief softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 * @param partZ = sum(exp(A), axis)
 */
void GPU_EXPORTS softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ);

/**
 * @brief softmax
 * @param A = softmax(A)
 * @param axis -> 0 - in row, 1 - in col
 * @param partZ = sum(exp(A), axis)
 */
void GPU_EXPORTS softmax(GpuMat& A, int axis, GpuMat& partZ);

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void GPU_EXPORTS sub_adamGrad(GpuMat& A, const GpuMat& gA, GpuMat &mA, GpuMat &vA,
				  double alpha, double sb1, double sb2, double betha1, double betha2);

/**
 * @brief adagrad
 * @param A
 * @param gA
 * @param hist_gA
 * @param alpha
 * @param betha
 */
void GPU_EXPORTS adagrad(GpuMat& A, GpuMat& hist_gA, const GpuMat& gA, double alpha, double betha);

/**
 * @brief subIndOne
 * @param A
 * @param Ind
 * @param B
 */
void GPU_EXPORTS subIndOne(const GpuMat& A, const GpuMat& Ind, GpuMat& B);

/**
 * @brief subIndOne
 * @param vA
 * @param Ind
 * @param B
 */
void GPU_EXPORTS subIndOne(const std::vector< GpuMat >& vA, const GpuMat& Ind, std::vector<GpuMat> &B);

/**
 * @brief hconcat
 * @param list
 * @param res
 */
void GPU_EXPORTS hconcat2(const std::vector<GpuMat> &list, gpumat::GpuMat& res);

/**
 * @brief hsplit
 * @param res
 * @param cols
 * @param list
 */
void GPU_EXPORTS hsplit2(const GpuMat& res, std::vector< int > cols, std::vector< GpuMat >& list);

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
void GPU_EXPORTS mul2deriv(const GpuMat& D, const gpumat::GpuMat& A, gpumat::etypefunction func, gpumat::GpuMat& DA,
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
void GPU_EXPORTS m2mpbaf(const GpuMat& A, const GpuMat& B, const GpuMat& C, etypefunction func, GpuMat& D,
			 double param1 = 0, double param2 = 0, double param3 = 0);

/**
 * @brief momentum_optimizer
 * @param W
 * @param M
 * @param G
 * @param alpha
 * @param betha
 */
void GPU_EXPORTS momentum_optimizer(GpuMat& W, GpuMat& M, const GpuMat& G, double alpha, double betha);

}

#endif // GPU_H
