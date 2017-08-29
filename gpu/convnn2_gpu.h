#ifndef CONV2_GPU_H
#define CONV2_GPU_H

#include "gpumat.h"
#include "helper_gpu.h"
#include "cuda_common.h"
#include <map>

namespace gpumat{

class convnn_gpu
{
public:
	gpumat::GpuMat W;		/// weights
	gpumat::GpuMat B;		/// biases
	int kernels;									/// kernels
	int channels;							/// input channels
	int stride;
	ct::Size szA0;							/// input size
	ct::Size szA1;							/// size after convolution
	ct::Size szA2;							/// size after pooling
	ct::Size szW;							/// size of weights
	ct::Size szK;							/// size of output data (set in forward)
	std::vector< gpumat::GpuMat >* pX;		/// input data
	std::vector< gpumat::GpuMat > Xc;		///
	std::vector< gpumat::GpuMat > A1;		/// out after appl nonlinear function
	std::vector< gpumat::GpuMat > A2;		/// out after pooling
	std::vector< gpumat::GpuMat > Dlt;		/// delta after backward pass
	std::vector< gpumat::GpuMat > Mask;		/// masks for bakward pass (created in forward pass)
	gpumat::GpuMat gW;		/// gradient for weights
	gpumat::GpuMat gB;		/// gradient for biases

	bool m_pool_dropout;
	double m_prob_dropout;

	convnn_gpu();

	void setLambda(double val);

	void setDropout(bool val);
	void setDropout(double val);

	void setParams(etypefunction type, double param);

	std::vector<gpumat::GpuMat> &XOut();
	/**
	 * @brief XOut1
	 * out after convolution
	 * @return
	 */
	std::vector< gpumat::GpuMat >& XOut1();
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< gpumat::GpuMat >& XOut2();

	bool use_pool() const;

	int outputFeatures() const;

	ct::Size szOut() const;

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW, etypefunction func, bool use_pool = true, bool use_transpose = true);

	void forward(const std::vector< gpumat::GpuMat >* _pX);

	void backcnv(const std::vector< gpumat::GpuMat >& D, std::vector< gpumat::GpuMat >& DS);

	void backward(const std::vector< gpumat::GpuMat >& D, bool last_level = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

	void write2(std::fstream& fs);
	void read2(std::fstream& fs);

private:
	bool m_use_pool;
	gpumat::etypefunction m_func;
	gpumat::GpuMat m_Dropout;
	double m_lambda;
	std::map< etypefunction, double > m_params;

	std::vector< gpumat::GpuMat > dSub2;
	std::vector< gpumat::GpuMat > Dc;		///
//	std::vector< gpumat::GpuMat > DA1;		///
	bool m_use_transpose;
};

//////////////////////

class CnvAdamOptimizer: public AdamOptimizer
{
public:
	CnvAdamOptimizer();

	bool init(std::vector<convnn_gpu> &cnv);
	bool pass(std::vector<convnn_gpu> &cnv);
};

class CnvMomentumOptimizer: public MomentumOptimizer
{
public:
	CnvMomentumOptimizer();

	bool init(std::vector<convnn_gpu> &cnv);
	bool pass(std::vector<convnn_gpu> &cnv);
};

//////////////////////

/**
 * @brief im2cols
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2colsT
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2cols
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

/**
 * @brief im2colsT
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

/////// same ///////

/**
 * @brief im2cols_same
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols_same(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2colsT_same
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT_same(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2cols_same
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols_same(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

/**
 * @brief im2colsT_same
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT_same(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

//////////////////// back convolution ////////////////
/**
 * @brief cols2im
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void cols2im(const gpumat::GpuMat& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief cols2im
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void cols2im(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size& szOut,
				const ct::Size& szA0,
				int channels,
				const ct::Size& szW,
				int stride,
				std::vector< gpumat::GpuMat >& X);

/**
 * @brief cols2imT
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void cols2imT(const gpumat::GpuMat& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief cols2imT
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void cols2imT(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size& szOut,
				const ct::Size& szA0,
				int channels,
				const ct::Size& szW,
				int stride,
				std::vector< gpumat::GpuMat >& X);

/////////////// conv2 ////////////////////////

enum TYPE_CONV{
	SAME,
	VALID
};

/**
 * @brief conv2
 * @param A
 * @param szA
 * @param channels
 * @param stride
 * @param B
 * @param szB
 * @param C
 */
void conv2(const GpuMat& A, const ct::Size &szA, int channels, int stride, const GpuMat &B,
		   const ct::Size &szB, GpuMat &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);

/////////////// subsample 2x2 ////////////////

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void subsample(const GpuMat& X, const ct::Size& szA, GpuMat& Y, GpuMat& Mask, ct::Size& szO);

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void subsample(const std::vector< GpuMat >& X, const ct::Size& szA, std::vector< GpuMat >& Y, std::vector< GpuMat >& Mask, ct::Size& szO);

//////////// upsample 2x2 /////////////////////

void upsample(const GpuMat& Y,int K, const GpuMat& Mask, const ct::Size& szO,
			  const ct::Size& szA, GpuMat& X);

void upsample(const std::vector< GpuMat >& Y, int K, const std::vector< GpuMat >& Mask, const ct::Size& szO,
			  const ct::Size& szA, std::vector< GpuMat >& X);

//////////// vector of row to matrix //////////////////////

/**
 * @brief vec2mat
 * @param vec
 * @param mat
 */
void vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat);

/**
 * @brief mat2vec
 * @param mat
 * @param szOut
 * @param vec
 */
void mat2vec(const GpuMat& mat, const ct::Size& szOut, std::vector< GpuMat >& vec);

////////// addition all matrices in vector /////////////

/**
 * @brief addvec
 * @param W
 * @param vW
 * @param alpha
 */
void addvec(GpuMat& W, const std::vector< GpuMat >& vW, double alpha);

////////// batch normalize /////////////
/**
 * @brief batch_normalize
 * @param X			- input matrices
 * @param Mean		- output mean
 * @param Sigma		- output sigma
 * @param Y			- output result
 * @param alpha
 * @param betha
 * Y = alpha * (X - Mean) / (sqrt(Sigma + 10e-8)) + betha
 */
void batch_normalize(const std::vector<GpuMat> &X, GpuMat &Mean, GpuMat &Sigma, std::vector<GpuMat> &Y,
					 double alpha = 1., double betha = 0., bool train = true);

void batch_denormalize(const std::vector<GpuMat> &D, const std::vector<GpuMat> &X, const GpuMat &Mean, const GpuMat &Sigma,
					   double& alpha, double &betha, std::vector<GpuMat> &Xout);

}

#endif // CONV2_GPU_H
