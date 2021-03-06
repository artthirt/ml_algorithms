#ifndef CONV2_GPU_H
#define CONV2_GPU_H

#include "gpumat.h"
#include "helper_gpu.h"
//#include "cuda_common.h"
#include <map>

namespace gpumat{

////////////////////////

class GPU_EXPORTS BN: public _BN{
public:
	BN();
	bool train;

	/**
	 * @brief normalize
	 */
	void normalize();
	/**
	 * @brief denormalize
	 */
	void denormalize();
	/**
	 * @brief initGammaAndBetha
	 */
	void initGammaAndBetha();
	/**
	 * @brief scaleAndShift
	 */
	void scaleAndShift();

	void read(std::fstream& fs);
	void write(std::fstream& fs);
};

//////////////////////

class GPU_EXPORTS convnn_gpu
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
	std::vector< gpumat::GpuMat > A3;		/// out after BN
	std::vector< gpumat::GpuMat > Dlt;		/// delta after backward pass
	std::vector< gpumat::GpuMat > Mask;		/// masks for bakward pass (created in forward pass)
	gpumat::GpuMat gW;		/// gradient for weights
	gpumat::GpuMat gB;		/// gradient for biases

	BN bn;					/// batch normalize

	bool m_pool_dropout;
	double m_prob_dropout;

	convnn_gpu();

	void setTrainMode(bool val);

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
	/**
	 * @brief XOut3
	 * @return after batch normalize
	 */
	std::vector<gpumat::GpuMat> &XOut3();

	bool use_pool() const;

	bool use_bn() const;

	int outputFeatures() const;

	ct::Size szOut() const;

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW,
			  etypefunction func, bool use_pool, bool use_bn, bool use_transpose, bool use_same = false);

	void forward(const std::vector< gpumat::GpuMat >* _pX);

	void backcnv(const std::vector< gpumat::GpuMat >& D, std::vector< gpumat::GpuMat >& DS);

	void backward(const std::vector< gpumat::GpuMat >& D, bool last_level = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

	void write2(std::fstream& fs);
	void read2(std::fstream& fs);

private:
	bool m_use_bn;
	bool m_use_pool;
	bool m_use_same;
	gpumat::etypefunction m_func;
	std::vector< gpumat::GpuMat > m_Dropout;
	double m_lambda;
	std::map< etypefunction, double > m_params;

	std::vector< gpumat::GpuMat > dSub2;
	std::vector< gpumat::GpuMat > Dc;		///
//	std::vector< gpumat::GpuMat > DA1;		///
	bool m_use_transpose;
};

//////////////////////

class GPU_EXPORTS CnvAdamOptimizer: public AdamOptimizer
{
public:
	CnvAdamOptimizer();

	int stop_layer;

	std::vector< GpuMat > mG, mB, vG, vB;

	bool init(std::vector<convnn_gpu> &cnv);
	bool pass(std::vector<convnn_gpu> &cnv);
};

class GPU_EXPORTS CnvMomentumOptimizer: public MomentumOptimizer
{
public:
	CnvMomentumOptimizer();

	int stop_layer;

	std::vector< GpuMat > mG, mB;

	bool init(std::vector<convnn_gpu> &cnv);
	bool pass(std::vector<convnn_gpu> &cnv);
};

class GPU_EXPORTS CnvAdaGradOptimizer: public AdaGradOptimizer
{
public:
    CnvAdaGradOptimizer();

    int stop_layer;

    std::vector< GpuMat > histG;
    std::vector< GpuMat > histB;

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
void GPU_EXPORTS im2cols(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2colsT(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2cols(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2colsT(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2cols_same(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2colsT_same(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2cols_same(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS im2colsT_same(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
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
void GPU_EXPORTS cols2im(const gpumat::GpuMat& Delta, const ct::Size& szOut, const ct::Size& szA0,
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
void GPU_EXPORTS cols2im(const std::vector< gpumat::GpuMat >& Delta,
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
void GPU_EXPORTS cols2imT(const gpumat::GpuMat& Delta, const ct::Size& szDelta, const ct::Size& szA0,
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
void GPU_EXPORTS cols2imT(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size& szOut,
				const ct::Size& szA0,
				int channels,
				const ct::Size& szW,
				int stride,
				std::vector< gpumat::GpuMat >& X);

/////////////// transpose convolution SAME ///

/**
 * @brief cols2im
 * @param Delta
 * @param szDelta
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void GPU_EXPORTS cols2im_same(const gpumat::GpuMat& Delta,
				  const ct::Size &szDelta, const ct::Size& szA0,
				  int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief cols2im
 * @param Delta
 * @param szDelta
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void GPU_EXPORTS cols2im_same(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size &szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride,
				std::vector< gpumat::GpuMat >& X);

/**
 * @brief cols2imT
 * @param Delta
 * @param szDelta
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void GPU_EXPORTS cols2imT_same(const gpumat::GpuMat& Delta, const ct::Size &szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief cols2imT
 * @param Delta
 * @param szDelta
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void GPU_EXPORTS cols2imT_same(const std::vector< gpumat::GpuMat >& Delta,
				   const ct::Size &szDelta, const ct::Size& szA0,
				   int channels, const ct::Size& szW, int stride,
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
void GPU_EXPORTS conv2(const GpuMat& A, const ct::Size &szA, int channels, int stride, const GpuMat &B,
		   const ct::Size &szB, GpuMat &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);

/**
 * @brief conv2_transpose
 * @param C
 * @param szA
 * @param channels
 * @param stride
 * @param B
 * @param szB
 * @param szOut
 * @param A
 * @param type
 * @param transpose
 */
void GPU_EXPORTS conv2_transpose(const GpuMat& C, const ct::Size &szA, int channels, int stride, const GpuMat &B,
		   const ct::Size &szB, const ct::Size &szOut, GpuMat &A, TYPE_CONV type = VALID, bool transpose = false);


/////////////// subsample 2x2 ////////////////

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void GPU_EXPORTS subsample(const GpuMat& X, const ct::Size& szA, GpuMat& Y, GpuMat& Mask, ct::Size& szO);

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void GPU_EXPORTS subsample(const std::vector< GpuMat >& X, const ct::Size& szA, std::vector< GpuMat >& Y, std::vector< GpuMat >& Mask, ct::Size& szO);

//////////// upsample 2x2 /////////////////////

void GPU_EXPORTS upsample(const GpuMat& Y,int K, const GpuMat& Mask, const ct::Size& szO,
			  const ct::Size& szA, GpuMat& X);

void GPU_EXPORTS upsample(const std::vector< GpuMat >& Y, int K, const std::vector< GpuMat >& Mask, const ct::Size& szO,
			  const ct::Size& szA, std::vector< GpuMat >& X);

//////////// vector of row to matrix //////////////////////

/**
 * @brief vec2mat
 * @param vec
 * @param mat
 */
void GPU_EXPORTS vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat);

/**
 * @brief mat2vec
 * @param mat
 * @param szOut
 * @param vec
 */
void GPU_EXPORTS mat2vec(const GpuMat& mat, const ct::Size& szOut, std::vector< GpuMat >& vec);

////////// addition all matrices in vector /////////////

/**
 * @brief addvec
 * @param W
 * @param vW
 * @param alpha
 */
void GPU_EXPORTS addvec(GpuMat& W, const std::vector< GpuMat >& vW, double alpha);

}

#endif // CONV2_GPU_H
