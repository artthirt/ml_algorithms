#ifndef CONVNN2_MIXED_H
#define CONVNN2_MIXED_H

#include "convnn2.h"
#include "nn.h"

#include "gpumat.h"

#include <vector>
#include "optim_mixed.h"

namespace conv2{

/////////////////////////////
/// \brief The convnn2_mixed class
///
class convnn2_mixed: public convnn_abstract<float>
{
public:
	ct::Matf W;			/// weights
	ct::Matf B;			/// biases
	int stride;
	ct::Size szW;							/// size of weights
	std::vector< ct::Matf>* pX;			/// input data
	std::vector< ct::Matf> Xc;			///
	std::vector< ct::Matf> A1;			/// out after appl nonlinear function
	std::vector< ct::Matf> A2;			/// out after pooling
	std::vector< ct::Matf> A3;			/// out after BN
	std::vector< ct::Matf> Dlt;			/// delta after backward pass
	std::vector< ct::Matf> Mask;		/// masks for bakward pass (created in forward pass)

	ct::BN<float> bn;

	ct::Matf gW;			/// gradient for weights
	ct::Matf gB;			/// gradient for biases

	std::vector< ct::Matf> dSub;
	std::vector< ct::Matf> Dc;

	convnn2_mixed();

	void setParams(ct::etypefunction type, float param);

	std::vector< ct::Matf>& XOut();

	const std::vector< ct::Matf>& XOut() const;
	/**
	 * @brief XOut1
	 * out after convolution
	 * @return
	 */
	std::vector< ct::Matf>& XOut1();
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< ct::Matf>& XOut2();
	/**
	 * @brief XOut3
	 * out after BN
	 * @return
	 */
	std::vector< ct::Matf>& XOut3();

	bool use_pool() const;

	bool use_bn() const;

	int outputFeatures() const;

	ct::Size szOut() const;

	void setLambda(float val);

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW, ct::etypefunction func,
			  bool use_pool, bool use_bn, bool use_transpose);

	void forward(const std::vector< ct::Matf>* _pX);

	void forward(const convnn2_mixed & conv);

	void backcnv(const gpumat::GpuMat& D, gpumat::GpuMat &A1, gpumat::GpuMat& DS);

	void backward(const std::vector< ct::Matf>& D, bool last_level = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

	void write2(std::fstream& fs);

	void read2(std::fstream& fs);

private:
	bool m_use_pool;
	bool m_use_bn;
	ct::etypefunction m_func;
	bool m_use_transpose;
	float m_Lambda;
	std::map< ct::etypefunction, float > m_params;
};

/////////////////////
/// \brief The CnvAdamOptimizerMixed class
///
class CnvAdamOptimizerMixed: public ct::AdamOptimizerMixed
{
public:
	CnvAdamOptimizerMixed();

	bool init(const std::vector<convnn2_mixed>& cnv);
	bool pass(std::vector<convnn2_mixed>& cnv);
};

/////////////////////
/// \brief The CnvMomentumOptimizerMixed class
///
class CnvMomentumOptimizerMixed: public ct::MomentumOptimizerMixed
{
public:
	CnvMomentumOptimizerMixed();

	bool init(const std::vector<convnn2_mixed>& cnv);
	bool pass(std::vector<convnn2_mixed>& cnv);
};

}

#endif // CONVNN2_MIXED_H
