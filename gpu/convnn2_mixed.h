#ifndef CONVNN2_MIXED_H
#define CONVNN2_MIXED_H

#include "convnn2.h"
#include "nn.h"

namespace conv2{

class AdamOptimizerMixed: public ct::Optimizer<float>{
public:
	AdamOptimizerMixed();

	float betha1() const;

	void setBetha1(float v);

	float betha2() const;

	void setBetha2(float v);

	bool init(const std::vector< ct::Matf >& W, const std::vector< ct::Matf >& B);

	bool pass(const std::vector< ct::Matf >& gradW, const std::vector< ct::Matf >& gradB,
			  std::vector< ct::Matf >& W, std::vector< ct::Matf >& b);

	bool empty() const;


protected:
	float m_betha1;
	float m_betha2;
	bool m_init;

	std::vector< ct::Matf > m_mW;
	std::vector< ct::Matf > m_mb;
	std::vector< ct::Matf > m_vW;
	std::vector< ct::Matf > m_vb;
};


/////////////////////////////

class convnn2_mixed: public convnn_abstract<float>
{
public:
	std::vector< ct::Matf> W;			/// weights
	std::vector< ct::Matf> B;			/// biases
	int stride;
	ct::Size szW;							/// size of weights
	std::vector< ct::Matf>* pX;			/// input data
	std::vector< ct::Matf> Xc;			///
	std::vector< ct::Matf> A1;			/// out after appl nonlinear function
	std::vector< ct::Matf> A2;			/// out after pooling
	std::vector< ct::Matf> Dlt;			/// delta after backward pass
	std::vector< ct::Matf> vgW;			/// for delta weights
	std::vector< ct::Matf> vgB;			/// for delta bias
	std::vector< ct::Matf> Mask;		/// masks for bakward pass (created in forward pass)
	ct::Optimizer< float > *m_optim;
	AdamOptimizerMixed m_adam;

	std::vector< ct::Matf> gW;			/// gradient for weights
	std::vector< ct::Matf> gB;			/// gradient for biases

	std::vector< ct::Matf> dSub;
	std::vector< ct::Matf> Dc;

	convnn2_mixed();

	void setOptimizer(ct::Optimizer<float>* optim);

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

	bool use_pool() const;

	int outputFeatures() const;

	ct::Size szOut() const;

	void setAlpha(float alpha);

	void setLambda(float val);

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW,
			  bool use_pool = true, bool use_transpose = true);

	void forward(const std::vector< ct::Matf>* _pX, ct::etypefunction func);

	void forward(const convnn<float> & conv, ct::etypefunction func);

	inline void backcnv(const std::vector< ct::Matf>& D, std::vector< ct::Matf>& DS);

	void backward(const std::vector< ct::Matf>& D, bool last_level = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

	void write2(std::fstream& fs);

	void read2(std::fstream& fs);

private:
	bool m_use_pool;
	ct::etypefunction m_func;
	bool m_use_transpose;
	float m_Lambda;
};

}

#endif // CONVNN2_MIXED_H
