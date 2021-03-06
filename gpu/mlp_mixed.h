#ifndef MLP_MIXED_H
#define MLP_   MIXED_H

#include "mlp.h"
#include "gpumat.h"
#include <map>

#include "optim_mixed.h"

namespace ct{

class GPU_EXPORTS mlp_mixed
{
public:
	Matf *pA0;
	Matf W;
	Matf B;
//	Matf Z;
	Matf A1;
	Matf DA1;
	Matf D1;
	Matf DltA0;
	Matf Dropout;
	Matf XDropout;
	Matf gW;
	Matf gB;

	std::vector< Matf > *pVecA0;
	std::vector< Matf > vecA1;
	std::vector< Matf > vecXDropout;
	std::vector< Matf > vecDltA0;

/////////////////////

	mlp_mixed();

	Matf& Y();

	void setLambda(float val);
	void setParams(etypefunction type, double params);

	void setDropout(bool val);
	void setDropout(float val);

	bool isInit() const;

	void init(int input, int output, etypefunction func);

	inline void apply_func(gpumat::GpuMat &Z, etypefunction func);
	inline void apply_back_func(gpumat::GpuMat& D1, const Matf& A1, etypefunction func);

	etypefunction funcType() const;

	/**
	 * @brief forward
	 * @param mat
	 * @param save_A0
	 */
	void forward(const ct::Matf *mat, bool save_A0 = true);
	/**
	 * @brief backward
	 * @param Delta
	 * @param last_layer
	 */
	void backward(const ct::Matf &Delta, bool last_layer = false);

	/**
	 * @brief forward
	 * @param mat
	 * @param func
	 * @param save_A0
	 */
	void forward(const std::vector<Matf> *mat, bool save_A0 = true);
	/**
	 * @brief backward
	 * @param Delta
	 * @param last_layer
	 */
	void backward(std::vector<Matf> &Delta, bool last_layer = false);

	void write(std::fstream& fs);

	void read(std::fstream& fs);

	void write2(std::fstream &fs);

	void read2(std::fstream &fs);

private:
	bool m_init;
	bool m_is_dropout;
	float m_prob;
	float m_lambda;
	etypefunction m_func;
	std::map< etypefunction, double > m_params;
};

////////////////////////

class GPU_EXPORTS MlpAdamOptimizerMixed: public ct::AdamOptimizerMixed
{
public:
	MlpAdamOptimizerMixed();

	bool init(const std::vector<mlp_mixed> &mlp);
	bool pass(std::vector<mlp_mixed>& mlp);
};

class GPU_EXPORTS MlpMomentumOptimizerMixed: public ct::MomentumOptimizerMixed
{
public:
	MlpMomentumOptimizerMixed();

	bool init(const std::vector<mlp_mixed> &mlp);
	bool pass(std::vector<mlp_mixed>& mlp);
};

};

#endif // MLP_MIXED_H
