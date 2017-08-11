#ifndef MLP_MIXED_H
#define MLP_   MIXED_H

#include "mlp.h"
#include "gpumat.h"
#include <map>

namespace ct{

class mlp_mixed
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

	mlp_mixed();

	Matf& Y();

	void setLambda(float val);
	void setParams(etypefunction type, double params);

	void setDropout(bool val);
	void setDropout(float val);

	bool isInit() const;

	void init(int input, int output);

	inline void apply_func(gpumat::GpuMat &Z, etypefunction func);
	inline void apply_back_func(gpumat::GpuMat& D1, etypefunction func);

	etypefunction funcType() const;

	void forward(const ct::Matf *mat, etypefunction func = RELU, bool save_A0 = true);
	void backward(const ct::Matf &Delta, bool last_layer = false);

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

class MlpOptimMixed: public AdamOptimizer<float>{
public:
	MlpOptimMixed();

#define AO this->

	bool init(std::vector<mlp_mixed> &Mlp);

	bool pass(std::vector<mlp_mixed> &Mlp);
};

}

#endif // MLP_MIXED_H
