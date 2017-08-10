#ifndef GPU_MLP_H
#define GPU_MLP_H

#include "custom_types.h"
#include "gpumat.h"
#include "cuda_common.h"
#include "helper_gpu.h"

namespace gpumat{

class mlp{
public:
	GpuMat *pA0;
	GpuMat W;
	GpuMat B;
	GpuMat Z;
	GpuMat A1;
//	GpuMat DA1;
	GpuMat PartZ;
	GpuMat DltA0;
	GpuMat Dropout;
	GpuMat XDropout;
	GpuMat gW;
	GpuMat gB;

	std::vector< GpuMat > vecXDropout;
	std::vector< GpuMat > *pVecA0;
	std::vector< GpuMat > vecA1;
//	std::vector< GpuMat > vecDA1;
	std::vector< GpuMat > vecDltA0;
	GpuMat gWi;
	GpuMat gBi;

	mlp();

	/**
	 * @brief setLambda
	 * @param val
	 */
	void setLambda(double val);
	/**
	 * @brief setDropout
	 * @param val
	 */
	void setDropout(bool val);
	/**
	 * @brief setDropout
	 * @param val
	 */
	void setDropout(double val);
	/**
	 * @brief isInit
	 * @return
	 */
	bool isInit() const;
	/**
	 * @brief init
	 * @param input
	 * @param output
	 * @param type
	 */
	void init(int input, int output, int type);
	/**
	 * @brief apply_func
	 * @param Z
	 * @param A
	 * @param func
	 */
	inline void apply_func(const GpuMat& Z, GpuMat& A, etypefunction func);
//	/**
//	 * @brief apply_func
//	 * @param A
//	 * @param func
//	 */
//	inline void apply_func(GpuMat& A, etypefunction func);
	/**
	 * @brief apply_back_func
	 * @param D1
	 * @param D2
	 * @param func
	 */
	inline void apply_back_func(const GpuMat& D1, const GpuMat &A1, GpuMat& D2, etypefunction func);
	/**
	 * @brief funcType
	 * @return
	 */
	etypefunction funcType() const;
	/**
	 * @brief forward
	 * @param mat
	 * @param func
	 * @param save_A0
	 */
	void forward(const GpuMat *mat, etypefunction func = RELU, bool save_A0 = true);
	/**
	 * @brief backward
	 * @param Delta
	 * @param last_layer
	 */
	void backward(const GpuMat &Delta, bool last_layer = false);

	//////////

	/**
	 * @brief forward
	 * @param mat
	 * @param func
	 * @param save_A0
	 */
	void forward(const std::vector< GpuMat > *mat, etypefunction func = RELU, bool save_A0 = true);
	/**
	 * @brief backward
	 * @param Delta
	 * @param last_layer
	 */
	void backward(const std::vector< GpuMat > &Delta, bool last_layer = false);

	//////////

	/**
	 * @brief Y
	 * @return
	 */
	inline GpuMat &Y(){
		return A1;
	}

	/**
	 * @brief write
	 * write only data
	 * @param fs
	 */
	void write(std::fstream& fs);
	/**
	 * @brief read
	 * read only data
	 * @param fs
	 */
	void read(std::fstream& fs);

	/**
	 * @brief write2
	 * write data with sizes
	 * @param fs
	 */
	void write2(std::fstream& fs);
	/**
	 * @brief read2
	 * read data with sizes
	 * @param fs
	 */
	void read2(std::fstream& fs);

private:
	bool m_init;
	bool m_is_dropout;
	double m_prob;
	double m_lambda;
	etypefunction m_func;
};

class MlpOptim: public AdamOptimizer{
public:
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

class MlpOptimSG: public StohasticGradientOptimizer{
public:
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

class MlpOptimMoment: public MomentumOptimizer{
public:
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

void maxnorm(GpuMat& A, double c);

}

#endif // GPU_MLP_H
