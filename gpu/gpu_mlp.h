#ifndef GPU_MLP_H
#define GPU_MLP_H

#include "custom_types.h"
#include "gpumat.h"
//#include "cuda_common.h"
#include "helper_gpu.h"

#include <map>

namespace gpumat{

class GPU_EXPORTS mlp{
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

	std::vector< GpuMat > *pVecA0;
	std::vector< GpuMat > vecA1;
	std::vector< GpuMat > vecXDropout;
	std::vector< GpuMat > vecDltA0;

	mlp();

	void setParams(etypefunction type, double param);
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
	void init(int input, int output, int type, etypefunction func);
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
    inline void apply_back_func(const GpuMat& D1, const GpuMat &A1, GpuMat& D2);
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
	void forward(const GpuMat *mat, bool save_A0 = true);
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
    void forward(const std::vector<GpuMat> *mat, bool save_A0 = true);
	/**
	 * @brief backward
	 * @param Delta
	 * @param last_layer
	 */
	void backward(std::vector<GpuMat> &Delta, bool last_layer = false);

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
	std::map< etypefunction, double > m_params;
	etypefunction m_func;
};

class GPU_EXPORTS MlpOptimAdam: public AdamOptimizer{
public:
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

class GPU_EXPORTS MlpOptimSG: public StohasticGradientOptimizer{
public:
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

class GPU_EXPORTS MlpOptimMoment: public MomentumOptimizer{
public:
	MlpOptimMoment();
	bool init(const std::vector< gpumat::mlp >& _mlp);
	bool pass(std::vector<mlp> &_mlp);
private:
};

class MlpOptimAdaGrad: public AdaGradOptimizer{
public:
    MlpOptimAdaGrad();
    bool init(const std::vector< gpumat::mlp >& _mlp);
    bool pass(std::vector<mlp> &_mlp);
private:
};

void maxnorm(GpuMat& A, double c);

}

#endif // GPU_MLP_H
