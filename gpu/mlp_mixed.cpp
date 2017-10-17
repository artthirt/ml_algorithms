#include "mlp_mixed.h"

#include "gpumat.h"
#include "helper_gpu.h"

using namespace ct;

mlp_mixed::mlp_mixed(){
	m_func = RELU;
	m_init = false;
	m_is_dropout = false;
	m_prob = (float)0.95;
	pA0 = nullptr;
	m_lambda = 0;
	m_params[LEAKYRELU] = 0.1;
}

Matf &mlp_mixed::Y(){
	return A1;
}

void mlp_mixed::setLambda(float val){
	m_lambda = val;
}

void mlp_mixed::setParams(etypefunction type, double params)
{
	m_params[LEAKYRELU] = params;
}

void mlp_mixed::setDropout(bool val){
	m_is_dropout = val;
}

void mlp_mixed::setDropout(float val){
	m_prob = val;
}

bool mlp_mixed::isInit() const{
	return m_init;
}

void mlp_mixed::init(int input, int output, etypefunction func){
	double n = 1./sqrt(input);
	m_func = func;

	W.setSize(input, output);
	W.randn(0., n);
	B.setSize(1, output);
	B.randn(0, n);

	m_init = true;
}

void mlp_mixed::apply_func(gpumat::GpuMat &Z, etypefunction func)
{
	gpumat::GpuMat partZ;
	switch (func) {
		default:
		case LINEAR:
			return;
		case RELU:
			gpumat::reLu(Z);
			break;
		case SOFTMAX:
			gpumat::softmax(Z, 1, partZ);
			break;
		case SIGMOID:
			gpumat::sigmoid(Z);
			break;
		case TANH:
			gpumat::tanh(Z);
			break;
		case LEAKYRELU:
			gpumat::leakyReLu(Z, m_params[LEAKYRELU]);
			break;
	}
}

void mlp_mixed::apply_back_func(gpumat::GpuMat &D1, const Matf &A1, etypefunction func)
{
	if(func == LINEAR || func == SOFTMAX)
		return;

	gpumat::GpuMat g_A1;
	gpumat::convert_to_gpu(A1, g_A1);

	gpumat::mul2deriv(D1, g_A1, (gpumat::etypefunction)func, D1, m_params[LEAKYRELU]);
}

etypefunction mlp_mixed::funcType() const{
	return m_func;
}

inline void apply_dropout(const Matf& X, double prob, Matf& XDropout, Matf& Dropout)
{
	ct::dropout(X.rows, X.cols, (float)prob, Dropout);
	elemwiseMult(X, Dropout, XDropout);
}

void mlp_mixed::forward(const Matf *mat, bool save_A0)
{
	if(!m_init || !mat)
			throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pA0 = (Matf*)mat;

	gpumat::GpuMat g_XDropout, g_Z, g_B;

	{
		gpumat::GpuMat g_A0, g_W;
		gpumat::GpuMat partZ;

		gpumat::convert_to_gpu(*pA0, g_A0);
		gpumat::convert_to_gpu(W, g_W);
		gpumat::convert_to_gpu(B, g_B);

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			gpumat::GpuMat g_Dropout;

			ct::dropout(pA0->rows, pA0->cols, m_prob, Dropout);
			gpumat::convert_to_gpu(Dropout, g_Dropout);

			gpumat::elemwiseMult(g_A0, g_Dropout, g_XDropout);
			if(m_func == SOFTMAX){
				gpumat::m2mpbaf(g_XDropout, g_W, g_B, gpumat::LINEAR, g_Z, m_params[LEAKYRELU]);
				gpumat::softmax(g_Z, 1, partZ);
			}else
				gpumat::m2mpbaf(g_XDropout, g_W, g_B, (gpumat::etypefunction)m_func, g_Z, m_params[LEAKYRELU]);
			gpumat::convert_to_mat(g_XDropout, XDropout);
		}else{
			//gpumat::matmul(g_A0, g_W, g_Z);
			if(m_func == SOFTMAX){
				gpumat::m2mpbaf(g_A0, g_W, g_B, gpumat::LINEAR, g_Z, m_params[LEAKYRELU]);
				gpumat::softmax(g_Z, 1, partZ);
			}else
				gpumat::m2mpbaf(g_A0, g_W, g_B, (gpumat::etypefunction)m_func, g_Z, m_params[LEAKYRELU]);
		}
	}

	{
//		gpumat::biasPlus(g_Z, g_B);

//		apply_func(g_Z, m_func);
		gpumat::convert_to_mat(g_Z, A1);
	}
	//gpumat::convert_to_mat(g_Z, Z);

	if(!save_A0)
		pA0 = nullptr;
}

void mlp_mixed::backward(const Matf &Delta, bool last_layer)
{
	if(!pA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

	gpumat::GpuMat g_DA1;

	gpumat::convert_to_gpu(Delta, g_DA1);

	apply_back_func(g_DA1, A1, m_func);

	float m = Delta.rows;

	{
		gpumat::GpuMat g_W;

		gpumat::convert_to_gpu(W, g_W);
	//	gpumat::convert_to_gpu(DA1, g_DA1);

		{
			gpumat::GpuMat g_gW;
			if(m_is_dropout && m_prob < 1){
				gpumat::GpuMat g_XDropout;
				gpumat::convert_to_gpu(XDropout, g_XDropout);
				gpumat::matmulT1(g_XDropout, g_DA1, g_gW, 1. / m);
			}else{
				gpumat::GpuMat g_A0;
				gpumat::convert_to_gpu(*pA0, g_A0);
				gpumat::matmulT1(g_A0, g_DA1, g_gW, 1. / m);
			}
			//mulval(g_gW, 1. / m);

			if(m_lambda > 0){
				gpumat::add(g_gW, g_W, 1, m_lambda / m);
			}
			gpumat::convert_to_mat(g_gW, gW);
		}
		if(!last_layer){
			gpumat::GpuMat g_DltA0;
			matmulT2(g_DA1, g_W, g_DltA0);
			gpumat::convert_to_mat(g_DltA0, DltA0);
		}
	}

	{
		gpumat::GpuMat g_gB;
//		g_gB.swap_dims();
		gpumat::sumRows(g_DA1, g_gB, 1.f / m);
//		g_gB.swap_dims();

		gpumat::convert_to_mat(g_gB, gB);
	}
}

void mlp_mixed::forward(const std::vector<Matf> *mat, bool save_A0)
{
	if(!m_init || !mat || mat->empty())
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pVecA0 = (std::vector< Matf >*)mat;

	vecA1.resize(mat->size());
	if(m_is_dropout && m_prob < 1){
		vecXDropout.resize(mat->size());
	}

	gpumat::GpuMat g_W, g_B;
	gpumat::convert_to_gpu(W, g_W);
	gpumat::convert_to_gpu(B, g_B);

	for(size_t i = 0; i < mat->size(); ++i){
		gpumat::GpuMat g_A0, g_vecA1;
		gpumat::convert_to_gpu((*pVecA0)[i], g_A0);

		if(m_is_dropout && m_prob < 1){
			gpumat::GpuMat g_Dropout;

			ct::dropout((*pVecA0)[i].rows, (*pVecA0)[i].cols, m_prob, Dropout);
			gpumat::convert_to_gpu(Dropout, g_Dropout);

			gpumat::elemwiseMult(g_A0, g_Dropout);
			gpumat::convert_to_mat(g_A0, vecXDropout[i]);
		}

		if(m_func == SOFTMAX){
			gpumat::GpuMat PartZ;
			gpumat::m2mpbaf(g_A0, g_W, g_B, gpumat::LINEAR, g_vecA1, m_params[ct::LEAKYRELU]);
			gpumat::softmax(g_vecA1, 1, PartZ);

			gpumat::convert_to_mat(g_vecA1, vecA1[i]);
		}else{
			gpumat::GpuMat g_vecA1;

			gpumat::m2mpbaf(g_A0, g_W, g_B, (gpumat::etypefunction)m_func, g_vecA1, m_params[ct::LEAKYRELU]);

			gpumat::convert_to_mat(g_vecA1, vecA1[i]);
		}
	}

	if(!save_A0)
		pVecA0 = nullptr;
}

void mlp_mixed::backward(std::vector<Matf> &Delta, bool last_layer)
{
	if(!pVecA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

	double m = Delta.size();

	std::vector< Matf >* pDA1 = &vecA1;

	if(m_func == ct::SOFTMAX || m_func == ct::LINEAR){
		pDA1 = (std::vector< Matf >*)&Delta;
	}

	{
		gpumat::GpuMat g_gW, g_gB, g_Dropout;

		g_gW.resize(W.size(), gpumat::GPU_FLOAT);
		g_gB.resize(B.size(), gpumat::GPU_FLOAT);

		g_gW.zeros();
		g_gB.zeros();

		for(size_t i = 0; i < Delta.size(); ++i){
			gpumat::GpuMat g_DA1;
			gpumat::convert_to_gpu((*pDA1)[i], g_DA1);
			if(m_func != ct::SOFTMAX && m_func != ct::LINEAR){
				apply_back_func(g_DA1, vecA1[i], m_func);
			}

			gpumat::GpuMat g_vecA0;

			if(m_is_dropout && m_prob < 1){
				gpumat::convert_to_gpu(vecXDropout[i], g_vecA0);
				gpumat::add2matmulT1(g_vecA0, g_DA1, g_gW);
			}else{
				gpumat::convert_to_gpu((*pVecA0)[i], g_vecA0);
				gpumat::add2matmulT1(g_vecA0, g_DA1, g_gW);
			}

	//		gBi.swap_dims();
			gpumat::add2sumRows(g_DA1, g_gB, 1);
	//		gBi.swap_dims();
		}

	//	gWi.release();
	//	gBi.release();

		gpumat::mulval(g_gW, 1. / m);
		gpumat::mulval(g_gB, 1. / m);

		if(m_lambda > 0){
			gpumat::GpuMat g_W;
			gpumat::convert_to_gpu(W, g_W);
			gpumat::add(g_gW, g_W, 1, m_lambda / m);
		}

		gpumat::convert_to_mat(g_gW, gW);
		gpumat::convert_to_mat(g_gB, gB);
	}

	if(!last_layer){
		vecDltA0.resize(Delta.size());
		gpumat::GpuMat g_W;
		gpumat::convert_to_gpu(W, g_W);
		for(size_t i = 0; i < Delta.size(); ++i){
			gpumat::GpuMat g_DA1, g_vecDltA0;
			gpumat::convert_to_gpu((*pDA1)[i], g_DA1);
			matmulT2(g_DA1, g_W, g_vecDltA0);
			gpumat::convert_to_mat(g_vecDltA0, vecDltA0[i]);
		}
	}
}

void mlp_mixed::write(std::fstream &fs){
	write_fs(fs, W);
	write_fs(fs, B);
}

void mlp_mixed::read(std::fstream &fs){
	read_fs(fs, W);
	read_fs(fs, B);
}

void mlp_mixed::write2(std::fstream &fs)
{
	write_fs2(fs, W);
	write_fs2(fs, B);
}

void mlp_mixed::read2(std::fstream &fs)
{
	read_fs2(fs, W);
	read_fs2(fs, B);

	if(B.rows != 1)
		B.swap_dims();
}

///////////////////////////////////
///////////////////////////////////

MlpAdamOptimizerMixed::MlpAdamOptimizerMixed() : AdamOptimizerMixed()
{
	init_iteration();
}

bool MlpAdamOptimizerMixed::init(const std::vector<mlp_mixed> &mlp)
{
	if(mlp.empty())
		return false;

	m_mW.resize(mlp.size());
	m_mb.resize(mlp.size());

	m_vW.resize(mlp.size());
	m_vb.resize(mlp.size());

	int index = 0;
	for(const mlp_mixed &item: mlp){
		initI(item.W, item.B, index++);
	}
	init_iteration();
	return true;
}

bool MlpAdamOptimizerMixed::pass(std::vector<mlp_mixed> &mlp)
{
	if(mlp.empty())
		return false;

	pass_iteration();

	int index = 0;
	for(mlp_mixed &item: mlp){
		passI(item.gW, item.gB, item.W, item.B, index++);
	}
	return true;
}

//////////////////////////////////

MlpMomentumOptimizerMixed::MlpMomentumOptimizerMixed(): MomentumOptimizerMixed()
{
	m_iteration = 0;
}

bool MlpMomentumOptimizerMixed::init(const std::vector<mlp_mixed> &mlp)
{
	if(mlp.empty())
		return false;

	m_mW.resize(mlp.size());
	m_mb.resize(mlp.size());

	int index = 0;
	for(const mlp_mixed &item: mlp){
		initI(item.W, item.B, index++);
	}
	return true;
}

bool MlpMomentumOptimizerMixed::pass(std::vector<mlp_mixed> &mlp)
{
	if(mlp.empty())
		return false;

	m_iteration++;

	int index = 0;
	for(mlp_mixed &item: mlp){
		passI(item.gW, item.gB, item.W, item.B, index++);
	}
	return true;
}
