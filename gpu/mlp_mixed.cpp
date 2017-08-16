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

void mlp_mixed::apply_back_func(gpumat::GpuMat &D1, etypefunction func)
{
	if(func == LINEAR || func == SOFTMAX)
		return;

	gpumat::GpuMat g_A1;
	gpumat::convert_to_gpu(A1, g_A1);

	gpumat::mul2deriv(D1, g_A1, (gpumat::etypefunction)func, D1, m_params[LEAKYRELU]);
//	switch (func) {
//		default:
//		case LINEAR:
////			D1.copyTo(D2);
//			return;
//		case RELU:
//			gpumat::convert_to_gpu(A1, g_A1);
//			gpumat::deriv_reLu(g_A1);
//			break;
//		case SOFTMAX:
//			//				A = softmax(A, 1);
////			D1.copyTo(D2);
//			return;
//		case SIGMOID:
//			gpumat::convert_to_gpu(A1, g_A1);
//			gpumat::deriv_sigmoid(g_A1);
//			break;
//		case TANH:
//			gpumat::convert_to_gpu(A1, g_A1);
//			gpumat::deriv_tanh(g_A1);
//			break;
//		case LEAKYRELU:
//			gpumat::convert_to_gpu(A1, g_A1);
//			gpumat::deriv_leakyReLu(g_A1, m_params[LEAKYRELU]);
//			break;
//	}
//	gpumat::elemwiseMult(D1, g_A1);
}

etypefunction mlp_mixed::funcType() const{
	return m_func;
}

void mlp_mixed::forward(const Matf *mat, bool save_A0){
	if(!m_init || !mat)
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pA0 = (Matf*)mat;

	gpumat::GpuMat g_XDropout, g_Z, g_B;

	{
		gpumat::GpuMat g_A0, g_W;

		gpumat::convert_to_gpu(*pA0, g_A0);
		gpumat::convert_to_gpu(W, g_W);
		gpumat::convert_to_gpu(B, g_B);

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			gpumat::GpuMat g_Dropout;

			ct::dropout(pA0->rows, pA0->cols, m_prob, Dropout);
			gpumat::convert_to_gpu(Dropout, g_Dropout);

			gpumat::elemwiseMult(g_A0, g_Dropout, g_XDropout);
			gpumat::matmul(g_XDropout, g_W, g_Z);
			gpumat::convert_to_mat(g_XDropout, XDropout);
		}else{
			gpumat::matmul(g_A0, g_W, g_Z);
		}
	}

	{
		gpumat::biasPlus(g_Z, g_B);

		apply_func(g_Z, m_func);
		gpumat::convert_to_mat(g_Z, A1);
	}
	//gpumat::convert_to_mat(g_Z, Z);

	if(!save_A0)
		pA0 = nullptr;
}

void mlp_mixed::backward(const Matf &Delta, bool last_layer){
	if(!pA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

	gpumat::GpuMat g_DA1;

	gpumat::convert_to_gpu(Delta, g_DA1);

	apply_back_func(g_DA1, m_func);

	float m = Delta.rows;

	{
		gpumat::GpuMat g_W;

		gpumat::convert_to_gpu(W, g_W);
	//	gpumat::convert_to_gpu(DA1, g_DA1);

		{
			gpumat::GpuMat g_gW;
			if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
				gpumat::GpuMat g_XDropout;
				gpumat::convert_to_gpu(XDropout, g_XDropout);
				gpumat::matmulT1(g_XDropout, g_DA1, g_gW);
			}else{
				gpumat::GpuMat g_A0;
				gpumat::convert_to_gpu(*pA0, g_A0);
				gpumat::matmulT1(g_A0, g_DA1, g_gW);
			}
			mulval(g_gW, 1. / m);

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

//////////////////////////////////////

MlpOptimMixed::MlpOptimMixed(): AdamOptimizer<float>(){

}

bool MlpOptimMixed::init(std::vector<ct::mlp_mixed > &Mlp){
	if(Mlp.empty())
		return false;

	AO m_iteration = 0;

	AO m_mW.resize(Mlp.size());
	AO m_mb.resize(Mlp.size());

	AO m_vW.resize(Mlp.size());
	AO m_vb.resize(Mlp.size());

	for(size_t i = 0; i < Mlp.size(); i++){
		ct::mlp_mixed& _mlp = Mlp[i];
		AO m_mW[i].setSize(_mlp.W.size());
		AO m_vW[i].setSize(_mlp.W.size());
		AO m_mW[i].fill(0);
		AO m_vW[i].fill(0);

		AO m_mb[i].setSize(_mlp.B.size());
		AO m_vb[i].setSize(_mlp.B.size());
		AO m_mb[i].fill(0);
		AO m_vb[i].fill(0);
	}
	AO m_init = true;
	return true;
}

bool MlpOptimMixed::pass(std::vector<ct::mlp_mixed > &Mlp){

	using namespace ct;

	AO m_iteration++;
	float sb1 = (float)(1. / (1. - pow(AO m_betha1, AO m_iteration)));
	float sb2 = (float)(1. / (1. - pow(AO m_betha2, AO m_iteration)));

	for(size_t i = 0; i < Mlp.size(); ++i){
		ct::mlp_mixed& _mlp = Mlp[i];

		gpumat::GpuMat g_m_mW, g_m_vW, g_m_mb, g_m_vb;

		{
			gpumat::GpuMat g_gW;
			gpumat::convert_to_gpu(m_mW[i], g_m_mW);
			gpumat::convert_to_gpu(m_vW[i], g_m_vW);
			gpumat::convert_to_gpu(_mlp.gW, g_gW);

			gpumat::add(g_m_mW, g_gW, m_betha1, (1. - m_betha1));
			gpumat::elemwiseSqr(g_gW, g_gW);
			gpumat::add(g_m_vW, g_gW, m_betha2, (1. - m_betha2));
		}

		{
			gpumat::GpuMat g_gB;
			gpumat::convert_to_gpu(_mlp.gB, g_gB);
			gpumat::convert_to_gpu(m_mb[i], g_m_mb);
			gpumat::convert_to_gpu(m_vb[i], g_m_vb);

			gpumat::add(g_m_mb, g_gB, m_betha1, (1. - m_betha1));
			gpumat::elemwiseSqr(g_gB, g_gB);
			gpumat::add(g_m_vb, g_gB, m_betha2, (1. - m_betha2));
		}

		/// W = -alpha * (sb1 * mW / (sqrt(sb2 * vW) + eps))

//		gpumat::add(W[i], m_mW[i], 1, -m_alpha);
//		gpumat::add(b[i], m_mb[i], 1, -m_alpha);
		{
			gpumat::GpuMat g_W, g_B;
			gpumat::convert_to_gpu(_mlp.W, g_W);
			gpumat::convert_to_gpu(_mlp.B, g_B);

			gpumat::sub_adamGrad(g_W, g_m_mW, g_m_vW, m_alpha, sb1, sb2);
			gpumat::sub_adamGrad(g_B, g_m_mb, g_m_vb, m_alpha, sb1, sb2);

			gpumat::convert_to_mat(g_W, _mlp.W);
			gpumat::convert_to_mat(g_B, _mlp.B);
		}

//		AO m_mW[i] = AO m_betha1 * AO m_mW[i] + (float)(1. - AO m_betha1) * _mlp.gW;
//		AO m_mb[i] = AO m_betha1 * AO m_mb[i] + (float)(1. - AO m_betha1) * _mlp.gB;

//		AO m_vW[i] = AO m_betha2 * AO m_vW[i] + (float)(1. - AO m_betha2) * elemwiseSqr(_mlp.gW);
//		AO m_vb[i] = AO m_betha2 * AO m_vb[i] + (float)(1. - AO m_betha2) * elemwiseSqr(_mlp.gB);

//		Matf mWs = AO m_mW[i] * sb1;
//		Matf mBs = AO m_mb[i] * sb1;
//		Matf vWs = AO m_vW[i] * sb2;
//		Matf vBs = AO m_vb[i] * sb2;

//		vWs.sqrt(); vBs.sqrt();
//		vWs += eps; vBs += eps;
//		mWs = elemwiseDiv(mWs, vWs);
//		mBs = elemwiseDiv(mBs, vBs);

//		_mlp.W -= AO m_alpha * mWs;
//		_mlp.B -= AO m_alpha * mBs;
	}
	return true;
}
