#include "convnn2_mixed.h"

#include "gpumat.h"
#include "convnn2_gpu.h"

using namespace conv2;

convnn2_mixed::convnn2_mixed(){
	m_use_pool = false;
	pX = nullptr;
	stride = 1;
	m_use_transpose = true;
	m_Lambda = 0;
	m_optim = &m_adam;
	m_params[ct::LEAKYRELU] = 0.1;
}

void convnn2_mixed::setOptimizer(ct::Optimizer<float> *optim){
	if(!optim)
		return;
	m_optim = optim;
}

void convnn2_mixed::setParams(ct::etypefunction type, float param)
{
	m_params[type] = param;
}

std::vector<ct::Matf> &convnn2_mixed::XOut(){
	if(m_use_pool)
		return A2;
	return A1;
}

const std::vector<ct::Matf> &convnn2_mixed::XOut() const{
	if(m_use_pool)
		return A2;
	return A1;
}

std::vector<ct::Matf> &convnn2_mixed::XOut1(){
	return A1;
}

std::vector<ct::Matf> &convnn2_mixed::XOut2(){
	return A2;
}

bool convnn2_mixed::use_pool() const{
	return m_use_pool;
}

int convnn2_mixed::outputFeatures() const{
	if(m_use_pool){
		int val = convnn_abstract<float>::szA2.area() * convnn_abstract<float>::kernels;
		return val;
	}else{
		int val= convnn_abstract<float>::szA1.area() * convnn_abstract<float>::kernels;
		return val;
	}
}

ct::Size convnn2_mixed::szOut() const{
	if(m_use_pool)
		return convnn_abstract<float>::szA2;
	else
		return convnn_abstract<float>::szA1;
}

void convnn2_mixed::setAlpha(float alpha){
	m_optim->setAlpha(alpha);
}

void convnn2_mixed::setLambda(float val){
	m_Lambda = val;
}

void convnn2_mixed::init(const ct::Size &_szA0, int _channels, int stride, int _K, const ct::Size &_szW, ct::etypefunction func, bool use_pool, bool use_transpose){
	szW = _szW;
	m_use_pool = use_pool;
	m_use_transpose = use_transpose;
	convnn_abstract<float>::kernels = _K;
	convnn_abstract<float>::channels = _channels;
	convnn_abstract<float>::szA0 = _szA0;
	this->stride = stride;
	this->m_func = func;

	int rows = szW.area() * convnn_abstract<float>::channels;
	int cols = convnn_abstract<float>::kernels;

	ct::get_cnv_sizes(convnn_abstract<float>::szA0, szW, stride, convnn_abstract<float>::szA1, convnn_abstract<float>::szA2);

	float n = (float)1./sqrt(kernels);

	W.resize(1);
	B.resize(1);
	gW.resize(1);
	gB.resize(1);

	W[0].setSize(rows, cols);
	W[0].randn(0, n);
	B[0].setSize(convnn_abstract<float>::kernels, 1);
	B[0].randn(0, n);

	m_optim->init(W, B);

	printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<float>::kernels);
}

void convnn2_mixed::forward(const std::vector<ct::Matf> *_pX)
{
	if(!_pX)
		return;
	pX = (std::vector< ct::Matf>*)_pX;

	Xc.resize(pX->size());
	A1.resize(pX->size());
	if(m_use_pool){
		A2.resize(A1.size());
		Mask.resize(Xc.size());
	}

	gpumat::GpuMat g_Xi, g_Xci, g_W, g_B, g_A1i;
	gpumat::convert_to_gpu(W[0], g_W);
	gpumat::convert_to_gpu(B[0], g_B);

	for(int i = 0; i < (int)Xc.size(); ++i){
		ct::Mat_<float>& Xi = (*pX)[i];
		ct::Size szOut;

		gpumat::convert_to_gpu(Xi, g_Xi);

		if(m_use_transpose){
			gpumat::im2colsT(g_Xi, convnn_abstract<float>::szA0,
								   convnn_abstract<float>::channels,
								   szW, stride, g_Xci, szOut);
		}else{
			gpumat::im2cols(g_Xi, convnn_abstract<float>::szA0,
								   convnn_abstract<float>::channels,
								   szW, stride, g_Xci, szOut);
		}
		gpumat::convert_to_mat(g_Xci, Xc[i]);

		gpumat::matmul(g_Xci, g_W, g_A1i);
		gpumat::biasPlus(g_A1i, g_B);

		switch (m_func) {
			case ct::RELU:
				gpumat::reLu(g_A1i);
				break;
			case ct::LEAKYRELU:
				gpumat::leakyReLu(g_A1i, m_params[ct::LEAKYRELU]);
				break;
			case ct::SIGMOID:
				gpumat::sigmoid(g_A1i);
				break;
			case ct::TANH:
				gpumat::tanh(g_A1i);
				break;
			default:
				break;
		}
		gpumat::convert_to_mat(g_A1i, A1[i]);

		if(m_use_pool){
			gpumat::GpuMat g_Mask, g_A2i;

			ct::Matf&A2i = A2[i];
			ct::Size szOut;
			gpumat::subsample(g_A1i, convnn_abstract<float>::szA1, g_A2i, g_Mask, szOut);
			gpumat::convert_to_mat(g_Mask, Mask[i]);
			gpumat::convert_to_mat(g_A2i, A2i);
		}
	}

	if(m_use_pool){
		convnn_abstract<float>::szK = A2[0].size();
	}else{
		convnn_abstract<float>::szK = A1[0].size();
	}
}

void convnn2_mixed::forward(const convnn2_mixed &conv)
{
	forward(&conv.XOut());
}

void convnn2_mixed::backcnv(const gpumat::GpuMat& D, gpumat::GpuMat& A1, gpumat::GpuMat& DS)
{
	switch (m_func) {
		case ct::RELU:
			gpumat::deriv_reLu(A1);
			gpumat::elemwiseMult(D, A1, DS);
			break;
		case ct::LEAKYRELU:
			gpumat::deriv_leakyReLu(A1, m_params[ct::LEAKYRELU]);
			gpumat::elemwiseMult(D, A1, DS);
			break;
		case ct::SIGMOID:
			gpumat::deriv_sigmoid(A1);
			gpumat::elemwiseMult(D, A1, DS);
			break;
		case ct::TANH:
			gpumat::deriv_tanh(A1);
			gpumat::elemwiseMult(D, A1, DS);
			break;
		default:
			if(D.data != DS.data){
				D.copyTo(DS);
			}
			break;
	}
}

void convnn2_mixed::backward(const std::vector<ct::Matf> &D, bool last_level){
	if(D.empty() || D.size() != Xc.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

	dSub.resize(D.size());


	//printf("1\n");
	gpumat::GpuMat g_Xci, g_Maski, g_A1i, g_dSubi, g_gW0, g_gB0, g_Di;

	gW[0].setSize(W[0].size());
	gW[0].fill(0);
	gB[0].setSize(B[0].size());
	gB[0].fill(0);

	gpumat::convert_to_gpu(gW[0], g_gW0);
	gpumat::convert_to_gpu(gB[0], g_gB0);

	for(int i = 0; i < (int)D.size(); ++i){
		const ct::Matf &Di = D[i];

		gpumat::convert_to_gpu(Di, g_Di);
		if(m_func != ct::LINEAR)
			gpumat::convert_to_gpu(A1[i], g_A1i);

		if(m_use_pool){
			gpumat::convert_to_gpu(Mask[i], g_Maski);
			gpumat::upsample(g_Di, convnn_abstract<float>::kernels, g_Maski,
							 convnn_abstract<float>::szA2, convnn_abstract<float>::szA1, g_dSubi);
			backcnv(g_dSubi, g_A1i, g_dSubi);
		}else{
			backcnv(g_Di, g_A1i, g_dSubi);
		}

		ct::Mat_<float>& Xci = Xc[i];
		gpumat::convert_to_gpu(Xci, g_Xci);

		{
			gpumat::GpuMat g_gWi;

			gpumat::matmulT1(g_Xci, g_dSubi, g_gWi);
			gpumat::add(g_gW0, g_gWi);
		}

		{
			gpumat::GpuMat g_gBi;

			gpumat::sumRows(g_dSubi, g_gBi, 1.f/g_dSubi.rows);
			g_gBi.swap_dims();
			gpumat::add(g_gB0, g_gBi);
		}

		gpumat::convert_to_mat(g_dSubi, dSub[i]);

	}
	//printf("3\n");

	gpumat::mulval(g_gW0, (float)1./(D.size() * channels));
	gpumat::mulval(g_gB0, (float)1./(D.size() * channels));

	//printf("4\n");
	if(m_Lambda > 0){
		gpumat::GpuMat g_W;
		gpumat::convert_to_gpu(W[0], g_W);
		gpumat::add(g_gW0,  g_W, 1., (m_Lambda / convnn_abstract<float>::kernels));
	}
	gpumat::convert_to_mat(g_gW0, gW[0]);
	gpumat::convert_to_mat(g_gB0, gB[0]);

	//printf("5\n");
	if(!last_level){
		Dlt.resize(D.size());

		//ct::MatfWf;
		//flipW(W, szW, channels, Wf);

		gpumat::GpuMat g_W, g_Dci, g_Dlti;

		Dc.resize(D.size());
		for(int i = 0; i < (int)D.size(); ++i){
			gpumat::convert_to_gpu(dSub[i], g_dSubi);
			gpumat::convert_to_gpu(W[0], g_W);
			gpumat::matmulT2(g_dSubi, g_W, g_Dci);
			gpumat::back_derivT(g_Dci, convnn_abstract<float>::szA1,
								convnn_abstract<float>::szA0, convnn_abstract<float>::channels,
								szW, stride, g_Dlti);
			gpumat::convert_to_mat(g_Dlti, Dlt[i]);
			//ct::Size sz = (*pX)[i].size();
			//Dlt[i].set_dims(sz);
		}
	}

	//printf("6\n");
	m_optim->pass(gW, gB, W, B);

	//printf("7\n");
}

void convnn2_mixed::write(std::fstream &fs){
	if(!W.size() || !B.size())
		return;
	ct::write_fs(fs, W[0]);
	ct::write_fs(fs, B[0]);
}

void convnn2_mixed::read(std::fstream &fs){
	if(!W.size() || !B.size())
		return;
	ct::read_fs(fs, W[0]);
	ct::read_fs(fs, B[0]);
}

void convnn2_mixed::write2(std::fstream &fs){
	fs.write((char*)&szW.width, sizeof(szW.width));
	fs.write((char*)&szW.height, sizeof(szW.height));
	fs.write((char*)&(convnn_abstract<float>::channels), sizeof(convnn_abstract<float>::channels));
	fs.write((char*)&(convnn_abstract<float>::kernels), sizeof(convnn_abstract<float>::kernels));

	ct::write_fs2(fs, W[0]);
	ct::write_fs2(fs, B[0]);
}

void convnn2_mixed::read2(std::fstream &fs){
	fs.read((char*)&szW.width, sizeof(szW.width));
	fs.read((char*)&szW.height, sizeof(szW.height));
	fs.read((char*)&(convnn_abstract<float>::channels), sizeof(convnn_abstract<float>::channels));
	fs.read((char*)&(convnn_abstract<float>::kernels), sizeof(convnn_abstract<float>::kernels));

	ct::read_fs2(fs, W[0]);
	ct::read_fs2(fs, B[0]);
}

/////////////////////////////
/////////////////////////////
///

AdamOptimizerMixed::AdamOptimizerMixed(): Optimizer<float>()
{
	m_betha1 = (float)0.9;
	m_betha2 = (float)0.999;
	m_init = false;
}

float AdamOptimizerMixed::betha1() const
{
	return m_betha1;
}

void AdamOptimizerMixed::setBetha1(float v){
	m_betha1 = v;
}

float AdamOptimizerMixed::betha2() const{
	return m_betha2;
}

void AdamOptimizerMixed::setBetha2(float v){
	m_betha2 = v;
}

bool AdamOptimizerMixed::init(const std::vector<ct::Matf> &W, const std::vector<ct::Matf> &B){
	if(W.empty() || B.empty())
		return false;

	using namespace ct;

	Optimizer<float>::m_iteration = 0;

	m_mW.resize(W.size());
	m_mb.resize(W.size());

	m_vW.resize(W.size());
	m_vb.resize(W.size());

	for(size_t i = 0; i < W.size(); i++){

		m_mW[i].setSize(W[i].size());
		m_vW[i].setSize(W[i].size());
		m_mW[i].fill(0);
		m_vW[i].fill(0);

		m_mb[i].setSize(B[i].size());
		m_vb[i].setSize(B[i].size());
		m_mb[i].fill(0);
		m_vb[i].fill(0);
	}
	m_init = true;
	return true;
}

bool AdamOptimizerMixed::pass(const std::vector<ct::Matf> &gradW, const std::vector<ct::Matf> &gradB, std::vector<ct::Matf> &W, std::vector<ct::Matf> &b){
	if(!gradW.size() || gradW.size() != gradB.size() || gradW.size() != W.size())
		return false;

	using namespace ct;

	Optimizer<float>::m_iteration++;
	float sb1 = (float)(1. / (1. - pow(m_betha1, Optimizer<float>::m_iteration)));
	float sb2 = (float)(1. / (1. - pow(m_betha2, Optimizer<float>::m_iteration)));

	for(size_t i = 0; i < gradW.size(); ++i){
		gpumat::GpuMat g_m_mW, g_m_vW, g_m_mb, g_m_vb;

		{
			gpumat::GpuMat g_gW;
			gpumat::convert_to_gpu(m_mW[i], g_m_mW);
			gpumat::convert_to_gpu(m_vW[i], g_m_vW);
			gpumat::convert_to_gpu(gradW[i], g_gW);

			gpumat::add(g_m_mW, g_gW, m_betha1, (1. - m_betha1));
			gpumat::elemwiseSqr(g_gW, g_gW);
			gpumat::add(g_m_vW, g_gW, m_betha2, (1. - m_betha2));
		}

		{
			gpumat::GpuMat g_gB;
			gpumat::convert_to_gpu(gradB[i], g_gB);
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
			gpumat::convert_to_gpu(W[i], g_W);
			gpumat::convert_to_gpu(b[i], g_B);

			gpumat::sub_adamGrad(g_W, g_m_mW, g_m_vW, m_alpha, sb1, sb2);
			gpumat::sub_adamGrad(g_B, g_m_mb, g_m_vb, m_alpha, sb1, sb2);

			gpumat::convert_to_mat(g_W, W[i]);
			gpumat::convert_to_mat(g_B, b[i]);
		}
	}
	return true;
}

//////////////////////////////////////

bool AdamOptimizerMixed::empty() const{
	return m_mW.empty() || m_mb.empty() || m_vW.empty() || m_vb.empty();
}

MomentOptimizerMixed::MomentOptimizerMixed(): Optimizer<float>(){
	Optimizer<float>::m_alpha = float(0.01);
	m_betha = float(0.9);
}

void MomentOptimizerMixed::setBetha(float val){
	m_betha = val;
}

bool MomentOptimizerMixed::pass(const std::vector<ct::Matf> &gradW, const std::vector<ct::Matf> &gradB, std::vector<ct::Matf> &W, std::vector<ct::Matf> &B)
{
	if(W.empty() || gradW.size() != W.size() || gradB.empty() || gradB.size() != gradW.size())
		throw new std::invalid_argument("MomentOptimizer: wrong parameters");
	if(m_mW.empty()){
		m_mW.resize(W.size());
		m_mb.resize(W.size());
		for(int i = 0; i < m_mW.size(); ++i){
			m_mW[i] = ct::Matf::zeros(W[i].rows, W[i].cols);
			m_mb[i] = ct::Matf::zeros(B[i].rows, B[i].cols);
		}
	}

	for(int i = 0; i < m_mW.size(); ++i){
		gpumat::GpuMat g_m_mW;
		gpumat::convert_to_gpu(m_mW[i], g_m_mW);
		{
			gpumat::GpuMat g_gradW;
			gpumat::convert_to_gpu(gradW[i], g_gradW);

			gpumat::add(g_m_mW, g_gradW, m_betha, 1. - m_betha);
		}

		{
			gpumat::GpuMat g_W;
			gpumat::convert_to_gpu(W[i], g_W);

			gpumat::add(g_W, g_m_mW, 1., -Optimizer<float>::m_alpha);

			gpumat::convert_to_mat(g_m_mW, m_mW[i]);
			gpumat::convert_to_mat(g_W, W[i]);
		}

	}
	for(int i = 0; i < m_mW.size(); ++i){
		gpumat::GpuMat g_m_mB;
		gpumat::convert_to_gpu(m_mb[i], g_m_mB);
		{
			gpumat::GpuMat g_gradB;
			gpumat::convert_to_gpu(gradB[i], g_gradB);

			gpumat::add(g_m_mB, g_gradB, m_betha, 1. - m_betha);
		}

		{
			gpumat::GpuMat g_B;
			gpumat::convert_to_gpu(B[i], g_B);

			gpumat::add(g_B, g_m_mB, 1., -Optimizer<float>::m_alpha);

			gpumat::convert_to_mat(g_m_mB, m_mb[i]);
			gpumat::convert_to_mat(g_B, B[i]);
		}

	}
	return true;
}
