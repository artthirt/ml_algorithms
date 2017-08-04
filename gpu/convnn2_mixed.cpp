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
}

void convnn2_mixed::setOptimizer(ct::Optimizer<float> *optim){
	if(!optim)
		return;
	m_optim = optim;
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

void convnn2_mixed::init(const ct::Size &_szA0, int _channels, int stride, int _K, const ct::Size &_szW, bool use_pool, bool use_transpose){
	szW = _szW;
	m_use_pool = use_pool;
	m_use_transpose = use_transpose;
	convnn_abstract<float>::kernels = _K;
	convnn_abstract<float>::channels = _channels;
	convnn_abstract<float>::szA0 = _szA0;
	this->stride = stride;

	int rows = szW.area() * convnn_abstract<float>::channels;
	int cols = convnn_abstract<float>::kernels;

	ct::get_cnv_sizes(convnn_abstract<float>::szA0, szW, stride, convnn_abstract<float>::szA1, convnn_abstract<float>::szA2);

	float n = (float)1./szW.area();

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

void convnn2_mixed::forward(const std::vector<ct::Matf> *_pX, ct::etypefunction func)
{
	if(!_pX)
		return;
	pX = (std::vector< ct::Matf>*)_pX;
	m_func = func;

	Xc.resize(pX->size());
	A1.resize(pX->size());

	gpumat::GpuMat g_Xi, g_Xci, g_W, g_B, g_A1i;
	if(m_use_transpose){
		for(int i = 0; i < (int)Xc.size(); ++i){
			ct::Mat_<float>& Xi = (*pX)[i];
			ct::Size szOut;

			gpumat::convert_to_gpu(Xi, g_Xi);

			gpumat::conv2::im2colsT(g_Xi, convnn_abstract<float>::szA0,
								   convnn_abstract<float>::channels,
								   szW, stride, g_Xci, szOut);
			gpumat::convert_to_mat(g_Xci, Xc[i]);
		}
	}else{
		for(int i = 0; i < (int)Xc.size(); ++i){
			ct::Mat_<float>& Xi = (*pX)[i];
			ct::Size szOut;

			gpumat::convert_to_gpu(Xi, g_Xi);

			gpumat::conv2::im2cols(g_Xi, convnn_abstract<float>::szA0,
								   convnn_abstract<float>::channels,
								   szW, stride, g_Xci, szOut);
			gpumat::convert_to_mat(g_Xci, Xc[i]);
		}
	}

	gpumat::convert_to_gpu(W[0], g_W);
	gpumat::convert_to_gpu(B[0], g_B);

	for(int i = 0; i < (int)Xc.size(); ++i){
		ct::Mat_<float>& Xi = Xc[i];
		ct::Mat_<float>& A1i = A1[i];

		gpumat::convert_to_gpu(Xi, g_Xi);

		gpumat::matmul(g_Xi, g_W, g_A1i);
		gpumat::biasPlus(g_A1i, g_B);
		gpumat::convert_to_mat(g_A1i, A1i);
	}

	for(int i = 0; i < (int)A1.size(); ++i){
		ct::Mat_<float>& Ao = A1[i];
		switch (m_func) {
			case ct::RELU:
				gpumat::convert_to_gpu(Ao, g_A1i);
				gpumat::reLu(g_A1i);
				break;
			case ct::SIGMOID:
				gpumat::convert_to_gpu(Ao, g_A1i);
				gpumat::sigmoid(g_A1i);
				break;
			case ct::TANH:
				gpumat::convert_to_gpu(Ao, g_A1i);
				gpumat::tanh(g_A1i);
				break;
			default:
				continue;
		}
		gpumat::convert_to_mat(g_A1i, Ao);
	}
	if(m_use_pool){
		gpumat::GpuMat g_Mask, g_A2i;

		Mask.resize(Xc.size());
		A2.resize(A1.size());
		for(int i = 0; i < (int)A1.size(); ++i){
			ct::Matf&A1i = A1[i];
			ct::Matf&A2i = A2[i];
			ct::Size szOut;

			gpumat::convert_to_gpu(A1i, g_A1i);

			gpumat::conv2::subsample(g_A1i, convnn_abstract<float>::szA1, g_A2i, g_Mask, szOut);
			gpumat::convert_to_mat(g_Mask, Mask[i]);
			gpumat::convert_to_mat(g_A2i, A2i);
		}
		convnn_abstract<float>::szK = A2[0].size();
	}else{
		convnn_abstract<float>::szK = A1[0].size();
	}
}

void convnn2_mixed::forward(const convnn<float> &conv, ct::etypefunction func)
{
	forward(&conv.XOut(), func);
}

void convnn2_mixed::backcnv(const std::vector<ct::Matf> &D, std::vector<ct::Matf> &DS)
{
	gpumat::GpuMat g_Di, g_A1i;
	if(D.data() != DS.data()){
		for(int i = 0; i < (int)D.size(); ++i){
			if(m_func == ct::LINEAR){
				DS[i] = D[i];
				continue;
			}
			gpumat::convert_to_gpu(D[i], g_Di);
			gpumat::convert_to_gpu(A1[i], g_A1i);
			switch (m_func) {
				case ct::RELU:
					gpumat::deriv_reLu(g_A1i);
//					ct::elemwiseMult(D[i], derivRelu(A1[i]), DS[i]);
					break;
				case ct::SIGMOID:
					gpumat::deriv_sigmoid(g_A1i);
//					ct::elemwiseMult(D[i], derivSigmoid(A1[i]), DS[i]);
					break;
				case ct::TANH:
					gpumat::deriv_tanh(g_A1i);
//					ct::elemwiseMult(D[i], derivTanh(A1[i]), DS[i]);
					break;
				default:
					break;
			}
			gpumat::elemwiseMult(g_Di, g_A1i);
			gpumat::convert_to_mat(g_Di, DS[i]);
		}
	}else{
		for(int i = 0; i < (int)D.size(); ++i){
			if(m_func == ct::LINEAR){
				continue;
			}
			gpumat::convert_to_gpu(D[i], g_Di);
			gpumat::convert_to_gpu(A1[i], g_A1i);
			switch (m_func) {
				case ct::RELU:
					gpumat::deriv_reLu(g_A1i);
//					ct::elemwiseMult(D[i], derivRelu(A1[i]), DS[i]);
					break;
				case ct::SIGMOID:
					gpumat::deriv_sigmoid(g_A1i);
//					ct::elemwiseMult(D[i], derivSigmoid(A1[i]), DS[i]);
					break;
				case ct::TANH:
					gpumat::deriv_tanh(g_A1i);
//					ct::elemwiseMult(D[i], derivTanh(A1[i]), DS[i]);
					break;
				default:
					break;
			}
			gpumat::elemwiseMult(g_Di, g_A1i);
			gpumat::convert_to_mat(g_Di, DS[i]);
		}
	}
}

void convnn2_mixed::backward(const std::vector<ct::Matf> &D, bool last_level){
	if(D.empty() || D.size() != Xc.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

	dSub.resize(D.size());

	//printf("1\n");
	if(m_use_pool){
		for(int i = 0; i < (int)D.size(); ++i){
			const ct::Matf &Di = D[i];
			//Di.set_dims(szA2.area(), K);
			upsample(Di, convnn_abstract<float>::kernels, Mask[i],convnn_abstract<float>:: szA2, convnn_abstract<float>::szA1, dSub[i]);
		}
		backcnv(dSub, dSub);
	}else{
		backcnv(D, dSub);
	}

	//printf("2\n");
	vgW.resize(D.size());
	vgB.resize(D.size());
	for(int i = 0; i < (int)D.size(); ++i){
		ct::Mat_<float>& Xci = Xc[i];
		ct::Mat_<float>& dSubi = dSub[i];
		ct::Mat_<float>& Wi = vgW[i];
		ct::Mat_<float>& vgBi = vgB[i];
		matmulT1(Xci, dSubi, Wi);
		vgBi = ct::sumRows(dSubi, 1.f/dSubi.rows);
		//Wi *= (1.f/dSubi.total());
		//vgBi.swap_dims();
	}
	//printf("3\n");
	gW[0].setSize(W[0].size());
	gW[0].fill(0);
	gB[0].setSize(B[0].size());
	gB[0].fill(0);
	for(size_t i = 0; i < D.size(); ++i){
		ct::add(gW[0], vgW[i]);
		ct::add(gB[0], vgB[i]);
	}
	gW[0] *= (float)1./(D.size());
	gB[0] *= (float)1./(D.size());

	//printf("4\n");
	if(m_Lambda > 0){
		ct::add<float>(gW[0],  W[0], 1., (m_Lambda / convnn_abstract<float>::kernels));
	}

	//printf("5\n");
	if(!last_level){
		Dlt.resize(D.size());

		//ct::MatfWf;
		//flipW(W, szW, channels, Wf);

		Dc.resize(D.size());
		for(int i = 0; i < (int)D.size(); ++i){
			ct::matmulT2(dSub[i], W[0], Dc[i]);
			back_derivT(Dc[i], convnn_abstract<float>::szA1, convnn_abstract<float>::szA0, convnn_abstract<float>::channels, szW, stride, Dlt[i]);
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

bool AdamOptimizerMixed::empty() const{
	return m_mW.empty() || m_mb.empty() || m_vW.empty() || m_vb.empty();
}
