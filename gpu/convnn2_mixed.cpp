#include "convnn2_mixed.h"

#include "gpumat.h"
#include "convnn2_gpu.h"

using namespace conv2;

convnn2_mixed::convnn2_mixed()
{
	m_use_pool = false;
	pX = nullptr;
	stride = 1;
	m_use_transpose = true;
	m_use_bn = false;
	m_use_same = false;
	m_Lambda = 0;
	m_params[ct::LEAKYRELU] = 0.1;
}

void convnn2_mixed::setParams(ct::etypefunction type, float param)
{
	m_params[type] = param;
}

std::vector<ct::Matf> &convnn2_mixed::XOut()
{
	if(m_use_bn)
		return A3;
	if(m_use_pool)
		return A2;
	return A1;
}

const std::vector<ct::Matf> &convnn2_mixed::XOut() const
{
	if(m_use_bn)
		return A3;
	if(m_use_pool)
		return A2;
	return A1;
}

std::vector<ct::Matf> &convnn2_mixed::XOut1()
{
	return A1;
}

std::vector<ct::Matf> &convnn2_mixed::XOut2()
{
	return A2;
}

std::vector<ct::Matf> &convnn2_mixed::XOut3()
{
	return A3;
}

bool convnn2_mixed::use_pool() const
{
	return m_use_pool;
}

bool convnn2_mixed::use_bn() const
{
	return m_use_bn;
}

int convnn2_mixed::outputFeatures() const
{
	if(m_use_pool){
		int val = convnn_abstract<float>::szA2.area() * convnn_abstract<float>::kernels;
		return val;
	}else{
		int val= convnn_abstract<float>::szA1.area() * convnn_abstract<float>::kernels;
		return val;
	}
}

ct::Size convnn2_mixed::szOut() const
{
	if(m_use_pool)
		return convnn_abstract<float>::szA2;
	else
		return convnn_abstract<float>::szA1;
}

void convnn2_mixed::setLambda(float val)
{
	m_Lambda = val;
}

void convnn2_mixed::init(const ct::Size &_szA0, int _channels, int stride,
						 int _K, const ct::Size &_szW, ct::etypefunction func,
						 bool use_pool, bool use_bn, bool use_transpose, bool use_same)
{
	szW = _szW;
	m_use_pool = use_pool;
	m_use_bn = use_bn;
	m_use_same = use_same;
	m_use_transpose = use_transpose;
	convnn_abstract<float>::kernels = _K;
	convnn_abstract<float>::channels = _channels;
	convnn_abstract<float>::szA0 = _szA0;
	this->stride = stride;
	this->m_func = func;

	bn.channels = _K;

	int rows = szW.area() * convnn_abstract<float>::channels;
	int cols = convnn_abstract<float>::kernels;

	if(use_same)
		ct::get_cnv_size_same(convnn_abstract<float>::szA0, stride,
						  convnn_abstract<float>::szA1, convnn_abstract<float>::szA2);
	else
		ct::get_cnv_sizes(convnn_abstract<float>::szA0, szW, stride,
						  convnn_abstract<float>::szA1, convnn_abstract<float>::szA2);

	float n = 0.05;//(float)1./sqrt(kernels);

	W.setSize(rows, cols);
	W.randn(0, n);
	B.setSize(1, convnn_abstract<float>::kernels);
	B.randn(0, n);

	printf("Out=[%dx%dx%d], W[%d, %d]\n", szOut().width, szOut().height, convnn_abstract<float>::kernels, W.rows, W.cols);
//	printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<float>::kernels);
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
	gpumat::convert_to_gpu(W, g_W);
	gpumat::convert_to_gpu(B, g_B);

	for(int i = 0; i < (int)Xc.size(); ++i){
		ct::Mat_<float>& Xi = (*pX)[i];
		ct::Size szOut;

		gpumat::convert_to_gpu(Xi, g_Xi);

		if(m_use_transpose){
			if(m_use_same)
				gpumat::im2colsT_same(g_Xi, convnn_abstract<float>::szA0,
									   convnn_abstract<float>::channels,
									   szW, stride, g_Xci, szOut);
			else
				gpumat::im2colsT(g_Xi, convnn_abstract<float>::szA0,
									   convnn_abstract<float>::channels,
									   szW, stride, g_Xci, szOut);
		}else{
			if(m_use_same)
				gpumat::im2cols_same(g_Xi, convnn_abstract<float>::szA0,
									   convnn_abstract<float>::channels,
									   szW, stride, g_Xci, szOut);
			else
				gpumat::im2cols(g_Xi, convnn_abstract<float>::szA0,
									   convnn_abstract<float>::channels,
									   szW, stride, g_Xci, szOut);
		}
		gpumat::convert_to_mat(g_Xci, Xc[i]);

		gpumat::m2mpbaf(g_Xci, g_W, g_B, (gpumat::etypefunction)m_func, g_A1i, m_params[ct::LEAKYRELU]);

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

		if(m_use_bn){
			bn.X = &A2;
			bn.Y = &A3;
			bn.normalize();
		}
	}else{
		convnn_abstract<float>::szK = A1[0].size();

		if(m_use_bn){
			bn.X = &A1;
			bn.Y = &A3;
			bn.normalize();
		}
	}

#if 0
	if(m_use_same){
		ct::save_mat((*pX)[0], "testPx26.txt");
		ct::save_mat(Xc[0], "testXc26.txt");
		ct::save_mat(A1[0], "testA126.txt");
		if(!A3.empty())ct::save_mat(A3[0], "testA326.txt");
		if(!A2.empty())ct::save_mat(A2[0], "testA226.txt");
		ct::save_mat(W, "testW.txt");
		ct::save_mat(B, "testB.txt");
		if(!Mask.empty())ct::save_mat(Mask[0], "testMask.txt");
		throw new std::string("ee");
	}
#endif
}

void convnn2_mixed::forward(const convnn2_mixed &conv)
{
	forward(&conv.XOut());
}

void convnn2_mixed::backcnv(const gpumat::GpuMat& D, gpumat::GpuMat& A1, gpumat::GpuMat& DS)
{
	gpumat::mul2deriv(D, A1, (gpumat::etypefunction)m_func, DS, m_params[ct::LEAKYRELU]);
}

void convnn2_mixed::backward(const std::vector<ct::Matf> &D, bool last_level){
	if(D.empty() || D.size() != Xc.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

	dSub.resize(D.size());


	//printf("1\n");
	gpumat::GpuMat g_Xci, g_Maski, g_A1i, g_dSubi, g_gW0, g_gB0, g_Di;

	gW.setSize(W.size());
	gW.fill(0);
	gB.setSize(B.size());
	gB.fill(0);

	gpumat::convert_to_gpu(gW, g_gW0);
	gpumat::convert_to_gpu(gB, g_gB0);

	if(m_use_bn){
		bn.D = (std::vector<ct::Matf>*)&D;
		bn.denormalize();
	}

	std::vector<ct::Matf> &_D = m_use_bn? bn.Dout : (std::vector<ct::Matf>&)D;

	for(int i = 0; i < (int)D.size(); ++i){
		const ct::Matf &Di = _D[i];

		gpumat::convert_to_gpu(Di, g_Di);
		if(m_func != ct::LINEAR)
			gpumat::convert_to_gpu(A1[i], g_A1i);

		if(m_use_pool){
			gpumat::convert_to_gpu(Mask[i], g_Maski);
			gpumat::upsample(g_Di, convnn_abstract<float>::kernels, g_Maski,
							 convnn_abstract<float>::szA2, convnn_abstract<float>::szA1, g_dSubi);

			if(m_func != ct::LINEAR && m_func != ct::SOFTMAX)
				backcnv(g_dSubi, g_A1i, g_dSubi);
		}else{
			if(m_func != ct::LINEAR && m_func != ct::SOFTMAX)
				backcnv(g_Di, g_A1i, g_dSubi);
			else
				g_Di.copyTo(g_dSubi);
		}

		ct::Mat_<float>& Xci = Xc[i];
		gpumat::convert_to_gpu(Xci, g_Xci);

		{
			//gpumat::GpuMat g_gWi;

			gpumat::add2matmulT1(g_Xci, g_dSubi, g_gW0);
//			gpumat::add(g_gW0, g_gWi);
		}

		{
			//gpumat::GpuMat g_gBi;

			gpumat::add2sumRows(g_dSubi, g_gB0, 1.f/g_dSubi.rows);
//			g_gBi.swap_dims();
			//gpumat::add(g_gB0, g_gBi);
		}

		gpumat::convert_to_mat(g_dSubi, dSub[i]);

	}
	//printf("3\n");

	gpumat::mulval(g_gW0, (float)1./(D.size()));
	gpumat::mulval(g_gB0, (float)1./(D.size()));

	//printf("4\n");
	if(m_Lambda > 0){
		gpumat::GpuMat g_W;
		gpumat::convert_to_gpu(W, g_W);
		gpumat::add(g_gW0,  g_W, 1., (m_Lambda / convnn_abstract<float>::kernels));
	}
	gpumat::convert_to_mat(g_gW0, gW);
	gpumat::convert_to_mat(g_gB0, gB);

	//printf("5\n");
	if(!last_level){
		Dlt.resize(D.size());

		//ct::MatfWf;
		//flipW(W, szW, channels, Wf);

		gpumat::GpuMat g_W, g_Dci, g_Dlti;

		Dc.resize(D.size());
		for(int i = 0; i < (int)D.size(); ++i){
			gpumat::convert_to_gpu(dSub[i], g_dSubi);
			gpumat::convert_to_gpu(W, g_W);
			gpumat::matmulT2(g_dSubi, g_W, g_Dci);

			if(m_use_transpose){
				if(m_use_same)
					gpumat::cols2imT_same(g_Dci, convnn_abstract<float>::szA1,
										convnn_abstract<float>::szA0, convnn_abstract<float>::channels,
										szW, stride, g_Dlti);
				else
					gpumat::cols2imT(g_Dci, convnn_abstract<float>::szA1,
										convnn_abstract<float>::szA0, convnn_abstract<float>::channels,
										szW, stride, g_Dlti);
			}else{
				if(m_use_same)
					gpumat::cols2im_same(g_Dci, convnn_abstract<float>::szA1,
										convnn_abstract<float>::szA0, convnn_abstract<float>::channels,
										szW, stride, g_Dlti);
				else
					gpumat::cols2im(g_Dci, convnn_abstract<float>::szA1,
										convnn_abstract<float>::szA0, convnn_abstract<float>::channels,
										szW, stride, g_Dlti);
			}
			gpumat::convert_to_mat(g_Dlti, Dlt[i]);
			//ct::Size sz = (*pX)[i].size();
			//Dlt[i].set_dims(sz);
		}
	}
}

void convnn2_mixed::write(std::fstream &fs){
	if(W.empty() || B.empty())
		return;
	ct::write_fs(fs, W);
	ct::write_fs(fs, B);
}

void convnn2_mixed::read(std::fstream &fs){
	if(W.empty() || B.empty())
		return;
	ct::read_fs(fs, W);
	ct::read_fs(fs, B);
}

void convnn2_mixed::write2(std::fstream &fs){
	fs.write((char*)&szW.width, sizeof(szW.width));
	fs.write((char*)&szW.height, sizeof(szW.height));
	fs.write((char*)&(convnn_abstract<float>::channels), sizeof(convnn_abstract<float>::channels));
	fs.write((char*)&(convnn_abstract<float>::kernels), sizeof(convnn_abstract<float>::kernels));

	ct::write_fs2(fs, W);
	ct::write_fs2(fs, B);
}

void convnn2_mixed::read2(std::fstream &fs){
	fs.read((char*)&szW.width, sizeof(szW.width));
	fs.read((char*)&szW.height, sizeof(szW.height));
	fs.read((char*)&(convnn_abstract<float>::channels), sizeof(convnn_abstract<float>::channels));
	fs.read((char*)&(convnn_abstract<float>::kernels), sizeof(convnn_abstract<float>::kernels));

	ct::read_fs2(fs, W);
	ct::read_fs2(fs, B);

	if(B.rows != 1)
		B.swap_dims();
}

///////////////////////////////////
///////////////////////////////////

CnvAdamOptimizerMixed::CnvAdamOptimizerMixed() : AdamOptimizerMixed()
{
	init_iteration();
}

bool CnvAdamOptimizerMixed::init(const std::vector<convnn2_mixed> &cnv)
{
	if(cnv.empty())
		return false;

	m_mW.resize(cnv.size());
	m_mb.resize(cnv.size());

	m_vW.resize(cnv.size());
	m_vb.resize(cnv.size());

	int index = 0;
	for(const convnn2_mixed &item: cnv){
		initI(item.W, item.B, index++);
	}
	init_iteration();
	return true;
}

bool CnvAdamOptimizerMixed::pass(std::vector<convnn2_mixed> &cnv)
{
	if(cnv.empty())
		return false;

	pass_iteration();

	int index = 0;
	for(convnn2_mixed &item: cnv){
		passI(item.gW, item.gB, item.W, item.B, index++);
	}
	return true;
}

//////////////////////////////

CnvMomentumOptimizerMixed::CnvMomentumOptimizerMixed(): MomentumOptimizerMixed()
{
	m_iteration = 0;
	stop_layer = 0;
}

bool CnvMomentumOptimizerMixed::init(const std::vector<convnn2_mixed> &cnv)
{
	if(cnv.empty())
		return false;

	m_mW.resize(cnv.size());
	m_mb.resize(cnv.size());

	int index = 0;
	for(const convnn2_mixed &item: cnv){
		if(index >= stop_layer){
			initI(item.W, item.B, index);
		}
		index++;
	}
	return true;
}

bool CnvMomentumOptimizerMixed::pass(std::vector<convnn2_mixed> &cnv)
{
	if(cnv.empty())
		return false;

	m_iteration++;

	int index = 0;
	for(convnn2_mixed &item: cnv){
		if(index >= stop_layer){
			passI(item.gW, item.gB, item.W, item.B, index);
		}
	}
	return true;
}
