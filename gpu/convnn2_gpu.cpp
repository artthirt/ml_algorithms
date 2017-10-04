#include "convnn2_gpu.h"
#include "nn.h"

#include "qt_work_mat.h"

using namespace gpumat;

//////////////////////////

void save_vec(const std::vector< gpumat::GpuMat >& Dlt)
{
	for(size_t i = 0; i < Dlt.size(); ++i){
		std::stringstream ss;
		ss << "data/" << i << ".txt";
		gpumat::save_gmat(Dlt[i], ss.str());
	}
}

//////////////////////////

#include "convnn2.h"

template< typename T>
void check(const ct::Mat_<T> &c1, const ct::Mat_<T>& c2)
{
	ct::Mat_<T> c3 = c2 - c1;
	ct::v_elemwiseSqr(c3);
	float s = c3.sum();
	ct::save_mat(c3, "c3.txt");
	assert(s < 1e-6);
}

void check_deriv(const std::vector< gpumat::GpuMat >& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 std::vector< gpumat::GpuMat >& X)
{
	gpumat::GpuMat Dlt;

	cols2imT(Delta[0], szOut, szA0, channels, szW, stride, Dlt);

	ct::Matf c1, c2, c3, c4;

	gpumat::convert_to_mat(X[0], c1);
//	gpumat::convert_to_mat(Dlt, c2);

//	check(c1, c2);

	gpumat::convert_to_mat(Delta[0], c4);

	conv2::cols2imT(c4, szOut, szA0, channels, szW, stride, c3);

	check(c3, c1);
}


//////////////////////////

convnn_gpu::convnn_gpu()
{
	m_use_pool = false;
	m_use_bn = false;
	m_use_same = false;
	pX = nullptr;
	stride = 1;
	m_use_transpose = true;
	m_pool_dropout = false;
	m_prob_dropout = 0.9;
	m_lambda = 0;

	m_params[LEAKYRELU] = 0.1;
}

void convnn_gpu::setTrainMode(bool val)
{
	bn.train = val;
}

void convnn_gpu::setLambda(double val)
{
	m_lambda = val;
}

void convnn_gpu::setDropout(bool val)
{
	m_pool_dropout = val;
}

void convnn_gpu::setDropout(double val)
{
	m_prob_dropout = val;
}

void convnn_gpu::setParams(etypefunction type, double param)
{
	m_params[type] = param;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut()
{
	if(m_use_bn)
		return A3;
	if(m_use_pool)
		return A2;
	return A1;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut1()
{
	return A1;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut2()
{
	return A2;
}

std::vector<GpuMat> &convnn_gpu::XOut3()
{
	return A3;
}

bool convnn_gpu::use_pool() const
{
	return m_use_pool;
}

bool convnn_gpu::use_bn() const
{
	return m_use_bn;
}

int convnn_gpu::outputFeatures() const
{
	if(m_use_pool){
		int val = szA2.area() * kernels;
		return val;
	}else{
		int val= szA1.area() * kernels;
		return val;
	}
}

ct::Size convnn_gpu::szOut() const
{
	if(m_use_pool)
		return szA2;
	else
		return szA1;
}

void convnn_gpu::init(const ct::Size &_szA0, int _channels, int stride, int _K,
					  const ct::Size &_szW, etypefunction func, bool use_pool, bool use_bn, bool use_transpose, bool use_same)
{
	szW = _szW;
	kernels = _K;
	channels = _channels;
	m_use_pool = use_pool;
	m_use_bn = use_bn;
	m_use_same = use_same;
	m_use_transpose = use_transpose;
	szA0 = _szA0;
	this->stride = stride;
	m_func = func;

	bn.channels = _K;

	int rows = szW.area() * channels;
	int cols = kernels;

	if(m_use_same){
		ct::get_cnv_size_same(szA0, stride, szA1, szA2);
	}else{
		ct::get_cnv_sizes(szA0, szW, stride, szA1, szA2);
	}
	float n = (float)1/(sqrt(szW.area() * channels));

	{
		ct::Matf Wi(rows, cols), Bi(1, kernels);
		Wi.randn(0, n);
		gpumat::convert_to_gpu(Wi, W);
		Bi.randn(0, n);
		gpumat::convert_to_gpu(Bi, B);
	}

	gW.resize(W);
	gB.resize(B);

	printf("Out=[%dx%dx%d], W[%d, %d]\n", szOut().width, szOut().height, kernels, W.rows, W.cols);
}

template< typename T >
void dropout_to_gpu(gpumat::GpuMat& Dropout, const ct::Size& sz, double prob)
{
	ct::Mat_<T> d;
	ct::dropout(sz.height, sz.width, (T)prob, d);
	gpumat::convert_to_gpu(d, Dropout);
}

void get_dropout(double prob, std::vector<gpumat::GpuMat>& X, gpumat::GpuMat& Dropout)
{
	if(X.empty() || std::abs(prob - 1.) < 1e-6)
		return;

	switch (X[0].type) {
		case gpumat::GPU_DOUBLE:
			dropout_to_gpu<double>(Dropout, X[0].sz(), prob);
			break;
		default:
		case gpumat::GPU_FLOAT:
			dropout_to_gpu<float>(Dropout, X[0].sz(), prob);
			break;
	}
	for(size_t i = 0; i < X.size(); ++i){
		gpumat::elemwiseMult(X[i], Dropout);
	}
//	qDebug("get_dropout: pool dropout generated and applied. prob=%f", prob);
}

void set_dropout(std::vector<gpumat::GpuMat>& X, const gpumat::GpuMat& Dropout)
{
	if(X.empty() || Dropout.empty())
		return;

	for(size_t i = 0; i < X.size(); ++i){
		gpumat::elemwiseMult(X[i], Dropout);
	}
//	qDebug("set_dropout: pool dropout applied");
}

void convnn_gpu::forward(const std::vector<gpumat::GpuMat> *_pX)
{
	if(!_pX)
		return;
	pX = (std::vector< gpumat::GpuMat >*)_pX;

	Xc.resize(pX->size());
	A1.resize(pX->size());

	ct::Size szOut;

	if(m_use_transpose){
		if(m_use_same)
			gpumat::im2colsT_same(*pX, szA0, channels, szW, stride, Xc, szOut);
		else
			gpumat::im2colsT(*pX, szA0, channels, szW, stride, Xc, szOut);
	}else{
		if(m_use_same)
			gpumat::im2cols_same(*pX, szA0, channels, szW, stride, Xc, szOut);
		else
			gpumat::im2cols(*pX, szA0, channels, szW, stride, Xc, szOut);
	}

	for(int i = 0; i < (int)Xc.size(); ++i){
		gpumat::GpuMat& Xi = Xc[i];
		gpumat::GpuMat& A1i = A1[i];
		gpumat::m2mpbaf(Xi, W, B, m_func, A1i, m_params[LEAKYRELU]);
	}

	if(m_pool_dropout){
		get_dropout(m_prob_dropout, A1, m_Dropout);
	}

	if(m_use_pool){
		Mask.resize(Xc.size());
		A2.resize(A1.size());
		ct::Size szOut;
		gpumat::subsample(A1, szA1, A2, Mask, szOut);
		szK = A2[0].sz();

		if(m_use_bn){
			bn.X = &A2;
			bn.Y = &A3;
			bn.normalize();
		}
	}else{
		szK = A1[0].sz();

		if(m_use_bn){
			bn.X = &A1;
			bn.Y = &A3;
			bn.normalize();
		}
	}

#if 0
	if(channels == 3){
		gpumat::save_gmat((*pX)[0], "testPx26.txt");
		gpumat::save_gmat(Xc[0], "testXc26.txt");
		gpumat::save_gmat(A1[0], "testA126.txt");
		if(!A3.empty())gpumat::save_gmat(A3[0], "testA326.txt");
		if(!A2.empty())gpumat::save_gmat(A2[0], "testA226.txt");
		gpumat::save_gmat(W, "testW.txt");
		gpumat::save_gmat(B, "testB.txt");
		if(!Mask.empty())gpumat::save_gmat(Mask[0], "testMask.txt");
		throw new std::string("ee");
	}
#endif

#if 0
	{
		QString pref = QString::number(channels) + "_" + QString::number(K);
		qt_work_mat::q_save_mat((*pX)[0], "Px_" + pref + ".txt");
		qt_work_mat::q_save_mat(Xc[0], "Xc_" + pref + ".txt");
		qt_work_mat::q_save_mat(A1[0], "A1_" + pref + ".txt");
		if(!A2.empty()){
			qt_work_mat::q_save_mat(Mask[0], "M_" + pref + ".txt");
			qt_work_mat::q_save_mat(A2[0], "A2_" + pref + ".txt");
		}
		qt_work_mat::q_save_mat(W[0], "W_" + pref + ".txt");
		qt_work_mat::q_save_mat(B[0], "B_" + pref + ".txt");
	}
#endif
}

void convnn_gpu::backcnv(const std::vector<gpumat::GpuMat> &D, std::vector<gpumat::GpuMat> &DS)
{
//	DA1.resize(A1.size());
	/// A1 -> DA1
	for(int i = 0; i < (int)D.size(); ++i){
		gpumat::mul2deriv(D[i], A1[i], m_func, DS[i], m_params[gpumat::LEAKYRELU]);
	}
}

void convnn_gpu::backward(const std::vector<gpumat::GpuMat> &D, bool last_level)
{
	if(D.empty() || D.size() != A1.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

	dSub2.resize(D.size());

	std::vector< GpuMat >& _D = (std::vector< GpuMat >&)D;

	if(m_use_pool){
		if(m_use_bn){
			bn.D = (std::vector< GpuMat >*)&D;
			bn.denormalize();
			_D = bn.Dout;
		}

		gpumat::upsample(_D, kernels, Mask, szA2, szA1, dSub2);

		backcnv(dSub2, dSub2);
	}else{
		if(m_use_bn){
			bn.D = (std::vector< GpuMat >*)&D;
			bn.denormalize();
			_D = bn.Dout;
		}

		backcnv(_D, dSub2);
	}

//	gpumat::save_gmat(dSub2[0], "gW.txt");

	if(m_pool_dropout){
		set_dropout(dSub2, m_Dropout);
	}
#if 0
	if(channels == 3){
		qt_work_mat::q_save_mat(D[26], "testD26.txt");
//		qt_work_mat::q_save_mat(dSub[26], "testDSub26.txt");
		qt_work_mat::q_save_mat(dSub2[26], "testDSub2_26.txt");
		//save_vec(dSub2);
	}
#endif

	gW.zeros();
	gB.zeros();
	for(int i = 0; i < (int)D.size(); ++i){
		gpumat::GpuMat& Xci		= Xc[i];
		gpumat::GpuMat& dSubi	= dSub2[i];
		gpumat::add2matmulT1(Xci, dSubi, gW);

//		vgB.swap_dims();
		add2sumRows(dSubi, gB, 1/dSubi.rows /*, (double)1. / (Xci.total())*/);
//		vgB.swap_dims();

		//gpumat::add(gW[0], vgW);
		//gpumat::add(gB[0], vgB);
	}
	gpumat::mulval(gW, (double)1./(D.size() * channels));
	gpumat::mulval(gB, (double)1./(D.size() * channels));

#if 0
#if 0
	gW[0].zeros();
	gB[0].zeros();
	for(size_t i = 0; i < D.size(); ++i){
		gpumat::add(gW[0], vgW[i]);
		gpumat::add(gB[0], vgB[i]);
	}
	gpumat::mulval(gW[0], (double)1./(D.size() * channels));
	gpumat::mulval(gB[0], (double)1./(D.size() * channels));
#else
	addvec(gW[0], vgW, (double)1./(D.size() * channels));
	addvec(gB[0], vgB, (double)1./(D.size() * channels));
#endif
#endif

	if(m_lambda > 0){
		gpumat::add(gW, W, 1, (double)m_lambda / kernels);
	}

#if 0
	if(1/*channels == 128*/){
		save_vec(vgW);
		gpumat::save_gmat(gW[0], "Wg.txt");
	}
#endif
	if(!last_level){
		Dlt.resize(D.size());

		Dc.resize(D.size());
		for(int i = 0; i < (int)D.size(); ++i){
			gpumat::GpuMat& Dci = Dc[i];
			gpumat::matmulT2(dSub2[i], W, Dci);
		}

		if(m_use_transpose){
			if(m_use_same)
				cols2imT_same(Dc, szA1, szA0, channels, szW, stride, Dlt);
			else
				cols2imT(Dc, szA1, szA0, channels, szW, stride, Dlt);
		}else{
			if(m_use_same)
				cols2im_same(Dc, szA1, szA0, channels, szW, stride, Dlt);
			else
				cols2im(Dc, szA1, szA0, channels, szW, stride, Dlt);
		}
#if 0
		check_deriv(Dc, szA1, szA0, channels, szW, stride, Dlt);
#endif

	}
}

void convnn_gpu::write(std::fstream &fs)
{
	gpumat::write_fs(fs, W);
	gpumat::write_fs(fs, B);
}

void convnn_gpu::read(std::fstream &fs)
{
	gpumat::read_fs(fs, W);
	gpumat::read_fs(fs, B);
}

void convnn_gpu::write2(std::fstream &fs)
{
//	int rows = szW.area() * channels;
//	int cols = K;

	fs.write((char*)&szW.width, sizeof(szW.width));
	fs.write((char*)&szW.height, sizeof(szW.height));
	fs.write((char*)&channels, sizeof(channels));
	fs.write((char*)&kernels, sizeof(kernels));

	gpumat::write_fs2(fs, W);
	gpumat::write_fs2(fs, B);
}

void convnn_gpu::read2(std::fstream &fs)
{
	fs.read((char*)&szW.width, sizeof(szW.width));
	fs.read((char*)&szW.height, sizeof(szW.height));
	fs.read((char*)&channels, sizeof(channels));
	fs.read((char*)&kernels, sizeof(kernels));

	gpumat::read_fs2(fs, W);
	gpumat::read_fs2(fs, B);

	if(B.rows != 1)
		B.swap_dims();
}

///////////////////////////////
///////////////////////////////

CnvAdamOptimizer::CnvAdamOptimizer() : AdamOptimizer()
{
	stop_layer = 0;
}

bool CnvAdamOptimizer::init(std::vector<convnn_gpu> &cnv)
{
	int index = 0;
	m_mW.resize(cnv.size());
	m_mb.resize(cnv.size());
	m_vW.resize(cnv.size());
	m_vb.resize(cnv.size());

	mG.resize(cnv.size());
	mB.resize(cnv.size());

	vG.resize(cnv.size());
	vB.resize(cnv.size());

	for(convnn_gpu& item: cnv){
		initI(item.W, item.B, index++);
	}
	init_iteration();
	return true;
}

bool CnvAdamOptimizer::pass(std::vector<convnn_gpu> &cnv)
{
	if(cnv.empty() || cnv.back().gW.empty() || cnv.back().gB.empty())
		return false;
	int index = 0;
	next_iteration();
	for(convnn_gpu& item: cnv){
		if(index >= stop_layer){
			if(item.use_bn()){
				if(mG[index].empty()){
					mG[index].resize(item.bn.dgamma); mG[index].zeros();
					mB[index].resize(item.bn.dbetha); mB[index].zeros();
					vG[index].resize(item.bn.dgamma); vG[index].zeros();
					vB[index].resize(item.bn.dbetha); vB[index].zeros();
				}
				sub_adamGrad(item.bn.gamma, item.bn.dgamma, mG[index], vG[index], m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
				sub_adamGrad(item.bn.betha, item.bn.dbetha, mB[index], vB[index], m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
			}

			passI(item.gW, item.gB, item.W, item.B, index);
		}
		index++;
	}
	return true;
}

///////////////////////////////


CnvMomentumOptimizer::CnvMomentumOptimizer() : MomentumOptimizer()
{
	stop_layer = 0;
}

bool CnvMomentumOptimizer::init(std::vector<convnn_gpu> &cnv)
{
	if(cnv.empty())
		return false;
	int index = 0;
	m_mW.resize(cnv.size());
	m_mb.resize(cnv.size());

	mG.resize(cnv.size());
	mB.resize(cnv.size());

	for(convnn_gpu& item: cnv){
		initI(item.W, item.B, index++);
	}
	m_iteration = 0;
	return true;
}

bool CnvMomentumOptimizer::pass(std::vector<convnn_gpu> &cnv)
{
	if(cnv.empty() || cnv.back().gW.empty() || cnv.back().gB.empty())
		return false;

	m_iteration++;
	int index = 0;
	for(convnn_gpu& item: cnv){
		if(index >= stop_layer){
			if(item.use_bn()){
				if(mG[index].empty()){
					mG[index].resize(item.bn.dgamma);
					mB[index].resize(item.bn.dbetha);
					mG[index].zeros();
					mB[index].zeros();
				}
				momentum_optimizer(item.bn.gamma, mG[index], item.bn.dgamma, m_alpha, m_betha);
				momentum_optimizer(item.bn.betha, mB[index], item.bn.dbetha, m_alpha, m_betha);
			}

			passI(item.gW, item.gB, item.W, item.B, index);
		}
		index++;
	}
	return true;
}

///////////////////////////////
///////////////////////////////

extern "C"
void cuda_batch_normalize(_BN &bn);

extern "C"
void cuda_batch_denormalize(_BN &bn);

extern "C"
void cuda_scale_and_shift_bn(_BN &bn);

///////////////////////////////

BN::BN(): _BN()
{
	train = true;
}

void gpumat::BN::normalize()
{
	if(!X || !Y || X->empty() || X->front().empty())
		throw new std::invalid_argument("batch_normalize: empty parameters");

	Mean.resize(1, channels, X->front().type);
	Var.resize(1, channels, X->front().type);
	Y->resize(X->size());
	Xu.resize(X->size());

	Mean.zeros();
	Var.zeros();

	if(gamma.empty() || betha.empty())
		initGammaAndBetha();

	if(1){
		int index = 0;
		for(const GpuMat& Xi: *X){
			Y->operator [](index).resize(Xi);
			Xu[index].resize(Xi);
			++index;
		}


		cuda_batch_normalize(*this);
//		index = 0;
//		for(gpumat::GpuMat& x : *X){
//			gpumat::save_gmat(x, "X" + std::to_string(index++) + ".txt");
//		}
//		gpumat::save_gmat(Mean, "mean.txt");
//		gpumat::save_gmat(Var, "var.txt");
//		gpumat::save_gmat(gamma, "g.txt");
//		gpumat::save_gmat(betha, "b.txt");
	}else{
		scaleAndShift();
	}
}

void gpumat::BN::denormalize()
{
	if(!D || D->empty() || D->front().empty() || Mean.empty() || Var.empty()
			|| Mean.cols != channels
			|| Var.cols != channels || gamma.empty() || betha.empty())
		throw new std::invalid_argument("batch_denormalize: empty parameters");

	Dout.resize(D->size());
	int index = 0;
	for(const GpuMat& Di: *D){
		Dout[index++].resize(Di);
	}

	cuda_batch_denormalize(*this);
}

void BN::initGammaAndBetha()
{
	gamma.resize(Mean);
	betha.resize(Mean);

	gamma.ones();
	betha.zeros();
}

void BN::scaleAndShift()
{
	if(gamma.empty() || betha.empty())
		initGammaAndBetha();

	Y->resize(X->size());
	cuda_scale_and_shift_bn(*this);
}

void BN::read(std::fstream &fs)
{
	gpumat::read_fs2(fs, gamma);
	gpumat::read_fs2(fs, betha);
}

void BN::write(std::fstream &fs)
{
	gpumat::write_fs2(fs, gamma);
	gpumat::write_fs2(fs, betha);
}

///////////////////////////////
///////////////////////////////
///////////////////////////////

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsT(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsT_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

///////////////

extern "C"
void cuda_im2colsSame(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2cols_vecSame(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsTSame(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsT_vecSame(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

/////////////

extern "C"
void cuda_cols2im(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X);

extern "C"
void cuda_cols2im_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X);

extern "C"
void cuda_cols2imT(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X);

extern "C"
void cuda_col2imT_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X);

///////////////////////////////////

extern "C"
void cuda_cols2im_same(const gpumat::GpuMat &Delta, const ct::Size &szDelta,
					   const ct::Size &szA0, int channels, const ct::Size &szW,
					   int stride, gpumat::GpuMat &X);

extern "C"
void cuda_cols2im_vec_same(const std::vector< gpumat::GpuMat > &Delta,
						   ct::Size szDelta, const ct::Size &szA0,
						   int channels, const ct::Size &szW, int stride,
						   std::vector< gpumat::GpuMat > &X);

extern "C"
void cuda_cols2imT_same(const gpumat::GpuMat &Delta,
						const ct::Size &szDelta, const ct::Size &szA0,
						int channels, const ct::Size &szW, int stride, gpumat::GpuMat &X);

extern "C"
void cuda_col2imT_vec_same(const std::vector< gpumat::GpuMat > &Delta,
						   const ct::Size &szDelta, const ct::Size &szA0,
						   int channels, const ct::Size &szW, int stride,
						   std::vector< gpumat::GpuMat > &X);

///////////////////////////////////

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
					const ct::Size &szA,
					gpumat::GpuMat &Y,
					gpumat::GpuMat &Mask,
					ct::Size &szO);

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO);

extern "C"
void cuda_vec2mat(const std::vector< gpumat::GpuMat >& vec, gpumat::GpuMat& mat);

extern "C"
void cuda_mat2vec(const gpumat::GpuMat& mat, const ct::Size& sz, std::vector< gpumat::GpuMat >& vec);

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X);

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X);

extern "C"
void cuda_addvec(gpumat::GpuMat &W, const std::vector<gpumat::GpuMat> &vW, double alpha);

///////////////////////////////

void gpumat::im2cols(const gpumat::GpuMat &X, const ct::Size &szA0,
							int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);

	cuda_im2cols(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::im2colsT(const gpumat::GpuMat &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);

	cuda_im2colsT(X, szA0, channels, szW, stride, Res, szOut);
}


void gpumat::im2cols(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							int channels, const ct::Size &szW, int stride,
							std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
	}

	cuda_im2cols_vec(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::im2colsT(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW, int stride,
							 std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2colsT: empty parameters");

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
	}

	cuda_im2colsT_vec(X, szA0, channels, szW, stride, Res, szOut);
}

//////////////////////

void gpumat::cols2im(const gpumat::GpuMat &Delta,
				const ct::Size &szDelta,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(channels, szA0.area(), Delta.type);
	X.zeros();

	cuda_cols2im(Delta, szDelta, szA0, channels, szW, stride, X);
}

void gpumat::cols2im(const std::vector<gpumat::GpuMat> &Delta,
							   const ct::Size &szOut, const ct::Size &szA0,
							   int channels, const ct::Size &szW, int stride,
							   std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(channels, szA0.area(), type);
		X[i].zeros();
	}

	cuda_cols2im_vec(Delta, szOut, szA0, channels, szW, stride, X);

}

void gpumat::cols2imT(const gpumat::GpuMat &Delta,
				const ct::Size &szDelta,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(szA0.area(), channels, Delta.type);
	X.zeros();

	cuda_cols2imT(Delta, szDelta, szA0, channels, szW, stride, X);
}

void gpumat::cols2imT(const std::vector<gpumat::GpuMat> &Delta,
								const ct::Size &szOut, const ct::Size &szA0,
								int channels, const ct::Size &szW, int stride,
								std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(szA0.area(), channels, type);
		X[i].zeros();
	}

	cuda_col2imT_vec(Delta, szOut, szA0, channels, szW, stride, X);

}

///////// SAME /////////////

void gpumat::cols2im_same(const gpumat::GpuMat &Delta,
						  const ct::Size &szDelta, const ct::Size &szA0,
						  int channels, const ct::Size &szW, int stride, gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(channels, szA0.area(), Delta.type);
	X.zeros();

	cuda_cols2im_same(Delta, szDelta, szA0, channels, szW, stride, X);
}

void gpumat::cols2im_same(const std::vector<gpumat::GpuMat> &Delta, const ct::Size &szDelta,
						  const ct::Size &szA0, int channels, const ct::Size &szW, int stride,
						  std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(channels, szA0.area(), type);
		X[i].zeros();
	}

	cuda_cols2im_vec_same(Delta, szDelta, szA0, channels, szW, stride, X);

}

void gpumat::cols2imT_same(const gpumat::GpuMat &Delta,
						   const ct::Size &szDelta, const ct::Size &szA0,
						   int channels, const ct::Size &szW, int stride, gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(szA0.area(), channels, Delta.type);
	X.zeros();

	cuda_cols2imT_same(Delta, szDelta, szA0, channels, szW, stride, X);
}

void gpumat::cols2imT_same(const std::vector<gpumat::GpuMat> &Delta,
						   const ct::Size &szDelta, const ct::Size &szA0,
						   int channels, const ct::Size &szW, int stride,
						   std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(szA0.area(), channels, type);
		X[i].zeros();
	}

	cuda_col2imT_vec_same(Delta, szDelta, szA0, channels, szW, stride, X);

}

////////////////////////////

void gpumat::subsample(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X.rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = (szA.width + 1) / 2;
	szO.height = (szA.height + 1) / 2;
	int K = X.cols;

	Y.resize(szO.area(), K, X.type);
	Mask.resize(X.rows, X.cols, X.type);
	Mask.zeros();

	cuda_subsample2(X, szA, Y, Mask, szO);
}

void gpumat::subsample(const std::vector<gpumat::GpuMat> &X,
							  const ct::Size &szA,
							  std::vector<gpumat::GpuMat> &Y,
							  std::vector<gpumat::GpuMat> &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X[0].rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = (szA.width + 1) / 2;
	szO.height = (szA.height + 1) / 2;

	int K = X[0].cols;

	Y.resize(X.size());
	Mask.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		Y[i].resize(szO.area(), K, X[i].type);
		Y[i].zeros();
		Mask[i].resize(X[i].rows, X[i].cols, X[i].type);
		Mask[i].zeros();
	}

	cuda_subsample2_vec(X, szA, Y, Mask, szO);
}

void gpumat::vec2mat(const std::vector<gpumat::GpuMat> &vec, gpumat::GpuMat &mat)
{
	if(vec.empty() || vec[0].empty())
		throw new std::invalid_argument("vec2mat: empty parameters");

	int rows = (int)vec.size();
	int cols = vec[0].total();

	mat.resize(rows, cols, vec[0].type);

	cuda_vec2mat(vec, mat);
}

void gpumat::mat2vec(const gpumat::GpuMat &mat, const ct::Size &szOut,
							std::vector<gpumat::GpuMat> &vec)
{
	if(mat.empty())
		throw new std::invalid_argument("mat2vec: empty parameters");

	int rows = mat.rows;

	vec.resize(rows);

	for(size_t i = 0; i < vec.size(); ++i){
		vec[i].resize(szOut.height, szOut.width, mat.type);
	}

	cuda_mat2vec(mat, szOut, vec);
}

void gpumat::upsample(const gpumat::GpuMat &Y, int K,
							 const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	if(Y.empty() || Mask.empty() || Y.total() != szO.area() * K)
		throw new std::invalid_argument("upsample: empty parameters");

	X.resize(szA.area(), K, Y.type);

	cuda_upsample2(Y, Mask, szO, szA, X);
}

void gpumat::upsample(const std::vector<gpumat::GpuMat> &Y,
							 int K, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	if(Y.empty() || Y[0].empty() || Mask.empty() || Mask[0].empty() || Y[0].total() != szO.area() * K)
		throw new std::invalid_argument("upsample: empty parameters");

	X.resize(Y.size());

	int type = Y[0].type;
	int area = szA.area();

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(area, K, type);
		X[i].zeros();
	}

	cuda_upsample2vec(Y, Mask, szO, szA, X);

}

void gpumat::addvec(gpumat::GpuMat &W, const std::vector<gpumat::GpuMat> &vW, double alpha)
{
	if(vW.empty() || vW[0].empty())
		throw new std::invalid_argument("addvec: empty parameters");

	const GpuMat& Wi = vW[0];

	W.resize(Wi);

	cuda_addvec(W, vW, alpha);
}

//////////////////

void gpumat::im2cols_same(const gpumat::GpuMat &X, const ct::Size &szA0,
							int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);
	Res.zeros();

	cuda_im2colsSame(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::im2colsT_same(const gpumat::GpuMat &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);
	Res.zeros();

	cuda_im2colsTSame(X, szA0, channels, szW, stride, Res, szOut);
}


void gpumat::im2cols_same(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							int channels, const ct::Size &szW, int stride,
							std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
		Res[i].zeros();
	}

	cuda_im2cols_vecSame(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::im2colsT_same(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW, int stride,
							 std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2colsT: empty parameters");

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
		Res[i].zeros();
	}

	cuda_im2colsT_vecSame(X, szA0, channels, szW, stride, Res, szOut);
}


void gpumat::conv2(const gpumat::GpuMat &A, const ct::Size &szA, int channels, int stride,
		   const gpumat::GpuMat &B, const ct::Size &szB, gpumat::GpuMat &C, ct::Size &szOut, TYPE_CONV type, bool transpose)
{
	if(A.empty() || B.empty())
		return;

	gpumat::GpuMat X;

	if(type == SAME){
		if(transpose)
			im2colsT_same(A, szA, channels, szB, stride, X, szOut);
		else
			im2cols_same(A, szA, channels, szB, stride, X, szOut);
	}else{
		if(transpose)
			im2colsT(A, szA, channels, szB, stride, X, szOut);
		else
			im2cols(A, szA, channels, szB, stride, X, szOut);
	}

	if(X.empty())
		return;

	gpumat::GpuMat& W = (gpumat::GpuMat &)B;
//	int rows = W.rows;
//	int cols = W.cols;
//	W.rows = szB.area() * channels;
//	W.cols = (rows * cols) / W.rows;

	gpumat::matmul(X, W, C);

//	W.rows = rows;
//	W.cols = cols;
}


void gpumat::conv2_transpose(const GpuMat &C, const ct::Size &szA, int channels, int stride,
							 const GpuMat &B, const ct::Size &szB, const ct::Size &szC,
							 GpuMat &A, TYPE_CONV type, bool transpose)
{
	if(C.empty() || B.empty())
		return;

	GpuMat D;
	GpuMat& W = (GpuMat&)B;
//	int rows = W.rows;
//	int cols = W.cols;
//	W.rows = szB.area() * channels;
//	W.cols = (rows * cols) / W.rows;

	gpumat::matmulT2(C, W, D);

//	W.rows = rows;
//	W.cols = cols;
	if(D.empty())
		return;

	if(type == SAME){
		if(transpose){
			cols2imT_same(D, szC, szA, channels, szB, stride, A);
		}else{
			cols2im_same(D, szC, szA, channels, szB, stride, A);
		}

	}else{
		if(transpose){
			cols2imT(D, szC, szA, channels, szB, 1, A);
		}else{
			cols2im(D, szC, szA, channels, szB, 1, A);
		}
	}
}


