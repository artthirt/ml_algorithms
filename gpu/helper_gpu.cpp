#include "helper_gpu.h"

#include "matops.h"

//#include <QDebug>

namespace gpumat{

void convert_to_gpu(const ct::Matf& mat, gpumat::GpuMat& gmat)
{
	if(mat.empty())
		return;
	gmat.resize(mat.rows, mat.cols, GPU_FLOAT);
	gmat.setData(mat.ptr());
}

void convert_to_gpu(const ct::Matd& mat, gpumat::GpuMat& gmat)
{
	if(mat.empty())
		return;
	gmat.resize(mat.rows, mat.cols, GPU_DOUBLE);
	gmat.setData(mat.ptr());
}

void convert_to_mat(const GpuMat &gmat, ct::Matf &mat)
{
	if(gmat.empty() || gmat.type != GPU_FLOAT)
		return;
	mat.setSize(gmat.rows, gmat.cols);
	gmat.getData((void*)mat.ptr());
}

void convert_to_mat(const GpuMat &gmat, ct::Matd &mat)
{
	if(gmat.empty() || gmat.type != GPU_DOUBLE)
		return;
	mat.setSize(gmat.rows, gmat.cols);
	gmat.getData((void*)mat.ptr());
}

///*****************

template< typename T >
void write_mat(std::fstream &fs, const gpumat::GpuMat& mat)
{
	ct::Mat_<T> mmat;
	convert_to_mat(mat, mmat);

	ct::write_fs(fs, mmat);
}

void write_fs(std::fstream &fs, const gpumat::GpuMat& mat)
{
	if(!mat.empty()){
		switch (mat.type) {
		case GPU_DOUBLE:
			write_mat<double>(fs, mat);
			break;
		case GPU_FLOAT:
			write_mat<float>(fs, mat);
			break;
		}
	}
}

//////////////////////////////////

template< typename T >
void write_mat2(std::fstream &fs, const gpumat::GpuMat& mat)
{
	ct::Mat_<T> mmat;
	convert_to_mat(mat, mmat);

	ct::write_fs2(fs, mmat);
}

void write_fs2(std::fstream &fs, const gpumat::GpuMat& mat)
{
	if(!mat.empty()){
		switch (mat.type) {
		case GPU_DOUBLE:
			write_mat2<double>(fs, mat);
			break;
		case GPU_FLOAT:
			write_mat2<float>(fs, mat);
			break;
		}
	}
}

////////////////////////////

template< typename T >
void read_mat(std::fstream &fs, gpumat::GpuMat& mat)
{
	ct::Mat_<T> mmat;
	mmat.setSize(mat.rows, mat.cols);
	ct::read_fs(fs, mmat);

	convert_to_gpu(mmat, mat);
}

void read_fs(std::fstream &fs, gpumat::GpuMat& mat)
{
	if(!mat.empty()){
		switch (mat.type) {
		case GPU_DOUBLE:
			read_mat<double>(fs, mat);
			break;
		case GPU_FLOAT:
			read_mat<float>(fs, mat);
			break;
		}
	}
}

///////////////////////////

template< typename T >
void read_mat2(std::fstream &fs, gpumat::GpuMat& mat)
{
	ct::Mat_<T> mmat;
	ct::read_fs2(fs, mmat);

	convert_to_gpu(mmat, mat);
}

void read_fs2(std::fstream &fs, gpumat::GpuMat& mat, int type)
{
	switch (type) {
	case GPU_DOUBLE:
		read_mat2<double>(fs, mat);
		break;
	case GPU_FLOAT:
		read_mat2<float>(fs, mat);
		break;
	}
}

////////////////////////////

void write_gmat(const std::string &name, const GpuMat &mat)
{
	std::fstream fs;
	fs.open(name, std::ios_base::out);

	write_fs(fs, mat);

	fs.close();
}

///////////////////////////////


Optimizer::Optimizer()
{
	m_alpha = 0.001;
	m_iteration = 1;
}

Optimizer::~Optimizer()
{

}

double Optimizer::alpha() const
{
	return m_alpha;
}

void Optimizer::setAlpha(double v)
{
	m_alpha = v;
}

uint32_t Optimizer::iteration() const
{
	return m_iteration;
}

///////////////////////////////

StohasticGradientOptimizer::StohasticGradientOptimizer(): Optimizer()
{

}

bool StohasticGradientOptimizer::init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB)
{
	gradW, gradB;
	m_iteration = 0;
	return true;
}

bool StohasticGradientOptimizer::pass(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB, std::vector<GpuMat> &W, std::vector<GpuMat> &b)
{
	if(gradW.empty() || gradB.empty() || W.empty() || b.empty() ||
			gradW.size() != W.size() || b.size() != gradB.size()){
		return false;
	}

	m_iteration++;

	for(size_t i = 0; i < W.size(); ++i){
		passI(gradW[i], gradB[i], W[i], b[i], i);
	}
	return true;
}

void StohasticGradientOptimizer::initI(const GpuMat &W, const GpuMat &B, int index)
{
	W, B, index;
}

void StohasticGradientOptimizer::passI(const GpuMat &gW, const GpuMat &gB, GpuMat &W, GpuMat &B, int index)
{
	index;
	gpumat::sub(W, gW, 1., m_alpha);
	gpumat::sub(B, gB, 1., m_alpha);
}

///////////////////////////////


MomentumOptimizer::MomentumOptimizer(): Optimizer()
{
	m_betha = 0.9;
}

double MomentumOptimizer::betha() const
{
	return m_betha;
}

void MomentumOptimizer::setBetha(double b)
{
	m_betha = b;
}

bool MomentumOptimizer::init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB)
{
	m_iteration = 0;

	m_mW.resize(gradW.size());
	m_mb.resize(gradW.size());

	for(size_t i = 0; i < gradW.size(); i++){
		initI(gradW[i], gradB[i], i);
	}
	return true;
}

bool MomentumOptimizer::pass(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB,
							 std::vector<GpuMat> &W, std::vector<GpuMat> &B)
{
	if(gradW.empty() || gradB.empty() || W.empty() || B.empty())
		return false;

	m_iteration++;
	for(size_t i = 0; i < gradW.size(); ++i){
		passI(gradW[i], gradB[i], W[i], B[i], i);
	}
	return true;

}

void MomentumOptimizer::initI(const GpuMat &W, const GpuMat &B, int index)
{
	m_mW[index].resize(W);
	m_mb[index].resize(B);

	m_mW[index].zeros();
	m_mb[index].zeros();
}

void MomentumOptimizer::passI(const GpuMat &gW, const GpuMat &gB, GpuMat &W, GpuMat &B, int index)
{
	gpumat::momentum_optimizer(W, m_mW[index], gW, m_alpha, m_betha);
	gpumat::momentum_optimizer(B, m_mb[index], gB, m_alpha, m_betha);
}

///////////////////////////////

AdamOptimizer::AdamOptimizer(): Optimizer()
{
	m_betha1 = 0.9;
	m_betha2 = 0.99;
	init_iteration();
	m_init_matB = false;
}


double AdamOptimizer::betha1() const
{
	return m_betha1;
}

void AdamOptimizer::setBetha1(double v)
{
	m_betha1 = v;
}

double AdamOptimizer::betha2() const{
	return m_betha2;
}

void AdamOptimizer::setBetha2(double v)
{
	m_betha2 = v;
}

bool AdamOptimizer::empty() const
{
	return m_mW.empty() || m_mb.empty();
}

bool AdamOptimizer::init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB)
{
//	qDebug("init...");
	m_iteration = 0;

	m_mW.resize(gradW.size());
	m_mb.resize(gradW.size());

	m_vW.resize(gradW.size());
	m_vb.resize(gradW.size());

	for(size_t i = 0; i < gradW.size(); i++){
		initI(gradW[i], gradB[i], i);
	}

	m_init_matB = true;

	init_iteration();

	return true;
}

void AdamOptimizer::next_iteration()
{
	m_iteration++;
	m_sb1 = (1. / (1. - pow(m_betha1, m_iteration)));
	m_sb2 = (1. / (1. - pow(m_betha2, m_iteration)));
}

void AdamOptimizer::init_iteration()
{
	m_iteration = 0;
	m_sb1 = 1;
	m_sb2 = 1;
}

bool AdamOptimizer::pass(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB,
						 std::vector<GpuMat> &W, std::vector<GpuMat> &b)
{
	if(gradW.empty() || gradB.empty() || W.empty() || b.empty())
		return false;

	if(!m_init_matB){
		init(gradW, gradB);
	}

	next_iteration();

	for(size_t i = 0; i < gradW.size(); ++i){
		passI(gradW[i], gradB[i], W[i], b[i], i);
	}
	return true;
}

void AdamOptimizer::initI(const GpuMat &W, const GpuMat &B, int index)
{
	m_mW[index].resize(W);
	m_mW[index].zeros();
	m_vW[index].resize(W);
	m_vW[index].zeros();

	m_mb[index].resize(B);
	m_mb[index].zeros();
	m_vb[index].resize(B);
	m_vb[index].zeros();

}

void AdamOptimizer::passI(const GpuMat &gW, const GpuMat &gB, GpuMat &W, GpuMat &B, int index)
{
	gpumat::sub_adamGrad(W, gW, m_mW[index], m_vW[index], m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
	gpumat::sub_adamGrad(B, gB, m_mb[index], m_vb[index], m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
}

///////////////////////////////////////////
///////////////////////////////////////////

SimpleAutoencoder::SimpleAutoencoder(){
	func = 0;
	deriv = 0;
	m_neurons = 0;
}

void SimpleAutoencoder::init(GpuMat &_W, GpuMat &_b, int samples, int neurons, SimpleAutoencoder::tfunc fn, SimpleAutoencoder::tfunc dfn)
{
	func = fn;
	deriv = dfn;
	m_neurons = neurons;

	std::vector< int > layers;
	layers.push_back(neurons);
	layers.push_back(samples);

	W.resize(2);
	b.resize(2);
	dW.resize(2);
	db.resize(2);

	W[0] = _W;
	b[0] = _b;

	transpose(_W, W[1]);
	b[1].resize(1, samples, _W.type);
	b[1].zeros();

	adam.init(W, b);
	//		W[0].randn(0, 0.1, 1);
	//		b[0].randn(0, 0.1, 1);
	//		W[1].randn(0, 0.1, 1);
	//		b[1].randn(0, 0.1, 1);
}

void SimpleAutoencoder::pass(const GpuMat &X)
{
	if(X.empty() || X.cols != W[0].rows || !func || !deriv)
		return;

	a[0] = X;
	for(int i = 0; i < 2; i++){
//		PRINT_GMAT10(a[i]);
//		PRINT_GMAT10(W[i]);
//		PRINT_GMAT10(b[i]);
		matmul_shared(a[i], W[i], z[i]);
//		W[i].save("W.txt");
//		a[i].save("a.txt");
//		z[i].save("z.txt");
//		PRINT_GMAT10(W[i]);
//		PRINT_GMAT10(z[i]);
		biasPlus(z[i], b[i]);
//		PRINT_GMAT10(z[i]);
		if(i == 0){
			(*func)(z[i], a[i + 1]);
//			PRINT_GMAT10(a[i + 1]);
		}else{
			a[i + 1] = z[i];
//			PRINT_GMAT10(a[i + 1]);
		}
	}

	double m = X.rows;

	sub(a[2], X, d);

//	PRINT_GMAT10(d);
	for(int i = 1; i > -1; --i){
		if(i > 0){
			(*deriv)(a[i], sz);
			matmulT2_shared(d, W[i], di);
//			PRINT_GMAT10(di);
			elemwiseMult(di, sz);
//			PRINT_GMAT10(di);
		}
//		a[i].save("ai.txt");
//		d.save("d.txt");
		matmulT1_shared(a[i], d, dW[i]);
		mulval(dW[i], 1./m);
//		dW[i].save("dWi.txt");
//		PRINT_GMAT10(d);
		sumRows_shared(d, db[i], 1./m);
//		PRINT_GMAT10(db[i]);
//		db[i].swap_dims();
		if(i > 0)
			d = di;
	}
	transpose(dW[1], tw1);
	add(dW[0], tw1);
	transpose(dW[0], dW[1]);

	db[1].zeros();

//	PRINT_GMAT10(dW[0]);
//	PRINT_GMAT10(dW[1]);
//	PRINT_GMAT10(db[0]);
//	PRINT_GMAT10(db[1]);
	adam.pass(dW, db, W, b);
}

double SimpleAutoencoder::l2(const GpuMat &X)
{
	if(X.empty() || W[0].empty())
		return -1.;

	a[0] = X;
	for(int i = 0; i < 2; i++){
		matmul(a[i], W[i], z[i]);
		biasPlus(z[i], b[i]);
		if(i == 0){
			(*func)(z[i], a[i + 1]);
		}else{
			a[i + 1] = z[i];
		}
	}
	double m = X.rows;
	sub(a[2], X, d);
	elemwiseMult(d, d);
	double res = 0;
	if(d.type == GPU_FLOAT){
		ct::Matf df;
		convert_to_mat(d, df);
		res = df.sum() / m;

	}
	if(d.type == GPU_DOUBLE){
		ct::Matf dd;
		convert_to_mat(d, dd);
		res = dd.sum() / m;

	}
	return res;
}

/////////////////////////////

void save_gmat(const GpuMat &mat, const std::string &fn)
{
	std::string s = mat.print(-1);			\
	std::fstream fs;
	fs.open(fn.c_str(), std::ios_base::out);

	fs << s;

	fs.close();
}

void save_gmat10(const GpuMat &mat, const std::string &fn)
{
	std::string s = mat.print(10);			\
	std::fstream fs;
	fs.open(fn.c_str(), std::ios_base::out);

	fs << s;

	fs.close();
}

template< typename T >
void _cnv2gpu(const std::vector<ct::Mat_<T>> &M, std::vector<GpuMat> &G)
{
	if(M.empty())
		return;

	G.resize(M.size());

	for(size_t i = 0; i < M.size(); ++i){
		const ct::Mat_<T>& Mi = M[i];
		gpumat::GpuMat& Gi = G[i];
		gpumat::convert_to_gpu(Mi, Gi);
	}
}

void cnv2gpu(const std::vector<ct::Matf> &M, std::vector<GpuMat> &G)
{
	_cnv2gpu<float>(M, G);
}

void cnv2gpu(const std::vector<ct::Matd> &M, std::vector<GpuMat> &G)
{
	_cnv2gpu<double>(M, G);
}

}
