#include "gpu_mlp.h"

#include "matops.h"

using namespace gpumat;

template< typename T >
inline void init(GpuMat& mat, T n){
	ct::Mat_<T> m;
	m.setSize(mat.rows, mat.cols);
	m.randn(0, n);
	gpumat::convert_to_gpu(m, mat);
}

////////////////////////////////////////////////

mlp::mlp(){
	m_func = RELU;
	m_init = false;
	m_is_dropout = false;
	m_prob = 0.95;
	pA0 = nullptr;
	pVecA0 = nullptr;
	m_lambda = 0.;

	m_params[LEAKYRELU] = 0.1;
}

void mlp::setParams(etypefunction type, double param)
{
	m_params[type] = param;
}

void mlp::setLambda(double val)
{
	m_lambda = val;
}

void mlp::setDropout(bool val)
{
	m_is_dropout = val;
}

void mlp::setDropout(double val)
{
	m_prob = val;
}

bool mlp::isInit() const
{
	return m_init;
}

void mlp::init(int input, int output, int type, etypefunction func)
{
	double n = 1./sqrt(input);
	m_func = func;

	W.resize(input, output, type);
	B.resize(1, output, type);

	switch (type) {
		case GPU_DOUBLE:
			::init<double>(W, n);
			::init<double>(B, n);
			break;
		case GPU_FLOAT:
			::init<float>(W, (float)n);
			::init<float>(B, (float)n);
			break;
	}

	m_init = true;
}

void mlp::apply_func(const GpuMat &Z, GpuMat &A, etypefunction func){
	switch (func) {
		default:
			return;
		case RELU:
			reLu(Z, A);
			break;
		case SOFTMAX:
			softmax(Z, 1, A, PartZ);
			break;
		case SIGMOID:
			sigmoid(Z, A);
			break;
		case TANH:
			tanh(Z, A);
			break;
		case LEAKYRELU:
			leakyReLu(Z, m_params[LEAKYRELU], A);
			break;
	}
}

void mlp::apply_back_func(const GpuMat &D1, const GpuMat& A1, GpuMat &D2)
{
	if(m_func == LINEAR || m_func == SOFTMAX){
		if(D1.data != D2.data)
			D1.copyTo(D2);
		return;
	}

	gpumat::mul2deriv(D1, A1, m_func, D2, m_params[LEAKYRELU]);
//	switch (func) {
//		case RELU:
//			deriv_reLu(A1, D2);
//			break;
//		case SIGMOID:
//			deriv_sigmoid(A1, D2);
//			break;
//		case TANH:
//			deriv_tanh(A1, D2);
//			break;
//		case LEAKYRELU:
//			deriv_leakyReLu(A1, m_params[LEAKYRELU], D2);
//			break;
//		default:
//			if(D1.data == D2.data)
//				return;
//			else
//				D1.copyTo(D2);
//	}
//	elemwiseMult(D1, D2, D2);
}

etypefunction mlp::funcType() const{
	return m_func;
}

inline void apply_dropout(const GpuMat& X, double prob, GpuMat& XDropout, GpuMat& Dropout)
{
	switch (X.type) {
		case GPU_DOUBLE:{
			ct::Matd _Dropout;
			ct::dropout(X.rows, X.cols, prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(X, Dropout, XDropout);
			break;
		}
		case GPU_FLOAT:{
			ct::Matf _Dropout;
			ct::dropout(X.rows, X.cols, (float)prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(X, Dropout, XDropout);
			break;
		}
	}
}

void mlp::forward(const GpuMat *mat, bool save_A0)
{
	if(!m_init || !mat)
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pA0 = (GpuMat*)mat;

	if(m_is_dropout && m_prob < 1){
		apply_dropout(*pA0, m_prob, XDropout, Dropout);
//		matmul(XDropout, W, A1);
		if(m_func == SOFTMAX){
			m2mpbaf(XDropout, W, B, LINEAR, A1, m_params[LEAKYRELU]);
			softmax(A1, 1, PartZ);
		}else{
			m2mpbaf(XDropout, W, B, m_func, A1, m_params[LEAKYRELU]);
		}
	}else{
//		matmul(*pA0, W, A1);
		if(m_func == SOFTMAX){
			m2mpbaf(*pA0, W, B, LINEAR, A1, m_params[LEAKYRELU]);
			softmax(A1, 1, PartZ);
		}else{
			m2mpbaf(*pA0, W, B, m_func, A1, m_params[LEAKYRELU]);
		}
	}

//	biasPlus(A1, B);
//	if(m_func != LINEAR)
//		apply_func(A1, A1, m_func);

	if(!save_A0)
		pA0 = nullptr;
}

void mlp::backward(const GpuMat &Delta, bool last_layer)
{
	if(!pA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

//	apply_back_func(Delta, DA1, m_func);

	double m = Delta.rows;

	gpumat::GpuMat* pDA1 = &A1;

	if(m_func != gpumat::SOFTMAX && m_func != gpumat::LINEAR){
        apply_back_func(Delta, A1, A1);
	}else{
		pDA1 = (GpuMat*)&Delta;
	}

	if(m_is_dropout && m_prob < 1){
		matmulT1(XDropout, *pDA1, gW, 1. / m);
	}else{
		matmulT1(*pA0, *pDA1, gW, 1. / m);
	}
//	mulval(gW, 1. / m);


	if(m_lambda > 0){
		gpumat::add(gW, W, 1, m_lambda / m);
	}

//	gB.swap_dims();
	sumRows(*pDA1, gB, 1.f / m);
//	gB.swap_dims();

	if(!last_layer){
		matmulT2(*pDA1, W, DltA0);
	}
}

void mlp::forward(const std::vector<GpuMat> *mat, bool save_A0)
{
	if(!m_init || !mat || mat->empty())
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pVecA0 = (std::vector<GpuMat>*)mat;

	if(m_is_dropout && m_prob < 1){
		vecXDropout.resize(mat->size());
	}
	vecA1.resize(mat->size());

	for(size_t i = 0; i < mat->size(); ++i){
		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			gpumat::GpuMat& XD = vecXDropout[i];
			apply_dropout((*pVecA0)[i], m_prob, XD, Dropout);
	//		matmul(XDropout, W, A1);
			if(m_func == SOFTMAX){
                m2mpbaf(XD, W, B, LINEAR, vecA1[i]);
				softmax(A1, 1, PartZ);
			}else{
				m2mpbaf(XD, W, B, m_func, vecA1[i], m_params[LEAKYRELU]);
			}
		}else{
	//		matmul(*pA0, W, A1);
			if(m_func == SOFTMAX){
                m2mpbaf((*pVecA0)[i], W, B, LINEAR, vecA1[i]);
				softmax(vecA1[i], 1, PartZ);
			}else{
				m2mpbaf((*pVecA0)[i], W, B, m_func, vecA1[i], m_params[LEAKYRELU]);
			}
		}
	}


	if(!save_A0)
        pVecA0 = nullptr;
}

void mlp::backward(std::vector<GpuMat> &Delta, bool last_layer)
{
	if(!pVecA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

//	apply_back_func(Delta, DA1, m_func);

	double m = Delta.size();

	std::vector< gpumat::GpuMat>* pDA1 = &vecA1;

	if(m_func == gpumat::SOFTMAX || m_func == gpumat::LINEAR){
		pDA1 = (std::vector< gpumat::GpuMat>*)&Delta;
	}

//	vecDA1.resize(Delta.size());

	gW.resize(W);
	gB.resize(B);

	gW.zeros();
	gB.zeros();

	for(size_t i = 0; i < Delta.size(); ++i){
		if(m_func != gpumat::SOFTMAX && m_func != gpumat::LINEAR){
            apply_back_func(Delta[i], vecA1[i], vecA1[i]);
		}

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			add2matmulT1(vecXDropout[i], (*pDA1)[i], gW);
		}else{
			add2matmulT1((*pVecA0)[i], (*pDA1)[i], gW);
		}

//		gBi.swap_dims();
		add2sumRows((*pDA1)[i], gB, 1);
//		gBi.swap_dims();
	}

//	gWi.release();
//	gBi.release();

	mulval(gW, 1. / m);
	mulval(gB, 1. / m);

	if(m_lambda > 0){
		gpumat::add(gW, W, 1, m_lambda / m);
	}

	if(!last_layer){
		vecDltA0.resize(Delta.size());
#pragma omp parallel for
		for(int i = 0; i < Delta.size(); ++i){
			matmulT2((*pDA1)[i], W, vecDltA0[i]);
		}
	}
}

void mlp::write(std::fstream &fs)
{
	gpumat::write_fs(fs, W);
	gpumat::write_fs(fs, B);
}

void mlp::read(std::fstream &fs)
{
	gpumat::read_fs(fs, W);
	gpumat::read_fs(fs, B);
}

void mlp::write2(std::fstream &fs)
{
	gpumat::write_fs2(fs, W);
	gpumat::write_fs2(fs, B);
}

void mlp::read2(std::fstream &fs)
{
	gpumat::read_fs2(fs, W);
	gpumat::read_fs2(fs, B);

	if(B.rows != 1)
		B.swap_dims();
}

///**************************

bool MlpOptimAdam::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration = 0;

	m_mW.resize(_mlp.size());
	m_mb.resize(_mlp.size());

	m_vW.resize(_mlp.size());
	m_vb.resize(_mlp.size());

	int index = 0;
	for(const mlp& item: _mlp){
		initI(item.W, item.B, index++);
	}
	init_iteration();

	m_init_matB = true;

	return true;
}

bool MlpOptimAdam::pass(std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	if(!m_init_matB){
		throw new std::invalid_argument("MlpOptim::pass: not initialized");
	}

	next_iteration();

#pragma omp parallel for
	for(int i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];

		passI(mlpi.gW, mlpi.gB, mlpi.W, mlpi.B, i);
	}
	return true;
}

/////////////////

#include "convnn_gpu.h"

template<typename T >
void _norm(GpuMat& A, T &s)
{
	gpumat::GpuMat res;
	ct::Mat_<T> mA;

	gpumat::legacy::reduce(A, res);

	gpumat::convert_to_mat(res, mA);
	s = sqrt(mA.at(0));
	std::cout << s << std::endl;
}

template<typename T >
void _maxnorm(GpuMat& A, double c, double s)
{
	if(s > c){
		gpumat::mulval(A, c / s);
	}
}

void gpumat::maxnorm(GpuMat &A, double c)
{
	GpuMat S;
	gpumat::elemwiseSqr(A, S);

	switch (A.type) {
		case GPU_DOUBLE:
		{
			double s;
			_norm<double>(S, s);
			_maxnorm<double>(A, c, s);
			break;
		}
		case GPU_FLOAT:
		default:
		{
			float s;
			_norm<float>(S, s);
			_maxnorm<float>(A, c, s);
			break;
		}
	}
}

//////////////////////////////////////////////////////

bool MlpOptimSG::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	return true;
}

bool MlpOptimSG::pass(std::vector<mlp> &_mlp)
{
	m_iteration++;
	for(size_t i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];
		gpumat::sub(mlpi.W, mlpi.gW, 1., m_alpha);
		gpumat::sub(mlpi.B, mlpi.gB, 1., m_alpha);

	}
	return true;
}

//////////////////////////////////////////////////

MlpOptimMoment::MlpOptimMoment() : MomentumOptimizer()
{
}

bool MlpOptimMoment::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration = 0;

	m_mW.resize(_mlp.size());
	m_mb.resize(_mlp.size());

	for(size_t i = 0; i < _mlp.size(); i++){
		const gpumat::mlp& _mlpi = _mlp[i];

		initI(_mlpi.W, _mlpi.B, i);
	}

	return true;

}

bool MlpOptimMoment::pass(std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration++;

#pragma omp parallel for
	for(int i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];

		passI(mlpi.gW, mlpi.gB, mlpi.W, mlpi.B, i);
	}
	return true;

}

//////////////////////////

MlpOptimAdaGrad::MlpOptimAdaGrad() : AdaGradOptimizer()
{

}

bool MlpOptimAdaGrad::init(const std::vector<mlp> &_mlp)
{
    if(_mlp.empty())
        return false;

    m_iteration = 0;

    initSize(_mlp.size());

    for(size_t i = 0; i < _mlp.size(); i++){
        const gpumat::mlp& _mlpi = _mlp[i];

        initI(_mlpi.W, _mlpi.B, i);
    }

    return true;

}

bool MlpOptimAdaGrad::pass(std::vector<mlp> &_mlp)
{
    if(_mlp.empty())
        return false;

    m_iteration++;

#pragma omp parallel for
	for(int i = 0; i < _mlp.size(); ++i){
        mlp& mlpi = _mlp[i];

        passI(mlpi.gW, mlpi.gB, mlpi.W, mlpi.B, i);
    }
    return true;

}
