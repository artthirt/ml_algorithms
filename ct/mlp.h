#ifndef MLP_H
#define MLP_H

#include "custom_types.h"
#include "common_types.h"
#include "nn.h"
#include <map>

namespace ct{

template< typename T >
class mlp;

template< typename T >
class MlpOptimAdam: public AdamOptimizer<T>{
public:
	MlpOptimAdam(): AdamOptimizer<T>(){

	}

#define AO this->

	bool init(const std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		AO m_iteration = 0;

		AO m_mW.resize(Mlp.size());
		AO m_mb.resize(Mlp.size());

		AO m_vW.resize(Mlp.size());
		AO m_vb.resize(Mlp.size());

		int index = 0;
		for(const ct::mlp<T>& item: Mlp){
			initI(item.W, item.B, index++);
		}
		init_iteration();
		AO m_init = true;
		return true;
	}

	bool pass(std::vector< ct::mlp<T> >& Mlp){

		using namespace ct;

		pass_iteration();
		int index = 0;
		for(ct::mlp<T>& item: Mlp){
			passI(item.gW, item.gB, item.W, item.B, index++);
		}

		return true;
	}
};

template< typename T >
class MlpOptimSG: public StohasticGradientOptimizer<T>{
public:
	MlpOptimSG(): StohasticGradientOptimizer<T>(){

	}
	bool pass(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		for(size_t i = 0; i < Mlp.size(); ++i){
			ct::mlp<T>& _mlp = Mlp[i];

			_mlp.W -= Optimizer<T>::m_alpha * _mlp.gW;
			_mlp.B -= Optimizer<T>::m_alpha * _mlp.gB;
		}

		return true;
	}
};

template< typename T >
class MlpOptimMoment: public MomentOptimizer<T>{
public:
	MlpOptimMoment(): MomentOptimizer<T>(){

	}
	bool init(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		Optimizer<T>::m_iteration = 0;

		MomentOptimizer<T>::m_mW.resize(Mlp.size());
		MomentOptimizer<T>::m_mb.resize(Mlp.size());

		int index = 0;
		for(const ct::mlp<T>& item: Mlp){
			initI(item.W, item.B, index++);
		}
		return true;
	}

	bool pass(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		Optimizer<T>::m_iteration++;
		int index = 0;
		for(ct::mlp<T>& item: Mlp){
			passI(item.gW, item.gB, item.W, item.B, index++);
		}

		return true;
	}
};

template< typename T >
class mlp{
public:
	Mat_<T> *pA0;
	Mat_<T> W;
	Mat_<T> B;
	Mat_<T> Z;
	Mat_<T> A1;
	Mat_<T> DA1;
	Mat_<T> D1;
	Mat_<T> DltA0;
	Mat_<T> Dropout;
	Mat_<T> XDropout;
	Mat_<T> gW;
	Mat_<T> gB;

	mlp(){
		m_func = RELU;
		m_init = false;
		m_is_dropout = false;
		m_prob = (T)0.95;
		pA0 = nullptr;
		m_lambda = 0;

		m_params[LEAKYRELU] = T(0.1);
	}

	Mat_<T>& Y(){
		return A1;
	}

	void setLambda(T val){
		m_lambda = val;
	}

	void setParams(etypefunction func, T param){
		m_params[LEAKYRELU] = param;
	}

	void setDropout(bool val){
		m_is_dropout = val;
	}
	void setDropout(T val){
		m_prob = val;
	}

	bool isInit() const{
		return m_init;
	}

	void init(int input, int output, etypefunction func){
		double n = 1./sqrt(input);
		m_func = func;

		W.setSize(input, output);
		W.randn(0., n);
		B.setSize(1, output);
		B.randn(0, n);

		m_init = true;
	}

	etypefunction funcType() const{
		return m_func;
	}

	void forward(const ct::Mat_<T> *mat, bool save_A0 = true){
		if(!m_init || !mat)
			throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
		pA0 = (Mat_<T>*)mat;

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			ct::dropout(pA0->rows, pA0->cols, m_prob, Dropout);
			elemwiseMult(*pA0, Dropout, XDropout);
			if(m_func == SOFTMAX){
				ct::m2mpaf(XDropout, W, B, LINEAR, A1, 0.f);
				ct::v_softmax(A1, A1, 1);
			}else{
				ct::m2mpaf(XDropout, W, B, m_func, A1, m_params[LEAKYRELU]);
			}
		}else{
			if(m_func == SOFTMAX){
				ct::m2mpaf(*pA0, W, B, LINEAR, A1, 0.f);
				ct::v_softmax(A1, A1, 1);
			}else{
				ct::m2mpaf(*pA0, W, B, m_func, A1, m_params[LEAKYRELU]);
			}
//			ct::matmul(*pA0, W, Z);
		}

//		Z.biasPlus(B);
//		apply_func(Z, A1, m_func);

		if(!save_A0)
			pA0 = nullptr;
	}
	void backward(const ct::Mat_<T> &Delta, bool last_layer = false){
		if(!pA0 || !m_init)
			throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

		if(m_func != SOFTMAX){
			mul2deriv(Delta, A1, m_func, DA1, m_params[LEAKYRELU]);
		}else{
			Delta.copyTo(DA1);
		}
//		apply_back_func(Delta, DA1, m_func);

		T m = Delta.rows;

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			matmulT1(XDropout, DA1, gW);
		}else{
			matmulT1(*pA0, DA1, gW);
		}
		gW *= (T) 1. / m;


		if(m_lambda > 0){
			gW += W * (m_lambda / m);
		}

		v_sumRows(DA1, gB, 1.f / m);
//		gB.swap_dims();

		if(!last_layer){
			matmulT2(DA1, W, DltA0);
		}
	}

	void write(std::fstream& fs){
		write_fs(fs, W);
		write_fs(fs, B);
	}

	void read(std::fstream& fs){
		read_fs(fs, W);
		read_fs(fs, B);
	}

	void write2(std::fstream &fs)
	{
		write_fs2(fs, W);
		write_fs2(fs, B);
	}

	void read2(std::fstream &fs)
	{
		read_fs2(fs, W);
		read_fs2(fs, B);
	}

private:
	bool m_init;
	bool m_is_dropout;
	T m_prob;
	T m_lambda;
	etypefunction m_func;
	std::map< etypefunction, T > m_params;
};

typedef mlp<float> mlpf;

}

#endif // MLP_H
