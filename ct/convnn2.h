#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"
#include <vector>
#include "nn.h"

#include <exception>

namespace conv2{

enum TYPE_CONV{
	SAME,
	VALID
};

/**
 * @brief im2col
 * @param X -> [channels, szA0.height * szA0.width]
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 */
void im2col(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2col(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief im2col_same
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2col_same(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2col_same(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief im2colT
 * @param X -> [szA0.height * szA0.width, channels]
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 */
void im2colT(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2colT(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief conv2
 * @param A
 * @param szA
 * @param channels
 * @param stride
 * @param B
 * @param szB
 * @param C
 */
void conv2(const ct::Matf& A, const ct::Size &szA, int channels, int stride, const ct::Matf &B,
		   ct::Size &szB, ct::Matf &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);
void conv2(const ct::Matd& A, const ct::Size &szA, int channels, int stride, const ct::Matd &B,
		   ct::Size &szB, ct::Matd &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);

/**
 * @brief back_deriv
 * @param Delta -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X -> [channels, szA0.height * szA0.width]
 */
void back_deriv(const ct::Matf& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void back_deriv(const ct::Matd& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief back_derivT
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void back_derivT(const ct::Matf& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void back_derivT(const ct::Matd& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void subsample(const ct::Matf& X, const ct::Size& szA, ct::Matf& Y, ct::Matf& Mask, ct::Size& szO);
void subsample(const ct::Matd& X, const ct::Size& szA, ct::Matd& Y, ct::Matd& Mask, ct::Size& szO);

/**
 * @brief upsample
 * @param Y
 * @param K
 * @param Mask
 * @param szO
 * @param szA
 * @param X
 */
void upsample(const ct::Matf& Y, int K, const ct::Matf& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Matf& X);
void upsample(const ct::Matd& Y, int K, const ct::Matd& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Matd& X);

/**
 * @brief vec2mat
 * @param vec
 * @param mat
 */
void vec2mat(const std::vector< ct::Matf >& vec, ct::Matf& mat);
void vec2mat(const std::vector< ct::Matd >& vec, ct::Matd& mat);

/**
 * @brief mat2vec
 * @param mat
 * @param szOut
 * @param vec
 */
void mat2vec(const ct::Matf& mat, const ct::Size& szOut, std::vector< ct::Matf >& vec);
void mat2vec(const ct::Matd& mat, const ct::Size& szOut, std::vector< ct::Matd >& vec);

/**
 * @brief flipW
 * @param W
 * @param sz
 * @param channels
 * @param Wr
 */
void flipW(const ct::Matf& W, const ct::Size& sz,int channels, ct::Matf& Wr);
void flipW(const ct::Matd& W, const ct::Size& sz,int channels, ct::Matd& Wr);
//-------------------------------------

template< typename T >
class convnn_abstract{
public:
	int kernels;									/// kernels
	int channels;							/// input channels

	ct::Size szA0;							/// input size
	ct::Size szA1;							/// size after convolution
	ct::Size szA2;							/// size after pooling
	ct::Size szK;							/// size of output data (set in forward)

	virtual std::vector< ct::Mat_<T> >& XOut() = 0;
	virtual int outputFeatures() const = 0;
	virtual ct::Size szOut() const = 0;
};

template< typename T >
class convnn: public convnn_abstract<T>{
public:
	std::vector< ct::Mat_<T> > W;			/// weights
	std::vector< ct::Mat_<T> > B;			/// biases
	int stride;
	ct::Size szW;							/// size of weights
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > Xc;			///
	std::vector< ct::Mat_<T> > A1;			/// out after appl nonlinear function
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > vgW;			/// for delta weights
	std::vector< ct::Mat_<T> > vgB;			/// for delta bias
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)
	ct::Optimizer< T > *m_optim;
	ct::AdamOptimizer<T> m_adam;

	std::vector< ct::Mat_<T> > gW;			/// gradient for weights
	std::vector< ct::Mat_<T> > gB;			/// gradient for biases

	std::vector< ct::Mat_<T> > dSub;
	std::vector< ct::Mat_<T> > Dc;

	convnn(){
		m_use_pool = false;
		pX = nullptr;
		stride = 1;
		m_use_transpose = true;
		m_Lambda = 0;
		m_optim = &m_adam;
	}

	void setOptimizer(ct::Optimizer<T>* optim){
		if(!optim)
			return;
		m_optim = optim;
	}

	std::vector< ct::Mat_<T> >& XOut(){
		if(m_use_pool)
			return A2;
		return A1;
	}

	const std::vector< ct::Mat_<T> >& XOut() const{
		if(m_use_pool)
			return A2;
		return A1;
	}
	/**
	 * @brief XOut1
	 * out after convolution
	 * @return
	 */
	std::vector< ct::Mat_<T> >& XOut1(){
		return A1;
	}
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< ct::Mat_<T> >& XOut2(){
		return A2;
	}

	bool use_pool() const{
		return m_use_pool;
	}

	int outputFeatures() const{
		if(m_use_pool){
			int val = convnn_abstract<T>::szA2.area() * convnn_abstract<T>::kernels;
			return val;
		}else{
			int val= convnn_abstract<T>::szA1.area() * convnn_abstract<T>::kernels;
			return val;
		}
	}

	ct::Size szOut() const{
		if(m_use_pool)
			return convnn_abstract<T>::szA2;
		else
			return convnn_abstract<T>::szA1;
	}

	void setAlpha(T alpha){
		m_optim->setAlpha(alpha);
	}

	void setLambda(T val){
		m_Lambda = val;
	}

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW,
			  bool use_pool = true, bool use_transpose = true){
		szW = _szW;
		m_use_pool = use_pool;
		m_use_transpose = use_transpose;
		convnn_abstract<T>::kernels = _K;
		convnn_abstract<T>::channels = _channels;
		convnn_abstract<T>::szA0 = _szA0;
		this->stride = stride;

		int rows = szW.area() * convnn_abstract<T>::channels;
		int cols = convnn_abstract<T>::kernels;

		ct::get_cnv_sizes(convnn_abstract<T>::szA0, szW, stride, convnn_abstract<T>::szA1, convnn_abstract<T>::szA2);

		T n = (T)1./szW.area();

		W.resize(1);
		B.resize(1);
		gW.resize(1);
		gB.resize(1);

		W[0].setSize(rows, cols);
		W[0].randn(0, n);
		B[0].setSize(convnn_abstract<T>::kernels, 1);
		B[0].randn(0, n);

		m_optim->init(W, B);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::kernels);
	}

	void forward(const std::vector< ct::Mat_<T> >* _pX, ct::etypefunction func){
		if(!_pX)
			return;
		pX = (std::vector< ct::Mat_<T> >*)_pX;
		m_func = func;

		Xc.resize(pX->size());
		A1.resize(pX->size());

		if(m_use_transpose){
			for(int i = 0; i < (int)Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2colT(Xi, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Xc[i], szOut);
			}
		}else{
			for(int i = 0; i < (int)Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2col(Xi, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Xc[i], szOut);
			}
		}


		for(int i = 0; i < (int)Xc.size(); ++i){
			ct::Mat_<T>& Xi = Xc[i];
			ct::Mat_<T>& A1i = A1[i];
			ct::matmul(Xi, W[0], A1i);
			A1i.biasPlus(B[0]);
		}

		for(int i = 0; i < (int)A1.size(); ++i){
			ct::Mat_<T>& Ao = A1[i];
			switch (m_func) {
				case ct::RELU:
					ct::v_relu(Ao);
					break;
				case ct::SIGMOID:
					ct::v_sigmoid(Ao);
					break;
				case ct::TANH:
					ct::v_tanh(Ao);
					break;
				default:
					break;
			}
		}
		if(m_use_pool){
			Mask.resize(Xc.size());
			A2.resize(A1.size());
			for(int i = 0; i < (int)A1.size(); ++i){
				ct::Mat_<T> &A1i = A1[i];
				ct::Mat_<T> &A2i = A2[i];
				ct::Size szOut;
				conv2::subsample(A1i, convnn_abstract<T>::szA1, A2i, Mask[i], szOut);
			}
			convnn_abstract<T>::szK = A2[0].size();
		}else{
			convnn_abstract<T>::szK = A1[0].size();
		}
	}

	void forward(const convnn<T> & conv, ct::etypefunction func){
		forward(&conv.XOut(), func);
	}

	inline void backcnv(const std::vector< ct::Mat_<T> >& D, std::vector< ct::Mat_<T> >& DS){
		if(D.data() != DS.data()){
			for(int i = 0; i < (int)D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::elemwiseMult(D[i], derivRelu(A1[i]), DS[i]);
						break;
					case ct::SIGMOID:
						ct::elemwiseMult(D[i], derivSigmoid(A1[i]), DS[i]);
						break;
					case ct::TANH:
						ct::elemwiseMult(D[i], derivTanh(A1[i]), DS[i]);
						break;
					default:
						break;
				}
			}
		}else{
			for(int i = 0; i < (int)D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::elemwiseMult(DS[i], ct::derivRelu(A1[i]));
						break;
					case ct::SIGMOID:
						ct::elemwiseMult(DS[i], ct::derivSigmoid(A1[i]));
						break;
					case ct::TANH:
						ct::elemwiseMult(DS[i], ct::derivTanh(A1[i]));
						break;
					default:
						break;
				}
			}
		}
	}

	void backward(const std::vector< ct::Mat_<T> >& D, bool last_level = false){
		if(D.empty() || D.size() != Xc.size()){
			throw new std::invalid_argument("vector D not complies saved parameters");
		}

		dSub.resize(D.size());

		//printf("1\n");
		if(m_use_pool){
			for(int i = 0; i < (int)D.size(); ++i){
				ct::Mat_<T> Di = D[i];
				//Di.set_dims(szA2.area(), K);
				upsample(Di, convnn_abstract<T>::kernels, Mask[i],convnn_abstract<T>:: szA2, convnn_abstract<T>::szA1, dSub[i]);
			}
			backcnv(dSub, dSub);
		}else{
			backcnv(D, dSub);
		}

		//printf("2\n");
		vgW.resize(D.size());
		vgB.resize(D.size());
		for(int i = 0; i < (int)D.size(); ++i){
			ct::Mat_<T>& Xci = Xc[i];
			ct::Mat_<T>& dSubi = dSub[i];
			ct::Mat_<T>& Wi = vgW[i];
			ct::Mat_<T>& vgBi = vgB[i];
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
		gW[0] *= (T)1./(D.size());
		gB[0] *= (T)1./(D.size());

		//printf("4\n");
		if(m_Lambda > 0){
			ct::add<float>(gW[0],  W[0], 1., (m_Lambda / convnn_abstract<T>::kernels));
		}

		//printf("5\n");
		if(!last_level){
			Dlt.resize(D.size());

			//ct::Mat_<T> Wf;
			//flipW(W, szW, channels, Wf);

			Dc.resize(D.size());
			for(int i = 0; i < (int)D.size(); ++i){
				ct::matmulT2(dSub[i], W[0], Dc[i]);
				back_derivT(Dc[i], convnn_abstract<T>::szA1, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Dlt[i]);
				//ct::Size sz = (*pX)[i].size();
				//Dlt[i].set_dims(sz);
			}
		}

		//printf("6\n");
		m_optim->pass(gW, gB, W, B);

		//printf("7\n");
	}

	void write(std::fstream& fs){
		if(!W.size() || !B.size())
			return;
		ct::write_fs(fs, W[0]);
		ct::write_fs(fs, B[0]);
	}
	void read(std::fstream& fs){
		if(!W.size() || !B.size())
			return;
		ct::read_fs(fs, W[0]);
		ct::read_fs(fs, B[0]);
	}

	void write2(std::fstream& fs){
		fs.write((char*)&szW.width, sizeof(szW.width));
		fs.write((char*)&szW.height, sizeof(szW.height));
		fs.write((char*)&(convnn_abstract<T>::channels), sizeof(convnn_abstract<T>::channels));
		fs.write((char*)&(convnn_abstract<T>::kernels), sizeof(convnn_abstract<T>::kernels));

		ct::write_fs2(fs, W[0]);
		ct::write_fs2(fs, B[0]);
	}

	void read2(std::fstream& fs){
		fs.read((char*)&szW.width, sizeof(szW.width));
		fs.read((char*)&szW.height, sizeof(szW.height));
		fs.read((char*)&(convnn_abstract<T>::channels), sizeof(convnn_abstract<T>::channels));
		fs.read((char*)&(convnn_abstract<T>::kernels), sizeof(convnn_abstract<T>::kernels));

		ct::read_fs2(fs, W[0]);
		ct::read_fs2(fs, B[0]);
	}

private:
	bool m_use_pool;
	ct::etypefunction m_func;
	bool m_use_transpose;
	T m_Lambda;
};

template< typename T >
class Pooling: public convnn_abstract<T>{
public:
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)
//	std::vector< ct::Mat_<T> > dSub;

	Pooling(){
		pX = nullptr;
		convnn_abstract<T>::channels = 0;
		convnn_abstract<T>::kernels = 0;
	}

	ct::Size szOut() const{
		return convnn_abstract<T>::szA2;
	}
	std::vector< ct::Mat_<T> >& XOut(){
		return A2;
	}
	std::vector< ct::Mat_<T> >* pXOut(){
		return &A2;
	}
	int outputFeatures() const{
			int val = convnn_abstract<T>::szA2.area() * convnn_abstract<T>::kernels;
			return val;
	}

	void init(const ct::Size& _szA0, int _channels, int _K){
		convnn_abstract<T>::kernels = _K;
		convnn_abstract<T>::channels = _channels;
		convnn_abstract<T>::szA0 = _szA0;

		convnn_abstract<T>::szA2 = ct::Size(convnn_abstract<T>::szA0.width/2, convnn_abstract<T>::szA0.height/2);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::kernels);
	}

	void init(const convnn<T>& conv){
		convnn_abstract<T>::kernels = conv.kernels;
		convnn_abstract<T>::channels = conv.channels;
		convnn_abstract<T>::szA0 = conv.szOut();

		convnn_abstract<T>::szA2 = ct::Size(convnn_abstract<T>::szA0.width/2, convnn_abstract<T>::szA0.height/2);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::kernels);
	}

	void forward(const std::vector< ct::Mat_<T> >* _pX){
		if(!_pX)
			return;
		pX = (std::vector< ct::Mat_<T> >*)_pX;

		std::vector< ct::Mat_<T> >& A1 = pX;			/// out after appl nonlinear function
		Mask.resize(A1.size());
		A2.resize(A1.size());
#pragma omp parallel for
		for(size_t i = 0; i < A1.size(); ++i){
			ct::Mat_<T> &A1i = A1[i];
			ct::Mat_<T> &A2i = A2[i];
			ct::Size szOut;
			conv2::subsample(A1i, convnn_abstract<T>::szA0, A2i, Mask[i], szOut);
		}
		convnn_abstract<T>::szK = A2[0].size();
	}

	void forward(convnn<T> & conv){
		pX = &conv.XOut();
		std::vector< ct::Mat_<T> >& A1 = conv.XOut();			/// out after appl nonlinear function
		Mask.resize(A1.size());
		A2.resize(A1.size());
#pragma omp parallel for
		for(size_t i = 0; i < A1.size(); ++i){
			ct::Mat_<T> &A1i = A1[i];
			ct::Mat_<T> &A2i = A2[i];
			ct::Size szOut;
			conv2::subsample(A1i, convnn_abstract<T>::szA0, A2i, Mask[i], szOut);
		}
		convnn_abstract<T>::szK = A2[0].size();
	}

	void backward(const std::vector< ct::Mat_<T> >& D){
		if(D.empty() || D.size() != pX->size()){
			throw new std::invalid_argument("vector D not complies saved parameters");
		}

		Dlt.resize(D.size());

		for(size_t i = 0; i < D.size(); ++i){
			ct::Mat_<T> Di = D[i];
			//Di.set_dims(szA2.area(), K);
			upsample(Di, convnn_abstract<T>::kernels, Mask[i], convnn_abstract<T>::szA2, convnn_abstract<T>::szA0, Dlt[i]);
		}
	}

};

template< typename T >
class Concat{
public:
	ct::Mat_<T> m_A1;
	ct::Mat_<T> m_A2;
	ct::Matf D1;
	ct::Matf D2;
	std::vector< ct::Matf > Dlt1;
	std::vector< ct::Matf > Dlt2;

	ct::Mat_<T> Y;

	convnn_abstract<T>* m_c1;
	convnn_abstract<T>* m_c2;

	Concat(){

	}

	void forward(convnn_abstract<T>* c1, convnn_abstract<T>* c2){
		if(!c1 || !c2)
			return;

		m_c1 = c1;
		m_c2 = c2;

		conv2::vec2mat(c1->XOut(), m_A1);
		conv2::vec2mat(c2->XOut(), m_A2);

		std::vector< ct::Matf* > concat;

		concat.push_back(&m_A1);
		concat.push_back(&m_A2);

		ct::hconcat(concat, Y);
	}
	void backward(const ct::Mat_<T>& Dlt){
		if(!m_c1 || !m_c2)
			return;

		std::vector< int > cols;
		std::vector< ct::Matf* > mats;
		cols.push_back(m_c1->outputFeatures());
		cols.push_back(m_c2->outputFeatures());
		mats.push_back(&D1);
		mats.push_back(&D2);
		ct::hsplit(Dlt, cols, mats);

		conv2::mat2vec(D1, m_c1->szK, Dlt1);
		conv2::mat2vec(D2, m_c2->szK, Dlt2);
	}
};

typedef convnn<float> convnnf;

}

#endif // NN2_H
