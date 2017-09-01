#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"
#include <vector>
#include <map>
#include "nn.h"

#include <exception>

namespace conv2{

enum TYPE_CONV{
	SAME,
	VALID
};

/**
 * @brief im2col
 * @param X -> [channels, szA0.height * szA0.width] input image with size szA0 and channels
 * @param szA0		- input window of image
 * @param channels  - channels
 * @param szW		- applyed window
 * @param stride	- stride
 * @param Res -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 */
void im2cols(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2cols(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief im2col_same
 * @param X			- input image with size szA0 and channels
 * @param szA0		- input window of image
 * @param channels	- channels
 * @param szW		- applyed window
 * @param stride	- stride
 * @param Res -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut		- szOut == szA
 */
void im2cols_same(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2cols_same(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief im2colT
 * image to columns
 * @param X -> [szA0.height * szA0.width, channels]
 * @param szA0
 * @param channels	- channels for image
 * @param szW		- which the window will be apply for the image
 * @param stride	- stride
 * @param Res -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 */
void im2colsT(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut);
void im2colsT(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut);

/**
 * @brief conv2
 * convolution for A
 * @param A			- input matrix with window szA and chanels (channels, szA.area()) for normal and (szA.area(), channels) for trasnspose view
 * @param szA		- window for A matrix
 * @param channels	- channels of matrix A
 * @param stride	- stride
 * @param B			- applied matrix
 * @param szB		- window of matrix B
 * @param C			- output of convolution with window szOut
 * @param szOut		- window of output matrix
 * @param type		- may be SAME(szOut = szA) of VALID
 * @param transpose - select of view matrix A
 */
void conv2(const ct::Matf& A, const ct::Size &szA, int channels, int stride, const ct::Matf &B,
		   const ct::Size &szB, ct::Matf &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);
void conv2(const ct::Matd& A, const ct::Size &szA, int channels, int stride, const ct::Matd &B,
		   const ct::Size &szB, ct::Matd &C, ct::Size &szOut, TYPE_CONV type = VALID, bool transpose = false);

/**
 * @brief cols2im
 * @param Delta -> [szOut.width * szOut.height, szW.width * szW.height * channels]
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X -> [channels, szA0.height * szA0.width]
 */
void cols2im(const ct::Matf& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void cols2im(const ct::Matd& Delta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief cols2im_same
 * columns to image . size saved
 * @param Delta		- columns (szA0.area(), szW.area() * channels)
 * @param szA0		- size of window for input and output
 * @param channels	- channels of matrix
 * @param szW		- size of applied window
 * @param stride	- stride
 * @param X			- output matrix (channels, szA0.area())
 */
void cols2im_same(const ct::Matf& Delta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void cols2im_same(const ct::Matd& Delta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief cols2imT
 * columns to image (transpose view). size output szA0
 * @param Delta		- columns (szA0.area(), szW.area() * channels)
 * @param szOut		- size of window for Delta
 * @param szA0		- size of window for output
 * @param channels	- channels of matrices
 * @param szW		- size of applied window
 * @param stride	- stride
 * @param X			- output matrix (szA0.area(), channels)
 */
void cols2imT(const ct::Matf& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void cols2imT(const ct::Matd& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief cols2imT_same
 * columns to image (transpose view). size saved
 * @param Delta		- columns
 * @param szA0		- size of window for input and output
 * @param channels	- channels of matrix
 * @param szW		- size of applied window
 * @param stride	- stride
 * @param X			- output matrix (szA0.area(), channels)
 */
void cols2imT_same(const ct::Matf& Delta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X);
void cols2imT_same(const ct::Matd& Delta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X);

/**
 * @brief conv2_transpose
 * @param C
 * @param szA
 * @param channels
 * @param stride
 * @param B
 * @param szB
 * @param szOut
 * @param A
 * @param type
 * @param transpose
 */
void conv2_transpose(const ct::Matf& C, const ct::Size &szA, int channels, int stride, const ct::Matf &B,
		   const ct::Size &szB, const ct::Size &szOut, ct::Matf &A, TYPE_CONV type = VALID, bool transpose = false);
void conv2_transpose(const ct::Matd& C, const ct::Size &szA, int channels, int stride, const ct::Matd &B,
		   const ct::Size &szB, const ct::Size &szOut, ct::Matd &A, TYPE_CONV type = VALID, bool transpose = false);



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
	ct::Mat_<T> W;							/// weights
	ct::Mat_<T> B;							/// biases
	int stride;
	ct::Size szW;							/// size of weights
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > Xc;			///
	std::vector< ct::Mat_<T> > A1;			/// out after appl nonlinear function
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > A3;			/// out after bn
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)

	ct::Mat_<T> gW;							/// gradient for weights
	ct::Mat_<T> gB;							/// gradient for biases

	std::vector< ct::Mat_<T> > dSub;
	std::vector< ct::Mat_<T> > Dc;

	convnn(){
		m_use_pool = false;
		pX = nullptr;
		stride = 1;
		m_use_transpose = true;
		m_Lambda = 0;
		m_params[ct::LEAKYRELU] = 0.1;
		m_use_bn = false;
	}

	void setParams(ct::etypefunction type, T param){
		m_params[type] = param;
	}

	std::vector< ct::Mat_<T> >& XOut(){
		if(m_use_bn)
			return A3;
		if(m_use_pool)
			return A2;
		return A1;
	}

	const std::vector< ct::Mat_<T> >& XOut() const{
		if(m_use_bn)
			return A3;
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
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< ct::Mat_<T> >& XOut3(){
		return A3;
	}

	bool use_pool() const{
		return m_use_pool;
	}
	bool use_bn(){
		return m_use_bn;
	}

	ct::BN<T>& bn(){
		return m_bn;
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

	/**
	 * @brief setLambda
	 * @param val
	 */
	void setLambda(T val){
		m_Lambda = val;
	}

	/**
	 * @brief init
	 * @param _szA0
	 * @param _channels
	 * @param stride
	 * @param _K
	 * @param _szW
	 * @param func
	 * @param use_pool
	 * @param use_transpose
	 * @param use_bn
	 */
	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW, ct::etypefunction func,
			  bool use_pool, bool use_bn, bool use_transpose){
		szW = _szW;
		m_use_pool = use_pool;
		m_use_bn = use_bn;
		m_use_transpose = use_transpose;
		m_func = func;
		convnn_abstract<T>::kernels = _K;
		convnn_abstract<T>::channels = _channels;
		convnn_abstract<T>::szA0 = _szA0;
		this->stride = stride;

		m_bn.channels = _K;

		int rows = szW.area() * convnn_abstract<T>::channels;
		int cols = convnn_abstract<T>::kernels;

		ct::get_cnv_sizes(convnn_abstract<T>::szA0, szW, stride, convnn_abstract<T>::szA1, convnn_abstract<T>::szA2);

		T n = (T)1/sqrt(szW.area() * convnn_abstract<T>::channels);

		W.setSize(rows, cols);
		W.randn(0, n);
		B.setSize(1, convnn_abstract<T>::kernels);
		B.randn(0, n);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::kernels);
	}

	void forward(const std::vector< ct::Mat_<T> >* _pX){
		if(!_pX)
			return;
		pX = (std::vector< ct::Mat_<T> >*)_pX;

		Xc.resize(pX->size());
		A1.resize(pX->size());

		if(m_use_transpose){
			for(int i = 0; i < (int)Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2colsT(Xi, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Xc[i], szOut);
			}
		}else{
			for(int i = 0; i < (int)Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2cols(Xi, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Xc[i], szOut);
			}
		}

		for(int i = 0; i < (int)Xc.size(); ++i){
			ct::Mat_<T>& Xi = Xc[i];
			ct::m2mpaf(Xi, W, B, m_func, A1[i], m_params[ct::LEAKYRELU]);
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

			if(m_use_bn){
				m_bn.X = &A2;
				m_bn.Y = &A3;
				m_bn.normalize();
			}
		}else{
			convnn_abstract<T>::szK = A1[0].size();
			if(m_use_bn){
				m_bn.X = &A1;
				m_bn.Y = &A3;
				m_bn.normalize();
			}
		}
	}

	void forward(const convnn<T> & conv){
		forward(&conv.XOut());
	}

	inline void backcnv(const std::vector< ct::Mat_<T> >& D, std::vector< ct::Mat_<T> >& DS){
		for(int i = 0; i < D.size(); ++i){
			ct::mul2deriv(D[i], A1[i], m_func, DS[i], m_params[ct::LEAKYRELU]);
		}
	}

	void backward(const std::vector< ct::Mat_<T> >& D, bool last_level = false){
		if(D.empty() || D.size() != Xc.size()){
			throw new std::invalid_argument("vector D not complies saved parameters");
		}

		dSub.resize(D.size());

		//printf("1\n");
		if(m_use_pool){

			std::vector< ct::Mat_<T> >& _D = (std::vector< ct::Mat_<T> >&)D;

			if(m_use_bn){
				m_bn.D = (std::vector< ct::Mat_<T> >*)&D;
				m_bn.denormalize();
				_D = m_bn.Dout;
			}

			for(int i = 0; i < (int)_D.size(); ++i){
				ct::Mat_<T> Di = _D[i];
				//Di.set_dims(szA2.area(), K);
				upsample(Di, convnn_abstract<T>::kernels, Mask[i],convnn_abstract<T>:: szA2, convnn_abstract<T>::szA1, dSub[i]);
			}
			backcnv(dSub, dSub);
		}else{
			std::vector< ct::Mat_<T> >& _D = (std::vector< ct::Mat_<T> >&)D;

			if(m_use_bn){
				m_bn.D = (std::vector< ct::Mat_<T> >*)&D;
				m_bn.denormalize();
				_D = m_bn.Dout;
			}

			backcnv(_D, dSub);
		}

		//printf("2\n");
		gW.setSize(W.size());
		gW.fill(0);
		gB.setSize(B.size());
		gB.fill(0);

		for(int i = 0; i < (int)D.size(); ++i){
			ct::Mat_<T>& Xci = Xc[i];
			ct::Mat_<T>& dSubi = dSub[i];
			//ct::Mat_<T>& Wi = vgW;
			//ct::Mat_<T>& vgBi = vgB;
			ct::add2matmulT1(Xci, dSubi, gW);
			ct::add2sumRows(dSubi, gB, 1.f/dSubi.rows);

//			ct::add(gW[0], vgW);
//			ct::add(gB[0], vgB);
			//Wi *= (1.f/dSubi.total());
			//vgBi.swap_dims();
		}
		//printf("3\n");
		gW *= (T)1./(D.size() * convnn_abstract<T>::channels);
		gB *= (T)1./(D.size() * convnn_abstract<T>::channels);

		//printf("4\n");
		if(m_Lambda > 0){
			ct::add<float>(gW,  W, 1., (m_Lambda / convnn_abstract<T>::kernels));
		}

		//printf("5\n");
		if(!last_level){
			Dlt.resize(D.size());

			//ct::Mat_<T> Wf;
			//flipW(W, szW, channels, Wf);

			Dc.resize(D.size());
			for(int i = 0; i < (int)D.size(); ++i){
				ct::matmulT2(dSub[i], W, Dc[i]);
				cols2imT(Dc[i], convnn_abstract<T>::szA1, convnn_abstract<T>::szA0, convnn_abstract<T>::channels, szW, stride, Dlt[i]);
				//ct::Size sz = (*pX)[i].size();
				//Dlt[i].set_dims(sz);
			}
		}
	}

	void write(std::fstream& fs){
		if(W.empty() || B.empty())
			return;
		ct::write_fs(fs, W);
		ct::write_fs(fs, B);
	}
	void read(std::fstream& fs){
		if(W.empty() || B.empty())
			return;
		ct::read_fs(fs, W);
		ct::read_fs(fs, B);
	}

	void write2(std::fstream& fs){
		fs.write((char*)&szW.width, sizeof(szW.width));
		fs.write((char*)&szW.height, sizeof(szW.height));
		fs.write((char*)&(convnn_abstract<T>::channels), sizeof(convnn_abstract<T>::channels));
		fs.write((char*)&(convnn_abstract<T>::kernels), sizeof(convnn_abstract<T>::kernels));

		ct::write_fs2(fs, W);
		ct::write_fs2(fs, B);
	}

	void read2(std::fstream& fs){
		fs.read((char*)&szW.width, sizeof(szW.width));
		fs.read((char*)&szW.height, sizeof(szW.height));
		fs.read((char*)&(convnn_abstract<T>::channels), sizeof(convnn_abstract<T>::channels));
		fs.read((char*)&(convnn_abstract<T>::kernels), sizeof(convnn_abstract<T>::kernels));

		ct::read_fs2(fs, W);
		ct::read_fs2(fs, B);
	}

private:
	bool m_use_pool;
	bool m_use_bn;
	ct::etypefunction m_func;
	bool m_use_transpose;
	std::map< ct::etypefunction, T > m_params;
	T m_Lambda;

	ct::BN<T> m_bn;
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

///////////////////////////////

template< typename T >
class CnvAdamOptimizer: public ct::AdamOptimizer<T>{
public:
	CnvAdamOptimizer() : ct::AdamOptimizer<T>(){

	}

	void init(const std::vector< convnn<T> >& cnv){
		int index = 0;
		ct::AdamOptimizer<T>::init_iteration();
		ct::AdamOptimizer<T>::m_mW.resize(cnv.size());
		ct::AdamOptimizer<T>::m_mb.resize(cnv.size());
		ct::AdamOptimizer<T>::m_vW.resize(cnv.size());
		ct::AdamOptimizer<T>::m_vb.resize(cnv.size());
		for(const convnn<T>& item: cnv){
			initI(item.W, item.B, index++);
		}
	}
	void pass(std::vector< convnn<T> >& cnv){
		ct::AdamOptimizer<T>::pass_iteration();
		int index = 0;
		for(convnn<T>& item: cnv){
			passI(item.gW, item.gB, item.W, item.B, index++);
		}
	}
};

template< typename T >
class CnvMomentumOptimizer: public ct::MomentumOptimizer<T>{
public:
	CnvMomentumOptimizer() : ct::MomentumOptimizer<T>(){

	}
	std::vector< ct::Mat_<T> > m_mG;
	std::vector< ct::Mat_<T> > m_mB;


	void init(const std::vector< convnn<T> >& cnv){
		int index = 0;
		ct::MomentumOptimizer<T>::m_mW.resize(cnv.size());
		ct::MomentumOptimizer<T>::m_mb.resize(cnv.size());

		m_mG.resize(cnv.size());
		m_mB.resize(cnv.size());
		for(const convnn<T>& item: cnv){
			ct::MomentumOptimizer<T>::initI(item.W, item.B, index++);
		}
	}
	void pass(std::vector< convnn<T> >& cnv){
		int index = 0;
		for(convnn<T>& item: cnv){
			if(item.use_bn()){
				if(m_mG[index].empty()){
					m_mG[index].setSize(item.bn().gamma.size());
					m_mG[index].fill(0);
					m_mB[index].setSize(item.bn().betha.size());
					m_mB[index].fill(0);
				}
				ct::momentumGrad(item.bn().dgamma, m_mG[index], item.bn().gamma, Optimizer<T>::m_alpha, m_betha);
				ct::momentumGrad(item.bn().dbetha, m_mB[index], item.bn().betha, Optimizer<T>::m_alpha, m_betha);

				ct::save_mat(item.bn().gamma, "gamma_" + std::to_string(index) + ".txt");
				ct::save_mat(item.bn().betha, "betha_" + std::to_string(index) + ".txt");

			}
			ct::MomentumOptimizer<T>::passI(item.gW, item.gB, item.W, item.B, index++);

		}
	}
};

///////////////////////////////

typedef convnn<float> convnnf;

}

#endif // NN2_H
