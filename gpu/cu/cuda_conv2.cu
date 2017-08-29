#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

#include "common_devices.h"
#include "cuda_types.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{

template< typename T >
__device__ void _im2cols(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Res.data;
		T *dXi = &dX[c * szA0area];

		for(int a = 0; a < szW.height; ++a){
			if(y0 + a < szA0.height){
				for(int b = 0; b < szW.width; ++b){
					int col2 = c * szWarea + (a * szW.width + b);
					if(x0 + b < szA0.width){
						dR[row2 * Res.cols + col2] = dXi[(y0 + a) * szA0.width + (x0 + b)];
					}
				}
			}
		}
	}
}

template< typename T >
__device__ void _im2colsT(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Res.data;
		T *dXi = (T*)X.data + c;

		for(int a = 0; a < szW.height; ++a){
			if(y0 + a < szA0.height){
				for(int b = 0; b < szW.width; ++b){
					int col2 = c * szWarea + (a * szW.width + b);
					if(x0 + b < szA0.width){
						dR[row2 * Res.cols + col2] = dXi[((y0 + a) * szA0.width + (x0 + b)) * channels];
					}
				}
			}
		}
	}
}

template< typename T >
__global__ void im2cols(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2cols<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2cols_vec(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2cols<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

////////

template< typename T >
__global__ void im2colsT(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2colsT<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2colsT_vec(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2colsT<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

//////// begin same //////

template< typename T >
__device__ void _im2colsSame(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Res.data;
		T *dXi = &dX[c * szA0area];

		for(int _a = 0; _a < szW.height; ++_a){
			int a = _a - szW.height/2;
			if(y0 + a >= 0 && y0 + a < szA0.height){
				for(int _b = 0; _b < szW.width; ++_b){
					int b = _b - szW.width/2;
					int col2 = c * szWarea + (_a * szW.width + _b);
					if(x0 + b >= 0 && x0 + b < szA0.width){
						dR[row2 * Res.cols + col2] = dXi[(y0 + a) * szA0.width + (x0 + b)];
					}
				}
			}
		}
	}
}

template< typename T >
__device__ void _im2colsTSame(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Res.data;
		T *dXi = (T*)X.data + c;

		for(int _a = 0; _a < szW.height; ++_a){
			int a = _a - szW.height/2;
			if(y0 + a >= 0 && y0 + a < szA0.height){
				for(int _b = 0; _b < szW.width; ++_b){
					int b = _b - szW.width/2;
					int col2 = c * szWarea + (_a * szW.width + _b);
					if(x0 + b >= 0 && x0 + b < szA0.width){
						dR[row2 * Res.cols + col2] = dXi[((y0 + a) * szA0.width + (x0 + b)) * channels];
					}
				}
			}
		}
	}
}

template< typename T >
__global__ void im2colsSame(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2colsSame<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2cols_vecSame(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2colsSame<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

////////

template< typename T >
__global__ void im2colsTSame(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2colsTSame<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2colsT_vecSame(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2colsTSame<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}


//////// end same ////////

template< typename T >
__device__ void _cols2im(const Mtx& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szA0Area = szA0.width * szA0.height;
	int all = szA0Area * channels;
	if(col < all){
//		int c = col / szOutArea;
//		int offset = col - c * szOutArea;

//		int y = offset / szOut.width;
//		int x = offset - y * szOut.width;

//		int x0 = x * stride;
//		int y0 = y * stride;
//		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
//		int szWarea = szW.width * szW.height;

//		T *dX = (T*)X.data;
//		T *dR = (T*)Delta.data;
//		T *dXi = &dX[c * szA0area];

//		for(int a = 0; a < szW.height; ++a){
//			for(int b = 0; b < szW.width; ++b){
//				int col2 = c * szWarea + (a * szW.width + b);
//				if(y0 + a < szA0.height && x0 + b < szA0.width){
//					dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row2 * Delta.cols + col2];
//				}
//			}
//		}
		int c = col / szA0Area;
		int offset = col - c * szA0Area;

		int y = offset / szA0.width;
		int x = offset - y * szA0.width;

		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Delta.data;
		T *dXi = &dX[c * szA0Area];

		T sum = 0;
		for(int a = 0; a < szW.height; ++a){
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				if(y0 >= 0 && y0 < szOut.height){
					for(int b = 0; b < szW.width; ++b){

						if((x - b) % stride == 0){

							int x0 = (x - b) / stride;

							if(x0 >= 0 && x0 < szOut.width){
								int row2 = y0 * szOut.width + x0;
								int col2 = c * szWarea + (a * szW.width + b);
								T val = dR[row2 * Delta.cols + col2];
								sum += val;
							}
						}
					}
				}
			}
		}
		dXi[y * szA0.width + x] = sum;

	}
}

//////

template< typename T >
__device__ void _cols2imT(const Mtx& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szA0Area = szA0.width * szA0.height;
	int all = szA0Area * channels;

	if(col < all){
		int c = col / szA0Area;
		int offset = col - c * szA0Area;

		int y = offset / szA0.width;
		int x = offset - y * szA0.width;

//		int x0 = x * stride;
//		int y0 = y * stride;
//		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Delta.data;
		T *dXi = (T*)X.data + c;

//		for(int a = 0; a < szW.height; ++a){
//			for(int b = 0; b < szW.width; ++b){
//				int col2 = c * szWarea + (a * szW.width + b);
//				if(y0 + a < szA0.height && x0 + b < szA0.width){
//					dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += dR[row2 * Delta.cols + col2];
//				}
//			}
//		}
		T sum = 0;
		for(int a = 0; a < szW.height; ++a){
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				if(y0 >= 0 && y0 < szOut.height){
					for(int b = 0; b < szW.width; ++b){

						if((x - b) % stride == 0){

							int x0 = (x - b) / stride;

							if(x0 >= 0 && x0 < szOut.width){
								int row2 = y0 * szOut.width + x0;
								int col2 = c * szWarea + (a * szW.width + b);
								T val = dR[row2 * Delta.cols + col2];
								sum += val;
							}
						}
					}
				}
			}
		}
		dXi[(y * szA0.width + x) * channels] = sum;
	}
}

/////

template< typename T >
__global__ void cols2im(Mtx Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_cols2im<T>(Delta, szOut, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void cols2im_vec(SmallMtxArray Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_cols2im<T>(Delta.mtx[row], szOut, szA0, channels, szW, stride, X.mtx[row]);
	}
}

////////

template< typename T >
__global__ void cols2imT(Mtx Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_cols2imT<T>(Delta, szOut, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void cols2imT_vec(SmallMtxArray Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_cols2imT<T>(Delta.mtx[row], szOut, szA0, channels, szW, stride, X.mtx[row]);
	}
}


////////////////

template< typename T >
__device__ void _subsample(const Mtx &X,
						   int K,
						   const ct::Size& szA,
						   Mtx Y,
						   Mtx Mask,
						   const ct::Size& szO)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	const int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)X.data + k;
		T* dM = (T*)Mask.data + k;
		T *dY = (T*)Y.data + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T mmax = dX[(y0 * szA.width + x0) * X.cols];
		int xm = x0, ym = y0;
		T resM = 0;

		for(int a = 0; a < stride; ++a){
			if(y0 + a < szA.height){
				for(int b = 0; b < stride; ++b){
					if(x0 + b < szA.width){
						T val = dX[((y0 + a) * szA.width + (x0 + b)) * X.cols];
						if(val > mmax){
							mmax = val;
							xm = x0 + b;
							ym = y0 + a;
							resM = 1;
						}
					}
				}
			}
		}

		dY[(y * szO.width + x) * Y.cols] = mmax;
		dM[(ym * szA.width + xm) * Mask.cols] = resM;
	}
}

template< typename T >
__global__ void subsample(Mtx X,
						  int K,
						  ct::Size szA,
						  Mtx Y,
						  Mtx Mask,
						  ct::Size szO)
{
	_subsample<T>(X, K, szA, Y, Mask, szO);
}

template< typename T >
__global__ void subsample_vec(SmallMtxArray X,
						  int K,
						  ct::Size szA,
						  SmallMtxArray Y,
						  SmallMtxArray Mask,
						  ct::Size szO)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_subsample<T>(X.mtx[row], K, szA, Y.mtx[row], Mask.mtx[row], szO);
	}
}

template< typename T >
__device__ void _upsample(const Mtx &Y,
						 const Mtx &Mask,
						 int K,
						 const ct::Size &szO,
						 const ct::Size &szA,
						 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)(X.data) + k;
		T* dM = (T*)(Mask.data) + k;
		T *dY = (T*)(Y.data) + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T val = dY[(y * szO.width + x) * K];

		for(int a = 0; a < stride; ++a){
			if(y0 + a < szA.height){
				for(int b = 0; b < stride; ++b){
					if(x0 + b < szA.width){
						T m = dM[((y0 + a) * szA.width + (x0 + b)) * Mask.cols];
						dX[((y0 + a) * szA.width + (x0 + b)) * X.cols] = val * m;
					}
				}
			}
		}
	}
}

template< typename T >
__global__ void upsample(Mtx Y,
						 Mtx Mask,
						 int K,
						 ct::Size szO,
						 ct::Size szA,
						 Mtx X)
{
	_upsample<T>(Y, Mask, K, szO, szA, X);
}

template< typename T >
__global__ void upsample_vec(SmallMtxArray Y,
							 SmallMtxArray Mask,
							 int K,
							 ct::Size szO,
							 ct::Size szA,
							 SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_upsample<T>(Y.mtx[row], Mask.mtx[row], K, szO, szA, X.mtx[row]);
	}
}

template< typename T >
__global__ void vec2mat(SmallMtxArray vec, Mtx mat)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		dM[row * mat.cols + col] = dV[col];
	}
}

template< typename T >
__global__ void mat2vec(Mtx mat, ct::Size sz, SmallMtxArray vec)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		int y = col/sz.width;
		int x = col - y * sz.width;

		dV[y * sz.width + x] = dM[row * mat.cols + col];
	}
}

template< typename T >
__global__ void addvec(Mtx mat,  SmallMtxArray vec, T alpha)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dM = (T*)mat.data;

		T val = 0;
		for(int i = 0; i < vec.count; ++i){
			T* dV = (T*)vec.mtx[i].data;
			val += dV[row * mat.cols + col];
		}

		dM[row * mat.cols + col] = val * alpha;
	}
}

template< typename T >
__global__ void meanAndSigma(SmallMtxArray X, Mtx Mean, Mtx Sigma)
{
//	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	int coli = threadIdx.x;
	int rowi = threadIdx.y;
	__shared__ T data[BLOCKSIZE][BLOCKSIZE];

	if(col < X.mtx[0].total()){
		T *dMean	= (T*)Mean.data;
		T *dSigma	= (T*)Sigma.data;

//		int off = coli * BLOCKSIZE;
//		int cnt = Xi.total() - off;
//		cnt = min(BLOCKSIZE, cnt);

		int offr = rowi * BLOCKSIZE;
		int cntr = X.count - offr;
		cntr = max(0, min(BLOCKSIZE, cntr));

		T val = 0;

		/// mean /////

		for(int i = 0; i < cntr; ++i){
			Mtx &Xi		= X.mtx[offr + i];
			T *dXi		= (T*)Xi.data;
			val			+= dXi[col];
		}
		data[rowi][coli] = val;

		__syncthreads();

//		if(coli == 0){
//			T val = 0;
//			for(int i = 0; i < cnt; ++i){
//				val += data[rowi][i];
//			}
//			val /= Xi.total();
//			data[rowi][0] = val;
//		}
		if(rowi == 0){
			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			val /= X.count;
			dMean[col] = val;
		}

		__syncthreads();
		/// sigma /////

		val = 0;
		for(int i = 0; i < cntr; ++i){
			Mtx &Xi		= X.mtx[offr + i];
			T *dXi		= (T*)Xi.data;
			T v2		= dXi[col] - dMean[col];
			val			+= v2 * v2;
		}
		data[rowi][coli] = val;

		__syncthreads();

//		if(coli == 0){
//			T val = 0;
//			for(int i = 0; i < cnt; ++i){
//				val += data[rowi][i];
//			}
//			val /= Xi.total();
//			data[rowi][0] = val;
//		}
		if(rowi == 0){
			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			val /= X.count;
			dSigma[col] = val;
		}
	}
}

template< typename T >
__global__ void batch_normalize(SmallMtxArray X, Mtx Mean, Mtx Sigma, SmallMtxArray Y, Mtx alpha, Mtx betha)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T eps = 10e-8;

	if(row < X.count && col < X.mtx[0].total()){
		Mtx &Xi = X.mtx[row];
		Mtx &Yi = Y.mtx[row];

		T *dXi		= (T*)Xi.data;
		T *dYi		= (T*)Yi.data;
		T *dMean	= (T*)Mean.data;
		T *dSigma	= (T*)Sigma.data;
		T *dAlpha	= (T*)alpha.data;
		T *dBetha	= (T*)betha.data;

		T val = (dXi[col] - dMean[col]) / (::sqrt(dSigma[col] + eps));
		dYi[col] = dAlpha[col] * val + dBetha[col];
	}
}

template< typename T >
__global__ void batch_denormalize(SmallMtxArray D, SmallMtxArray X, Mtx Mean, Mtx Sigma, Mtx alpha, Mtx betha,
									SmallMtxArray Xout)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(row < D.count && col < D.mtx[0].total()){

	}
}

}	/// @endnamespace internal

}	/// @endnamespace gpumat

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2cols<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2cols_vec<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols_vec<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

//////////

extern "C"
void cuda_im2colsT(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2colsT<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsT<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2colsT_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2colsT_vec<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsT_vec<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

////////// same

extern "C"
void cuda_im2colsSame(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2colsSame<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsSame<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2cols_vecSame(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2cols_vecSame<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols_vecSame<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

//////////

extern "C"
void cuda_im2colsTSame(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2colsTSame<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsTSame<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2colsT_vecSame(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2colsT_vecSame<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsT_vecSame<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

////////// end same

extern "C"
void cuda_cols2im(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::cols2im<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2im<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_cols2im_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::cols2im_vec<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2im_vec<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

//////////////////

extern "C"
void cuda_cols2imT(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::cols2imT<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2imT<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_col2imT_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::cols2imT_vec<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2imT_vec<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

//////////////////

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::subsample<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::subsample_vec<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample_vec<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::upsample<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::upsample_vec<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample_vec<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}


extern "C"
void cuda_vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat)
{
	int rows = mat.rows;
	int cols = mat.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::vec2mat<double> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
		case GPU_FLOAT:
			internal::vec2mat<float> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
	}
}

extern "C"
void cuda_mat2vec(const GpuMat& mat, const ct::Size& sz, std::vector< GpuMat >& vec)
{
	int rows = mat.rows;
	int cols = mat.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::mat2vec<double> <<<dimGrid, dimBlock>>>(mat, sz, vec);
			break;
		case GPU_FLOAT:
			internal::mat2vec<float> <<<dimGrid, dimBlock>>>(mat, sz, vec);
			break;
	}
}

extern "C"
void cuda_addvec(gpumat::GpuMat &W, const std::vector<gpumat::GpuMat> &vW, double alpha)
{
	int rows = W.rows;
	int cols = W.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (W.type) {
		case GPU_DOUBLE:
			internal::addvec<double> <<<dimGrid, dimBlock>>>(W, vW, alpha);
			break;
		case GPU_FLOAT:
			internal::addvec<float> <<<dimGrid, dimBlock>>>(W, vW, alpha);
			break;
	}
}

extern "C"
void cuda_batch_normalize(const std::vector<GpuMat> &X, const GpuMat &alpha, const GpuMat &betha, GpuMat &Mean,
					 GpuMat &Sigma, std::vector<GpuMat> &Y)
{
	int rows = X.size();
	int cols = X[0].total();

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::meanAndSigma<double><<<dim3(x1, 1), dimBlock>>>(X, Mean, Sigma);
			internal::batch_normalize<double> <<<dimGrid, dimBlock>>>(X, Mean, Sigma, Y, alpha, betha);
			break;
		case GPU_FLOAT:
			internal::meanAndSigma<float><<<dim3(x1, 1), dimBlock>>>(X, Mean, Sigma);
			internal::batch_normalize<float> <<<dimGrid, dimBlock>>>(X, Mean, Sigma, Y, alpha, betha);
			break;
	}
}

extern "C"
void cuda_batch_denormalize(const std::vector<GpuMat> &D, const std::vector<GpuMat> &X,
							const GpuMat &Mean, const GpuMat &Sigma,
							GpuMat &alpha, GpuMat &betha, std::vector<GpuMat> &Xout)
{
	int rows = D.size();
	int cols = D[0].total();

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (D[0].type) {
		case GPU_DOUBLE:
			internal::batch_denormalize<double> <<<dimGrid, dimBlock>>>(D, Xout, Mean, Sigma, alpha, betha, Xout);
			break;
		case GPU_FLOAT:
			internal::batch_denormalize<float> <<<dimGrid, dimBlock>>>(D, Xout, Mean, Sigma, alpha, betha, Xout);
			break;
	}
}
