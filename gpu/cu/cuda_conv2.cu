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

			for(int b = 0; b < szW.width; ++b){

				int col2 = c * szWarea + (a * szW.width + b);
				T val = 0;
				if(y0 + a < szA0.height && x0 + b < szA0.width){
					val = dXi[(y0 + a) * szA0.width + (x0 + b)];
				}
				if(col2 < Res.cols)
					dR[row2 * Res.cols + col2] = val;
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
			
			for(int b = 0; b < szW.width; ++b){
				
				int col2 = c * szWarea + (a * szW.width + b);
				T val = 0;
				if(y0 + a < szA0.height && x0 + b < szA0.width){
					val = dXi[((y0 + a) * szA0.width + (x0 + b)) * channels];
				}
				if(col2 < Res.cols)
					dR[row2 * Res.cols + col2] = val;
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

		if(row2 >= Res.rows)
			return;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Res.data;
		T *dXi = &dX[c * szA0area];

		for(int _a = 0; _a < szW.height; ++_a){
			int a = _a - szW.height/2;
			for(int _b = 0; _b < szW.width; ++_b){
				int b = _b - szW.width/2;
				int col2 = c * szWarea + (_a * szW.width + _b);
				T val = 0;
				if(y0 + a >= 0 && y0 + a < szA0.height && x0 + b >= 0 && x0 + b < szA0.width){
					val = dXi[(y0 + a) * szA0.width + (x0 + b)];
				}
				if(col2 < Res.cols)
					dR[row2 * Res.cols + col2] = val;
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

		if(row2 >= Res.rows)
			return;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Res.data;
		T *dXi = (T*)X.data + c;

		for(int _a = 0; _a < szW.height; ++_a){
            int ya = y0 + _a - szW.height/2;
            if(ya >= 0 && ya < szA0.height){
                int yaw = (ya) * szA0.width;
                for(int _b = 0; _b < szW.width; ++_b){
                    int xb = x0 + _b - szW.width/2;
                    int col2 = c * szWarea + (_a * szW.width + _b);
                    T val = 0;
                    if(xb >= 0 && xb < szA0.width){
                        val = dXi[(yaw + xb) * channels];
                    }
                    if(col2 < Res.cols)
                        dR[row2 * Res.cols + col2] = val;
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

//////////////// SAME ///////////

template< typename T >
__device__ void _cols2im_same(const Mtx& Delta,
				 ct::Size szDelta, const ct::Size& szA0,
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

		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Delta.data;
		T *dXi = &dX[c * szA0Area];

		T sum = 0;
		for(int _a = 0; _a < szW.height; ++_a){
			int a = _a - szW.height / 2;
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				if(y0 >= 0 && y0 < szDelta.height){
					for(int _b = 0; _b < szW.width; ++_b){

						int b = _b - szW.width/2;
						if((x - b) % stride == 0){

							int x0 = (x - b) / stride;
							T val = 0;
							if(x0 >= 0 && x0 < szDelta.width){
								int row2 = y0 * szDelta.width + x0;
								int col2 = c * szWarea + (_a * szW.width + _b);
								if(row2 < Delta.rows && col2 < Delta.cols)
									val = dR[row2 * Delta.cols + col2];
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

template< typename T >
__device__ void _cols2imT_same(const Mtx& Delta,
				 ct::Size szDelta, const ct::Size& szA0,
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
		for(int _a = 0; _a < szW.height; ++_a){
			int a = _a - szW.height/2;
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				if(y0 >= 0 && y0 < szDelta.height){
					for(int _b = 0; _b < szW.width; ++_b){

						int b = _b - szW.width/2;
						if((x - b) % stride == 0){

							int x0 = (x - b) / stride;
							T val = 0;
							if(x0 >= 0 && x0 < szDelta.width){
								int row2 = y0 * szDelta.width + x0;
								int col2 = c * szWarea + (_a * szW.width + _b);
								if(row2 < Delta.rows && col2 < Delta.cols)
									val = dR[row2 * Delta.cols + col2];
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

template< typename T >
__global__ void cols2im_same(Mtx Delta,
						   ct::Size szDelta, ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_cols2im_same<T>(Delta, szDelta, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void cols2im_vec_same(SmallMtxArray Delta,
						   ct::Size szDelta, ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_cols2im_same<T>(Delta.mtx[row], szDelta, szA0, channels, szW, stride, X.mtx[row]);
	}
}

//////

template< typename T >
__global__ void cols2imT_same(Mtx Delta,
						   ct::Size szDelta, ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_cols2imT_same<T>(Delta, szDelta, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void cols2imT_vec_same(SmallMtxArray Delta,
						   ct::Size szDelta, ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_cols2imT_same<T>(Delta.mtx[row], szDelta, szA0, channels, szW, stride, X.mtx[row]);
	}
}

/////////////////////////////////

template< typename T>
__device__ T min(T a, T b)
{
    return a < b? a : b;
}

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

        int yr = min(y0 + stride, szA.height);
        int xr = min(x0 + stride, szA.width);
        for(int ya = y0; ya < yr; ++ya){
            int y0a = (ya) * szA.width;
            for(int xb = x0; xb < xr; ++xb){
                T val = dX[(y0a + xb) * X.cols];
                if(val > mmax){
                    mmax = val;
                    xm = xb;
                    ym = ya;
                }
            }
		}

		dY[(y * szO.width + x) * Y.cols] = mmax;
		dM[(ym * szA.width + xm) * Mask.cols] = 1;
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

        int yr = min(y0 + stride, szA.height);
        int xr = min(x0 + stride, szA.width);
        for(int ya = y0; ya < yr; ++ya){
            int y0a = (ya) * szA.width;
            for(int xb = x0; xb < xr; ++xb){
                T m = dM[(y0a + (xb)) * Mask.cols];
                dX[(y0a + (xb)) * X.cols] = val * m;
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

__device__ void get_offset_and_cnt(int& off, int& cnt, int count, int rowi)
{
    off = rowi * BLOCKSIZE_CNV;
	cnt = count - off;
    cnt = max(0, min(BLOCKSIZE_CNV, cnt));
}

template< typename T >
__device__ T sqr(T val)
{
	return val * val;
}

template< typename T >
__global__ void get_mean(const SmallMtxArray X,  Mtx Mean, int spatial, int channels)
{
//	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T N = X.count * spatial, val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, X.count, rowi);

    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	if(col < channels){

		T *dMean	= (T*)Mean.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dX = (T*)X.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s) val += dX[col + s * channels];
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			val = 0;
			for(int i = 0; i < cntr; ++i) val += data[i][coli];
			dMean[col] = val / N;
		}
	}
}

template< typename T >
__global__ void get_var(const SmallMtxArray X, const Mtx Mean, SmallMtxArray Xmu, Mtx Var, int spatial, int channels)
{
	T eps = 1e-8;
//	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T N = X.count * spatial, val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, X.count, rowi);

    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	if(col < channels){

		T *dMean	= (T*)Mean.data;
		T *dV		= (T*)Var.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dX = (T*)X.mtx[offr + i].data;
			T *dXmu = (T*)Xmu.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s){
				T w = dX[col + s * channels] - dMean[col];
				val += w * w;
				dXmu[col + s * channels] = w;
			}
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			val = 0;
			for(int i = 0; i < cntr; ++i) val += data[i][coli];
			dV[col] = sqrt((val / (N - 1)) + eps);
		}
	}
}

template< typename T >
__global__ void get_shift(const SmallMtxArray Xmu, const Mtx gamma, const Mtx betha,
						   const Mtx Var, SmallMtxArray Y, int spatial, int channels)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(row < Xmu.count && col < channels){
		T *dG	= (T*)gamma.data;
		T *dB	= (T*)betha.data;
		T *dXmu		= (T*)Xmu.mtx[row].data;
		T *dVar		= (T*)Var.data;
		T *dY		= (T*)Y.mtx[row].data;

		for(int s = 0; s < spatial; ++s){
			T val = dXmu[col + s * channels] / dVar[col];
			dY[col + s * channels] = dG[col] * val + dB[col];
		}
	}
}


template< typename T >
__global__ void batch_normalize(SmallMtxArray X, Mtx Mean, Mtx Var, SmallMtxArray Xu,
								SmallMtxArray Y, Mtx alpha, Mtx betha)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(row < X.count && col < X.mtx[0].total()){
		T *dXi		= (T*)X.mtx[row].data;
		T *dYi		= (T*)Y.mtx[row].data;
		T *dMean	= (T*)Mean.data;
		T *dAlpha	= (T*)alpha.data;
		T *dBetha	= (T*)betha.data;
		T *dXu		= (T*)Xu.mtx[row].data;
		T *dVar		= (T*)Var.data;

		T xu = dXi[col] - dMean[col];

		T val = xu / dVar[col];
		dYi[col] = dAlpha[col] * val + dBetha[col];
		dXu[col] = xu;
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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szOut.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::cols2imT_vec<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2imT_vec<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

/////////// SAME //////////

extern "C"
void cuda_cols2im_same(const gpumat::GpuMat &Delta, const ct::Size &szDelta,
					   const ct::Size &szA0, int channels, const ct::Size &szW,
					   int stride, gpumat::GpuMat &X)
{
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::cols2im_same<double> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2im_same<float> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_cols2im_vec_same(const std::vector< gpumat::GpuMat > &Delta,
						   const ct::Size &szDelta, const ct::Size &szA0,
						   int channels, const ct::Size &szW, int stride,
						   std::vector< gpumat::GpuMat > &X)
{
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::cols2im_vec_same<double> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2im_vec_same<float> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
	}
}

//////////////////

extern "C"
void cuda_cols2imT_same(const gpumat::GpuMat &Delta,
						const ct::Size &szDelta, const ct::Size &szA0,
						int channels,
						const ct::Size &szW,
						int stride,
						gpumat::GpuMat &X)
{
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::cols2imT_same<double> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2imT_same<float> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_col2imT_vec_same(const std::vector< gpumat::GpuMat > &Delta,
						   const ct::Size &szDelta, const ct::Size &szA0,
						   int channels,
						   const ct::Size &szW,
						   int stride,
						   std::vector< gpumat::GpuMat > &X)
{
    int x1 = szA0.area() * channels / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::cols2imT_vec_same<double> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::cols2imT_vec_same<float> <<<dimGrid, dimBlock>>>(Delta, szDelta, szA0, channels, szW, stride, X);
			break;
	}
}

///////////////////////////

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	int K = X.cols;
    int x1 = szO.area() * K / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szO.area() * K / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
    int x1 = szO.area() * K / BLOCKSIZE_CNV + 1;
	int x2 = 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, 1);

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
    int x1 = szO.area() * K / BLOCKSIZE_CNV + 1;
    int x2 = (int)X.size() / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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

    int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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

    int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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

    int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV);

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
void cuda_batch_normalize(_BN &bn)
{
	int spatial = bn.X->front().total() / bn.channels;

	int rows = bn.X->size();
	int cols = bn.X->front().total();

    int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV), dimGridC(bn.channels / BLOCKSIZE_CNV + 1, x2);

	switch (bn.X->front().type) {
		case GPU_DOUBLE:
			internal::get_mean	<double><<<dimGridC, dimBlock>>>(*bn.X, bn.Mean, spatial, bn.channels);
			internal::get_var	<double><<<dimGridC, dimBlock>>>(*bn.X, bn.Mean, bn.Xu, bn.Var, spatial, bn.channels);
			internal::get_shift	<double><<<dimGridC, dimBlock>>>(bn.Xu, bn.gamma, bn.betha, bn.Var, *bn.Y, spatial, bn.channels);
//			internal::batch_normalize<double> <<<dimGrid, dimBlock>>>(*bn.X, bn.Mean, bn.Var, bn.Xu, *bn.Y, bn.gamma, bn.betha);
			break;
		case GPU_FLOAT:
			internal::get_mean	<float ><<<dimGridC, dimBlock>>>(*bn.X, bn.Mean, spatial, bn.channels);
			internal::get_var	<float ><<<dimGridC, dimBlock>>>(*bn.X, bn.Mean, bn.Xu, bn.Var, spatial, bn.channels);
			internal::get_shift	<float ><<<dimGridC, dimBlock>>>(bn.Xu, bn.gamma, bn.betha, bn.Var, *bn.Y, spatial, bn.channels);
//			internal::batch_normalize<float> <<<dimGrid, dimBlock>>>(*bn.X, bn.Mean, bn.Var, bn.Xu, *bn.Y, bn.gamma, bn.betha);
			break;
	}
}

/////////////////////////////////

namespace gpumat{

namespace internal{

template< typename T >
__global__ void get_dbetha(const SmallMtxArray D, const Mtx gamma, Mtx dbetha, SmallMtxArray Dout, int spatial, int channels)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	T val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, D.count, rowi);

	if(col < channels){
		T *dG	= (T*)gamma.data;
		T *ddB	= (T*)dbetha.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dD		= (T*)D.mtx[offr + i].data;
			T *dDout		= (T*)Dout.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s){
				val						   += dD[col + s * channels];
				dDout[col + s * channels]	= dD[col + s * channels] * dG[col];
			}
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			ddB[col] = val;
		}
	}
}

template< typename T >
__global__ void get_dgamma(const SmallMtxArray D, const SmallMtxArray Xu, const Mtx Var, Mtx dgamma, int spatial, int channels)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, D.count, rowi);
    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	if(col < channels){
		T *dG	= (T*)dgamma.data;
		T *dV = (T*)Var.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dDi		= (T*)D.mtx[offr + i].data;
			T *dXu		= (T*)Xu.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s){
				val			+= dDi[col + s * channels] * (dXu[col + s * channels] / dV[col]);
			}
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			dG[col] = val;
		}
	}
}

template< typename T >
__global__ void get_dxmu(const SmallMtxArray D, const Mtx Var, SmallMtxArray Dout, int spatial, int channels)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(col < channels && row < Dout.count){
		T *dV	= (T*)Var.data;

		T *dD		= (T*)D.mtx[row].data;
		T *dDout	= (T*)Dout.mtx[row].data;
		for(int s = 0; s < spatial; ++s){
			dDout[col + s * channels]	= dD[col + s * channels] / dV[col];
		}
	}
}

template< typename T >
__global__ void get_dsigma(const SmallMtxArray D, const SmallMtxArray Xmu, const Mtx Var, Mtx DVar, int spatial, int channels)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T N = D.count * spatial, val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, D.count, rowi);
    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	if(col < channels){
		T *dV	= (T*)Var.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dD		= (T*)D.mtx[offr + i].data;
			T *dXmu		= (T*)Xmu.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s){
				val += dD[col + s * channels] * dXmu[col + s * channels];
			}
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			T *ddV	= (T*)DVar.data;

			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			T s = dV[col];
			T res = -val * (1/(s * s));
			res = res * 0.5 * (1/s);
			ddV[col] = res / (N - 1);
		}
	}
}

template< typename T >
__global__ void get_dmean(const SmallMtxArray D, const SmallMtxArray Xmu, const Mtx DVar, Mtx dMean, SmallMtxArray Dout, int spatial, int channels)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	T N = Dout.count * spatial, val;
	int offr, cntr;
	int coli = threadIdx.x; int rowi = threadIdx.y;
	get_offset_and_cnt(offr, cntr, Dout.count, rowi);
    __shared__ T data[BLOCKSIZE_CNV][BLOCKSIZE_CNV];

	if(col < channels){
		T *dA	= (T*)dMean.data;
		T *dDVar = (T*)DVar.data;

		val = 0;
		for(int i = 0; i < cntr; ++i){
			T *dD			= (T*)D.mtx[offr + i].data;
			T *dDout		= (T*)Dout.mtx[offr + i].data;
			T *dXmu			= (T*)Xmu.mtx[offr + i].data;
			for(int s = 0; s < spatial; ++s){
				T dsq						= 2 * dDVar[col] * dXmu[col + s * channels];
				val							-= (dsq + dD[col + s * channels]);
				dDout[col + s * channels]	= dD[col + s * channels] + dsq;
			}
		}
		data[rowi][coli] = val;

		__syncthreads();

		if(rowi == 0){
			T val = 0;
			for(int i = 0; i < cntr; ++i){
				val += data[i][coli];
			}
			dA[col] = val/N;
		}
	}
}

template< typename T >
__global__ void add2mean(const SmallMtxArray Dout, const Mtx DMean, SmallMtxArray Xout, int spatial, int channels)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(col < channels && row < Dout.count){
		T *dDMean	= (T*)DMean.data;

		T *dDout	= (T*)Dout.mtx[row].data;
		T *dXout	= (T*)Xout.mtx[row].data;
		for(int s = 0; s < spatial; ++s){
			T xout = dDout[col + s * channels] + dDMean[col];
			dXout[col + s * channels] = xout;
		}
	}
}
template< typename T >
__global__ void scale_and_shift(const SmallMtxArray X, const Mtx gamma, Mtx betha, SmallMtxArray Y, int spatial, int channels)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if(col < channels && row < X.count){
		T *dG	= (T*)gamma.data;
		T *dB	= (T*)betha.data;

		T *dX	= (T*)X.mtx[row].data;
		T *dY	= (T*)Y.mtx[row].data;
		for(int s = 0; s < spatial; ++s){
			T xout = dX[col + s * channels];
			dY[col + s * channels] = dG[col] * xout + dB[col];
		}
	}
}

}	/** @endnamespace internal */

}	/** @endnamespace gpumat */

extern "C"
void cuda_batch_denormalize(_BN &bn)
{
	int spatial = bn.X->front().total() / bn.channels;

	int rows = bn.X->size();
	int cols = bn.X->front().total();

//	int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV), dimGridC(bn.channels / BLOCKSIZE_CNV + 1, x2);

	bn.dgamma.resize(bn.gamma);
	bn.dbetha.resize(bn.betha);
//	bn.dMean.resize(bn.Mean);
	bn.dVar.resize(bn.Var);

	switch (bn.D->front().type) {
		case GPU_DOUBLE:
			internal::get_dbetha	<double> <<<dimGridC, dimBlock>>>(*bn.D, bn.gamma, bn.dbetha, bn.Dout, spatial, bn.channels);
			internal::get_dgamma	<double> <<<dimGridC, dimBlock>>>(*bn.D, bn.Xu, bn.Var, bn.dgamma, spatial, bn.channels);
			internal::get_dxmu		<double> <<<dimGridC, dimBlock>>>(bn.Dout, bn.Var, *bn.D, spatial, bn.channels);
			internal::get_dsigma	<double> <<<dimGridC, dimBlock>>>(bn.Dout, bn.Xu, bn.Var, bn.dVar, spatial, bn.channels);
			internal::get_dmean		<double> <<<dimGridC, dimBlock>>>(*bn.D, bn.Xu, bn.dVar, bn.Mean, bn.Dout, spatial, bn.channels);
			internal::add2mean		<double> <<<dimGridC, dimBlock>>>(bn.Dout, bn.Mean, bn.Dout, spatial, bn.channels);
			break;
		case GPU_FLOAT:
			internal::get_dbetha	<float > <<<dimGridC, dimBlock>>>(*bn.D, bn.gamma, bn.dbetha, bn.Dout, spatial, bn.channels);
			internal::get_dgamma	<float > <<<dimGridC, dimBlock>>>(*bn.D, bn.Xu, bn.Var, bn.dgamma, spatial, bn.channels);
			internal::get_dxmu		<float > <<<dimGridC, dimBlock>>>(bn.Dout, bn.Var, *bn.D, spatial, bn.channels);
			internal::get_dsigma	<float > <<<dimGridC, dimBlock>>>(bn.Dout, bn.Xu, bn.Var, bn.dVar, spatial, bn.channels);
			internal::get_dmean		<float > <<<dimGridC, dimBlock>>>(*bn.D, bn.Xu, bn.dVar, bn.Mean, bn.Dout, spatial, bn.channels);
			internal::add2mean		<float > <<<dimGridC, dimBlock>>>(bn.Dout, bn.Mean, bn.Dout, spatial, bn.channels);
			break;
	}
//	int index = 0;
//	for(GpuMat &D: *bn.D){
//		D.copyTo(bn.Dout[index++]);
//	}
}

extern "C"
void cuda_scale_and_shift_bn(_BN &bn)
{
	int spatial = bn.X->front().total() / bn.channels;

	int rows = bn.X->size();
	int cols = bn.X->front().total();

//	int x1 = cols / BLOCKSIZE_CNV + 1;
    int x2 = rows / BLOCKSIZE_CNV + 1;

    dim3 dimBlock(BLOCKSIZE_CNV, BLOCKSIZE_CNV), dimGridC(bn.channels / BLOCKSIZE_CNV + 1, x2);

	switch (bn.D->front().type) {
		case GPU_DOUBLE:
			internal::scale_and_shift	<double> <<<dimGridC, dimBlock>>>(*bn.X, bn.gamma, bn.betha, *bn.Y, spatial, bn.channels);
			break;
		case GPU_FLOAT:
			internal::scale_and_shift	<float > <<<dimGridC, dimBlock>>>(*bn.X, bn.gamma, bn.betha, *bn.Y, spatial, bn.channels);
			break;
	}

}
