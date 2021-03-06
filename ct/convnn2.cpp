#include "convnn2.h"

namespace conv2{

template< typename T >
void _im2cols(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	T *dX = X.ptr();
	T *dR = Res.ptr();

#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];

#pragma omp parallel for
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						T val = 0;
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							val = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
						if(col < Res.cols)
							dR[row * Res.cols + col] = val;
					}
				}

			}
		}
	}
}

void im2cols(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut)
{
	_im2cols(X, szA0, channels, szW, stride, Res, szOut);
}

void im2cols(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut)
{
	_im2cols(X, szA0, channels, szW, stride, Res, szOut);
}

/////////////////////////////////////////

template< typename T >
void _im2colsT(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	ct::get_cnv_sizes(szA0, szW, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	int colsX = channels;

	T *dR = Res.ptr();

#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;

#pragma omp parallel for
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						T val = 0;
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							val = dXi[((y0 + a) * szA0.width + (x0 + b)) * colsX];
						}
						if(col < Res.cols)
							dR[row * Res.cols + col] = val;
					}
				}

			}
		}
	}
}

void im2colsT(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut)
{
	_im2colsT(X, szA0, channels, szW, stride, Res, szOut);
}

void im2colsT(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut)
{
	_im2colsT(X, szA0, channels, szW, stride, Res, szOut);
}

//////////////////////////////////////////

template< typename T >
void _im2cols_same(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	T *dX = X.ptr();
	T *dR = Res.ptr();

#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];

#pragma omp parallel for
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int _a = 0; _a < szW.height; ++_a){
					int a = _a - szW.height/2;
					for(int _b = 0; _b < szW.width; ++_b){
						int b = _b - szW.width/2;
						T val = 0;
						int col = c * szW.area() + (_a * szW.width + _b);
						if(y0 + a >= 0 && y0 + a < szA0.height && x0 + b >= 0 && x0 + b < szA0.width){
							val = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
						if(col < Res.cols)
							dR[row * Res.cols + col] = val;
					}
				}

			}
		}
	}
}

void im2cols_same(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut)
{
	_im2cols_same(X, szA0, channels, szW, stride, Res, szOut);
}

void im2cols_same(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut)
{
	_im2cols_same(X, szA0, channels, szW, stride, Res, szOut);
}

/////////////////////////////////////////

template< typename T >
void _im2colsT_same(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	ct::get_cnv_size_same(szA0, stride, szOut);

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	int colsX = channels;

	T *dR = Res.ptr();

//#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;

//#pragma omp parallel for
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int _a = 0; _a < szW.height; ++_a){
					int a = _a - szW.height/2;
					for(int _b = 0; _b < szW.width; ++_b){
						int b = _b - szW.width/2;
						T val = 0;
						int col = c * szW.area() + (_a * szW.width + _b);
						if(y0 + a >= 0 && y0 + a < szA0.height && x0 + b >= 0 && x0 + b < szA0.width){
							val = dXi[((y0 + a) * szA0.width + (x0 + b)) * colsX];
						}
						if(col < Res.cols)
							dR[row * Res.cols + col] = val;
					}
				}

			}
		}
	}
}

void im2colsT_same(const ct::Matf& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matf& Res, ct::Size& szOut)
{
	_im2colsT_same(X, szA0, channels, szW, stride, Res, szOut);
}

void im2colsT_same(const ct::Matd& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Matd& Res, ct::Size& szOut)
{
	_im2colsT_same(X, szA0, channels, szW, stride, Res, szOut);
}

//////////////////////////////////////////

template<typename T>
void _conv2(const ct::Mat_<T> &A, const ct::Size &szA, int channels, int stride, const ct::Mat_<T> &B, const ct::Size &szB,
			ct::Mat_<T> &C, ct::Size &szOut, TYPE_CONV type, bool transpose)
{
	if(A.empty() || B.empty())
		return;

	ct::Mat_<T> X;

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

	ct::Mat_<T>& W = (ct::Mat_<T>&)B;
//	int rows = W.rows;
//	int cols = W.cols;
//	W.rows = szB.area() * channels;
//	W.cols = (rows * cols) / W.rows;

	ct::matmul(X, W, C);

//	W.rows = rows;
//	W.cols = cols;
}

void conv2(const ct::Matf &A, const ct::Size &szA, int channels, int stride, const ct::Matf &B,
		   const ct::Size &szB, ct::Matf &C, ct::Size &szOut, TYPE_CONV type, bool transpose)
{
	_conv2<float>(A, szA, channels, stride, B, szB, C, szOut, type, transpose);
}

void conv2(const ct::Matd &A, const ct::Size &szA, int channels, int stride, const ct::Matd &B,
		   const ct::Size &szB, ct::Matd &C, ct::Size& szOut, TYPE_CONV type, bool transpose)
{
	_conv2<double>(A, szA, channels, stride, B, szB, C, szOut, type, transpose);
}

//////////////////////////////////////////

template< typename T >
void _cols2im(const ct::Mat_<T>& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(channels, szA0.area());
	X.fill(0);

	T *dX = X.ptr();
	T *dR = Delta.ptr();

#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];
#pragma omp parallel for
		for(int y = 0; y < szDelta.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szDelta.width; ++x){
				int x0 = x * stride;
				int row = y * szDelta.width + x;

				for(int a = 0; a < szW.height; ++a){
					if(y0 + a < szA0.height){
#ifdef __GNUC__
#pragma omp simd
#endif
						for(int b = 0; b < szW.width; ++b){
							int col = c * szW.area() + (a * szW.width + b);
							if(x0 + b < szA0.width){
								dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row * Delta.cols + col];
							}
						}
					}
				}

			}
		}
	}
}

void cols2im(const ct::Matf& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X)
{
	_cols2im(Delta, szDelta, szA0, channels, szW, stride, X);
}

void cols2im(const ct::Matd& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X)
{
	_cols2im(Delta, szDelta, szA0, channels, szW, stride, X);
}

////////////// back_deriv_same //////////////

template< typename T >
void _cols2im_same(const ct::Mat_<T>& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(channels, szA0.area());
	X.fill(0);

	T *dX = X.ptr();
	T *dR = Delta.ptr();

#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];
//#pragma omp parallel for
		for(int y = 0; y < szDelta.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szDelta.width; ++x){
				int x0 = x * stride;
				int row = y * szDelta.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int _a = 0; _a < szW.height; ++_a){
					int a = _a - szW.height/2;
					if(y0 + a >= 0 && y0 + a < szA0.height){
						for(int _b = 0; _b < szW.width; ++_b){
							int b = _b - szW.width/2;
							int col = c * szW.area() + (_a * szW.width + _b);
							if(x0 + b >= 0 && x0 + b < szA0.width && col < Delta.cols){
								T val = dR[row * Delta.cols + col];
								dXi[(y0 + a) * szA0.width + (x0 + b)] += val;
							}
						}
					}
				}

			}
		}
	}
}

void cols2im_same(const ct::Matf& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X)
{
	_cols2im_same(Delta, szDelta, szA0, channels, szW, stride, X);
}

void cols2im_same(const ct::Matd& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X)
{
	_cols2im_same(Delta, szDelta, szA0, channels, szW, stride, X);
}

/////////////////////////////////////////////

template< typename T >
void _col2imT(const ct::Mat_<T>& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(szA0.area(), channels);
	X.fill(0);

	T *dR = Delta.ptr();
//#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;
//#pragma omp parallel for
		for(int y = 0; y < szDelta.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szDelta.width; ++x){
				int x0 = x * stride;
				int row = y * szDelta.width + x;

				for(int a = 0; a < szW.height; ++a){
					if(y0 + a < szA0.height){
#ifdef __GNUC__
#pragma omp simd
#endif
						for(int b = 0; b < szW.width; ++b){
							int col = c * szW.area() + (a * szW.width + b);
							if(x0 + b < szA0.width){
								dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += dR[row * Delta.cols + col];
							}
						}
					}
				}

			}
		}
	}
}

void cols2imT(const ct::Matf& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X)
{
	_col2imT(Delta, szDelta, szA0, channels, szW, stride, X);
}

void cols2imT(const ct::Matd& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X)
{
	_col2imT(Delta, szDelta, szA0, channels, szW, stride, X);
}

///////////// back_derivT_same /////////////////////

/////////////////////////////////////////////

template< typename T >
void _col2imT_same(const ct::Mat_<T>& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(szA0.area(), channels);
	X.fill(0);

	T *dR = Delta.ptr();
#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;
#pragma omp parallel for
		for(int y = 0; y < szDelta.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szDelta.width; ++x){
				int x0 = x * stride;
				int row = y * szDelta.width + x;

				for(int _a = 0; _a < szW.height; ++_a){
					int a = _a - szW.height/2;
					if(y0 + a >= 0 && y0 + a < szA0.height){
#ifdef __GNUC__
#pragma omp simd
#endif
						for(int _b = 0; _b < szW.width; ++_b){
							int b = _b - szW.width/2;
							int col = c * szW.area() + (_a * szW.width + _b);
							if(x0 + b >= 0 && x0 + b < szA0.width && col < Delta.cols){
								T val = dR[row * Delta.cols + col];
								dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += val;
							}
						}
					}
				}

			}
		}
	}
}

void cols2imT_same(const ct::Matf& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matf& X)
{
	_col2imT_same(Delta, szDelta, szA0, channels, szW, stride, X);
}

void cols2imT_same(const ct::Matd& Delta, const ct::Size& szDelta, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Matd& X)
{
	_col2imT_same(Delta, szDelta, szA0, channels, szW, stride, X);
}


///////////// conv2_transpose //////////////////////

template<typename T>
void _conv2_transpose(const ct::Mat_<T> &C, const ct::Size &szA, int channels, int stride,
					 const ct::Mat_<T> &B, const ct::Size &szB, const ct::Size &szC, ct::Mat_<T> &A,
					 TYPE_CONV type, bool transpose)
{
	if(C.empty() || B.empty())
		return;

	ct::Mat_<T> D;
	ct::Mat_<T>& W = (ct::Mat_<T>&)B;
//	int rows = W.rows;
//	int cols = W.cols;
//	W.rows = szB.area();
//	W.cols = (rows * cols) / W.rows;

	ct::matmulT2(C, W, D);

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


void conv2_transpose(const ct::Matf &C, const ct::Size &szA, int channels, int stride,
					 const ct::Matf &B, const ct::Size &szB, const ct::Size &szC, ct::Matf &A,
					 TYPE_CONV type, bool transpose)
{
	_conv2_transpose<float>(C, szA, channels, stride, B, szB, szC, A, type, transpose);
}

void conv2_transpose(const ct::Matd &C, const ct::Size &szA, int channels, int stride,
					 const ct::Matd &B, const ct::Size &szB, const ct::Size &szC, ct::Matd &A,
					 TYPE_CONV type, bool transpose)
{
	_conv2_transpose<double>(C, szA, channels, stride, B, szB, szC, A, type, transpose);
}

////////////////////////////////////////////////////

template< typename T >
void _subsample(const ct::Mat_<T>& X, const ct::Size& szA, ct::Mat_<T>& Y, ct::Mat_<T>& Mask, ct::Size& szO)
{
	if(X.empty() || X.rows != szA.area())
		return;

	szO.width = (szA.width + 1) / 2;
	szO.height = (szA.height + 1) / 2;
	int K = X.cols;

	Y.setSize(szO.area(), K);
	Mask.setSize(X.size());
	Mask.fill(0);

	int stride = 2;

#pragma omp parallel for
	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

#pragma omp parallel for
		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T mmax = dX[(y0 * szA.width + x0) * X.cols];
				int xm = x0, ym = y0;
#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < stride; ++a){
					for(int b = 0; b < stride; ++b){
						if(y0 + a < szA.height && x0 + b < szA.width){
							T val = dX[((y0 + a) * szA.width + (x0 + b)) * X.cols];
							if(val > mmax){
								mmax = val;
								xm = x0 + b;
								ym = y0 + a;
							}
						}
					}
				}

				dY[(y * szO.width + x) * Y.cols] = mmax;
				dM[(ym * szA.width + xm) * Mask.cols] = 1;
			}
		}
	}
}

void subsample(const ct::Matf& X, const ct::Size& szA, ct::Matf& Y, ct::Matf& Mask, ct::Size& szO)
{
	_subsample(X, szA, Y, Mask, szO);
}

void subsample(const ct::Matd& X, const ct::Size& szA, ct::Matd& Y, ct::Matd& Mask, ct::Size& szO)
{
	_subsample(X, szA, Y, Mask, szO);
}

////////////////////////////////////////////

template< typename T >
void _upsample(const ct::Mat_<T>& Y, int K, const ct::Mat_<T>& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Mat_<T>& X)
{
	if(Y.empty() || Mask.empty() || Y.total() != szO.area() * K)
		return;

	X.setSize(szA.area(), K);

	int stride = 2;

#pragma omp parallel for
	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

#pragma omp parallel for
		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T val = dY[(y * szO.width + x) * K];

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < stride; ++a){
					for(int b = 0; b < stride; ++b){
						if(y0 + a < szA.height && x0 + b < szA.width){
							T m = dM[((y0 + a) * szA.width + (x0 + b)) * Mask.cols];
							dX[((y0 + a) * szA.width + (x0 + b)) * X.cols] = val * m;
						}
					}
				}
			}
		}
	}
}

void upsample(const ct::Matf& Y, int K, const ct::Matf& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Matf& X)
{
	_upsample(Y, K, Mask, szO, szA, X);
}

void upsample(const ct::Matd& Y, int K, const ct::Matd& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Matd& X)
{
	_upsample(Y, K, Mask, szO, szA, X);
}

//////////////////////////////////////////

template< typename T >
void _vec2mat(const std::vector< ct::Mat_<T> >& vec, ct::Mat_<T>& mat)
{
	if(vec.empty() || vec[0].empty())
		return;

	int rows = (int)vec.size();
	int cols = vec[0].total();

	mat.setSize(rows, cols);

	T *dM = mat.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		const ct::Mat_<T>& V = vec[i];
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dM[i * cols + j] = dV[j];
		}
	}
}

void vec2mat(const std::vector< ct::Matf >& vec, ct::Matf& mat)
{
	_vec2mat(vec, mat);
}

void vec2mat(const std::vector< ct::Matd >& vec, ct::Matd& mat)
{
	_vec2mat(vec, mat);
}

////////////////////////////////////////////

template< typename T >
void _mat2vec(const ct::Mat_<T>& mat, const ct::Size& szOut, std::vector< ct::Mat_<T> >& vec)
{
	if(mat.empty())
		return;

	int rows = mat.rows;
	int cols = mat.cols;

	vec.resize(rows);

	T *dM = mat.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		ct::Mat_<T>& V = vec[i];
		V.setSize(szOut);
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dV[j] = dM[i * cols + j];
		}
	}
}

void mat2vec(const ct::Matf& mat, const ct::Size& szOut, std::vector< ct::Matf >& vec)
{
	_mat2vec<float>(mat, szOut, vec);
}

void mat2vec(const ct::Matd& mat, const ct::Size& szOut, std::vector< ct::Matd >& vec)
{
	_mat2vec<double>(mat, szOut, vec);
}

//////////////////////////////////////

template< typename T >
void _flipW(const ct::Mat_<T>& W, const ct::Size& sz,int channels, ct::Mat_<T>& Wr)
{
	if(W.empty() || W.rows != sz.area() * channels)
		return;

	Wr.setSize(W.size());

#pragma omp parallel for
	for(int k = 0; k < W.cols; ++k){
		for(int c = 0; c < channels; ++c){
			T *dW = W.ptr() + c * sz.area() * W.cols + k;
			T *dWr = Wr.ptr() + c * sz.area() * W.cols + k;

#ifdef __GNUC__
#pragma omp simd
#endif
			for(int a = 0; a < sz.height; ++a){
				for(int b = 0; b < sz.width; ++b){
					dWr[((sz.height - a - 1) * sz.width + b) * W.cols] = dW[((a) * sz.width + b) * W.cols];
				}
			}

		}
	}
}

void flipW(const ct::Matf& W, const ct::Size& sz,int channels, ct::Matf& Wr)
{
	_flipW<float>(W, sz, channels, Wr);
}

void flipW(const ct::Matd& W, const ct::Size& sz,int channels, ct::Matd& Wr)
{
	_flipW<double>(W, sz, channels, Wr);
}

}
