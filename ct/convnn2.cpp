#include "convnn2.h"

namespace conv2{

template< typename T >
void im2col(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

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
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void im2colT(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

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
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[((y0 + a) * szA0.width + (x0 + b)) * colsX];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void back_deriv(const ct::Mat_<T>& Delta, const ct::Size& szOut, const ct::Size& szA0,
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
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				for(int a = 0; a < szW.height; ++a){
#ifdef __GNUC__
#pragma omp simd
#endif
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row * Delta.cols + col];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void back_derivT(const ct::Mat_<T>& Delta, const ct::Size& szOut, const ct::Size& szA0,
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
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				for(int a = 0; a < szW.height; ++a){
#ifdef __GNUC__
#pragma omp simd
#endif
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += dR[row * Delta.cols + col];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void subsample(const ct::Mat_<T>& X, const ct::Size& szA, ct::Mat_<T>& Y, ct::Mat_<T>& Mask, ct::Size& szO)
{
	if(X.empty() || X.rows != szA.area())
		return;

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
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
				T resM = 0;
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
								resM = 1;
							}
						}
					}
				}

				dY[(y * szO.width + x) * Y.cols] = mmax;
				dM[(ym * szA.width + xm) * Mask.cols] = resM;
			}
		}
	}
}

template< typename T >
void upsample(const ct::Mat_<T>& Y, int K, const ct::Mat_<T>& Mask, const ct::Size& szO,
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

template< typename T >
void vec2mat(const std::vector< ct::Mat_<T> >& vec, ct::Mat_<T>& mat)
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
	vec2mat(vec, mat);
}

void vec2mat(const std::vector< ct::Matd >& vec, ct::Matd& mat)
{
	vec2mat(vec, mat);
}

////////////////////////////////////////////

template< typename T >
void mat2vec(const ct::Mat_<T>& mat, const ct::Size& szOut, std::vector< ct::Mat_<T> >& vec)
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
	mat2vec<float>(mat, szOut, vec);
}

void mat2vec(const ct::Matd& mat, const ct::Size& szOut, std::vector< ct::Matd >& vec)
{
	mat2vec<double>(mat, szOut, vec);
}

//////////////////////////////////////

template< typename T >
void flipW(const ct::Mat_<T>& W, const ct::Size& sz,int channels, ct::Mat_<T>& Wr)
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
	flipW<float>(W, sz, channels, Wr);
}

void flipW(const ct::Matd& W, const ct::Size& sz,int channels, ct::Matd& Wr)
{
	flipW<double>(W, sz, channels, Wr);
}


}