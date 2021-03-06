#ifndef MATOPS_H
#define MATOPS_H

#include "custom_types.h"

namespace ct{

template< typename T >
inline Mat_<T> operator* (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.rows)
		return Mat_<T>();
	int r = m1.rows;
	int c = m2.cols;
	Mat_<T> res(r, c);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){

//#pragma omp parallel for
		for(int k = 0; k < m2.cols; k++){
			T s = 0;
			for(int j = 0; j < m1.cols; j++){
				s += val1[i * m1.cols + j]/*at(i, j)*/ * val2[j * m2.cols + k]/*at(j, k)*/;
			}
			valr[i * res.cols + k] = s;
//			res.at(i, k) = s;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator* (const Mat_<T>& m1, T v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			valr[offset] = val1[offset] * v;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator* (T v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			valr[offset] = val1[offset] * v;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator+ (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			valr[offset] = val1[offset] + val2[offset];
		}
	}

	return res;
}

template< typename T >
/**
 * @brief add
 * @param m1
 * @param m2
 * @param res = m1 + m2
 */
inline void add(const Mat_<T>& m1, const Mat_<T>& m2, Mat_<T>& res)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return;
	res.setSize(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			valr[offset] = val1[offset] + val2[offset];
		}
	}
}

/**
 * @brief add
 * @param m1 = a1 * m1 + a2 * m2
 * @param m2
 * @param a1
 * @param a2
 */
template< typename T >
inline void add(Mat_<T>& m1, const Mat_<T>& m2, T a1 = 1., T a2 = 1.)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return;

	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			val1[offset] = a1 * val1[offset] + a2 * val2[offset];
		}
	}
}


template< typename T >
inline Mat_<T> operator+ (const Mat_<T>& m1, const T& v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] + v;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator+ (const T& v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] + v;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
	T* m2_val = &(*m2.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] - m2_val[offset];
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const Mat_<T>& m1, const T& v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] - v;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const T& v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j <  m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = v - m1_val[offset];
		}
	}

	return res;
}

template< typename T, int count >
inline Mat_<T> operator* (const Mat_<T>& m1, const Vec_< T, count >& v)
{
	Mat_<T> res(m1.rows, 1);

	if(m1.cols != count)
		return res;

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
		T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			s += m1_val[i * m1.cols + j] * v.val[j];
		}
		res_val[i] = s;
	}

	return res;
}

////***************************

template< typename T >
bool operator==(const Mat_<T>& A, const Mat_<T>& B)
{
	if(A.cols != B.cols || A.rows != B.rows)
		return false;

	T* val1 = &(*A.val)[0];
	T* val2 = &(*B.val)[0];
	T eps = 0;
#pragma omp parallel for shared(eps)
	for(int i = 0; i < A.total(); i++){
		eps += std::abs(val1[i] - val2[i]);
	}
	if(eps < 1e-9)
		return true;
	return false;
}

////*************************

/**
 * @brief elemMult
 * @param A = A .* B
 * @param B
 */
template< typename T >
inline void elemwiseMult(Mat_<T > &A, const Mat_<T > &B)
{
	if(A.cols != B.cols || A.rows != B.rows)
		return;

	T* dA = A.ptr();
	T* dB = B.ptr();
#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; j++){
			int offset = i * A.cols + j;
			dA[offset] *= dB[offset];
		}
	}
}

/**
 * @brief elemwiseMult
 * @param m1
 * @param m2
 * @param C  = m1 .* m2
 */
template< typename T >
inline void elemwiseMult(const Mat_<T > &m1, const Mat_<T > &m2, Mat_<T>& C)
{
	if(m1.empty() || m2.empty() || m1.cols != m2.cols || m1.rows != m2.rows)
		return;
	if(C.ptr() != m1.ptr() && C.ptr() != m2.ptr()){
		C.setSize(m1.rows, m1.cols);
	}else{
		if(C.ptr() == m1.ptr())
			elemwiseMult(C, m2);
		if(C.ptr() == m2.ptr())
			elemwiseMult(C, m1);
		return;
	}

	T* res_val = C.ptr();
	T* m1_val = m1.ptr();
	T* m2_val = m2.ptr();
#pragma omp parallel for
	for(int i = 0; i < m1.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] * m2_val[offset];
		}
	}
}

template< typename T >
inline void flip(const Mat_<T > &A, Mat_<T > &B)
{
	if(A.empty())
		return;

	B.setSize(A.rows, A.cols);

	T *dA = A.ptr();
	T *dB = B.ptr();

	for(int i = 0; i < A.rows; ++i){
		for(int j = 0; j < A.cols; ++j){
			dB[(B.rows - i - 1) * B.cols + j] = dA[i * A.cols + j];
		}
	}
}

template< typename T >
inline Mat_<T> sumRows(const Mat_<T > &m, T alpha = 1.)
{
	Mat_<T> res;
	if(m.rows == 0 || m.cols == 0)
		return res;
	res = Mat_<T>::zeros(1, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#else
//#pragma omp parallel for
#endif
		for(int j = 0; j < m.cols; j++)
			res_val[j] += m_val[i * m.cols + j] * alpha;
	}
	return res;
}

template< typename T >
void add2sumRows(const Mat_<T > &m, Mat_<T > &res, T alpha = 1.)
{
	if(m.rows == 0 || m.cols == 0)
		return;
	if(res.empty()){
		res.setSize(m.size());
		res.fill(0);
	}

	T* res_val	= res.ptr();
	T* m_val	= m.ptr();

#pragma omp parallel for
	for(int j = 0; j < m.cols; j++){
		T val = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int i = 0; i < m.rows; i++){
			val += m_val[i * m.cols + j];
		}
		res_val[j] = val * alpha;
	}
}

/**
 * @brief v_sumRows
 * @param m
 * @param res
 * @param alpha
 */
template< typename T >
inline void v_sumRows(const Mat_<T > &m, Mat_<T>& res, T alpha = 1.)
{
	if(m.rows == 0 || m.cols == 0)
		return;
	res.setSize(1, m.cols);
	res.fill(0);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#else
//#pragma omp parallel for
#endif
		for(int j = 0; j < m.cols; j++)
			res_val[j] += m_val[i * m.cols + j] * alpha;
	}
}

/**
 * @brief exp
 * @param m
 * @return exp(m)
 */
template< typename T >
inline Mat_<T> exp(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = std::exp(m_val[offset]);
		}
	}
	return res;
}

/**
 * @brief log
 * @param m
 * @return log(m)
 */
template< typename T >
inline Mat_<T> log(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = std::log(m_val[offset]);
		}
	}
	return res;
}

/**
 * @brief expi
 * @param m
 * @return exp(-m)
 */
template< typename T >
inline Mat_<T> expi(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

//#pragma omp parallel for
	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = std::exp(-m_val[offset]);
		}
	}
	return res;
}

/**
 * @brief sigmoid
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> sigmoid(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = (T)(1. / (1. + std::exp(-m_val[offset])));
		}
	}
	return res;
}

/**
 * @brief v_sigmoid
 * @param m
 */
template< typename T >
void v_sigmoid(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = (T)(1. / (1. + std::exp(-m_val[offset])));
		}
	}
}

/**
 * @brief v_sigmoid
 * @param m
 * @param r - return matrix
 */
template< typename T >
void v_sigmoid(const Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T* m_val = m.ptr();
	T* r_val = r.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			r_val[offset] = 1. / (1. + std::exp(-m_val[offset]));
		}
	}
}

/**
 * @brief v_derivSigmoid
 * @param m
 * @return
 */
template< typename T >
inline void v_derivSigmoid(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] * (1 - m_val[offset]);
		}
	}
}

/**
 * @brief v_derivSigmoid
 * @param m
 * @return
 */
template< typename T >
inline void v_derivSigmoid(Mat_<T>& m)
{
	T* m_val = m.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] * (1 - m_val[offset]);
		}
	}
}

/**
 * @brief tanh
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> tanh(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T e = std::exp(2 * m_val[offset]);
			res_val[offset] = (T)((e - 1.) / (e + 1.));
		}
	}
	return res;
}

/**
 * @brief v_tanh
 * @param m
 */
template< typename T >
void v_tanh(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T e = std::exp(2 * m_val[offset]);
			m_val[offset] = (T)((e - 1.) / (e + 1.));
		}
	}
}

/**
 * @brief v_tanh
 * @param m
 * @param r - return matrix
 */
template< typename T >
void v_tanh(const Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T* m_val = m.ptr();
	T* r_val = r.ptr();
//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T e = std::exp(2 * m_val[offset]);
			r_val[offset] = (e - 1.) / (e + 1.);
		}
	}
}

/**
 * @brief v_derivTanh
 * @param m
 * @return
 */
template< typename T >
inline void v_derivTanh(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = (1 - m_val[offset] * m_val[offset]);
		}
	}
}

/**
 * @brief v_derivTanh
 * @param m
 * @return
 */
template< typename T >
inline void v_derivTanh(Mat_<T>& m)
{
	T* m_val = m.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = (1 - m_val[offset] * m_val[offset]);
		}
	}
}

///////// relu ///////////////

/**
 * @brief relu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> relu(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T val = m_val[offset];
			res_val[offset] = val * (val > 0);
		}
	}
	return res;
}

/**
 * @brief v_relu
 * @param m
 * @return
 */
template< typename T >
inline void v_relu(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T val = m_val[offset];
			m_val[offset] = val * (val > 0);
		}
	}
}

/**
 * @brief v_relu
 * @param m
 * @return
 */
template< typename T >
inline void v_relu(const Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T *m_val = m.ptr();
	T *r_val = r.ptr();
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T val = m_val[offset];
			r_val[offset] = val * (val > 0);
		}
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivRelu(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] > T(0) ? T(1) : T(0);
		}
	}
	return res;
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivRelu(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] > T(0);
		}
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivRelu(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] > T(0);
		}
	}
}

///////////// leaky relu ///

/**
 * @brief leakyRelu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> leakyRelu(const Mat_<T>& m, T x)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] =  m_val[offset] >= 0? m_val[offset] : x * m_val[offset];
		}
	}
	return res;
}

/**
 * @brief v_leakyRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_leakyRelu(const Mat_<T>& m, T x, Mat_<T>& r)
{
	r.setSize(m.size());
	T *m_val = m.ptr();
	T *r_val = r.ptr();
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			r_val[offset] = m_val[offset] >= 0? m_val[offset] : x * m_val[offset];
		}
	}
}

/**
 * @brief v_leakyRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_leakyRelu(Mat_<T>& m, T x)
{
	T *m_val = m.ptr();
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] >= 0? m_val[offset] : x * m_val[offset];
		}
	}
}

/**
 * @brief derivLeakyRelu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivLeakyRelu(const Mat_<T>& m, T x)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] >= T(0) ? T(1) : T(x);
		}
	}
	return res;
}

/**
 * @brief derivLeakyRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivLeakyRelu(Mat_<T>& m, T x)
{
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] >= T(0) ? T(1) : T(x);
		}
	}
}

/**
 * @brief v_derivLeakyRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivLeakyRelu(const Mat_<T>& m, T x, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] >= T(0) ? T(1) : T(x);
		}
	}
}

////////// derivative sigmoid ///////////////

/**
 * @brief derivSigmoid
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivSigmoid(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T val = m_val[offset];
			res_val[offset] = val * (1 - val);
		}
	}
	return res;
}

/**
 * @brief derivTanh
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivTanh(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			T val = m_val[offset];
			res_val[offset] = (1 - val * val);
		}
	}
	return res;
}

namespace math{

template<typename T >
inline void max_rows(const Mat_<T>& A, Mat_<T>& Max)
{
	Max.setSize(1, A.cols);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int j = 0; j < A.cols; j++){
		T sC = dA[0 * A.cols + j];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int i = 1; i < A.rows; i++){
			sC = std::max(dA[i * A.cols + j], sC);
		}
		dM[j] = sC;
	}
}

template<typename T >
inline void max_cols(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;
	Max.setSize(A.rows, 1);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){
		T sC = dA[i * A.cols + 0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 1; j < A.cols; j++){
			sC = std::max(dA[i * A.cols + j], sC);
		}
		dM[i] = sC;
	}
}

///

template<typename T >
inline void sum_rows(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;

	Max.setSize(1, A.cols);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int j = 0; j < A.cols; j++){
		T sC = dA[0 * A.cols + j];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int i= 1; i < A.rows; i++){
			sC += dA[i * A.cols + j];
		}
		dM[j] = sC;
	}
}

template<typename T >
inline void sum_cols(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;

	Max.setSize(A.rows, 1);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){
		T sC = dA[i * A.cols + 0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 1; j < A.cols; j++){
			sC += dA[i * A.cols + j];
		}
		dM[i] = sC;
	}
}

///

template< typename T >
inline void exp_rows(const Mat_<T>& A, Mat_<T>& Max, Mat_<T>& C)
{
	if(A.empty() || Max.empty())
		return;

	C.setSize(A.rows, A.cols);

	T* dA = &(*A.val)[0];
	T* dC = &(*C.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = dA[i * A.cols + j] - dM[j];
			val = std::exp(val);
			dC[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void exp_cols(const Mat_<T>& A, Mat_<T>& Max, Mat_<T>& C)
{
	if(A.empty() || Max.empty())
		return;

	C.setSize(A.rows, A.cols);

	T* dA = &(*A.val)[0];
	T* dC = &(*C.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = dA[i * A.cols + j] - dM[i];
			val = std::exp(val);
			dC[i * A.cols + j] = val;
		}
	}
}

////

template< typename T >
inline void sub_ln_rows(Mat_<T>& A, const Mat_<T>& Sum)
{
	if(A.empty() || Sum.empty())
		return;

	T* dA = &(*A.val)[0];
	T* dM = &(*Sum.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::log(dA[i * A.cols + j]) - std::log(dM[j]);
			dA[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void sub_ln_cols(Mat_<T>& A, const Mat_<T>& Sum)
{
	if(A.empty() || Sum.empty())
		return;

	T* dA = &(*A.val)[0];
	T* dM = &(*Sum.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::log(dA[i * A.cols + j]) - std::log(dM[i]);
			dA[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void _exp(Mat_<T>& A)
{
	if(A.empty())
		return;

	T* dA = &(*A.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::exp(dA[i * A.cols + j]);
			dA[i * A.cols + j] = val;
		}
	}
}

}

/**
 * @brief softmax
 * @param m
 * @return softmax(m)
 */
template< typename T >
inline Mat_<T> softmax(const Mat_<T>& m, int axis = 0)
{
	Mat_<T> res(m.rows, m.cols);

	Mat_<T> Max;
//#pragma omp parallel for

	if(axis == 0){
		math::max_rows<T>(m, Max);
		math::exp_rows<T>(m, Max, res);
		math::sum_rows<T>(res, Max);
		math::sub_ln_rows<T>(res, Max);
		math::_exp(res);
	}else
	if(axis == 1){
		math::max_cols<T>(m, Max);
		math::exp_cols<T>(m, Max, res);
		math::sum_cols<T>(res, Max);
		math::sub_ln_cols<T>(res, Max);
		math::_exp(res);
	}

	return res;
}

/**
 * @brief v_softmax
 * @param m
 * @param res = softmax(m)
 */
template< typename T >
inline void v_softmax(const Mat_<T>& m, Mat_<T>& res, int axis = 0)
{
	res.setSize(m.rows, m.cols);

	Mat_<T> Max;
//#pragma omp parallel for

	if(axis == 0){
		math::max_rows<T>(m, Max);
		math::exp_rows<T>(m, Max, res);
		math::sum_rows<T>(res, Max);
		math::sub_ln_rows<T>(res, Max);
		math::_exp(res);
	}else
	if(axis == 1){
		math::max_cols<T>(m, Max);
		math::exp_cols<T>(m, Max, res);
		math::sum_cols<T>(res, Max);
		math::sub_ln_cols<T>(res, Max);
		math::_exp(res);
	}
}

/**
 * @brief sqrt
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqrt(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = std::sqrt(m_val[offset]);
		}
	}
	return res;
}

/**
 * @brief sqrt
 * @param m
 * @return
 */
template< typename T >
void v_elemwiseSqrt(Mat_<T>& m)
{
	if(m.empty())
		throw new std::invalid_argument("v_elemwiseSqrt: matrix is empty");

	T* m_val = m.ptr();

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = sqrt(m_val[offset]);
		}
	}
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqr(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] * m_val[offset];
		}
	}
	return res;
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqr(const Mat_<T>& m, T eps)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			res_val[offset] = m_val[offset] * m_val[offset] + eps;
		}
	}
	return res;
}

/**
 * @brief v_mulColumns
 * @param m
 * @return
 */
template< typename T >
void v_mulColumns(const Mat_<T>& m, const Mat_<T>& c, Mat_<T>& res, int col = 0)
{
	if(m.empty() || c.empty() || c.rows != m.rows || col >= c.cols)
		throw new std::invalid_argument("v_elemwiseSqr: matrix is empty");

	res.setSize(m.size());

	T *m_val = m.ptr();
	T *c_val = c.ptr();
	T *r_val = res.ptr();


#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			int c_off = i * c.cols + col;
			r_val[offset] = m_val[offset] * c_val[c_off];
		}
	}
}

/**
 * @brief v_cropValues
 * @param m
 * @return
 */
template< typename T >
void v_cropValues(const Mat_<T>& m, T val)
{
	if(m.empty())
		throw new std::invalid_argument("v_elemwiseSqr: matrix is empty");

	T* m_val = m.ptr();

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] < val? 0 : m_val[offset];
		}
	}
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
void v_elemwiseSqr(Mat_<T>& m)
{
	if(m.empty())
		throw new std::invalid_argument("v_elemwiseSqr: matrix is empty");

	T* m_val = m.ptr();

#pragma omp parallel for
	for(int i = 0; i < m.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m.cols; j++){
			int offset = i * m.cols + j;
			m_val[offset] = m_val[offset] * m_val[offset];
		}
	}
}

/**
 * @brief division
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseDiv(const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.rows != m2.rows || m1.cols != m2.cols)
		return Mat_<T>();

	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
	T* m2_val = &(*m2.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			int offset = i * m1.cols + j;
			res_val[offset] = m1_val[offset] / m2_val[offset];
		}
	}
	return res;
}

template< typename T >
inline void matmul(const Mat_<T>& m1, const Mat_<T>& m2, Mat_<T>& res)
{
	if(m1.cols != m2.rows)
		return;

	int r = m1.rows;
	int c = m2.cols;
//	Mat_<T> res(r, c);
	res.setSize(r, c);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){

//#pragma omp parallel for
		for(int k = 0; k < m2.cols; k++){
			T s = 0;
			for(int j = 0; j < m1.cols; j++){
				s += val1[i * m1.cols + j]/*at(i, j)*/ * val2[j * m2.cols + k]/*at(j, k)*/;
			}
			valr[i * res.cols + k] = s;
//			res.at(i, k) = s;
		}
	}
}

/**
 * @brief matmulT1
 * @param At
 * @param B
 * @param C = A' * B
 */
template< typename T >
void matmulT1(const Mat_<T>& At, const Mat_<T>& B, Mat_<T>& C)
{
	if(At.rows != B.rows)
		return;
	int r = At.cols;
	int c = B.cols;
	if(C.rows != r || C.cols != c)
		C.setSize(r, c);

	T* valr = &(*C.val)[0];
	T* val1 = &(*At.val)[0];
	T* val2 = &(*B.val)[0];

#pragma omp parallel for
	for(int i = 0; i < At.cols; i++){

//#pragma omp parallel for
		for(int k = 0; k < B.cols; k++){
			T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < At.rows; j++){
				s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k]/*at(j, k)*/;
			}
			valr[i * C.cols + k] = s;
//			res.at(i, k) = s;
		}
	}

}

/**
 * @brief matmulT1
 * @param A
 * @param Bt
 * @param C = A * B'
 */
template< typename T >
void matmulT2(const Mat_<T>& A, const Mat_<T>& Bt, Mat_<T>& C)
{
	if(A.cols != Bt.cols)
		return;
	int r = A.rows;
	int c = Bt.rows;
	if(C.rows != r || C.cols != c)
		C.setSize(r, c);

	T* valr = &(*C.val)[0];
	T* val1 = &(*A.val)[0];
	T* val2 = &(*Bt.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){

//#pragma omp parallel for
		for(int k = 0; k < Bt.rows; k++){
			T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < A.cols; j++){
				s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
			}
			valr[i * C.cols + k] = s;
		}
	}

}

///////////

/**
 * @brief add2matmulT1
 * @param At
 * @param B
 * @param C = A' * B
 */
template< typename T >
void add2matmulT1(const Mat_<T>& At, const Mat_<T>& B, Mat_<T>& C)
{
	if(At.rows != B.rows)
		return;
	int r = At.cols;
	int c = B.cols;
	if(C.rows != r || C.cols != c)
		C.setSize(r, c);

	T* valr = C.ptr();
	T* val1 = At.ptr();
	T* val2 = B.ptr();

#pragma omp parallel for
	for(int i = 0; i < At.cols; i++){

//#pragma omp parallel for
		for(int k = 0; k < B.cols; k++){
			T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < At.rows; j++){
				s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k]/*at(j, k)*/;
			}
			valr[i * C.cols + k] += s;
//			res.at(i, k) = s;
		}
	}

}

///////////

template< typename T >
void dropout(Mat_<T>& mat, T p, Mat_<T>& D, Mat_<T>& Dt, int seed = 0)
{
	std::binomial_distribution<int> bi(1, p);
	//std::normal_distribution< double > nrm(0, 1);
	if(seed > 0)
		generator.seed(seed);

	D = Mat_<T>::ones(mat.rows, mat.cols);
	Dt = Mat_<T>::ones(mat.cols, mat.rows);

	T* val1 = &(*D.val)[0];
	T* val2 = &(*Dt.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < mat.rows; i++){
//#ifdef __GNUC__
//#pragma omp simd
//#endif
		for(int j = 0; j < mat.cols; j++){
			int pi = bi(generator);
			val1[i * D.cols + j] = (T)pi;
			val2[j * D.rows + i] = (T)pi;
		}
	}
	elemwiseMult(mat, D);
}

template< typename T >
void dropout(int rows, int cols, T p, Mat_<T>& D, int seed = 0)
{
	if(rows <= 0 || cols <= 0 || p < 0 || p > 1)
		throw new std::invalid_argument("wrong parameters");
	std::binomial_distribution<int> bi(1, p);
	//std::normal_distribution< double > nrm(0, 1);
	if(seed > 0)
		generator.seed(seed);

    D = Mat_<T>::zeros(rows, cols);

	T* dD = D.ptr();

#pragma omp parallel for
	for(int i = 0; i < D.total(); ++i){
		dD[i] = bi(generator);
	}

}

/**
 * @brief dropout_transpose
 * @param mat
 * @param D
 */
template< typename T >
void dropout_transpose(Mat_<T>& mat, const Mat_<T>& D)
{
	elemwiseMult(mat, D);
}


/**
 * @brief subInd
 * @param mat
 * @param ind
 * @return mat[ind] - 1
 */
template< typename T >
inline Mat_<T> subIndOne(const Mat_<T>& mat, const Mat_<T>& ind)
{
	Mat_<T> res(mat.rows, mat.cols, mat.ptr());

	T* dI = ind.ptr();
	T* dR = res.ptr();

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < mat.rows; ++i){
		int index = (int)dI[i * ind.cols];
		dR[i * mat.cols + index] -= 1.;
	}
	return res;
}

template< typename T >
void subWithColumn(ct::Mat_<T>& t, const ct::Mat_<T>& y, const ct::Mat_<T> column)
{
	if(t.empty() || y.empty() || column.empty() || t.rows != y.rows || t.cols != y.cols || column.rows != t.rows || column.cols != 1)
		throw new std::invalid_argument("subWithColumn: wring parameters");

#pragma omp parallel for
	for(int i = 0; i < t.rows; ++i){
		T* dT = t.ptr(i);
		T* dY = y.ptr(i);
		T* dC = column.ptr(i);
		T c = dC[0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < t.cols; ++j){
			T v1 = dT[j];
			T v2 = dY[j];
			v1 = (v1 - v2) * c;
			dT[j] = v1;
		}
	}
}

template< typename T >
void back_delta_sigmoid(ct::Mat_<T>& t, const ct::Mat_<T>& y, const ct::Mat_<T> column)
{
	if(t.empty() || y.empty() || column.empty() || t.rows != y.rows || t.cols != y.cols || column.rows != t.rows || column.cols != 1)
		throw new std::invalid_argument("back_delta_sigmoid: wring parameters");

#pragma omp parallel for
	for(int i = 0; i < t.rows; ++i){
		T* dT = t.ptr(i);
		T* dY = y.ptr(i);
		T* dC = column.ptr(i);
		T c = dC[0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < t.cols; ++j){
			T val = dT[j];
			T target = dY[j];
			T d = val - target;
			T res = d * val * (1 - val);
			dT[j] = res * c;
		}
	}
}

template< typename T >
inline void m2mpaf(const Mat_<T>& m1, const Mat_<T>& m2, const Mat_<T>& b, etypefunction func, Mat_<T>& res, T param)
{
	if(m1.cols != m2.rows || b.cols != m2.cols || b.rows != 1)
		return;

	int r = m1.rows;
	int c = m2.cols;
//	Mat_<T> res(r, c);
	res.setSize(r, c);

	T* valr = res.ptr();
	T* val1 = m1.ptr();
	T* val2 = m2.ptr();
	T *valb = b.ptr();

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){

//#pragma omp parallel for
		for(int k = 0; k < m2.cols; k++){
			T s = 0;
			for(int j = 0; j < m1.cols; j++){
				s += val1[i * m1.cols + j]/*at(i, j)*/ * val2[j * m2.cols + k]/*at(j, k)*/;
			}

			s += valb[k];
			int off = i * res.cols + k;

			switch (func) {
				default:
				case LINEAR:{
					valr[off] = s;
					break;
				}
				case RELU:{
					valr[off] = s * (s > 0);
					break;
				}
				case LEAKYRELU:{
					valr[off] = s > 0 ? s : param * s;
					break;
				}
				case SIGMOID:{
					valr[off] = (T)(1 / (1 + ::exp(-s)));
					break;
				}
				case TANH:{
					s = (T)::exp(2 * s);
					s = (s - 1) / (s + 1);
					valr[off] = s;
					break;
				}
			}
			//valr[i * res.cols + k] = s;
//			res.at(i, k) = s;
		}
	}
}

/**
 * @brief mul2deriv
 * @param D
 * @param A
 * @param func
 * @param DA
 * @param param1
 * @param param2
 * @param param3
 */
template< typename T >
void mul2deriv(const Mat_<T> &D, const Mat_<T> &A, etypefunction func, Mat_<T> &DA, T param1 = 0, T param2 = 0, T param3 = 0)
{
	if(D.empty() || A.empty() || D.cols != A.cols || D.rows != A.rows)
		throw new std::invalid_argument("mul2deriv: wrong parameters");

	DA.setSize(D.size());

#pragma omp parallel for
	for(int i = 0; i < D.rows; ++i){
		T* dD = D.ptr(i);
		T* dA = A.ptr(i);
		T *dDA = DA.ptr(i);
		for(int j = 0; j < D.cols; ++j){
			T d = dD[j];
			T a = dA[j];
			switch (func) {
				default:
				case LINEAR:{
					dDA[j] = d;
					break;
				}
				case RELU:{
					dDA[j] = d * (a > 0);
					break;
				}
				case LEAKYRELU:{
					dDA[j] = (a > 0)? d : d * param1;
					break;
				}
				case SIGMOID:{
					dDA[j] = d * a * (1 - a);
					break;
				}
				case TANH:{
					dDA[j] = d * (1 - a * a);
					break;
				}
			}
		}
	}
}

template< typename T >
void adamGrad(const ct::Mat_<T>& gW, ct::Mat_<T>& mW, ct::Mat_<T>& vW, ct::Mat_<T>& W, T sb1, T sb2, T alpha, T betha1, T betha2)
{
	if(gW.empty() || mW.empty() || vW.empty() || W.empty() || gW.size() != mW.size()
			|| gW.size() != vW.size() || gW.size() != W.size())
		throw new std::invalid_argument("adamGrad: wrong parameters");

	const T eps = (T)(10e-8);

#pragma omp parallel for
	for(int i = 0; i < gW.rows; ++i){
		const T *dgW	= gW.ptr(i);
		T *dmW			= mW.ptr(i);
		T *dvW			= vW.ptr(i);
		T *dW			= W.ptr(i);
		for(int j = 0; j < gW.cols; ++j){
			T g = dgW[j];
			T g2 = g * g;
			T m = betha1 * dmW[j] + (1 - betha1) * g;
			T v = betha2 * dvW[j] + (1 - betha2) * g2;
			dmW[j] = m;
			dvW[j] = v;

			dW[j] -= alpha * (sb1 * m) / (sqrt(sb2 * v) + eps);
		}
	}
}

template< typename T >
void momentumGrad(const ct::Mat_<T>& gW, ct::Mat_<T>& mW,ct::Mat_<T>& W, T alpha, T betha)
{
	if(gW.empty() || mW.empty() || W.empty() || gW.size() != mW.size()
			|| gW.size() != W.size())
		throw new std::invalid_argument("momentumGrad: wrong parameters");

#pragma omp parallel for
	for(int i = 0; i < gW.rows; ++i){
		const T *dgW	= gW.ptr(i);
		T *dmW			= mW.ptr(i);
		T *dW			= W.ptr(i);
		for(int j = 0; j < gW.cols; ++j){
			T g = dgW[j];
			T m = betha * dmW[j] + (1 - betha) * g;
			dmW[j] = m;

			dW[j] -= alpha * m;
		}
	}
}

template< typename T >
void get_mean(const ct::Mat_<T>& X, ct::Mat_<T>& Y, int axis = 0)
{
	if(X.empty())
		throw new std::invalid_argument("median: empty input matrix");

	if(axis == 0){

		T m = (T)X.rows;

		Y.setSize(1, X.cols);
		Y.fill(0);

		T *dY = Y.ptr();
		T *dX = X.ptr();
		for(int j = 0; j < X.cols; ++j){
			for(int i = 0; i < X.rows; ++i){
				dY[j] += dX[i * X.cols + j];
			}
		}
		for(int j = 0; j < X.cols; ++j){
			dY[j] /= m;
		}
	}else{
		T m = (T)X.cols;

		Y.setSize(X.rows, 1);
		Y.fill(0);

		T *dY = Y.ptr();
		T *dX = X.ptr();
		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i] += dX[i * X.cols + j];
			}
		}
		for(int i = 0; i < X.rows; ++i){
			dY[i] /= m;
		}
	}
}

template< typename T >
void get_std(const ct::Mat_<T>& X, const ct::Mat_<T>& mean, ct::Mat_<T>& Y, int axis = 0)
{
	if(X.empty() || mean.empty() || !(X.cols == mean.cols && mean.rows == 1 && axis == 0)
			&& !(X.rows == mean.rows && mean.cols == 1 && axis == 1))
	{
		throw new std::invalid_argument("get_std: dimensions error");
	}

	Y.setSize(mean.size());
	Y.fill(0);

	T *dX = X.ptr();
	T *dM = mean.ptr();
	T *dY = Y.ptr();

	if(axis == 0){
		T m = (T)X.rows;
		for(int j = 0; j < X.cols; ++j){
			for(int i = 0; i < X.rows; ++i){
				T val1 = dX[i * X.cols + j];
				T val2 = dM[j];
				T val3 = val1 - val2;
				dY[j] += val3 * val3;
			}
		}
		for(int j = 0; j < X.cols; ++j){
			dY[j] = dY[j] / (m - 1);
		}
	}else{
		T m = (T)X.cols;
		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				T val = (dX[i * X.cols + j] - dM[i]);
				dY[i] += val * val;
			}
		}
		for(int i = 0; i < X.rows; ++i){
			dY[i] = dY[i] / (m - 1);
		}
	}
}

template< typename T >
void get_norm(const ct::Mat_<T>& X, const ct::Mat_<T>& mean, const ct::Mat_<T>& std, ct::Mat_<T>& Y, int axis = 0, T eps = 1e-8)
{
	if(X.empty() || mean.empty() || std.empty())
		throw new ::std::invalid_argument("get_norm: empty matrices");

	Y.setSize(X.size());

	T *dX = X.ptr();
	T *dM = mean.ptr();
	T *dS = std.ptr();
	T *dY = Y.ptr();

	if(axis == 0){
		if(X.cols != mean.cols || mean.rows != 1)
			throw new ::std::invalid_argument("get_norm: dimensions wrong");

		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i * Y.cols + j] = (dX[i * X.cols + j] - dM[j])/(sqrt(dS[j] + eps));
			}
		}

	}else{
		if(X.rows != mean.rows || mean.cols != 1)
			throw new ::std::invalid_argument("get_norm: dimensions wrong");

		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i * Y.cols + j] = (dX[i * X.cols + j] - dM[i])/(sqrt(dS[i] + eps));
			}
		}

	}
}

/**
 * @brief write_fs
 * @param fs
 * @param mat
 */
template< typename T >
void write_fs(std::fstream& fs, const ct::Mat_<T>& mat)
{
	fs.write((char*)mat.ptr(), sizeof(T) * mat.total());
}

/**
 * @brief read_fs
 * @param fs
 * @param mat
 */
template< typename T >
void read_fs(std::fstream& fs, ct::Mat_<T>& mat)
{
	fs.read((char*)mat.ptr(), sizeof(T) * mat.total());
}

/////////////////////////////

/**
 * @brief write_fs2
 * write matrix woth info about rows and cols
 * @param fs
 * @param mat
 */
template< typename T >
void write_fs2(std::fstream& fs, const ct::Mat_<T>& mat)
{
	int rows = mat.rows, cols = mat.cols;
	fs.write((char*)&rows, sizeof(rows));
	fs.write((char*)&cols, sizeof(rows));
	fs.write((char*)mat.ptr(), sizeof(T) * mat.total());
}

/**
 * @brief read_fs2
 * read matrix with info about rows and cols
 * @param fs
 * @param mat
 */
template< typename T >
void read_fs2(std::fstream& fs, ct::Mat_<T>& mat)
{
	int rows, cols;
	fs.read((char*)&rows, sizeof(rows));
	fs.read((char*)&cols, sizeof(rows));
	mat.setSize(rows, cols);
	fs.read((char*)mat.ptr(), sizeof(T) * mat.total());
}

////////////////////////////////

template< typename T >
void read_mat(const std::string& name, ct::Mat_<T>& mat)
{
	std::fstream fs;
	fs.open(name, std::ios_base::in);

	read_fs<T>(fs, mat);

	fs.close();
}

}

#endif // MATOPS_H
