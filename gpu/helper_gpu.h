#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include "custom_types.h"
#include "gpumat.h"

#define PRINT_GMAT10(mat) {		\
	std::string s = mat.print(10);			\
	qDebug("%s\n", s.c_str());	\
}

namespace gpumat{

/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void GPU_EXPORTS convert_to_gpu(const ct::Matf& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void GPU_EXPORTS convert_to_gpu(const ct::Matd& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void GPU_EXPORTS convert_to_mat(const gpumat::GpuMat& gmat, ct::Matf& mat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void GPU_EXPORTS convert_to_mat(const gpumat::GpuMat& gmat, ct::Matd& mat);

/**
 * @brief write_fs
 * write to fstream
 * @param fs
 * @param mat
 */
void GPU_EXPORTS write_fs(std::fstream &fs, const GpuMat &mat);

/**
 * @brief write_fs2
 * @param fs
 * @param mat
 */
void GPU_EXPORTS write_fs2(std::fstream &fs, const GpuMat &mat);

/**
 * @brief write_gmat
 * @param name
 * @param mat
 */
void GPU_EXPORTS write_gmat(const std::string &name, const GpuMat &mat);

/**
 * @brief read_fs
 * read from fstream
 * @param fs
 * @param mat
 */
void GPU_EXPORTS read_fs(std::fstream &fs, gpumat::GpuMat& mat);

/**
 * @brief read_fs2
 * @param fs
 * @param mat
 */
void GPU_EXPORTS read_fs2(std::fstream &fs, gpumat::GpuMat& mat, int type = GPU_FLOAT);

/////////////////////////////////////////

class GPU_EXPORTS Optimizer{
public:
	Optimizer();
	virtual ~Optimizer();

	double alpha()const;

	void setAlpha(double v);

	uint32_t iteration() const;

    virtual void initSize(int size);
    virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB) = 0;
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b) = 0;
	virtual void initI(const GpuMat &W, const GpuMat &B, int index) = 0;
	virtual void passI(const GpuMat &gW, const GpuMat &gB, gpumat::GpuMat& W, gpumat::GpuMat& B, int index) = 0;

protected:
	uint32_t m_iteration;
	double m_alpha;

private:
};

//////////////////////////////////////////

class GPU_EXPORTS StohasticGradientOptimizer: public Optimizer{
public:
	StohasticGradientOptimizer();

    void initSize(int size);
    virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);
	void initI(const GpuMat &W, const GpuMat &B, int index);
	void passI(const GpuMat &gW, const GpuMat &gB, gpumat::GpuMat& W, gpumat::GpuMat& B, int index);

private:

};

//////////////////////////////////////////

class GPU_EXPORTS MomentumOptimizer: public Optimizer{
public:
	MomentumOptimizer();

	double betha() const;
	void setBetha(double b);

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& B);

    void initSize(int size);
    void initI(const GpuMat &W, const GpuMat &B, int index);
	void passI(const GpuMat &gW, const GpuMat &gB, gpumat::GpuMat& W, gpumat::GpuMat& B, int index);

protected:
	double m_betha;
	std::vector< gpumat::GpuMat > m_mW;
	std::vector< gpumat::GpuMat > m_mb;
};

/////////////////////////////////////////

class GPU_EXPORTS AdamOptimizer: public Optimizer{
public:
	AdamOptimizer();

	double betha1() const;

	void setBetha1(double v);

	double betha2() const;

	void setBetha2(double v);

	/**
	 * @brief setDelimiterIteration
	 * set use for iter = m_iteration / val
	 * @param val
	 */
	void setDelimiterIteration(double val);

	bool empty() const;

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);

    void initSize(int size);
	void initI(const GpuMat &W, const GpuMat &B, int index);
	void passI(const GpuMat &gW, const GpuMat &gB, gpumat::GpuMat& W, gpumat::GpuMat& B, int index);

protected:
	double m_betha1;
	double m_betha2;
	bool m_init_matB;
	double m_sb1;
	double m_sb2;
	double m_delim_iter;

	std::vector< gpumat::GpuMat > m_mW;
	std::vector< gpumat::GpuMat > m_mb;
	std::vector< gpumat::GpuMat > m_vW;
	std::vector< gpumat::GpuMat > m_vb;

	void next_iteration();
	void init_iteration();
};

class GPU_EXPORTS AdaGradOptimizer: public Optimizer{
public:
    AdaGradOptimizer();

    double betha() const;
    void setBetha(double b);

    virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
    virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
              std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& B);

    void initSize(int size);
    void initI(const GpuMat &W, const GpuMat &B, int index);
    void passI(const GpuMat &gW, const GpuMat &gB, gpumat::GpuMat& W, gpumat::GpuMat& B, int index);

protected:
    double m_betha;
    std::vector< gpumat::GpuMat > m_histW;
    std::vector< gpumat::GpuMat > m_histB;
};

/////////////////////////////////////////////

class GPU_EXPORTS SimpleAutoencoder
{
public:

	typedef void (*tfunc)(const GpuMat& _in, GpuMat& _out);

	SimpleAutoencoder();

	double m_alpha;
	int m_neurons;

	std::vector<GpuMat> W;
	std::vector<GpuMat> b;
	std::vector<GpuMat> dW;
	std::vector<GpuMat> db;

	tfunc func;
	tfunc deriv;

	void init(GpuMat& _W, GpuMat& _b, int samples, int neurons, tfunc fn, tfunc dfn);

	void pass(const GpuMat& X);
	double l2(const GpuMat& X);
private:
	AdamOptimizer adam;
	GpuMat a[3], tw1;
	GpuMat z[2], d, di, sz;
};

/**
 * @brief save_gmat
 * @param mat
 * @param fn
 */
void GPU_EXPORTS save_gmat(const GpuMat &mat, const std::string &fn);
/**
 * @brief save_gmat10
 * @param mat
 * @param fn
 */
void GPU_EXPORTS save_gmat10(const GpuMat& mat, const std::string& fn);

/**
 * @brief cnv2gpu
 * @param M
 * @param G
 */
void GPU_EXPORTS cnv2gpu(const std::vector<ct::Matf> &M, std::vector< gpumat::GpuMat > &G);
void GPU_EXPORTS cnv2gpu(const std::vector<ct::Matd> &M, std::vector< gpumat::GpuMat > &G);

/**
 * @brief cnv2mat
 * @param G
 * @param M
 */
void GPU_EXPORTS cnv2mat(const std::vector< gpumat::GpuMat > &G, std::vector<ct::Matf> &M);
void GPU_EXPORTS cnv2mat(const std::vector< gpumat::GpuMat > &G, std::vector<ct::Matd> &M);

}

#endif // HELPER_GPU_H
