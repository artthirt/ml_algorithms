#include "optim_mixed.h"

#include "gpumat.h"
#include "helper_gpu.h"

using namespace ct;

/////////////////////////////
/////////////////////////////
///

AdamOptimizerMixed::AdamOptimizerMixed(): AdamOptimizer<float>()
{
}

void AdamOptimizerMixed::passI(const ct::Matf &gW, const ct::Matf &gB, ct::Matf &W, ct::Matf &B, int index)
{
	{
		gpumat::GpuMat g_mW, g_vW, g_W, g_gW;
		gpumat::convert_to_gpu(W, g_W);
		gpumat::convert_to_gpu(gW, g_gW);
		gpumat::convert_to_gpu(m_mW[index], g_mW);
		gpumat::convert_to_gpu(m_vW[index], g_vW);
		gpumat::sub_adamGrad(g_W, g_gW, g_mW, g_vW, ct::Optimizer<float>::m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
		gpumat::convert_to_mat(g_W, W);
		gpumat::convert_to_mat(g_mW, m_mW[index]);
		gpumat::convert_to_mat(g_vW, m_vW[index]);
	}

	{
		gpumat::GpuMat g_mB, g_vB, g_B, g_gB;
		gpumat::convert_to_gpu(B, g_B);
		gpumat::convert_to_gpu(gB, g_gB);
		gpumat::convert_to_gpu(m_mb[index], g_mB);
		gpumat::convert_to_gpu(m_vb[index], g_vB);
		gpumat::sub_adamGrad(g_B, g_gB, g_mB, g_vB, ct::Optimizer<float>::m_alpha, m_sb1, m_sb2, m_betha1, m_betha2);
		gpumat::convert_to_mat(g_B, B);
		gpumat::convert_to_mat(g_mB, m_mb[index]);
		gpumat::convert_to_mat(g_vB, m_vb[index]);
	}
//	ct::adamGrad(gW, m_mW[index], m_vW[index], W, m_sb1, m_sb2, ct::Optimizer<T>::m_alpha, m_betha1, m_betha2);
//	ct::adamGrad(gB, m_mb[index], m_vb[index], B, m_sb1, m_sb2, ct::Optimizer<T>::m_alpha, m_betha1, m_betha2);
}

bool AdamOptimizerMixed::empty() const
{
	return m_mW.empty() || m_mb.empty() || m_vW.empty() || m_vb.empty();
}

//////////////////////////////////////

MomentumOptimizerMixed::MomentumOptimizerMixed(): MomentumOptimizer<float>()
{
}

void MomentumOptimizerMixed::passI(const ct::Matf &gW, const ct::Matf &gB, ct::Matf &W, ct::Matf &B, int index)
{
	{
		gpumat::GpuMat g_mW, g_W, g_gW;
		gpumat::convert_to_gpu(W, g_W);
		gpumat::convert_to_gpu(gW, g_gW);
		gpumat::convert_to_gpu(m_mW[index], g_mW);
		gpumat::momentum_optimizer(g_W, g_mW, g_gW, ct::Optimizer<float>::m_alpha, m_betha);
		gpumat::convert_to_mat(g_W, W);
		gpumat::convert_to_mat(g_mW, m_mW[index]);
	}

	{
		gpumat::GpuMat g_mB, g_B, g_gB;
		gpumat::convert_to_gpu(B, g_B);
		gpumat::convert_to_gpu(gB, g_gB);
		gpumat::convert_to_gpu(m_mb[index], g_mB);
		gpumat::momentum_optimizer(g_B, g_mB, g_gB, ct::Optimizer<float>::m_alpha, m_betha);
		gpumat::convert_to_mat(g_B, B);
		gpumat::convert_to_mat(g_mB, m_mb[index]);
	}
//	ct::momentumGrad(gW, m_mW[index], W, Optimizer<T>::m_alpha, m_betha);
//	ct::momentumGrad(gB, m_mb[index], B, Optimizer<T>::m_alpha, m_betha);
}
