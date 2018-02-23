#ifndef OPTIM_MIXED_H
#define OPTIM_MIXED_H

#include "custom_types.h"
#include "nn.h"

namespace ct{

class GPU_EXPORTS AdamOptimizerMixed: public ct::AdamOptimizer<float>{
public:
	AdamOptimizerMixed();

	virtual void passI(const ct::Matf& gW, const ct::Matf& gB, ct::Matf& W, ct::Matf& B, int index);

	bool empty() const;
};

/////////////////////////////

class GPU_EXPORTS MomentumOptimizerMixed: public ct::MomentumOptimizer<float>{
public:
	MomentumOptimizerMixed();

	virtual void passI(const ct::Matf& gW, const ct::Matf& gB, ct::Matf& W, ct::Matf& B, int index);


protected:
};

}


#endif // OPTIM_MIXED_H
