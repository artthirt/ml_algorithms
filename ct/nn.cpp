#include "nn.h"

namespace ct{

void get_cnv_sizes(const ct::Size sizeIn, const ct::Size szW, int stride, ct::Size &szA1, ct::Size &szA2)
{
	get_cnv_sizes(sizeIn, szW, stride, szA1);

	szA2 = ct::Size((szA1.width + 1)/2, (szA1.height + 1)/2);
}

void get_cnv_sizes(const Size sizeIn, const Size szW, int stride, Size &szA1)
{
	int dw = sizeIn.width - szW.width;
	int dh = sizeIn.height - szW.height;
	szA1.width		= (dw) / stride + 1 + (dw % 2);
	szA1.height		= (dh) / stride + 1 + (dh % 2);
}

void get_cnv_size_same(const Size szA0, int stride, Size &szA1)
{
	szA1.width = (szA0.width)/stride + (szA0.width % 2);
	szA1.height = (szA0.height)/stride + (szA0.height % 2);

}

}
