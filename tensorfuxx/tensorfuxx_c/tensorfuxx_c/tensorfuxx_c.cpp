#include "stdafx.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>


#define DLLEXPORT extern "C" __declspec(dllexport)


inline void im2col(float *ans,float *origin,const int &o_h, const int &o_w, const int &i_c,const int & f_h, const int &z_fh_size, const int &cs, const int &ms)
{
	float *img = ans;
	float *origin_h = origin;
	for (int i_h = 0; i_h < o_h; i_h++, origin_h += z_fh_size)
	{
		float *origin_h_w = origin_h;
		for (int i_w = 0; i_w < o_w; i_w++, origin_h_w += i_c)
		{
			float *origin_h_w_h2 = origin_h_w;
			for (int i_h2 = 0; i_h2 < f_h; i_h2++, origin_h_w_h2 += z_fh_size)
			{
				memcpy(img, origin_h_w_h2, cs);
				img += ms;
			}
		}
	}
}

DLLEXPORT
int im2col_Conv2D(float* z, int batchs, int z_h, int z_w, int i_c, int h_step, int w_step,
	float* f, int f_h, int f_w, int o_c,
	float* o, int o_h, int o_w)
{
	int o_b_size = o_h * o_w * o_c;
	int o_oh_size = o_w * o_c;
	int f_fh_size = f_w * i_c * o_c;
	int f_fw_size = i_c * o_c;
	int z_b_size = z_h * z_w * i_c;
	int z_oh_size = h_step * z_w*i_c;
	int z_ow_size = w_step * i_c;
	int z_fh_size = z_w * i_c;
	int dim1 = o_h * o_w;
	int dim2 = f_h * f_w * i_c;
	int dim3 = o_c;
	float *ans = new float[dim1 * dim2];
	int ms = f_w * i_c;
	int cs = ms * sizeof(float);

	float *zz = z;
	float *oo = o;

	for (int i = 0; i < batchs; i++, zz += z_b_size, oo += o_b_size) // for each patch
	{
		im2col(ans, zz, o_h, o_w, i_c, f_h, z_fh_size, cs, ms);
		//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim1, dim3, dim2, 1.0, ans, dim2, f, dim3, 0.0, oo, dim3);
	}
	delete[] ans;
	return 0;
}


DLLEXPORT
int correlate2d(
	float* z,
	int batchs,
	int z_h,
	int z_w,
	int i_c,
	int h_step,
	int w_step,
	float* f,
	int f_h,
	int f_w,
	int o_c,
	float* o,
	int o_h,
	int o_w
)
{
	/*
	enumarate dims
	b,oh,ow,ih,iw,ic,oc,
	*/
	int o_b_size = o_h * o_w*o_c;
	int o_oh_size = o_w * o_c;
	int f_fh_size = f_w * i_c*o_c;
	int f_fw_size = i_c * o_c;
	int z_b_size = z_h * z_w*i_c;
	int z_oh_size = h_step * z_w*i_c;
	int z_ow_size = w_step * i_c;
	int z_fh_size = z_w * i_c;

	//std::cout << "Receive" << std::endl;
	float* o_b = o;
	float* z_b = z;
	for (int b = 0; b < batchs; ++b)
	{
		float* o_oh = o_b;
		float* z_oh = z_b;
		for (int oh = 0; oh < o_h; ++oh)
		{

			float* o_ow = o_oh;
			float* z_ow = z_oh;
			for (int ow = 0; ow < o_w; ++ow)
			{

				float* f_fh = f;
				float* z_fh = z_ow;
				for (int fh = 0; fh < f_h; ++fh)
				{
					float* f_fw = f_fh;
					float* z_fw = z_fh;
					for (int fw = 0; fw < f_w; ++fw)
					{

						float* f_ic = f_fw;
						for (int ic = 0; ic < i_c; ++ic)
						{
							for (int oc = 0; oc < o_c; ++oc)
							{
								// o[b][oh][ow][oc] += z[b][ih][iw][ic] * f[fh][fw][ic][oc]
								o_ow[oc] += z_fw[ic] * f_ic[oc];
								// std::cout << (o_ow - o) + oc << '=' << (z_fw - z) + ic << '+' << (f_ic - f) + oc << std::endl;
							}
							f_ic += o_c;
						}

						f_fw += f_fw_size;
						z_fw += i_c;
					}

					f_fh += f_fh_size;
					z_fh += z_fh_size;
				}

				o_ow += o_c;
				z_ow += z_ow_size;
			}
			o_oh += o_oh_size;
			z_oh += z_oh_size;
		}
		o_b += o_b_size;
		z_b += z_b_size;
	}
	//std::cout << "Receive" << std::endl;
	return 0;
}


DLLEXPORT
int conv2d_filter_gradient(
	float* z,
	int batchs,
	int z_h,
	int z_w,
	int i_c,
	float* f,
	int f_h,
	int f_w,
	int o_c,
	float* o,
	int o_h,
	int o_w
)
{
	int o_oh_size = o_w * i_c*o_c;
	int o_ow_size = i_c * o_c;
	int z_b_size = z_h * z_w*i_c;
	int z_oh_size = z_w * i_c;
	int z_fh_size = z_w * i_c;
	int f_b_size = f_h * f_w*o_c;
	int f_fh_size = f_w * o_c;

	float* z_b = z;
	float* f_b = f;
	for (int b = 0; b < batchs; ++b)
	{
		float* o_oh = o;
		float* z_oh = z_b;
		for (int oh = 0; oh < o_h; ++oh)
		{
			float* o_ow = o_oh;
			float* z_ow = z_oh;
			for (int ow = 0; ow < o_w; ++ow)
			{
				float* z_fh = z_ow;
				float* f_fh = f_b;
				for (int fh = 0; fh < f_h; ++fh)
				{
					float* z_fw = z_fh;
					float* f_fw = f_fh;
					for (int fw = 0; fw < f_w; ++fw)
					{
						float* o_ic = o_ow;
						for (int ic = 0; ic < i_c; ++ic)
						{
							for (int oc = 0; oc < o_c; ++oc)
							{
								o_ic[oc] += z_fw[ic] * f_fw[oc];
							}
							o_ic += o_c;
						}
						z_fw += i_c;
						f_fw += o_c;
					}
					z_fh += z_fh_size;
					f_fh += f_fh_size;
				}
				o_ow += o_ow_size;
				z_ow += i_c;
			}
			o_oh += o_oh_size;
			z_oh += z_oh_size;
		}
		z_b += z_b_size;
		f_b += f_b_size;
	}
	return 0;
}


DLLEXPORT
int max_pool_gradient(float* g,int batchs,int g_h,int g_w,int i_c,
	float* o,int h_step,int w_step,
	float* z,int z_h,int z_w)
{
	int g_b_size = g_h * g_w*i_c;
	int g_gh_size = g_w * i_c;
	int z_b_size = z_h * z_w*i_c;
	int z_gh_size = h_step * z_w*i_c;
	int z_gw_size = w_step * i_c;
	int z_h_size = z_w * i_c;
	for (int b = 0; b < batchs; ++b)
	{
		float* g_b = g + b * g_b_size;
		float* z_b = z + b * z_b_size;
		for (int gh = 0; gh < g_h; ++gh)
		{
			float* g_gh = g_b + gh * g_gh_size;
			float* z_gh = z_b + gh * z_gh_size;
			for (int gw = 0; gw < g_w; ++gw)
			{
				float* g_gw = g_gh + gw * i_c;
				float* z_gw = z_gh + gw * z_gw_size;
				for (int ic = 0; ic < i_c; ++ic)
				{
					float* z_ic = z_gw + ic;
					float* max_loc = z_ic;
					for (int h = 0; h < h_step; ++h)
					{
						float* z_h = z_ic + h * z_h_size;
						for (int w = 0; w < w_step; ++w)
						{
							float* z_w = z_h + w * i_c;
							if ((*z_w) > (*max_loc))
							{
								max_loc = z_w;
							}
						}
					}
					o[max_loc - z] += g_gw[ic];
				}
			}
		}
	}
	return 0;
}
