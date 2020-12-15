#include <stdio.h>
#include <string.h>
#include <thread>
//#include <iostream>
#include "mkl.h"
#define max(x,y) (((x) > (y)) ? (x) : (y))

#define DLLEXPORT extern "C" //__declspec(dllexport)


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
int mat_mul(float *A,int an,int am,
	float *B,int bn,int bm,
	float *C,int n,int m,int tranA,int tranB)
{
	CBLAS_TRANSPOSE ta=CblasNoTrans,tb=CblasNoTrans;
	int k=am;
	if(tranA) ta=CblasTrans,k=an;
	if(tranB) tb=CblasTrans;
	cblas_sgemm(CblasRowMajor, ta, tb, n, m, k, 1.0, A, am, B, bm, 0.0, C, m);
	return 0;
}

DLLEXPORT
int im2col_Conv2D(float* z, int batchs, int z_h, int z_w, int i_c, int h_step, int w_step,
	float* f, int f_h, int f_w, int o_c,
	float* o, int o_h, int o_w)
{
	//#pragma comment(linker, "/STACK:10737418240,10737418240") 
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
	int a_size = dim1 * dim2;
	int cpsize=a_size*sizeof(float);
	float *ans = (float *)mkl_malloc(cpsize,sizeof(float));
	int ms = f_w * i_c;
	int cs = ms * sizeof(float);
	float *zz = z;
	float *oo = o;
	int nsize=batchs*sizeof(float**);
	
	/*float **ba=(float**)mkl_malloc(nsize,sizeof(float**));
	float **bf=(float**)mkl_malloc(nsize,sizeof(float**));
	float **bo=(float**)mkl_malloc(nsize,sizeof(float**));
	
	MKL_INT m[1];
	MKL_INT k[1];
	MKL_INT n[1];
	
	CBLAS_TRANSPOSE transA[1];
	CBLAS_TRANSPOSE transB[1];
	
	float alpha[1];
	float beta[1];
	MKL_INT gs[1];
	
	m[0]=dim1;
	k[0]=dim2;
	n[0]=dim3;
		
	transA[0]=CblasNoTrans;
	transB[0]=CblasNoTrans;
	
	alpha[0]=1.0;
	beta[0]=0.0;
	gs[0]=batchs;*/
	
	for (int i = 0; i < batchs; i++, zz += z_b_size, oo += o_b_size)
	{
		im2col(ans, zz, o_h, o_w, i_c, f_h, z_fh_size, cs, ms);
		//ba[i]=(float *)mkl_malloc(cpsize,sizeof(float));
		//memcpy(ba[i],ans,cpsize);
		//bf[i]=f;
		//bo[i]=oo;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim1, dim3, dim2, 1.0, ans, dim2, f, dim3, 0.0, oo, dim3);
	}
	//using namespace std;
	//cout<<"------------------"<<endl<<bo[0][0]<<endl;
	//cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, (const float**)ba, k, (const float**)bf, n, beta, bo, n, 1, gs);
	//cout<<"------------------"<<endl<<bo[0][0]<<endl;
	mkl_free(ans);
	/*for(int i = 0; i < batchs; i++)
	{
		mkl_free(ba[i]);
	}
	mkl_free(ba);
    mkl_free(bf);
    mkl_free(bo);*/
	return 0;
}

void Tconv2d_filter_gradient(float* z,int L,int R,int z_h,int z_w,int i_c,
	float* f,int f_h,int f_w,int o_c,
	float* o,int o_h,int o_w)
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
	for (int b = L; b < R; ++b)
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
}

DLLEXPORT
int conv2d_filter_gradient(float* z,int batchs,int z_h,int z_w,int i_c,
	float* f,int f_h,int f_w,int o_c,
	float* o,int o_h,int o_w)
{
	//puts("FUCK");
	int o_b_size = o_h * o_w*o_c;
	int o_ow_size = i_c * o_c;
	int z_b_size = z_h * z_w*i_c;
	int z_oh_size = z_w * i_c;
	int z_fh_size = z_w * i_c;
	int f_b_size = f_h * f_w*o_c;
	int f_fh_size = f_w * o_c;

	float* z_b = z;
	float* f_b = f;
	float* o_b = o;
	
	int L,R,st=batchs/10;
	L=0;R=st;
	
	std::thread *T[10];
	for(int i=0;i<10;i++)
	{
		T[i]=new std::thread(Tconv2d_filter_gradient,z_b,L,R,z_h,z_w,i_c,
		f_b,f_h,f_w,o_c,
		o,o_h,o_w);
		L+=st;
		R+=st;
		//using namespace std;
		//R=min(R,batchs);
		//cout<<L<<" "<<R<<" "<<batchs<<endl;
		if(i==8) R=batchs;
		if(i==9) continue;
		z_b+=st*z_b_size;
		f_b+=st*f_b_size;
		//o_b+=st*o_b_size;
	}
	for(int i=0;i<10;i++)
	{
		T[i]->join();
		delete T[i];
	}
	return 0;
}

DLLEXPORT
int max_pool(float* z,int batchs,int z_h,int z_w,int i_c,
	float* o,int o_h,int o_w,int h_step,int w_step,int up, int left)
{
	int o_b_size = o_h*o_w*i_c;
	int o_h_size = o_w*i_c;
	float* o_b=o;
	float* zp=z;
	for (int b=0;b<batchs;b++,o_b+=o_b_size)
	{
		for(int h=0;h<z_h;h++)
			for(int w=0;w<z_w;w++)
			{
				float* op=o_b+(h+up)/h_step*o_h_size+(w+left)/w_step*i_c;
				for(int c=0;c<i_c;c++)
				{
					*op=max(*op,*zp);
					zp++;
					op++;
				}
			}
	}
	return 0;
}

DLLEXPORT
int max_pool_gradient(float* g,int batchs,int g_h,int g_w,int i_c,
	float* o,int o_h,int o_w,int h_step,int w_step,
	float* z,int z_h,int z_w)
{
	//printf("%d %d\n",h_step,w_step);
	
	int g_b_size = g_h*g_w*i_c;
	int g_gh_size = g_w*i_c;
	int z_b_size = z_h*z_w*i_c;
	int z_gh_size = h_step*z_w*i_c;
	int z_gw_size = w_step*i_c;
	int z_h_size = z_w*i_c;
	float* g_b = g;
	float* z_b = z;
	for (int b = 0; b < batchs;b++,g_b+=g_b_size,z_b+=z_b_size)
	{
		float* g_gh = g_b;
		float* z_gh = z_b;
		for (int gh = 0; gh < g_h;gh++,g_gh+=g_gh_size,z_gh+=z_gh_size)
		{
			float* g_gw = g_gh;
			float* z_gw = z_gh;
			for (int gw = 0; gw < g_w;gw++,g_gw+=i_c,z_gw+=z_gw_size)
			{
				for (int ic = 0; ic < i_c; ic++) 
				{
					float* z_ic = z_gw + ic;
					float* max_loc = z_ic;

					float* z_w=z_ic;
					if ((*z_w) > (*max_loc)) 
					{
						max_loc = z_w;
					}
					z_w+=i_c;
					if ((*z_w) > (*max_loc)) 
					{
						max_loc = z_w;
					}
					z_ic+=z_h_size;
					if ((*z_ic) > (*max_loc)) 
					{
						max_loc = z_ic;
					}
					z_ic+=i_c;
					if ((*z_ic) > (*max_loc)) 
					{
						max_loc = z_ic;
					}
					
					/*for (int h = 0; h < h_step; ++h) {
						float* z_h = z_ic + h*z_h_size;

						for (int w = 0; w < w_step; ++w) {
							float* z_w = z_h + w*i_c;

							if ((*z_w) > (*max_loc)) {
								max_loc = z_w;
							}
						}
					}*/
					o[max_loc - z] += g_gw[ic];
				}
			}
		}
	}
	return 0;
}
