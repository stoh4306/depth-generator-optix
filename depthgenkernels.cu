//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cuda_runtime.h>

#include "depthgenkernels.h"

#include <sutil/vec_math.h>
#include <iostream>


inline int idivCeil( int x, int y )
{
    return ( x + y - 1 ) / y;
}


__global__ void createRaysOrthoKernel( Ray* rays, int width, int height, float x0, float y0, float z, float dx, float dy )
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if( rayx >= width || rayy >= height )
        return;

    const int idx    = rayx + rayy * width;
    rays[idx].origin = make_float3( x0 + rayx * dx, y0 + rayy * dy, z );
    rays[idx].tmin   = 0.0f;
    rays[idx].dir    = make_float3( 0, 0, -1 );
    rays[idx].tmax   = 1e34f;
}


// Note: uses left handed coordinate system
void createRaysOrthoOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, 
	float padding,	float& z0, float& z1, 
	float& dx, float& dy, float& xmin, float& ymin, float& zmin, float& zmax )
{
	std::cout << "- Object bounding box : " << "\n"
		<< " . x : [ " << bbmin.x << ", " << bbmax.x << "\n"
		<< " . y : [ " << bbmin.y << ", " << bbmax.y << "\n"
		<< " . z : [ " << bbmin.z << ", " << bbmax.z << std::endl;

    const float3 bbspan = bbmax - bbmin;
	dx = bbspan.x * (1 + 2 * padding) / width;
	dy = bbspan.y * (1 + 2 * padding) / height;
    float        x0     = bbmin.x - bbspan.x * padding + dx / 2;
    float        y0     = bbmin.y - bbspan.y * padding + dy / 2;

	zmin = bbmin.z-fmaxf(bbspan.z, 1.0f) * .1f;
	zmax = bbmax.z+fmaxf(bbspan.z, 1.0f) * .1f;

	std::cout << " . extended z-bound : " << zmin << " ~ " << zmax << std::endl;

    //float        z      = zmin;
	//float  z = zmax;

	if (zmin - z0 < 0.0f)
	{
		printf("- Warning, the input image plane(z=%f) intersects with the object bounding box zmin=%f, zmax=%f\n", z0, zmin, zmax);
		z0 = zmin - 0.5*(zmax - zmin);
		printf(" -> The image plane will be automatically changed to z=%f.\n", z0);
	}

	z1 = zmax + (zmin - z0);

	std::cout << "- Ray generation plane : z1=" << z1 << std::endl;

	//z_img = z;
	xmin = x0;
	ymin = y0;

    dim3 blockSize( 32, 16 );
    dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
    createRaysOrthoKernel<<<gridSize, blockSize>>>( rays_device, width, height, x0, y0, z1, dx, dy );
}


__global__ void translateRaysKernel( Ray* rays, int count, float3 offset )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx >= count )
        return;

    rays[idx].origin = rays[idx].origin + offset;
}


void translateRaysOnDevice( Ray* rays_device, int count, float3 offset )
{
    const int blockSize  = 512;
    const int blockCount = idivCeil( count, blockSize );
    translateRaysKernel<<<blockCount, blockSize>>>( rays_device, count, offset );
}


__global__ void shadeHitsKernel( float3* image, int count, const Hit* hits, float z0, float z1, float zmax)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx >= count )
        return;

    const float3 backgroundColor = make_float3( 0.0f, 0.0f, 0.0f );
    if( hits[idx].t < 0.0f )
    {
        image[idx] = backgroundColor;
    }
    else
    {
        //image[idx] = 0.5f * hits[idx].geom_normal + make_float3( 0.5f, 0.5f, 0.5f );
		//image[idx] = make_float3(1.0f, 0.0f, 0.0f);

		image[idx] = make_float3((z1 - z0 - hits[idx].z_depth) / (zmax - z0), 
								 (z1 - z0 - hits[idx].z_depth) / (zmax - z0), 
								 (z1 - z0 - hits[idx].z_depth) / (zmax - z0));
    }
}


void shadeHitsOnDevice( float3* image_device, int count, const Hit* hits_device, float z0, float z1, float zmax)
{
    const int blockSize  = 512;
    const int blockCount = idivCeil( count, blockSize );
    shadeHitsKernel<<<blockCount, blockSize>>>( image_device, count, hits_device, z0, z1, zmax );
}

__global__ void shadeHitsKernel2(uchar3* image, int w, int h, int count, const Hit* hits, int tw, int th, uchar3* texData)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= count)
		return;

	//const float3 backgroundColor = make_float3(0.0f, 0.0f, 0.0f);
	const uchar3 backgroundColor = make_uchar3(0, 0, 0);

	int i = idx % w;
	int j = idx / w;

	int pid = (h-1-j) * w + i;

	if (hits[idx].t < 0.0f)
	{
		image[pid] = backgroundColor;
	}
	else
	{
		//image[idx] = 0.5f * hits[idx].geom_normal + make_float3( 0.5f, 0.5f, 0.5f );
		//image[idx] = make_float3(1.0f, 0.0f, 0.0f);

		int ti = (int)(hits[idx].uv.x * tw);
		int tj = (int)(hits[idx].uv.y * th);
		uchar3 c = texData[tj*tw + ti];

		//image[idx] = make_float3(c.z / 255.f, c.y / 255.f, c.x / 255.f);
		image[pid] = make_uchar3(c.x, c.y, c.z);
	}
}

void shadeHitsOnDevice2(uchar3* image_device, int w, int h, const Hit* hits_device, 
	int texWidth, int texHeight, uchar3* texImgData_d)
{
	const int blockSize = 512;
	const int blockCount = idivCeil(w*h, blockSize);
	shadeHitsKernel2 << <blockCount, blockSize >> > (image_device, w, h, w*h, hits_device, texWidth, texHeight, texImgData_d);
}
