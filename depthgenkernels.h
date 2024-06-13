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

#pragma once

//
// Kernels for processing hits and rays outside of OptiX
//

struct Ray
{
    float3 origin;
    float  tmin;
    float3 dir;
    float  tmax;
};

struct Hit
{
    float  t;
    float3 geom_normal;
	float  z_depth;
	float2 uv;
};

void createRaysOrthoOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding, 
	float& z0, float& z1, float& dx, float& dy, float& xmin, float& ymin, float& zmin, float& zmax);

void translateRaysOnDevice( Ray* rays_device, int count, float3 offset );

void shadeHitsOnDevice( float3* image_device, int count, const Hit* hits_device, float z0, float z1, float zmax);
void shadeHitsOnDevice2(uchar3* image_device, int width, int height, const Hit* hits_device,
	int texWidth, int texHeight, uchar3* texImgData_d);

