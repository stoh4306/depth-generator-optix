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

#include <optix.h>

#include "depthgen.h"
#include "depthgenkernels.h"

#include "cuda/LocalGeometry.h"
#include "cuda/whitted.h"

#include <sutil/vec_math.h>

#include <stdint.h>


extern "C" {
__constant__ Params params;
}


extern "C" __global__ void __raygen__from_buffer()
{
    const uint3    idx        = optixGetLaunchIndex();
    const uint3    dim        = optixGetLaunchDimensions();
    const uint32_t linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    uint32_t t, nx, ny, nz, z_depth, u, v;
    Ray      ray = params.rays[linear_idx];
    optixTrace( params.handle, ray.origin, ray.dir, ray.tmin, ray.tmax, 0.0f, OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, t, nx, ny, nz, z_depth, u, v );

	//if (int_as_float(t) < 0.0f)
	//	printf("linear_idx=%d, t=%d(%f), normal=(%f, %f, %f)", linear_idx,
	//		t, int_as_float(t), int_as_float(nx), int_as_float(ny), int_as_float(nz));

    Hit hit;
    hit.t                   = int_as_float( t );
    hit.geom_normal.x       = int_as_float( nx );
    hit.geom_normal.y       = int_as_float( ny );
    hit.geom_normal.z       = int_as_float( nz );
	hit.z_depth				= int_as_float( z_depth );
	hit.uv.x				= int_as_float( u );
	hit.uv.y				= int_as_float( v );

    params.hits[linear_idx] = hit;
}


extern "C" __global__ void __miss__buffer_miss()
{
    optixSetPayload_0( float_as_int( -1.0f ) );
    optixSetPayload_1( float_as_int( 1.0f ) );
    optixSetPayload_2( float_as_int( 0.0f ) );
    optixSetPayload_3( float_as_int( 0.0f ) );
	optixSetPayload_4( float_as_int( 1e34f ) );
	optixSetPayload_5( float_as_int(-1.0f) );
	optixSetPayload_6( float_as_int(-1.0f) );
}


extern "C" __global__ void __closesthit__buffer_hit()
{
    const uint32_t t = optixGetRayTmax();

    whitted::HitGroupData* rt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    LocalGeometry          geom    = getLocalGeometry( rt_data->geometry_data );

    // Set the hit data
    optixSetPayload_0( float_as_int( t ) );
    optixSetPayload_1( float_as_int( geom.N.x ) );
    optixSetPayload_2( float_as_int( geom.N.y ) );
    optixSetPayload_3( float_as_int( geom.N.z ) );

	const uint3    idx = optixGetLaunchIndex();
	const uint3    dim = optixGetLaunchDimensions();
	const uint32_t linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

	Ray      ray = params.rays[linear_idx];

	float z_depth = geom.P.z - ray.origin.z;
	if (z_depth < 0.0) z_depth = -z_depth;
	optixSetPayload_4( float_as_int( z_depth ) );

	// texture coordinates (u,v)
	optixSetPayload_5( float_as_int( geom.UV.x ) );
    optixSetPayload_6( float_as_int( geom.UV.y ) );

	//if (t != linear_idx)
	//{
	//	printf("(t, id)=(%d, %d)\n", t, linear_idx);
	//}

	//if (z_depth < 0.0)
	//{
	//	printf("z-depth[%d]=%f\n", linear_idx, z_depth);
	//}
}

