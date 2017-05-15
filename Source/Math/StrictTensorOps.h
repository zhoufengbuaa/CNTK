//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements strict elementwise tensor operations, compiled without fastmath.
//

#pragma once

#pragma push_macro("TENSOR_OPS_DECL")
#ifndef TENSOR_OPS_DECL // to make these accessible to CUDA kernels, say '#define TENSOR_OPS_DECL __device__ __host__'
#define TENSOR_OPS_DECL
#endif

#pragma push_macro("DECL")
#define DECL TENSOR_OPS_DECL

#pragma push_macro("OverloadBinaryMathFns")
#define OverloadBinaryMathFns(x)         \
    DECL float x##_(float f, float y, int=0);   \
    DECL double x##_(double f, double y, char='\0');

OverloadBinaryMathFns(pow);

#pragma pop_macro("OverloadBinaryMathFns")
#pragma pop_macro("DECL")
#pragma pop_macro("TENSOR_OPS_DECL")
