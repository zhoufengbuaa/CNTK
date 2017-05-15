#pragma warning(disable : 4505)
#include <cuda.h>
#include <math.h>
#define TENSOR_OPS_DECL __device__ __host__
#include "StrictTensorOps.h"

TENSOR_OPS_DECL float pow_(float f, float y, int)
{                                    
    return powf(f, y);
}

TENSOR_OPS_DECL double pow_(double f, double y, char)
{                                    
    return pow(f, y);
}
