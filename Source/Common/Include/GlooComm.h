#pragma once

#include "Matrix.h"
#include "MPIWrapper.h"

class GlooComm
{
#ifdef USE_GLOO
private:
    void AllReduceImpl(void *inputbuffer, void *outputbuffer, size_t count);
    void BroadcastImpl(void *buffer, size_t count, int root);
    glooComm_t m_glooComm;
#endif

public:
    GlooComm(int deviceId, const MPIWrapperPtr& mpiComm);
    ~GlooComm();
    
    template <typename ElemType>
    void AllReduce(ElemType* inputBuffer)
}
