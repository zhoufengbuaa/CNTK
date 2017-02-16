//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReaderShim.h: Currently we are preserving the old interface in SGD. So this shim exposes the old interface and calls into the 
// reader implemented with the new interfaces (reader/packer/transforms/serializers)
//

#pragma once

#include <unordered_map>
#include <string>
#include <future>
#include "DataReader.h"
#include "Reader.h"

namespace CNTK
{
    class CompositeMinibatchSource;
}

namespace Microsoft { namespace MSR { namespace CNTK {

// The class uses a fixed thread pool to execute async works to avoid thread creation in std::async
template<size_t NumThreads>
class AsyncFixedThreadPool
{
private:
    std::deque<std::function<void()>> m_queue;
    std::mutex m_queueMutex;
    std::condition_variable m_wakeup;
    std::vector<std::thread> m_threads;
    std::atomic<bool> m_shouldExit;

    void ThreadProc()
    {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        while (!(m_shouldExit && m_queue.empty())) // note that when m_shouldExit sets, queued works would still be finished to avoid broken promise
        {
            if (m_queue.empty())
            {
                m_wakeup.wait(lock); // release mutex and wait on signal to wake up
            }
            else
            {
                auto work = m_queue.front();
                m_queue.pop_front();
                lock.unlock();
                work();
                lock.lock();
            }
        }
    }

public:
    AsyncFixedThreadPool() :
        m_shouldExit(false)
    {
        for (size_t i = 0; i < NumThreads; ++i)
        {
            m_threads.emplace_back(std::thread(&AsyncFixedThreadPool::ThreadProc, this));
        }
    }

    ~AsyncFixedThreadPool()
    {
        m_shouldExit = true;
        m_wakeup.notify_all();

        for (auto& t : m_threads) 
        {
            t.join();
        }
    }

    template <typename Function>
    auto async(Function&& f) -> std::future<decltype(f())>
    {
        using ReturnType = decltype(f());

        if (m_shouldExit)
        {
            RuntimeError("Cannot run more async works when exiting");
        }

        struct PromiseFunc
        {
            std::promise<ReturnType> promise;
            std::function<ReturnType()> function;
            PromiseFunc(std::function<ReturnType()>&& f) :
                promise(std::promise<ReturnType>()), function(f)
            {
            }
        };

        auto work = std::make_shared<PromiseFunc>(std::move(f));
        auto future = work->promise.get_future();

        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_queue.emplace_back(
                [work]()
                {
                    try {
                        work->promise.set_value(work->function());
                    }
                    catch (...) {
                        work->promise.set_exception(std::current_exception());
                    }
                });
        }
        m_wakeup.notify_one();

        return future;
    }
};

typedef ReaderPtr (*ReaderFactory)(const ConfigParameters& parameters);

template <class ElemType>
class ReaderShim : public IDataReader
{
    friend class ::CNTK::CompositeMinibatchSource;
private:
    ReaderShim();

public:
    explicit ReaderShim(ReaderFactory factory);
    explicit ReaderShim(ReaderPtr reader);

    virtual ~ReaderShim() { }

    virtual void Init(const ScriptableObjects::IConfigRecord& /*config*/) override
    {
        assert(false);
    }
    virtual void Init(const ConfigParameters& config) override;

    virtual void Destroy() override
    {
        // Make sure there are no outstanding reads.
        // Future destructor does not wait as of 2013 so probably it is not in VS2013:
        // More info can be found here http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3679.html.
        if (m_prefetchTask.valid())
        {
            // If there are some, give them time to finish.
            m_prefetchTask.wait_for(std::chrono::seconds(5));
            // TODO: if the prefetch is still valid, print a warning here!
        }

        delete this;
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>& inputs, size_t requestedEpochSamples = requestDataSize) override;
    virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, const std::unordered_set<InputStreamDescription>& inputs, size_t requestedEpochSamples) override;

    void StartEpoch(const EpochConfiguration& epoch, const std::unordered_set<InputStreamDescription>& inputs);

    virtual void StartMinibatchLoop(size_t, size_t, size_t) override
    {
        LogicError("Legacy StartMinibatchLoop is not implemented.");
    }

    virtual void StartDistributedMinibatchLoop(size_t, size_t, size_t, size_t, size_t) override
    {
        LogicError("Legacy StartDistributedMinibatchLoop is not implemented.");
    }

    virtual bool SupportsDistributedMBRead() const override
    {
        return true;
    }

    virtual bool IsLegacyReader() const override
    {
        return false;
    }

    virtual bool GetMinibatch(StreamMinibatchInputs& matrices) override;

    virtual bool DataEnd() override;

    void CopyMBLayoutTo(MBLayoutPtr) override;

    virtual size_t GetNumParallelSequencesForFixingBPTTMode() override;

    virtual size_t GetCurrentSamplePosition() override;

    void SetCurrentSamplePosition(size_t currentSamplePosition);

    void SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>& inputDescriptions);

    bool IsEndOfEpoch() const
    {
        return m_endOfEpoch;
    }

    bool IsEndOfSweep() const
    {
        return m_endOfSweep;
    }

private:
    struct PrefetchResult
    {
        bool m_isEndOfSweep;
        bool m_isEndOfEpoch;
        bool m_isDataAvailable;
    };

    PrefetchResult PrefetchMinibatch(size_t currentDataTransferIndex);

    std::future<PrefetchResult> m_prefetchTask;
    ReaderPtr m_reader;
    ReaderFactory m_factory;
    bool m_endOfEpoch;
    bool m_endOfSweep;

    size_t m_numParallelSequences;

    std::unordered_map<std::wstring, size_t> m_nameToStreamId;
    std::vector<StreamDescriptionPtr> m_streams;
    launch m_launchType;
    AsyncFixedThreadPool<2> m_asyncPrefetchThreadPool;

    inline void StartPrefetchTask()
    {
        // Starting the prefetch task. There is always a single async read in flight.
        // When the network requests a new minibatch, we wait for the current async to finish, swap the buffers
        // and kick off the new prefetch.
        auto localCurrentDataTransferIndex = m_currentDataTransferIndex;
        auto prefetchFunc = [this, localCurrentDataTransferIndex]() { return PrefetchMinibatch(localCurrentDataTransferIndex); };
        if (m_launchType == std::launch::async)
        {
            // use fixed thread pool for async prefetch to avoid thread creation which causes Philly perf drop
            m_prefetchTask = m_asyncPrefetchThreadPool.async(prefetchFunc);
        }
        else
        {
            m_prefetchTask = std::async(m_launchType, prefetchFunc);
        }
    }

    // Data structure required for prefetch.
    struct StreamPrefetchBuffer
    {
        std::shared_ptr<Matrix<ElemType>> m_matrix;
        MBLayoutPtr m_mbLayout;
    };

    // Intermediate buffer where the prefetch thread puts its data to.
    // When the main thread enters GetMinibatch it swaps the matrices from this buffer,
    // triggers the next prefetch and waits if memCpy is still in progress.
    std::unordered_map<std::wstring, StreamPrefetchBuffer> m_prefetchBuffers;

    // Alternating data transfer operations. In the current version these are only two - 
    // currently waiting on the main thread and the one that can be started by the prefetch thread 
    // in the meantime.
    std::vector<DataTransfererPtr> m_dataTransferers;

    // Current data transfer. Flips 0 and 1.
    // Can be changed only from the main thread with no ongoing prefetch.
    size_t m_currentDataTransferIndex; 

    // Device id.
    int m_deviceId;

    // Current sample position of the reader on the global timeline.
    // We have to remember the value locally before starting prefetch.
    // The value is updated only from the main thread (in StartEpoch/GetMinibatch)
    size_t m_currentSamplePosition;

    static void FillMatrixFromStream(
        StorageType type,
        Matrix<ElemType>* matrix,
        size_t numRows,
        const StreamMinibatchPtr& stream,
        DataTransferer* transferer);
};

}}}
