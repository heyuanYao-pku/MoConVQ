#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "CudaConfig.h"

namespace CudaConfig
{
    int num_devices = 0;
    std::vector<int> max_threads_per_block;

    void init()
    {
        cudaGetDeviceCount(&num_devices);
        max_threads_per_block.resize(num_devices);
        for (int i = 0; i < num_devices; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            max_threads_per_block[i] = prop.maxThreadsPerBlock;
        }
    }

    int tableSizeFor(int cap)
    {
        int n = cap - 1;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        int ret = (n < 0) ? 1 : n + 1;
        return ret;
    }

    void get_thread_block(int n, int & num_grid_, int & num_threads_, int device_id)
    {
        n = tableSizeFor(n);
        num_threads_ = std::min(n, max_threads_per_block[device_id]);
        if (n % num_threads_ == 0)
        {
            num_grid_ = n / num_threads_;
        }
        else
        {
            num_grid_ = (n + num_threads_) / num_threads_;
        }
        // std::cout << "n = " << n << ", num grid = " << num_grid_ << ", " << "num threads = " << num_threads_ << std::endl;
    }

    void print_info()
    {
        // document from https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
        // print cuda information
        std::cout << "Total Devices: " << num_devices << std::endl;
        for (int i = 0; i < num_devices; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "Device Number: " << i << std::endl;
            std::cout << "  Device name: " << prop.name << std::endl;
            // std::cout << "  asyncEngineCount: " << prop.asyncEngineCount << std::endl;
            std::cout << "  concurrentKernels: " << prop.concurrentKernels << std::endl;
            std::cout << "  concurrentManagedAccess: " << prop.concurrentManagedAccess << std::endl;
            std::cout << "  cooperativeLaunch: " << prop.cooperativeLaunch << std::endl;
            // std::cout << "  canMapHostMemory: " << prop.canMapHostMemory << std::endl;
            std::cout << "  deviceOverlap: " << prop.deviceOverlap << std::endl;
            // std::cout << "  directManagedMemAccessFromHost: " << prop.directManagedMemAccessFromHost << std::endl;
            std::cout << "  major: " << prop.major << ", ";
            std::cout << "  minor: " << prop.minor << std::endl;
            // std::cout << "  managedMemory: " << prop.managedMemory << std::endl;
            std::cout << "  maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
            // std::cout << "  maxGridSize: " << prop.maxGridSize << std::endl;
            // std::cout << "  maxThreadsDim: " << prop.maxThreadsDim << std::endl;
            std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "  maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
            std::cout << "  multiProcessorCount: " << prop.multiProcessorCount << std::endl;
            // std::cout << "  warpSize: " << prop.warpSize << std::endl;
            std::cout << std::endl;
        }
    }
}