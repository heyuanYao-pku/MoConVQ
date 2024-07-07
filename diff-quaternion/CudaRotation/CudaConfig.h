#pragma once
namespace CudaConfig
{
    void init();
    void print_info();
    void get_thread_block(int n, int & num_grid_, int & num_threads_, int device_id);
}