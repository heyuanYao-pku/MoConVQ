#include <iostream>
#include <omp.h>
#include <chrono>

int parallel_flag = 1;

template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
void quat_multiply_single(
    const data_type * q1,
    const data_type * q2,
    data_type * q
)
{
    data_type x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
    data_type x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
    q[0] = + w1 * x2 - z1 * y2 + y1 * z2 + x1 * w2;
    q[1] = + z1 * x2 + w1 * y2 - x1 * z2 + y1 * w2;
    q[2] = - y1 * x2 + x1 * y2 + w1 * z2 + z1 * w2;
    q[3] = - x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2;
}

template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
void quat_multiply(
    const data_type * q1,
    const data_type * q2,
    data_type * q,
    size_t num_quat
)
{
    #pragma omp parallel for if (parallel_flag)
    for(long long i=0; i<num_quat; i++)
    {
        quat_multiply_single(q1 + 4 * i, q2 + 4 * i, q + 4 * i);
    }
}

int main()
{
    using namespace std;
    using namespace std::chrono;
    const int batch = 100000000;
    double * a = new double[batch * 4];
    double * b = new double[batch * 4];
    double * c = new double[batch * 4];
    #if _OPENMP
        std::cout << "support openmp " << std::endl;
    #else
        std::cout << "not support openmp" << std::endl;
    #endif

    auto t1 = steady_clock::now();
    quat_multiply(a, b, c, batch);
    auto t2 = steady_clock::now();
    std::cout << duration_cast<microseconds>(t2-t1).count() << std::endl;
    
    parallel_flag = 0;
    t1 = steady_clock::now();
    quat_multiply(a, b, c, batch);
    t2 = steady_clock::now();
    std::cout << duration_cast<microseconds>(t2-t1).count() << std::endl;

    return 0;
}