// config.h
#ifndef CONFIG_H
#define CONFIG_H

// Optimization Techniques (1: Enable, 0: Disable)
#define ENABLE_MULTI_THREADING 1
#define ENABLE_SIMD 1
#define ENABLE_CACHE_MISS_MINIMIZATION 1

// Number of Threads
#define THREAD_NUM 4

// Matrix Sizes
#define MATRIX_SIZE 1000

// Matrix Sparsity
#define SPARSITY 0.01

//1: DENSE_DENSE 2: DENSE_SPARSE 3: SPARSE_SPARSE
#define TYPE  1

#endif