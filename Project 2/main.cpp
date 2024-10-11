// main.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <immintrin.h>
#include "config.h"

// Define Matrix Types
enum MatrixType {
    DENSE_DENSE,
    DENSE_SPARSE,
    SPARSE_SPARSE
};

// Sparse Matrix in CSR Format
struct SparseMatrix {
    int rows;
    int cols;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;

    SparseMatrix(int r, int c, double sparsity) : rows(r), cols(c) {
        row_ptr.reserve(rows + 1);
        row_ptr.push_back(0);
        std::default_random_engine eng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (dist(eng) < sparsity) {
                    values.push_back(dist(eng));
                    col_indices.push_back(j);
                }
            }
            row_ptr.push_back(values.size());
        }
    }
};

// Dense Matrix
struct DenseMatrix {
    int rows;
    int cols;
    std::vector<double> data;

    DenseMatrix(int r, int c) : rows(r), cols(c), data(r* c, 0.0) {}
};

// Function to generate Dense Matrix with random values
DenseMatrix generate_dense(int rows, int cols) {
    DenseMatrix mat(rows, cols);
    std::default_random_engine eng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& val : mat.data) {
        val = dist(eng);
    }
    return mat;
}

// Function to transpose a Dense Matrix
DenseMatrix transpose_dense(const DenseMatrix& A) {
    DenseMatrix At(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            At.data[j * A.rows + i] = A.data[i * A.cols + j];
        }
    }
    return At;
}

// Matrix Multiplication Functions

// Dense x Dense
DenseMatrix multiply_dense_dense(const DenseMatrix& A, const DenseMatrix& B) {
    DenseMatrix C(A.rows, B.cols);
    DenseMatrix Bt = B; // Default copy

    if (ENABLE_CACHE_MISS_MINIMIZATION) {
        Bt = transpose_dense(B);
    }

    auto multiply = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                double sum = 0.0;
#if ENABLE_SIMD
                __m256d vsum = _mm256_setzero_pd();
                int k = 0;
                for (; k + 3 < A.cols; k += 4) {
                    __m256d va = _mm256_loadu_pd(&A.data[i * A.cols + k]);
                    __m256d vb = ENABLE_CACHE_MISS_MINIMIZATION ?
                        _mm256_loadu_pd(&Bt.data[j * Bt.cols + k]) :
                        _mm256_loadu_pd(&B.data[k * B.cols + j]);
                    __m256d vmul = _mm256_mul_pd(va, vb);
                    vsum = _mm256_add_pd(vsum, vmul);
                }
                // Horizontal addition of vsum
                double temp[4];
                _mm256_storeu_pd(temp, vsum);
                sum += temp[0] + temp[1] + temp[2] + temp[3];
                // Handle remaining elements
                for (; k < A.cols; ++k) {
                    double a_val = A.data[i * A.cols + k];
                    double b_val = ENABLE_CACHE_MISS_MINIMIZATION ?
                        Bt.data[j * Bt.cols + k] :
                        B.data[k * B.cols + j];
                    sum += a_val * b_val;
                }
#else
                for (int k = 0; k < A.cols; ++k) {
                    double a_val = A.data[i * A.cols + k];
                    double b_val = ENABLE_CACHE_MISS_MINIMIZATION ?
                        Bt.data[j * Bt.cols + k] :
                        B.data[k * B.cols + j];
                    sum += a_val * b_val;
                }
#endif
                C.data[i * C.cols + j] = sum;
            }
        }
        };

    if (ENABLE_MULTI_THREADING) {
        std::vector<std::thread> threads;
        int chunk = A.rows / THREAD_NUM;
        for (int t = 0; t < THREAD_NUM; ++t) {
            int start = t * chunk;
            int end = (t == THREAD_NUM - 1) ? A.rows : start + chunk;
            threads.emplace_back(multiply, start, end);
        }
        for (auto& th : threads) th.join();
    }
    else {
        multiply(0, A.rows);
    }
    return C;
}

// Dense x Sparse
DenseMatrix multiply_dense_sparse(const DenseMatrix& A, const SparseMatrix& B) {
    DenseMatrix C(A.rows, B.cols);
    auto multiply = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int k = 0; k < A.cols; ++k) {
                double a_val = A.data[i * A.cols + k];
                for (int idx = B.row_ptr[k]; idx < B.row_ptr[k + 1]; ++idx) {
                    int j = B.col_indices[idx];
                    C.data[i * B.cols + j] += a_val * B.values[idx];
                }
            }
        }
        };

    if (ENABLE_MULTI_THREADING) {
        std::vector<std::thread> threads;
        int chunk = A.rows / THREAD_NUM;
        for (int t = 0; t < THREAD_NUM; ++t) {
            int start = t * chunk;
            int end = (t == THREAD_NUM - 1) ? A.rows : start + chunk;
            threads.emplace_back(multiply, start, end);
        }
        for (auto& th : threads) th.join();
    }
    else {
        multiply(0, A.rows);
    }
    return C;
}

// Sparse x Sparse
SparseMatrix multiply_sparse_sparse(const SparseMatrix& A, const SparseMatrix& B) {
    // Simple implementation (not optimized)
    // For demonstration purposes
    SparseMatrix C(A.rows, B.cols, 0.0);
    C.row_ptr.reserve(A.rows + 1);
    C.row_ptr.push_back(0);
    for (int i = 0; i < A.rows; ++i) {
        std::vector<double> temp(C.cols, 0.0);
        for (int idxA = A.row_ptr[i]; idxA < A.row_ptr[i + 1]; ++idxA) {
            int k = A.col_indices[idxA];
            double a_val = A.values[idxA];
            for (int idxB = B.row_ptr[k]; idxB < B.row_ptr[k + 1]; ++idxB) {
                int j = B.col_indices[idxB];
                temp[j] += a_val * B.values[idxB];
            }
        }
        // Convert temporary row to CSR format
        for (int j = 0; j < C.cols; ++j) {
            if (temp[j] != 0.0) {
                C.values.push_back(temp[j]);
                C.col_indices.push_back(j);
            }
        }
        C.row_ptr.push_back(C.values.size());
    }
    return C;
}

// Function to perform and time multiplication based on type
void perform_multiplication(int type, int size, double sparsity) {
    std::cout << "Multiplication Type: ";
    switch (type) {
    case 1: std::cout << "Dense x Dense"; break;
    case 2: std::cout << "Dense x Sparse"; break;
    case 3: std::cout << "Sparse x Sparse"; break;
    }
    std::cout << ", Size: " << size << "x" << size << ", Sparsity: " << sparsity * 100 << "%" << std::endl;

    // Generate Matrices
    DenseMatrix A = generate_dense(size, size);
    DenseMatrix B_dense = generate_dense(size, size);
    SparseMatrix B_sparse(size, size, sparsity);
    SparseMatrix C_sparse(size, size, sparsity); // For sparse x sparse multiplication

    // Perform Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    if (type == 1) {
        DenseMatrix C = multiply_dense_dense(A, B_dense);
    }
    else if (type == 2) {
        DenseMatrix C = multiply_dense_sparse(A, B_sparse);
    }
    else if (type == 3) {
        SparseMatrix C = multiply_sparse_sparse(B_sparse, C_sparse);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time Taken: " << diff.count() << " s\n" << std::endl;
}

int main() {

    perform_multiplication(TYPE, MATRIX_SIZE, SPARSITY);
  
    return 0;
}
