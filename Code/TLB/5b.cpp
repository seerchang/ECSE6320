// ProgramB.cpp
// Compile with: cl /O2 /EHsc ProgramB.cpp

#include <iostream>
#include <chrono>

int main() {
    const size_t N = 256 * 1024 * 1024; // Total number of integers (~1GB)
    const size_t PAGE_SIZE = 4096;      // Page size in bytes
    const size_t INT_SIZE = sizeof(int);
    const size_t INTS_PER_PAGE = PAGE_SIZE / INT_SIZE; // Number of integers per page
    const size_t STRIDE = INTS_PER_PAGE; // Stride to jump to the next page

    // Allocate a contiguous array
    int* data = new int[N];

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i);
    }

    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Strided access to increase TLB misses
    for (size_t i = 0; i < STRIDE; ++i) {
        for (size_t j = i; j < N; j += STRIDE) {
            sum += data[j] * 2;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Output results
    std::cout << "Sum: " << sum << std::endl;

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    delete[] data;

    return 0;
}
