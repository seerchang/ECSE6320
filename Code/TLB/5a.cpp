#include <iostream>
#include <chrono>

int main() {
    const size_t N = 256 * 1024 * 1024; // Total number of integers (~1GB)

    // Allocate a contiguous array
    int* data = new int[N];

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i);
    }

    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Sequential access
    for (size_t i = 0; i < N; ++i) {
        sum += data[i] * 2;
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Output results
    std::cout << "Sum: " << sum << std::endl;

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    delete[] data;

    return 0;
}