#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

int main() {
    const size_t N = 128 * 1024 * 1024; // Total number of integers (~1GB)

    // Allocate a contiguous array
    int* data = new int[N];

    // Initialize data
    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i);
    }

    // Generate a random access pattern
    std::vector<size_t> indices(N);
    for (size_t i = 0; i < N; ++i) {
        indices[i] = i;
    }

    // Shuffle indices to create randomness
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Random access
    for (size_t i = 0; i < N; ++i) {
        size_t index = indices[i];
        sum += data[index] * 2;
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Output results

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    delete[] data;

    return 0;
}