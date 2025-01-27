#include <benchmark/benchmark.h>
#include <array>
#include <random>
#include <smmintrin.h>
#include <immintrin.h>

// No AVX
void add(int size, int *a,int *b) {
    for (int i=0;i<size;i++) {
        a[i] += b[i];
    }
}

// with AVX2
void add_avx(int size, int *a, int*b) {
    int i=0;
    for (;i<size;i+=8) {
        // load 256-bit chunks of each array
        __m256i av = _mm256_loadu_si256((__m256i*) &a[i]);
        __m256i bv = _mm256_loadu_si256((__m256i*) &b[i]);
        // add each pair of 32-bit integers in chunks
        av = _mm256_add_epi32(av, bv);

        // store 256-bit chunk to a
        _mm256_storeu_si256((__m256i*) &a[i], av);
    }
    // clean up
    for(;i<size;i++) {
        a[i] += b[i];
    }
}

static void BM_AddNoAvx(benchmark::State& state) {
    // Perform setup here
    // Initialize random engine and distribution
    std::random_device rd;  // Get a seed from the random hardware (if available)
    std::mt19937 gen(rd()); // Mersenne Twister random number engine
    std::uniform_int_distribution<> distrib(1, 1000);  // Distribution range [1, 100]

    // Allocated two int arrays of size 10000 and initialized it with random number
    static constexpr int kArrSize { 10000 };
    std::array<int, kArrSize> arr_a {};
    std::array<int, kArrSize> arr_b {};

    for (int i = 0; i < kArrSize; i++) {
        arr_a[i] = distrib(gen);
        arr_b[i] = distrib(gen);
    }

    for (auto _ : state) {
        // This code gets timed
        add(kArrSize, arr_a.data(), arr_b.data());
    }
}

static void BM_AddAvx(benchmark::State& state) {
    // Perform setup here
    // Initialize random engine and distribution
    std::random_device rd;  // Get a seed from the random hardware (if available)
    std::mt19937 gen(rd()); // Mersenne Twister random number engine
    std::uniform_int_distribution<> distrib(1, 1000);  // Distribution range [1, 100]

    // Allocated two int arrays of size 10000 and initialized it with random number
    static constexpr int kArrSize { 10000 };
    std::array<int, kArrSize> arr_a {};
    std::array<int, kArrSize> arr_b {};

    for (int i = 0; i < kArrSize; i++) {
        arr_a[i] = distrib(gen);
        arr_b[i] = distrib(gen);
    }

    for (auto _ : state) {
        // This code gets timed
        add_avx(kArrSize, arr_a.data(), arr_b.data());
    }
}
// Register the function as a benchmark
BENCHMARK(BM_AddNoAvx);
// Register the function as a benchmark
BENCHMARK(BM_AddAvx);
// Run the benchmark
BENCHMARK_MAIN();