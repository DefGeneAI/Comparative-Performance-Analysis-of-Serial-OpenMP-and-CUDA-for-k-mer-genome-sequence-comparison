/*
 * kmer_cuda.cu
 * ============================================================
 * CUDA implementation of k-mer-based genome sequence comparison
 *
 * Algorithm:
 *   1. Read DNA sequence from FASTA/plain-text file
 *   2. Encode each k-mer using 2-bit DNA encoding
 *      A = 00
 *      C = 01
 *      G = 10
 *      T = 11
 *   3. Generate k-mers on GPU
 *      One CUDA thread -> one k-mer
 *   4. Sort k-mers on GPU using Thrust
 *   5. Remove duplicate k-mers on GPU
 *   6. Copy unique k-mers back to CPU
 *   7. Calculate Jaccard similarity on CPU
 *
 * Jaccard:
 *
 *              |A intersection B|
 *       J = -------------------------
 *              |A union B|
 *
 * This implementation is intended to use the SAME algorithm
 * as the serial/OpenMP versions for fair comparison.
 *
 * Usage:
 *
 * ./kmer_cuda <reference.fa> <query1.fa> [query2.fa ...]
 *             -k <15|21|31>
 *             [-r <repetitions>]
 *             [-s <serial_total_time>]
 *
 * Example:
 *
 * ./kmer_cuda \
 * "Reference/Escherichia coli str. K-12 substr. MG1655, complete genome.txt" \
 * "Queries/E coli Sakai.txt" \
 * -k 21 -r 1 -s 81.363749
 *
 * Compile:
 *
 * nvcc -O2 -std=c++17 -o kmer_cuda src/kmer_cuda.cu
 *
 * ============================================================
 */

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/* ============================================================
 * Configuration
 * ============================================================
 */

static const uint64_t INVALID_KMER = UINT64_MAX;

/*
 * Maximum sequence length.
 *
 * The genomes used in this project are approximately
 * 2.8M - 6.3M bases, so this is more than sufficient.
 */
static const size_t MAX_SEQ_LEN = (size_t)1 << 27;

/* ============================================================
 * CUDA error checking
 * ============================================================
 */

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr,                                                \
                    "CUDA ERROR at %s:%d\n"                                \
                    "  %s\n",                                              \
                    __FILE__, __LINE__, cudaGetErrorString(err__));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* ============================================================
 * Timing helper
 * ============================================================
 */

static double wall_time_seconds()
{
    using clock = chrono::steady_clock;

    static const auto start = clock::now();

    auto now = clock::now();

    return chrono::duration<double>(now - start).count();
}

/* ============================================================
 * Host nucleotide lookup table
 *
 * IMPORTANT:
 * Do NOT use C designated initializers here.
 *
 * Older/different nvcc configurations can reject:
 *
 *     ['A'] = 0
 *
 * So we initialize the table using a normal function.
 * ============================================================
 */

static uint8_t HOST_NUC2BIT[256];

/*
 * Initialize nucleotide lookup table.
 *
 * 255 = invalid
 *
 * A/a -> 0
 * C/c -> 1
 * G/g -> 2
 * T/t -> 3
 */
static void init_host_nuc2bit()
{
    for (int i = 0; i < 256; ++i) {
        HOST_NUC2BIT[i] = 255;
    }

    HOST_NUC2BIT[(unsigned char)'A'] = 0;
    HOST_NUC2BIT[(unsigned char)'C'] = 1;
    HOST_NUC2BIT[(unsigned char)'G'] = 2;
    HOST_NUC2BIT[(unsigned char)'T'] = 3;

    HOST_NUC2BIT[(unsigned char)'a'] = 0;
    HOST_NUC2BIT[(unsigned char)'c'] = 1;
    HOST_NUC2BIT[(unsigned char)'g'] = 2;
    HOST_NUC2BIT[(unsigned char)'t'] = 3;
}

/* ============================================================
 * Device nucleotide conversion
 * ============================================================
 */

__device__ __forceinline__
uint8_t device_nuc2bit(char c)
{
    switch (c) {

        case 'A':
        case 'a':
            return 0;

        case 'C':
        case 'c':
            return 1;

        case 'G':
        case 'g':
            return 2;

        case 'T':
        case 't':
            return 3;

        default:
            return 255;
    }
}

/* ============================================================
 * FASTA/plain-text reader
 *
 * Header lines beginning with '>' are ignored.
 *
 * Only A/C/G/T are retained.
 *
 * This matches the basic behavior of the serial/OpenMP
 * implementations.
 * ============================================================
 */

static bool read_sequence(
    const string &filename,
    string &sequence,
    double &io_time)
{
    auto t0 = chrono::steady_clock::now();

    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file:\n"
             << filename << endl;

        return false;
    }

    sequence.clear();

    /*
     * Reserve enough memory for the genomes used in this project.
     */
    sequence.reserve(8000000);

    string line;

    while (getline(file, line)) {

        /*
         * Ignore FASTA headers.
         */
        if (!line.empty() && line[0] == '>') {
            continue;
        }

        for (unsigned char c : line) {

            if (HOST_NUC2BIT[c] != 255) {

                sequence.push_back(
                    static_cast<char>(toupper(c))
                );
            }
        }
    }

    file.close();

    auto t1 = chrono::steady_clock::now();

    io_time =
        chrono::duration<double>(t1 - t0).count();

    return true;
}

/* ============================================================
 * CUDA k-mer generation kernel
 *
 * One CUDA thread generates one k-mer.
 *
 * For each starting position:
 *
 *   sequence[i ... i+k-1]
 *
 * is converted into a 2-bit encoded uint64_t.
 *
 * Example:
 *
 * A C G T
 *
 * 00 01 10 11
 *
 * becomes:
 *
 * 00011011
 *
 * ============================================================
 */

__global__
void encode_kmers_kernel(
    const char *sequence,
    uint64_t *kmers,
    size_t seq_len,
    int k)
{
    size_t idx =
        (size_t)blockIdx.x *
        (size_t)blockDim.x +
        (size_t)threadIdx.x;

    /*
     * Number of possible k-mers.
     */
    size_t num_kmers = 0;

    if (seq_len >= (size_t)k) {
        num_kmers = seq_len - (size_t)k + 1;
    }

    if (idx >= num_kmers) {
        return;
    }

    uint64_t encoded = 0;

    bool valid = true;

    /*
     * Build the k-mer.
     *
     * encoded:
     *
     * encoded << 2
     *     |
     * nucleotide bits
     */
    for (int j = 0; j < k; ++j) {

        uint8_t bits =
            device_nuc2bit(sequence[idx + j]);

        if (bits == 255) {
            valid = false;
            break;
        }

        encoded =
            (encoded << 2) |
            (uint64_t)bits;
    }

    if (valid) {
        kmers[idx] = encoded;
    }
    else {
        kmers[idx] = INVALID_KMER;
    }
}

/* ============================================================
 * Remove INVALID_KMER
 *
 * After sorting, invalid values will be at the end because
 * INVALID_KMER = UINT64_MAX.
 *
 * We remove them using erase.
 * ============================================================
 */

/* ============================================================
 * Calculate Jaccard similarity
 *
 * Inputs MUST be:
 *
 *   - sorted
 *   - unique
 *
 * Uses two-pointer intersection.
 *
 * ============================================================
 */

static double calculate_jaccard(
    const vector<uint64_t> &A,
    const vector<uint64_t> &B,
    size_t &intersection,
    size_t &union_count)
{
    size_t i = 0;
    size_t j = 0;

    intersection = 0;

    while (i < A.size() && j < B.size()) {

        if (A[i] == B[j]) {

            ++intersection;
            ++i;
            ++j;
        }
        else if (A[i] < B[j]) {

            ++i;
        }
        else {

            ++j;
        }
    }

    union_count =
        A.size() +
        B.size() -
        intersection;

    if (union_count == 0) {
        return 1.0;
    }

    return static_cast<double>(intersection) /
           static_cast<double>(union_count);
}

/* ============================================================
 * GPU result structure
 * ============================================================
 */

struct GPUKmerResult {

    vector<uint64_t> unique_kmers;

    size_t sequence_length = 0;

    size_t raw_kmers = 0;

    double io_time = 0.0;

    double h2d_time = 0.0;

    double kmer_generation_time = 0.0;

    double gpu_sort_time = 0.0;

    double gpu_unique_time = 0.0;

    double d2h_time = 0.0;

    double total_gpu_time = 0.0;
};

/* ============================================================
 * Generate unique k-mers on GPU
 * ============================================================
 */

static GPUKmerResult process_sequence_gpu(
    const string &sequence,
    int k)
{
    GPUKmerResult result;

    result.sequence_length =
        sequence.size();

    /*
     * Number of raw k-mers.
     *
     * N - k + 1
     */
    if (sequence.size() >= (size_t)k) {

        result.raw_kmers =
            sequence.size() -
            (size_t)k +
            1;
    }
    else {

        result.raw_kmers = 0;
    }

    if (result.raw_kmers == 0) {

        result.unique_kmers.clear();

        return result;
    }

    /*
     * Allocate device sequence.
     */
    char *d_sequence = nullptr;

    uint64_t *d_kmers = nullptr;

    /*
     * -------------------------
     * H2D
     * -------------------------
     */

    auto h2d_start =
        chrono::steady_clock::now();

    CUDA_CHECK(
        cudaMalloc(
            (void **)&d_sequence,
            sequence.size() * sizeof(char)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_sequence,
            sequence.data(),
            sequence.size() * sizeof(char),
            cudaMemcpyHostToDevice
        )
    );

    /*
     * Allocate k-mer array.
     */
    CUDA_CHECK(
        cudaMalloc(
            (void **)&d_kmers,
            result.raw_kmers * sizeof(uint64_t)
        )
    );

    auto h2d_end =
        chrono::steady_clock::now();

    result.h2d_time =
        chrono::duration<double>(
            h2d_end - h2d_start
        ).count();

    /*
     * -------------------------
     * GPU k-mer generation
     * -------------------------
     */

    const int BLOCK_SIZE = 256;

    size_t grid_size =
        (result.raw_kmers +
         BLOCK_SIZE - 1) /
        BLOCK_SIZE;

    cudaEvent_t gen_start;
    cudaEvent_t gen_stop;

    CUDA_CHECK(cudaEventCreate(&gen_start));
    CUDA_CHECK(cudaEventCreate(&gen_stop));

    CUDA_CHECK(cudaEventRecord(gen_start));

    encode_kmers_kernel<<<
        static_cast<unsigned int>(grid_size),
        BLOCK_SIZE
    >>>(
        d_sequence,
        d_kmers,
        sequence.size(),
        k
    );

    CUDA_CHECK(cudaEventRecord(gen_stop));

    CUDA_CHECK(cudaEventSynchronize(gen_stop));

    CUDA_CHECK(cudaGetLastError());

    float generation_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &generation_ms,
            gen_start,
            gen_stop
        )
    );

    result.kmer_generation_time =
        generation_ms / 1000.0;

    CUDA_CHECK(cudaEventDestroy(gen_start));
    CUDA_CHECK(cudaEventDestroy(gen_stop));

    /*
     * -------------------------
     * Wrap raw CUDA memory in
     * a Thrust device vector
     *
     * We need Thrust sorting.
     *
     * To avoid copying the entire array into another
     * allocation, we use thrust::device_ptr.
     * -------------------------
     */

    thrust::device_ptr<uint64_t> thrust_begin(d_kmers);

    thrust::device_ptr<uint64_t> thrust_end =
        thrust_begin + result.raw_kmers;

    /*
     * -------------------------
     * GPU SORT
     * -------------------------
     */

    cudaEvent_t sort_start;
    cudaEvent_t sort_stop;

    CUDA_CHECK(cudaEventCreate(&sort_start));
    CUDA_CHECK(cudaEventCreate(&sort_stop));

    CUDA_CHECK(cudaEventRecord(sort_start));

    thrust::sort(
        thrust_begin,
        thrust_end
    );

    CUDA_CHECK(cudaEventRecord(sort_stop));

    CUDA_CHECK(cudaEventSynchronize(sort_stop));

    float sort_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &sort_ms,
            sort_start,
            sort_stop
        )
    );

    result.gpu_sort_time =
        sort_ms / 1000.0;

    CUDA_CHECK(cudaEventDestroy(sort_start));
    CUDA_CHECK(cudaEventDestroy(sort_stop));

    /*
     * -------------------------
     * GPU UNIQUE
     * -------------------------
     *
     * thrust::unique moves duplicate values together and
     * returns the new logical end.
     */

    cudaEvent_t unique_start;
    cudaEvent_t unique_stop;

    CUDA_CHECK(cudaEventCreate(&unique_start));
    CUDA_CHECK(cudaEventCreate(&unique_stop));

    CUDA_CHECK(cudaEventRecord(unique_start));

    thrust::device_ptr<uint64_t> unique_end =
        thrust::unique(
            thrust_begin,
            thrust_end
        );

    CUDA_CHECK(cudaEventRecord(unique_stop));

    CUDA_CHECK(cudaEventSynchronize(unique_stop));

    float unique_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &unique_ms,
            unique_start,
            unique_stop
        )
    );

    result.gpu_unique_time =
        unique_ms / 1000.0;

    CUDA_CHECK(cudaEventDestroy(unique_start));
    CUDA_CHECK(cudaEventDestroy(unique_stop));

    /*
     * Number of unique k-mers.
     */
    size_t unique_count =
        static_cast<size_t>(
            unique_end - thrust_begin
        );

    /*
     * INVALID_KMER values were sorted to the end.
     *
     * If the sequence contained invalid bases, they would be
     * represented as INVALID_KMER.
     *
     * Since our reader already removes non-ACGT characters,
     * this normally should not occur.
     *
     * Still, we explicitly remove them for safety.
     */

    /*
     * Copy the unique values back to CPU.
     */
    vector<uint64_t> temp_unique(
        unique_count
    );

    /*
     * -------------------------
     * D2H
     * -------------------------
     */

    auto d2h_start =
        chrono::steady_clock::now();

    if (unique_count > 0) {

        CUDA_CHECK(
            cudaMemcpy(
                temp_unique.data(),
                d_kmers,
                unique_count * sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            )
        );
    }

    auto d2h_end =
        chrono::steady_clock::now();

    result.d2h_time =
        chrono::duration<double>(
            d2h_end - d2h_start
        ).count();

    /*
     * Remove INVALID_KMER from the host vector if present.
     */
    auto invalid_it =
        lower_bound(
            temp_unique.begin(),
            temp_unique.end(),
            INVALID_KMER
        );

    temp_unique.erase(
        invalid_it,
        temp_unique.end()
    );

    result.unique_kmers =
        std::move(temp_unique);

    /*
     * Free GPU memory.
     */
    CUDA_CHECK(cudaFree(d_sequence));
    CUDA_CHECK(cudaFree(d_kmers));

    /*
     * Total GPU-side processing time.
     *
     * This excludes the CPU Jaccard calculation.
     */
    result.total_gpu_time =
        result.h2d_time +
        result.kmer_generation_time +
        result.gpu_sort_time +
        result.gpu_unique_time +
        result.d2h_time;

    return result;
}

/* ============================================================
 * Print GPU information
 * ============================================================
 */

static void print_gpu_information()
{
    int device_count = 0;

    CUDA_CHECK(
        cudaGetDeviceCount(&device_count)
    );

    if (device_count == 0) {

        cerr << "ERROR: No CUDA GPU detected."
             << endl;

        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    cout << "\n============================================================\n";
    cout << "CUDA GPU INFORMATION\n";
    cout << "============================================================\n";

    cout << "GPU              : "
         << prop.name
         << "\n";

    cout << "Compute Capability: "
         << prop.major
         << "."
         << prop.minor
         << "\n";

    cout << "SM Count          : "
         << prop.multiProcessorCount
         << "\n";

    cout << "Global Memory     : "
         << fixed
         << setprecision(2)
         << static_cast<double>(
                prop.totalGlobalMem
            ) / (1024.0 * 1024.0 * 1024.0)
         << " GB\n";

    cout << "Max Threads/Block : "
         << prop.maxThreadsPerBlock
         << "\n";

    cout << "============================================================\n";
}

/* ============================================================
 * Parse integer safely
 * ============================================================
 */

static bool parse_int(
    const string &s,
    int &value)
{
    try {

        size_t pos = 0;

        int temp =
            stoi(s, &pos);

        if (pos != s.size()) {
            return false;
        }

        value = temp;

        return true;
    }
    catch (...) {

        return false;
    }
}

/* ============================================================
 * Parse double safely
 * ============================================================
 */

static bool parse_double(
    const string &s,
    double &value)
{
    try {

        size_t pos = 0;

        double temp =
            stod(s, &pos);

        if (pos != s.size()) {
            return false;
        }

        value = temp;

        return true;
    }
    catch (...) {

        return false;
    }
}

/* ============================================================
 * Print usage
 * ============================================================
 */

static void print_usage(
    const char *program)
{
    cerr << "\nUsage:\n";

    cerr << program
         << " <reference.fa>"
         << " <query1.fa> [query2.fa ...]"
         << " -k <15|21|31>"
         << " [-r repetitions]"
         << " [-s serial_total_time]\n";

    cerr << "\nExample:\n";

    cerr << program
         << " \"Reference/reference.txt\""
         << " \"Queries/query1.fa\""
         << " -k 21"
         << " -r 5"
         << " -s 81.363749\n";
}

/* ============================================================
 * MAIN
 * ============================================================
 */

int main(
    int argc,
    char **argv)
{
    /*
     * Initialize host nucleotide lookup table.
     *
     * This is the important compatibility fix for the nvcc
     * error you encountered.
     */
    init_host_nuc2bit();

    /*
     * Need at least:
     *
     * program
     * reference
     * query
     * -k
     * value
     */
    if (argc < 5) {

        print_usage(argv[0]);

        return EXIT_FAILURE;
    }

    /*
     * --------------------------------------------------------
     * Parse arguments
     * --------------------------------------------------------
     */

    string reference_file;

    vector<string> query_files;

    int k = 0;

    int repetitions = 1;

    double serial_total_time = -1.0;

    reference_file = argv[1];

    int i = 2;

    /*
     * Everything before "-k" is treated as a query file.
     */
    while (i < argc) {

        string arg = argv[i];

        if (arg == "-k") {

            if (i + 1 >= argc) {

                cerr << "ERROR: Missing value after -k\n";

                return EXIT_FAILURE;
            }

            if (!parse_int(argv[i + 1], k)) {

                cerr << "ERROR: Invalid k value\n";

                return EXIT_FAILURE;
            }

            i += 2;

            break;
        }

        query_files.push_back(arg);

        ++i;
    }

    /*
     * Parse remaining optional arguments.
     */
    while (i < argc) {

        string arg = argv[i];

        if (arg == "-r") {

            if (i + 1 >= argc) {

                cerr << "ERROR: Missing value after -r\n";

                return EXIT_FAILURE;
            }

            if (!parse_int(
                    argv[i + 1],
                    repetitions)) {

                cerr << "ERROR: Invalid repetition count\n";

                return EXIT_FAILURE;
            }

            i += 2;
        }
        else if (arg == "-s") {

            if (i + 1 >= argc) {

                cerr << "ERROR: Missing value after -s\n";

                return EXIT_FAILURE;
            }

            if (!parse_double(
                    argv[i + 1],
                    serial_total_time)) {

                cerr << "ERROR: Invalid serial time\n";

                return EXIT_FAILURE;
            }

            i += 2;
        }
        else {

            cerr << "ERROR: Unknown argument: "
                 << arg
                 << "\n";

            return EXIT_FAILURE;
        }
    }

    /*
     * --------------------------------------------------------
     * Validate parameters
     * --------------------------------------------------------
     */

    if (query_files.empty()) {

        cerr << "ERROR: No query files supplied.\n";

        print_usage(argv[0]);

        return EXIT_FAILURE;
    }

    if (!(k == 15 || k == 21 || k == 31)) {

        cerr << "ERROR: k must be 15, 21, or 31.\n";

        return EXIT_FAILURE;
    }

    if (repetitions <= 0) {

        cerr << "ERROR: repetitions must be >= 1.\n";

        return EXIT_FAILURE;
    }

    /*
     * --------------------------------------------------------
     * Initialize CUDA context
     *
     * This is done before benchmarking so that CUDA context
     * initialization is not accidentally included in the
     * algorithm timing.
     * --------------------------------------------------------
     */

    CUDA_CHECK(cudaFree(0));

    print_gpu_information();

    /*
     * --------------------------------------------------------
     * Configuration output
     * --------------------------------------------------------
     */

    cout << "\n============================================================\n";
    cout << "CUDA K-MER GENOME COMPARISON\n";
    cout << "============================================================\n";

    cout << "Reference file     : "
         << reference_file
         << "\n";

    cout << "Number of queries  : "
         << query_files.size()
         << "\n";

    cout << "k                  : "
         << k
         << "\n";

    cout << "Repetitions        : "
         << repetitions
         << "\n";

    if (serial_total_time >= 0.0) {

        cout << "Serial total time  : "
             << fixed
             << setprecision(6)
             << serial_total_time
             << " s\n";
    }
    else {

        cout << "Serial total time  : not supplied\n";
    }

    cout << "============================================================\n";

    /*
     * --------------------------------------------------------
     * Read reference
     *
     * The reference is read ONCE.
     * --------------------------------------------------------
     */

    string reference_sequence;

    double reference_io_time = 0.0;

    if (!read_sequence(
            reference_file,
            reference_sequence,
            reference_io_time)) {

        return EXIT_FAILURE;
    }

    cout << "\nReference sequence length: "
         << reference_sequence.size()
         << " bp\n";

    size_t reference_raw_kmers = 0;

    if (reference_sequence.size() >= (size_t)k) {

        reference_raw_kmers =
            reference_sequence.size() -
            (size_t)k +
            1;
    }

    cout << "Reference raw k-mers      : "
         << reference_raw_kmers
         << "\n";

    cout << "Reference I/O time       : "
         << fixed
         << setprecision(6)
         << reference_io_time
         << " s\n";

    /*
     * --------------------------------------------------------
     * Process reference
     * --------------------------------------------------------
     */

    auto reference_gpu_start =
        chrono::steady_clock::now();

    GPUKmerResult reference_result =
        process_sequence_gpu(
            reference_sequence,
            k
        );

    auto reference_gpu_end =
        chrono::steady_clock::now();

    double reference_gpu_wall =
        chrono::duration<double>(
            reference_gpu_end -
            reference_gpu_start
        ).count();

    cout << "Reference unique k-mers  : "
         << reference_result.unique_kmers.size()
         << "\n";

    cout << "Reference GPU time       : "
         << reference_gpu_wall
         << " s\n";

    /*
     * We no longer need the reference DNA sequence.
     *
     * Freeing this memory reduces host RAM usage.
     */
    reference_sequence.clear();
    reference_sequence.shrink_to_fit();

    /*
     * --------------------------------------------------------
     * Store timing information
     * --------------------------------------------------------
     */

    double total_program_start =
        wall_time_seconds();

    double total_query_wall_time = 0.0;

    /*
     * --------------------------------------------------------
     * Process every query
     * --------------------------------------------------------
     */

    for (size_t query_index = 0;
         query_index < query_files.size();
         ++query_index) {

        const string &query_file =
            query_files[query_index];

        cout << "\n------------------------------------------------------------\n";

        cout << "QUERY "
             << (query_index + 1)
             << "/"
             << query_files.size()
             << "\n";

        cout << "File: "
             << query_file
             << "\n";

        cout << "------------------------------------------------------------\n";

        /*
         * ----------------------------------------------------
         * Repeat benchmark
         * ----------------------------------------------------
         */

        vector<double> repetition_times;

        repetition_times.reserve(
            repetitions
        );

        double last_io_time = 0.0;

        size_t final_sequence_length = 0;

        size_t final_raw_kmers = 0;

        size_t final_unique_kmers = 0;

        size_t final_intersection = 0;

        size_t final_union = 0;

        double final_jaccard = 0.0;

        double final_h2d = 0.0;

        double final_generation = 0.0;

        double final_sort = 0.0;

        double final_unique = 0.0;

        double final_d2h = 0.0;

        double final_gpu_time = 0.0;

        /*
         * Each repetition reads the query again.
         *
         * This follows the current serial/OpenMP benchmark
         * structure where each repetition processes the query
         * independently.
         */

        for (int rep = 0;
             rep < repetitions;
             ++rep) {

            string query_sequence;

            double query_io_time = 0.0;

            /*
             * ------------------------------------------------
             * Read query
             * ------------------------------------------------
             */

            if (!read_sequence(
                    query_file,
                    query_sequence,
                    query_io_time)) {

                return EXIT_FAILURE;
            }

            last_io_time =
                query_io_time;

            /*
             * ------------------------------------------------
             * Full query timing
             *
             * This includes:
             *
             *   GPU processing
             *   D2H
             *   CPU Jaccard
             *
             * but excludes query file reading.
             * ------------------------------------------------
             */

            auto query_start =
                chrono::steady_clock::now();

            /*
             * ------------------------------------------------
             * GPU k-mer processing
             * ------------------------------------------------
             */

            GPUKmerResult query_result =
                process_sequence_gpu(
                    query_sequence,
                    k
                );

            /*
             * ------------------------------------------------
             * CPU Jaccard
             * ------------------------------------------------
             */

            size_t intersection = 0;

            size_t union_count = 0;

            double jaccard =
                calculate_jaccard(
                    reference_result.unique_kmers,
                    query_result.unique_kmers,
                    intersection,
                    union_count
                );

            auto query_end =
                chrono::steady_clock::now();

            double query_time =
                chrono::duration<double>(
                    query_end -
                    query_start
                ).count();

            repetition_times.push_back(
                query_time
            );

            /*
             * Save values from final repetition.
             */
            final_sequence_length =
                query_result.sequence_length;

            final_raw_kmers =
                query_result.raw_kmers;

            final_unique_kmers =
                query_result.unique_kmers.size();

            final_intersection =
                intersection;

            final_union =
                union_count;

            final_jaccard =
                jaccard;

            final_h2d =
                query_result.h2d_time;

            final_generation =
                query_result.kmer_generation_time;

            final_sort =
                query_result.gpu_sort_time;

            final_unique =
                query_result.gpu_unique_time;

            final_d2h =
                query_result.d2h_time;

            final_gpu_time =
                query_result.total_gpu_time;

            /*
             * Print every repetition.
             */
            cout << "Rep "
                 << (rep + 1)
                 << "/"
                 << repetitions
                 << ": "
                 << fixed
                 << setprecision(6)
                 << query_time
                 << " s\n";

            /*
             * Free query sequence before next repetition.
             */
            query_sequence.clear();

            query_sequence.shrink_to_fit();
        }

        /*
         * ----------------------------------------------------
         * Median
         * ----------------------------------------------------
         */

        vector<double> sorted_times =
            repetition_times;

        sort(
            sorted_times.begin(),
            sorted_times.end()
        );

        double median_time = 0.0;

        size_t n =
            sorted_times.size();

        if (n % 2 == 1) {

            median_time =
                sorted_times[n / 2];
        }
        else {

            median_time =
                (
                    sorted_times[n / 2 - 1] +
                    sorted_times[n / 2]
                ) / 2.0;
        }

        total_query_wall_time +=
            median_time;

        /*
         * ----------------------------------------------------
         * Print query results
         * ----------------------------------------------------
         */

        cout << "\nRESULTS\n";

        cout << "Sequence length      : "
             << final_sequence_length
             << " bp\n";

        cout << "Raw k-mers           : "
             << final_raw_kmers
             << "\n";

        cout << "Unique k-mers        : "
             << final_unique_kmers
             << "\n";

        cout << "Intersection         : "
             << final_intersection
             << "\n";

        cout << "Union                : "
             << final_union
             << "\n";

        cout << "Jaccard similarity   : "
             << fixed
             << setprecision(6)
             << final_jaccard
             << "\n";

        cout << "Jaccard percentage   : "
             << fixed
             << setprecision(4)
             << final_jaccard * 100.0
             << "%\n";

        cout << "\nTIMING BREAKDOWN\n";

        cout << "Query I/O            : "
             << fixed
             << setprecision(6)
             << last_io_time
             << " s\n";

        cout << "H2D                   : "
             << final_h2d
             << " s\n";

        cout << "GPU k-mer generation : "
             << final_generation
             << " s\n";

        cout << "GPU sort             : "
             << final_sort
             << " s\n";

        cout << "GPU unique           : "
             << final_unique
             << " s\n";

        cout << "D2H                  : "
             << final_d2h
             << " s\n";

        cout << "GPU processing       : "
             << final_gpu_time
             << " s\n";

        cout << "Median query time    : "
             << median_time
             << " s\n";
    }

    /*
     * --------------------------------------------------------
     * End total timing
     * --------------------------------------------------------
     */

    double total_program_end =
        wall_time_seconds();

    double total_wall_time =
        total_program_end -
        total_program_start;

    /*
     * --------------------------------------------------------
     * Print overall benchmark
     * --------------------------------------------------------
     */

    cout << "\n============================================================\n";
    cout << "OVERALL CUDA BENCHMARK\n";
    cout << "============================================================\n";

    cout << "k                  : "
         << k
         << "\n";

    cout << "Queries processed  : "
         << query_files.size()
         << "\n";

    cout << "Repetitions/query  : "
         << repetitions
         << "\n";

    cout << "Reference I/O      : "
         << fixed
         << setprecision(6)
         << reference_io_time
         << " s\n";

    cout << "Total wall time    : "
         << total_wall_time
         << " s\n";

    /*
     * --------------------------------------------------------
     * Speedup
     * --------------------------------------------------------
     */

    if (serial_total_time > 0.0) {

        double speedup =
            serial_total_time /
            total_wall_time;

        cout << "Serial total time  : "
             << serial_total_time
             << " s\n";

        cout << "CUDA speedup       : "
             << fixed
             << setprecision(4)
             << speedup
             << "x\n";

        /*
         * IMPORTANT:
         *
         * Do NOT calculate:
         *
         *     speedup / GPU thread count
         *
         * as CUDA efficiency.
         *
         * A GPU has thousands of hardware threads and the
         * concept is not directly comparable to OpenMP CPU
         * thread efficiency.
         */
    }
    else {

        cout << "CUDA speedup       : serial time not supplied\n";
    }

    cout << "============================================================\n";

    /*
     * --------------------------------------------------------
     * Final note
     * --------------------------------------------------------
     */

    cout << "\nBenchmark completed successfully.\n";

    return EXIT_SUCCESS;
}