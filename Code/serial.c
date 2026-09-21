/*
 * kmer_serial.c
 * ============================================================
 * Serial C implementation of k-mer-based genome sequence
 * comparison using 2-bit encoding and Jaccard similarity.
 *
 * Compatible base for OpenMP and CUDA extensions:
 *   - All parallelisable loops are clearly marked
 *   - Timing macros capture T_IO, T_compute, T_total separately
 *   - Memory-usage reporting via /proc/self/status (Linux)
 *
 * Compile (serial):
 *   gcc -O2 -o kmer_serial kmer_serial.c -lm
 *
 * Compile (OpenMP – replace parallel loops with #pragma omp parallel for):
 *   gcc -O2 -fopenmp -o kmer_omp kmer_serial.c -lm
 *
 * For CUDA: port the kmer_encode_all() and jaccard_intersect() sections
 * to CUDA kernels; the host logic and I/O remain identical.
 *
 * Usage:
 *   ./kmer_serial <reference.fa> <query1.fa> [query2.fa ...] -k <15|21|31> [-r <reps>]
 *
 * Example:
 *   ./kmer_serial ref.fa q1.fa q2.fa -k 21 -r 5
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

/* ── Configuration ─────────────────────────────────────────── */
#define MAX_SEQ_LEN   (1UL << 27)   /* 128 MB of nucleotides   */
#define MAX_KMERS     (1UL << 27)   /* max k-mers per genome   */
#define MAX_PATH_LEN  512
#define DEFAULT_K     21
#define DEFAULT_REPS  5

/* ── 2-bit encoding lookup ─────────────────────────────────── */
/*
 * Step 4 (document): Encode each nucleotide with 2 bits:
 *   A → 00, C → 01, G → 10, T → 11
 * Invalid characters return 0xFF (sentinel).
 */
static const uint8_t NUC2BIT[256] = {
    ['A'] = 0, ['a'] = 0,
    ['C'] = 1, ['c'] = 1,
    ['G'] = 2, ['g'] = 2,
    ['T'] = 3, ['t'] = 3,
    /* everything else stays 0 (default) – detected via IS_VALID below */
};

/* Returns 1 if the character is a recognised nucleotide */
static inline int is_valid_nuc(char c) {
    return (c=='A'||c=='a'||c=='C'||c=='c'||
            c=='G'||c=='g'||c=='T'||c=='t');
}

/* ── Timing helpers ────────────────────────────────────────── */
typedef struct { double wall; } Timer;

static inline void timer_start(Timer *t) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t->wall = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double timer_stop(Timer *t) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9) - t->wall;
}

/* ── Memory usage (Linux) ──────────────────────────────────── */
static long get_peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long peak = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmPeak:", 7) == 0) {
            sscanf(line + 7, "%ld", &peak);
            break;
        }
    }
    fclose(f);
    return peak;  /* kB */
}

/* ── FASTA / plain-text reader ─────────────────────────────── */
/*
 * Step 1 (document): Load genome sequence from a FASTA or plain-text file.
 * Returns the number of nucleotide characters read, or 0 on error.
 * Skips '>' header lines and whitespace.
 */
static size_t read_sequence(const char *path, char *buf, size_t maxlen) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open file: %s\n", path);
        return 0;
    }

    size_t n = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp) && n < maxlen) {
        if (line[0] == '>') continue;          /* skip FASTA header */
        for (size_t i = 0; line[i] && n < maxlen; i++) {
            char c = line[i];
            if (is_valid_nuc(c)) buf[n++] = c;
        }
    }
    fclose(fp);
    buf[n] = '\0';
    return n;
}

/* ── k-mer type: packed 64-bit integer ─────────────────────── */
typedef uint64_t kmer_t;

/* ── Step 3 + 4: Generate AND encode all k-mers (sliding window) ─────── */
/*
 * For a sequence of length N and a given k, there are (N - k + 1) k-mers.
 * Each k-mer is encoded into a 64-bit integer using 2-bit nucleotide encoding.
 * k must be <= 32 to fit in 64 bits.
 *
 * PARALLELISM NOTE (OpenMP):
 *   The loop below can be parallelised with:
 *       #pragma omp parallel for schedule(static)
 *   followed by a thread-private local buffer merged after the loop,
 *   because each iteration writes to a unique index.
 *
 * PARALLELISM NOTE (CUDA):
 *   This loop maps directly to a 1-D CUDA kernel where each thread
 *   computes one k-mer: threadIdx.x + blockIdx.x * blockDim.x → i.
 */
static size_t encode_kmers(const char *seq, size_t seqlen, int k,
                           kmer_t *out, size_t out_max) {
    if ((size_t)k > seqlen) return 0;

    const size_t num_kmers = seqlen - (size_t)k + 1;
    if (num_kmers > out_max) {
        fprintf(stderr, "[WARN] Too many k-mers (%zu); truncating.\n", num_kmers);
    }
    const size_t count = (num_kmers < out_max) ? num_kmers : out_max;

    /* Build the first k-mer */
    kmer_t mask = (k < 32) ? ((kmer_t)1 << (2 * k)) - 1 : ~(kmer_t)0;

    /* === PARALLELISABLE LOOP (serial here) === */
    for (size_t i = 0; i < count; i++) {
        kmer_t val = 0;
        int valid = 1;
        for (int j = 0; j < k; j++) {
            char c = seq[i + j];
            if (!is_valid_nuc(c)) { valid = 0; break; }
            val = (val << 2) | NUC2BIT[(unsigned char)c];
        }
        out[i] = valid ? (val & mask) : (kmer_t)UINT64_MAX; /* sentinel for invalid */
    }
    return count;
}

/* ── Comparison function for qsort ────────────────────────── */
static int cmp_kmer(const void *a, const void *b) {
    kmer_t ka = *(const kmer_t *)a;
    kmer_t kb = *(const kmer_t *)b;
    return (ka > kb) - (ka < kb);
}

/* ── Step 5: Sort and deduplicate k-mer array ─────────────── */
/*
 * Returns the number of unique valid k-mers.
 * Invalid sentinels (UINT64_MAX) end up at the tail after sorting
 * and are excluded.
 *
 * PARALLELISM NOTE (OpenMP):
 *   qsort is single-threaded; replace with a parallel sort (e.g.,
 *   OpenMP-based merge sort or __gnu_parallel::sort) for the OMP build.
 *
 * PARALLELISM NOTE (CUDA):
 *   Use thrust::sort_by_key or CUB device sort on the GPU.
 */
static size_t sort_unique(kmer_t *arr, size_t n) {
    if (n == 0) return 0;

    /* Sort ascending */
    qsort(arr, n, sizeof(kmer_t), cmp_kmer);

    /* Remove invalid sentinels (they sort to the top) */
    size_t valid_end = n;
    while (valid_end > 0 && arr[valid_end - 1] == UINT64_MAX) valid_end--;

    /* Deduplicate */
    size_t unique = 0;
    for (size_t i = 0; i < valid_end; i++) {
        if (i == 0 || arr[i] != arr[i - 1]) arr[unique++] = arr[i];
    }
    return unique;
}

/* ── Step 6 + 7: Intersection and Jaccard similarity ─────── */
/*
 * Both arrays must be sorted and deduplicated (Step 5).
 * Uses a two-pointer merge: O(|A| + |B|).
 *
 * PARALLELISM NOTE (OpenMP / CUDA):
 *   This sequential two-pointer merge is cache-friendly but inherently
 *   serial. For parallel builds, partition the sorted arrays by key
 *   ranges and count intersections per partition independently, then
 *   reduce the per-partition counts.
 *
 * J(A, B) = |A ∩ B| / |A ∪ B|
 *          = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 */
static double jaccard(const kmer_t *A, size_t sA,
                      const kmer_t *B, size_t sB,
                      size_t *out_intersect, size_t *out_union) {
    size_t i = 0, j = 0, inter = 0;
    while (i < sA && j < sB) {
        if (A[i] == B[j]) { inter++; i++; j++; }
        else if (A[i] < B[j]) i++;
        else                   j++;
    }
    size_t uni = sA + sB - inter;
    if (out_intersect) *out_intersect = inter;
    if (out_union)     *out_union     = uni;
    return (uni == 0) ? 0.0 : (double)inter / (double)uni;
}

/* ── Pretty-print a horizontal separator ───────────────────── */
static void separator(void) {
    printf("%-80s\n", "────────────────────────────────────────────────────────────────────────────────");
}

/* ── Main ───────────────────────────────────────────────────── */
int main(int argc, char **argv) {

    /* ── Argument parsing ─────────────────────────────────── */
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <reference.fa> <query1.fa> [query2.fa ...] -k <15|21|31> [-r <reps>]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *ref_path = argv[1];
    int k = DEFAULT_K;
    int reps = DEFAULT_REPS;

    /* Collect query paths and parse flags */
    const char *query_paths[64];
    int n_queries = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else {
            query_paths[n_queries++] = argv[i];
        }
    }

    if (n_queries == 0) {
        fprintf(stderr, "[ERROR] At least one query genome must be specified.\n");
        return EXIT_FAILURE;
    }
    if (k < 1 || k > 32) {
        fprintf(stderr, "[ERROR] k must be in [1, 32]. Got %d.\n", k);
        return EXIT_FAILURE;
    }
    if (reps < 1) reps = 1;

    printf("\n");
    separator();
    printf("  k-mer Serial Genome Comparison  (k=%d, reps=%d)\n", k, reps);
    separator();
    printf("  Reference : %s\n", ref_path);
    printf("  Queries   : %d genome(s)\n", n_queries);
    printf("  k-mer size: %d\n", k);
    printf("  Repeats   : %d\n", reps);
    separator();

    /* ── Allocate sequence buffer ─────────────────────────── */
    char *seq_buf = (char *)malloc(MAX_SEQ_LEN + 1);
    if (!seq_buf) { perror("malloc seq_buf"); return EXIT_FAILURE; }

    kmer_t *kmers_ref   = (kmer_t *)malloc(MAX_KMERS * sizeof(kmer_t));
    kmer_t *kmers_query = (kmer_t *)malloc(MAX_KMERS * sizeof(kmer_t));
    if (!kmers_ref || !kmers_query) { perror("malloc kmers"); return EXIT_FAILURE; }

    /* ── Step 1: Load reference genome ───────────────────── */
    Timer t_total, t_io, t_compute;
    timer_start(&t_total);
    timer_start(&t_io);

    printf("\n[Step 1] Loading reference genome ...\n");
    size_t ref_len = read_sequence(ref_path, seq_buf, MAX_SEQ_LEN);
    if (ref_len == 0) { free(seq_buf); free(kmers_ref); free(kmers_query); return EXIT_FAILURE; }
    printf("         Reference length: %zu bp\n", ref_len);

    /* Step 3 + 4: Encode reference k-mers */
    size_t ref_raw = encode_kmers(seq_buf, ref_len, k, kmers_ref, MAX_KMERS);

    double t_io_elapsed = timer_stop(&t_io);
    printf("         Raw k-mers (ref): %zu\n", ref_raw);
    printf("         T_IO (ref load): %.6f s\n", t_io_elapsed);

    /* Step 5: Sort and deduplicate reference */
    timer_start(&t_compute);
    size_t ref_unique = sort_unique(kmers_ref, ref_raw);
    printf("         Unique k-mers (ref): %zu\n\n", ref_unique);

    /* ── Process each query genome ─────────────────────── */
    /*
     * Step 2 (document): Select k-mer size (already done above via -k flag).
     */

    for (int q = 0; q < n_queries; q++) {
        const char *qpath = query_paths[q];
        printf("[Query %d] %s\n", q + 1, qpath);

        /* Accumulate timing across repetitions (Step 10) */
        double times[64];
        int valid_reps = 0;
        double jaccard_val = 0.0;
        size_t inter = 0, uni = 0, q_unique = 0;

        for (int rep = 0; rep < reps; rep++) {

            /* Step 1: Load query */
            Timer t_rep;
            timer_start(&t_rep);

            timer_start(&t_io);
            size_t q_len = read_sequence(qpath, seq_buf, MAX_SEQ_LEN);
            t_io_elapsed = timer_stop(&t_io);

            if (q_len == 0) continue;

            /* Step 3 + 4: Encode query k-mers (=== PARALLELISABLE ===) */
            size_t q_raw = encode_kmers(seq_buf, q_len, k, kmers_query, MAX_KMERS);

            /* Step 5: Sort and deduplicate query */
            q_unique = sort_unique(kmers_query, q_raw);

            /* Step 6 + 7: Jaccard similarity */
            jaccard_val = jaccard(kmers_ref, ref_unique,
                                  kmers_query, q_unique,
                                  &inter, &uni);

            times[valid_reps++] = timer_stop(&t_rep);

            if (rep == 0) {
                /* Print genome info only on first rep */
                printf("         Query length    : %zu bp\n", q_len);
                printf("         Raw k-mers      : %zu\n", q_raw);
                printf("         Unique k-mers   : %zu\n", q_unique);
            }
        }

        if (valid_reps == 0) {
            printf("         [SKIP] Could not read query.\n\n");
            continue;
        }

        /* Step 10: Compute median over repetitions */
        /* Simple insertion sort on small array */
        for (int a = 1; a < valid_reps; a++) {
            double key = times[a];
            int b = a - 1;
            while (b >= 0 && times[b] > key) { times[b+1] = times[b]; b--; }
            times[b+1] = key;
        }
        double median_time = times[valid_reps / 2];

        /* ── Step 7: Report Jaccard result ─────────────── */
        printf("\n  ┌─ Results ─────────────────────────────────\n");
        printf("  │  |A ∩ B| (intersection) : %zu k-mers\n", inter);
        printf("  │  |A ∪ B| (union)        : %zu k-mers\n", uni);
        printf("  │  Jaccard similarity      : %.6f  (%.2f%%)\n",
               jaccard_val, jaccard_val * 100.0);
        printf("  │\n");
        printf("  │  Median exec time (%d reps): %.6f s\n", valid_reps, median_time);
        printf("  │  T_IO (last rep)           : %.6f s\n", t_io_elapsed);
        printf("  └───────────────────────────────────────────\n\n");
    }

    double total_wall = timer_stop(&t_total);
    double compute_elapsed = timer_stop(&t_compute);

    /* ── Step 9: Performance summary ──────────────────────── */
    separator();
    printf("  Performance Summary\n");
    separator();
    printf("  T_total  (wall clock)  : %.6f s\n", total_wall);
    printf("  T_compute (approx)     : %.6f s\n", compute_elapsed);
    printf("  T_IO      (last query) : %.6f s\n", t_io_elapsed);

    /* Step 9: Memory usage */
    long peak_kb = get_peak_rss_kb();
    if (peak_kb >= 0)
        printf("  Peak CPU RAM (VmPeak)  : %.2f MB\n", (double)peak_kb / 1024.0);
    else
        printf("  Peak CPU RAM           : (unavailable on this OS)\n");

    separator();
    printf("\n  NOTES FOR PARALLEL EXTENSIONS\n");
    printf("  ─────────────────────────────\n");
    printf("  OpenMP : Add -fopenmp and insert #pragma omp parallel for\n");
    printf("           above the loops in encode_kmers() and sort_unique().\n");
    printf("           Set OMP_NUM_THREADS=<n> before running.\n");
    printf("           Speedup  S_p = T_serial / T_omp\n");
    printf("           Efficiency E_p = S_p / p  (p = thread count)\n");
    printf("\n");
    printf("  CUDA   : Port encode_kmers() to a 1-D kernel (one thread per\n");
    printf("           k-mer position). Sort with thrust::sort. Measure\n");
    printf("           T_H2D, T_kernel, T_D2H separately with CUDA events.\n");
    printf("           T_CUDA_total = T_H2D + T_kernel + T_D2H\n");
    separator();
    printf("\n");

    free(seq_buf);
    free(kmers_ref);
    free(kmers_query);
    return EXIT_SUCCESS;
}