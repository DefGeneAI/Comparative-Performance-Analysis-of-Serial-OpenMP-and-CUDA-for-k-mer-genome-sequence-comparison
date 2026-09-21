/*
 * kmer_openmp.c
 * ============================================================
 * OpenMP implementation of k-mer-based genome sequence
 * comparison using:
 *
 *   1. FASTA/plain-text input
 *   2. 2-bit nucleotide encoding
 *   3. k-mer generation
 *   4. Parallel merge sort
 *   5. k-mer deduplication
 *   6. Jaccard similarity
 *
 * IMPORTANT:
 * The computational methodology is kept the SAME as the
 * serial implementation. OpenMP is used only to parallelize
 * suitable computational sections.
 *
 * OpenMP parallel sections:
 *
 *   - k-mer generation
 *   - merge-sort
 *
 * Jaccard intersection remains sequential because it is already
 * a linear two-pointer traversal of sorted arrays.
 *
 * Compile:
 *   gcc -O2 -fopenmp -o kmer_openmp kmer_openmp.c -lm
 *
 * Usage:
 *
 *   ./kmer_openmp <reference.fa> <query1.fa> [query2.fa ...]
 *       -k <15|21|31>
 *       [-r <reps>]
 *       [-s <serial_total_time>]
 *
 * Example:
 *
 *   ./kmer_openmp \
 *       "Reference/Escherichia coli str. K-12 substr. MG1655, complete genome.txt" \
 *       "Queries/Bacillus subtilis subsp. subtilis str. 168 complete genome.fa" \
 *       "Queries/E coli Sakai.txt" \
 *       -k 15 -r 5 -s 82.991948
 *
 * Thread count:
 *
 *   OMP_NUM_THREADS=4 ./kmer_openmp ...
 *
 *   OMP_NUM_THREADS=8 ./kmer_openmp ...
 *
 *   OMP_NUM_THREADS=16 ./kmer_openmp ...
 *
 * Speedup:
 *
 *   S_p = T_serial / T_openmp
 *
 * Efficiency:
 *
 *   E_p = S_p / p
 *
 * where:
 *
 *   p = number of OpenMP threads
 *
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>


/* ============================================================
 * CONFIGURATION
 * ============================================================ */

#define MAX_SEQ_LEN   (1UL << 27)   /* 128 MB */
#define MAX_KMERS     (1UL << 27)   /* 134 million k-mers */
#define DEFAULT_K     21
#define DEFAULT_REPS  5

/*
 * Parallel merge-sort cutoff.
 *
 * Small partitions are sorted using qsort because creating
 * OpenMP tasks for tiny arrays would create unnecessary overhead.
 */
#define SORT_CUTOFF 100000


/* ============================================================
 * 2-BIT NUCLEOTIDE ENCODING
 *
 * A = 00
 * C = 01
 * G = 10
 * T = 11
 * ============================================================ */

static const uint8_t NUC2BIT[256] = {
    ['A'] = 0, ['a'] = 0,
    ['C'] = 1, ['c'] = 1,
    ['G'] = 2, ['g'] = 2,
    ['T'] = 3, ['t'] = 3
};


/* ============================================================
 * NUCLEOTIDE VALIDATION
 * ============================================================ */

static inline int is_valid_nuc(char c)
{
    return (c=='A'||c=='a'||
            c=='C'||c=='c'||
            c=='G'||c=='g'||
            c=='T'||c=='t');
}


/* ============================================================
 * TIMER
 * ============================================================ */

typedef struct {
    double wall;
} Timer;


static inline void timer_start(Timer *t)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    t->wall =
        (double)ts.tv_sec +
        (double)ts.tv_nsec * 1e-9;
}


static inline double timer_stop(Timer *t)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    double now =
        (double)ts.tv_sec +
        (double)ts.tv_nsec * 1e-9;

    return now - t->wall;
}


/* ============================================================
 * MEMORY USAGE
 *
 * Linux:
 * /proc/self/status -> VmPeak
 * ============================================================ */

static long get_peak_rss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");

    if (!f)
        return -1;

    char line[256];

    long peak = -1;

    while (fgets(line, sizeof(line), f)) {

        if (strncmp(line, "VmPeak:", 7) == 0) {

            sscanf(line + 7, "%ld", &peak);

            break;
        }
    }

    fclose(f);

    return peak;
}


/* ============================================================
 * FASTA / PLAIN TEXT READER
 * ============================================================ */

static size_t read_sequence(
    const char *path,
    char *buf,
    size_t maxlen)
{
    FILE *fp = fopen(path, "r");

    if (!fp) {

        fprintf(stderr,
                "[ERROR] Cannot open file: %s\n",
                path);

        return 0;
    }

    size_t n = 0;

    char line[4096];

    while (fgets(line, sizeof(line), fp)
           && n < maxlen) {

        /* Skip FASTA header */
        if (line[0] == '>')
            continue;

        for (size_t i = 0;
             line[i] && n < maxlen;
             i++) {

            char c = line[i];

            if (is_valid_nuc(c))
                buf[n++] = c;
        }
    }

    fclose(fp);

    buf[n] = '\0';

    return n;
}


/* ============================================================
 * K-MER TYPE
 * ============================================================ */

typedef uint64_t kmer_t;


/* ============================================================
 * OPENMP K-MER GENERATION
 *
 * SAME ALGORITHM AS SERIAL:
 *
 * For every position i:
 *
 *   take k nucleotides
 *   convert each nucleotide to 2 bits
 *   combine into 64-bit value
 *
 * Each iteration writes to out[i].
 *
 * Therefore iterations are independent and can safely be
 * executed in parallel.
 * ============================================================ */

static size_t encode_kmers(
    const char *seq,
    size_t seqlen,
    int k,
    kmer_t *out,
    size_t out_max)
{
    if ((size_t)k > seqlen)
        return 0;

    const size_t num_kmers =
        seqlen - (size_t)k + 1;

    if (num_kmers > out_max) {

        fprintf(stderr,
                "[WARN] Too many k-mers (%zu); truncating.\n",
                num_kmers);
    }

    const size_t count =
        (num_kmers < out_max)
        ? num_kmers
        : out_max;


    /*
     * Same mask as serial implementation.
     */
    kmer_t mask;

    if (k < 32)
        mask = ((kmer_t)1 << (2 * k)) - 1;
    else
        mask = ~(kmer_t)0;


    /*
     * ========================================================
     * OPENMP PARALLEL LOOP
     * ========================================================
     *
     * Each iteration calculates ONE k-mer.
     *
     * Example:
     *
     * Thread 0 -> i = 0,1,2...
     * Thread 1 -> other positions
     * Thread 2 -> other positions
     *
     * Every iteration writes to a unique out[i].
     *
     * Therefore there is no race condition.
     */

    #pragma omp parallel for schedule(static)

    for (size_t i = 0;
         i < count;
         i++) {

        kmer_t val = 0;

        int valid = 1;

        for (int j = 0;
             j < k;
             j++) {

            char c = seq[i + j];

            if (!is_valid_nuc(c)) {

                valid = 0;

                break;
            }

            val =
                (val << 2) |
                NUC2BIT[(unsigned char)c];
        }

        if (valid)
            out[i] = val & mask;
        else
            out[i] = (kmer_t)UINT64_MAX;
    }

    return count;
}


/* ============================================================
 * K-MER COMPARATOR
 * ============================================================ */

static int cmp_kmer(
    const void *a,
    const void *b)
{
    kmer_t ka = *(const kmer_t *)a;

    kmer_t kb = *(const kmer_t *)b;

    return (ka > kb) - (ka < kb);
}


/* ============================================================
 * MERGE TWO SORTED ARRAYS
 * ============================================================ */

static void merge_arrays(
    kmer_t *arr,
    kmer_t *temp,
    size_t left,
    size_t mid,
    size_t right)
{
    size_t i = left;
    size_t j = mid;
    size_t k = left;


    while (i < mid && j < right) {

        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }


    while (i < mid)
        temp[k++] = arr[i++];


    while (j < right)
        temp[k++] = arr[j++];


    for (size_t x = left;
         x < right;
         x++) {

        arr[x] = temp[x];
    }
}


/* ============================================================
 * SERIAL MERGE SORT HELPER
 *
 * Used for small partitions.
 * ============================================================ */

static void merge_sort_serial(
    kmer_t *arr,
    kmer_t *temp,
    size_t left,
    size_t right)
{
    if (right - left <= 1)
        return;

    size_t mid =
        left + (right - left) / 2;


    merge_sort_serial(
        arr,
        temp,
        left,
        mid);


    merge_sort_serial(
        arr,
        temp,
        mid,
        right);


    merge_arrays(
        arr,
        temp,
        left,
        mid,
        right);
}


/* ============================================================
 * PARALLEL MERGE SORT
 *
 * Same final operation as qsort:
 *
 *     sorted k-mer array
 *
 * But sorting work is divided among OpenMP tasks.
 * ============================================================ */

static void parallel_merge_sort_recursive(
    kmer_t *arr,
    kmer_t *temp,
    size_t left,
    size_t right)
{
    /*
     * Small partitions:
     * use serial qsort to avoid excessive task overhead.
     */

    if (right - left <= SORT_CUTOFF) {

        qsort(
            arr + left,
            right - left,
            sizeof(kmer_t),
            cmp_kmer);

        return;
    }


    size_t mid =
        left + (right - left) / 2;


    /*
     * Two independent sorting operations.
     */

    #pragma omp task shared(arr, temp)
    {
        parallel_merge_sort_recursive(
            arr,
            temp,
            left,
            mid);
    }


    #pragma omp task shared(arr, temp)
    {
        parallel_merge_sort_recursive(
            arr,
            temp,
            mid,
            right);
    }


    /*
     * Wait for both halves to finish.
     */

    #pragma omp taskwait


    /*
     * Merge sorted halves.
     */

    merge_arrays(
        arr,
        temp,
        left,
        mid,
        right);
}


/* ============================================================
 * PARALLEL SORT + DEDUPLICATION
 * ============================================================ */

static size_t sort_unique(
    kmer_t *arr,
    size_t n)
{
    if (n == 0)
        return 0;


    /*
     * Temporary array required for merge.
     */

    kmer_t *temp =
        (kmer_t *)malloc(
            n * sizeof(kmer_t));


    if (!temp) {

        fprintf(stderr,
                "[ERROR] Cannot allocate temporary "
                "sorting buffer.\n");

        exit(EXIT_FAILURE);
    }


    /*
     * Start one OpenMP parallel region.
     */

    #pragma omp parallel
    {

        #pragma omp single
        {

            parallel_merge_sort_recursive(
                arr,
                temp,
                0,
                n);
        }
    }


    free(temp);


    /*
     * ========================================================
     * Remove invalid sentinels.
     *
     * UINT64_MAX sorts to the end.
     * ========================================================
     */

    size_t valid_end = n;

    while (valid_end > 0 &&
           arr[valid_end - 1] == UINT64_MAX) {

        valid_end--;
    }


    /*
     * ========================================================
     * Deduplicate.
     *
     * This remains sequential.
     * ========================================================
     */

    size_t unique = 0;

    for (size_t i = 0;
         i < valid_end;
         i++) {

        if (i == 0 ||
            arr[i] != arr[i - 1]) {

            arr[unique++] = arr[i];
        }
    }


    return unique;
}


/* ============================================================
 * JACCARD SIMILARITY
 *
 * SAME AS SERIAL IMPLEMENTATION
 *
 * J(A,B) =
 *
 *        |A ∩ B|
 * -------------------
 *        |A ∪ B|
 *
 * ============================================================ */

static double jaccard(
    const kmer_t *A,
    size_t sA,
    const kmer_t *B,
    size_t sB,
    size_t *out_intersect,
    size_t *out_union)
{
    size_t i = 0;

    size_t j = 0;

    size_t inter = 0;


    while (i < sA &&
           j < sB) {

        if (A[i] == B[j]) {

            inter++;

            i++;

            j++;
        }

        else if (A[i] < B[j]) {

            i++;
        }

        else {

            j++;
        }
    }


    size_t uni =
        sA + sB - inter;


    if (out_intersect)
        *out_intersect = inter;


    if (out_union)
        *out_union = uni;


    if (uni == 0)
        return 0.0;


    return
        (double)inter /
        (double)uni;
}


/* ============================================================
 * SEPARATOR
 * ============================================================ */

static void separator(void)
{
    printf(
        "────────────────────────────────────────────────────────────────────────────────\n");
}


/* ============================================================
 * MAIN
 * ============================================================ */

int main(
    int argc,
    char **argv)
{
    /*
     * --------------------------------------------------------
     * Argument checking
     * --------------------------------------------------------
     */

    if (argc < 3) {

        fprintf(stderr,
            "Usage: %s <reference.fa> "
            "<query1.fa> [query2.fa ...] "
            "-k <15|21|31> [-r <reps>] "
            "[-s <serial_total_time>]\n",
            argv[0]);

        return EXIT_FAILURE;
    }


    const char *ref_path =
        argv[1];


    int k =
        DEFAULT_K;


    int reps =
        DEFAULT_REPS;


    /*
     * Serial baseline.
     *
     * If supplied:
     *
     * -s 82.991948
     *
     * then speedup can be calculated.
     */

    double serial_time =
        0.0;


    int serial_time_given =
        0;


    /*
     * --------------------------------------------------------
     * Query paths
     * --------------------------------------------------------
     */

    const char *query_paths[64];

    int n_queries = 0;


    /*
     * --------------------------------------------------------
     * Parse command-line arguments
     * --------------------------------------------------------
     */

    for (int i = 2;
         i < argc;
         i++) {

        if (strcmp(argv[i], "-k") == 0 &&
            i + 1 < argc) {

            k =
                atoi(argv[++i]);
        }

        else if (strcmp(argv[i], "-r") == 0 &&
                 i + 1 < argc) {

            reps =
                atoi(argv[++i]);
        }

        else if (strcmp(argv[i], "-s") == 0 &&
                 i + 1 < argc) {

            serial_time =
                atof(argv[++i]);

            serial_time_given = 1;
        }

        else {

            query_paths[n_queries++] =
                argv[i];
        }
    }


    /*
     * --------------------------------------------------------
     * Validate arguments
     * --------------------------------------------------------
     */

    if (n_queries == 0) {

        fprintf(stderr,
                "[ERROR] At least one query genome "
                "must be specified.\n");

        return EXIT_FAILURE;
    }


    if (k < 1 || k > 32) {

        fprintf(stderr,
                "[ERROR] k must be in [1,32].\n");

        return EXIT_FAILURE;
    }


    if (reps < 1)
        reps = 1;


    /*
     * --------------------------------------------------------
     * OpenMP information
     * --------------------------------------------------------
     */

    int max_threads =
        omp_get_max_threads();


    int processors =
        omp_get_num_procs();


    /*
     * --------------------------------------------------------
     * Header
     * --------------------------------------------------------
     */

    printf("\n");

    separator();

    printf(
        "  k-mer OpenMP Genome Comparison\n");

    printf(
        "  k=%d, repetitions=%d\n",
        k,
        reps);

    separator();


    printf(
        "  Reference : %s\n",
        ref_path);


    printf(
        "  Queries   : %d genome(s)\n",
        n_queries);


    printf(
        "  k-mer size: %d\n",
        k);


    printf(
        "  Repeats   : %d\n",
        reps);


    printf(
        "  OpenMP maximum threads : %d\n",
        max_threads);


    printf(
        "  Available processors   : %d\n",
        processors);


    /*
     * Actual OMP_NUM_THREADS setting can be shown
     * using omp_get_max_threads().
     */

    printf(
        "  Actual OpenMP threads  : %d\n",
        max_threads);


    if (serial_time_given) {

        printf(
            "  Serial baseline        : %.6f s\n",
            serial_time);
    }

    else {

        printf(
            "  Serial baseline        : not supplied\n");
    }


    separator();


    /*
     * --------------------------------------------------------
     * Allocate sequence buffer
     * --------------------------------------------------------
     */

    char *seq_buf =
        (char *)malloc(
            MAX_SEQ_LEN + 1);


    if (!seq_buf) {

        perror("malloc seq_buf");

        return EXIT_FAILURE;
    }


    /*
     * Allocate reference and query k-mer arrays.
     */

    kmer_t *kmers_ref =
        (kmer_t *)malloc(
            MAX_KMERS *
            sizeof(kmer_t));


    kmer_t *kmers_query =
        (kmer_t *)malloc(
            MAX_KMERS *
            sizeof(kmer_t));


    if (!kmers_ref ||
        !kmers_query) {

        perror("malloc kmers");

        free(seq_buf);

        free(kmers_ref);

        free(kmers_query);

        return EXIT_FAILURE;
    }


    /*
     * --------------------------------------------------------
     * TOTAL TIMER
     * --------------------------------------------------------
     */

    Timer t_total;

    Timer t_io;

    Timer t_compute;


    timer_start(&t_total);


    /*
     * --------------------------------------------------------
     * STEP 1
     * Load reference
     * --------------------------------------------------------
     */

    timer_start(&t_io);


    printf(
        "\n[Step 1] Loading reference genome ...\n");


    size_t ref_len =
        read_sequence(
            ref_path,
            seq_buf,
            MAX_SEQ_LEN);


    double ref_io =
        timer_stop(&t_io);


    if (ref_len == 0) {

        free(seq_buf);

        free(kmers_ref);

        free(kmers_query);

        return EXIT_FAILURE;
    }


    printf(
        "         Reference length: %zu bp\n",
        ref_len);


    /*
     * --------------------------------------------------------
     * Generate reference k-mers
     * --------------------------------------------------------
     */

    size_t ref_raw =
        encode_kmers(
            seq_buf,
            ref_len,
            k,
            kmers_ref,
            MAX_KMERS);


    printf(
        "         Raw k-mers (ref): %zu\n",
        ref_raw);


    printf(
        "         T_IO (ref load): %.6f s\n",
        ref_io);


    /*
     * --------------------------------------------------------
     * Reference sorting
     *
     * Start compute timer here.
     * --------------------------------------------------------
     */

    timer_start(&t_compute);


    size_t ref_unique =
        sort_unique(
            kmers_ref,
            ref_raw);


    printf(
        "         Unique k-mers (ref): %zu\n\n",
        ref_unique);


    /*
     * --------------------------------------------------------
     * QUERY PROCESSING
     * --------------------------------------------------------
     */

    for (int q = 0;
         q < n_queries;
         q++) {

        const char *qpath =
            query_paths[q];


        printf(
            "[Query %d] %s\n",
            q + 1,
            qpath);


        /*
         * Store timing of each repetition.
         */

        double times[64];


        int valid_reps = 0;


        double jaccard_val = 0.0;


        size_t inter = 0;

        size_t uni = 0;

        size_t q_unique = 0;

        size_t q_len_last = 0;

        size_t q_raw_last = 0;


        /*
         * ----------------------------------------------------
         * REPETITIONS
         * ----------------------------------------------------
         */

        for (int rep = 0;
             rep < reps;
             rep++) {


            Timer t_rep;

            timer_start(&t_rep);


            /*
             * -----------------------------------------------
             * Load query genome
             * -----------------------------------------------
             */

            timer_start(&t_io);


            size_t q_len =
                read_sequence(
                    qpath,
                    seq_buf,
                    MAX_SEQ_LEN);


            double query_io =
                timer_stop(&t_io);


            if (q_len == 0)
                continue;


            /*
             * -----------------------------------------------
             * Encode query k-mers
             *
             * OPENMP PARALLEL
             * -----------------------------------------------
             */

            size_t q_raw =
                encode_kmers(
                    seq_buf,
                    q_len,
                    k,
                    kmers_query,
                    MAX_KMERS);


            /*
             * -----------------------------------------------
             * Sort + deduplicate
             *
             * OPENMP PARALLEL SORT
             * -----------------------------------------------
             */

            q_unique =
                sort_unique(
                    kmers_query,
                    q_raw);


            /*
             * -----------------------------------------------
             * Jaccard
             *
             * Same sequential algorithm as serial.
             * -----------------------------------------------
             */

            jaccard_val =
                jaccard(
                    kmers_ref,
                    ref_unique,
                    kmers_query,
                    q_unique,
                    &inter,
                    &uni);


            /*
             * Record complete query processing time.
             */

            times[valid_reps++] =
                timer_stop(&t_rep);


            /*
             * Save last values.
             */

            q_len_last = q_len;

            q_raw_last = q_raw;


            /*
             * Print only first repetition.
             */

            if (rep == 0) {

                printf(
                    "         Query length    : %zu bp\n",
                    q_len);

                printf(
                    "         Raw k-mers      : %zu\n",
                    q_raw);

                printf(
                    "         Unique k-mers   : %zu\n",
                    q_unique);
            }
        }


        /*
         * ----------------------------------------------------
         * Check valid repetitions
         * ----------------------------------------------------
         */

        if (valid_reps == 0) {

            printf(
                "         [SKIP] Could not read query.\n\n");

            continue;
        }


        /*
         * ----------------------------------------------------
         * Calculate median
         * ----------------------------------------------------
         */

        for (int a = 1;
             a < valid_reps;
             a++) {

            double key =
                times[a];

            int b = a - 1;


            while (b >= 0 &&
                   times[b] > key) {

                times[b + 1] =
                    times[b];

                b--;
            }


            times[b + 1] =
                key;
        }


        double median_time =
            times[valid_reps / 2];


        /*
         * ----------------------------------------------------
         * RESULTS
         * ----------------------------------------------------
         */

        printf(
            "\n  ┌─ Results ─────────────────────────────────\n");


        printf(
            "  │  |A ∩ B| (intersection) : %zu k-mers\n",
            inter);


        printf(
            "  │  |A ∪ B| (union)        : %zu k-mers\n",
            uni);


        printf(
            "  │  Jaccard similarity      : %.6f (%.2f%%)\n",
            jaccard_val,
            jaccard_val * 100.0);


        printf(
            "  │\n");


        printf(
            "  │  Median OpenMP time (%d reps): %.6f s\n",
            valid_reps,
            median_time);


        printf(
            "  └───────────────────────────────────────────\n\n");
    }


    /*
     * --------------------------------------------------------
     * STOP TOTAL TIMERS
     * --------------------------------------------------------
     */

    double total_wall =
        timer_stop(&t_total);


    double compute_elapsed =
        timer_stop(&t_compute);


    /*
     * --------------------------------------------------------
     * MEMORY
     * --------------------------------------------------------
     */

    long peak_kb =
        get_peak_rss_kb();


    /*
     * --------------------------------------------------------
     * PERFORMANCE SUMMARY
     * --------------------------------------------------------
     */

    separator();


    printf(
        "  OpenMP Performance Summary\n");


    separator();


    printf(
        "  OpenMP threads        : %d\n",
        max_threads);


    printf(
        "  Available processors   : %d\n",
        processors);


    printf(
        "  T_total (wall clock)  : %.6f s\n",
        total_wall);


    printf(
        "  T_compute (approx)    : %.6f s\n",
        compute_elapsed);


    printf(
        "  T_IO (reference)      : %.6f s\n",
        ref_io);


    /*
     * --------------------------------------------------------
     * SPEEDUP
     * --------------------------------------------------------
     */

    if (serial_time_given &&
        total_wall > 0.0) {


        double speedup =
            serial_time /
            total_wall;


        double efficiency =
            speedup /
            (double)max_threads;


        double efficiency_percent =
            efficiency * 100.0;


        printf(
            "\n  Parallel Performance\n");


        printf(
            "  ───────────────────────────────────────────\n");


        printf(
            "  Serial time           : %.6f s\n",
            serial_time);


        printf(
            "  OpenMP time           : %.6f s\n",
            total_wall);


        printf(
            "  Speedup               : %.4fx\n",
            speedup);


        printf(
            "  Efficiency            : %.4f\n",
            efficiency);


        printf(
            "  Efficiency            : %.2f%%\n",
            efficiency_percent);
    }


    else {

        printf(
            "\n  Speedup               : N/A\n");


        printf(
            "  Efficiency            : N/A\n");


        printf(
            "  Use -s <serial_time> to calculate these.\n");
    }


    /*
     * --------------------------------------------------------
     * MEMORY
     * --------------------------------------------------------
     */

    if (peak_kb >= 0) {

        printf(
            "\n  Peak CPU RAM (VmPeak) : %.2f MB\n",
            (double)peak_kb / 1024.0);
    }

    else {

        printf(
            "\n  Peak CPU RAM          : unavailable\n");
    }


    separator();


    /*
     * --------------------------------------------------------
     * END
     * --------------------------------------------------------
     */

    printf(
        "\n  OpenMP implementation completed successfully.\n");


    printf(
        "  Algorithmic results should match the serial version.\n");


    separator();


    /*
     * --------------------------------------------------------
     * FREE MEMORY
     * --------------------------------------------------------
     */

    free(seq_buf);

    free(kmers_ref);

    free(kmers_query);


    return EXIT_SUCCESS;
}