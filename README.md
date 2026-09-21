# Comparative Performance Analysis of Serial, OpenMP & CUDA Implementations for k-mer-Based Genome Sequence Comparison

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/c++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-brightgreen.svg)

## 📌 Project Overview

Comparing massive genomic sequences using traditional alignment methods (e.g., Needleman-Wunsch, Smith-Waterman) scales poorly with large dataset sizes. Alignment-free methods based on **k-mers**—dividing a DNA sequence into overlapping substrings of length $k$—provide a scalable, highly parallelizable alternative.

This project implements and compares a **k-mer-based genome sequence comparison pipeline** across three execution paradigms:
1. **Serial CPU Execution** (Single-threaded baseline)
2. **OpenMP Multicore CPU Parallelism** (Multi-threaded shared memory)
3. **CUDA GPU Acceleration** (Massively parallel thread processing)

By unifying 2-bit nucleotide encoding, k-mer extraction, deduplication, and Jaccard similarity estimation across all three models, this study systematically isolates performance differences, scaling behaviors, speedups, and hardware resource utilization.

---

## 👥 Authors & Team
* **Karen Francina Morais** (`26MCB1019`)
* **Janet Aikya K** (`26MCB1006`)
* **Numa Jalal** (`26MCB1018`)

---

## 🧬 Algorithm & Methodology

The pipeline follows a standardized workflow across all three execution modes to ensure exact numerical equivalence:

```
                  +---------------------+
                  |   FASTA Sequences   |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |  2-Bit Nucleotide   |
                  |  Encoding (A,C,G,T) |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |  k-mer Extraction   |
                  |    (k = 15,21,31)   |
                  +----------+----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
  [Serial Engine]    [OpenMP Engine]      [CUDA Engine]
   Sequential         Multi-threaded       NVIDIA Thrust
  unordered_set    Thread-Local Buffers   Sort & Deduplicate
         |                   |                   |
         +-------------------+-------------------+
                             |
                             v
                  +---------------------+
                  |  Unique k-mer Sets  |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |   Set Intersection  |
                  |     & Jaccard       |
                  | Similarity Computation|
                  +---------------------+
```

### Jaccard Similarity Formula
The similarity between a reference genome set $A$ and query genome set $B$ is calculated as:
$$\text{Jaccard Similarity } J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

---

## 💻 System Configuration & Hardware Setup

| Component | Hardware / Software Specification |
| :--- | :--- |
| **GPU** | NVIDIA Tesla T4 (40 SMs, Compute Capability 7.5, 14.56 GB Global Memory) |
| **CPU** | Intel(R) Xeon(R) CPU @ 2.00GHz (2 Cores / 4 Threads visible via Hyperthreading) |
| **CUDA Driver / Runtime** | Driver 580.82.07 / CUDA 12.8 / `nvcc` Compiler 12.8.93 |
| **Reference Genome** | *Escherichia coli* K-12 MG1655 (4,641,652 bp) |
| **Query Genomes** | 10 Bacterial Query Genomes |

---

## 📊 Experimental Results & Comparison

### 1. Genome-Wise Similarity (Validation Across Execution Modes)
All three models (Serial, OpenMP, CUDA) produced identical similarity values, proving numerical correctness.

| # | Query Genome | Jaccard ($k=15$) | Jaccard ($k=21$) | Jaccard ($k=31$) |
| :-: | :--- | :-: | :-: | :-: |
| 1 | *Bacillus subtilis* | 0.004242 (0.42%) | 0.000075 (0.01%) | 0.000023 (0.00%) |
| 2 | *E. coli* Sakai | 0.508365 (50.84%) | 0.452032 (45.20%) | 0.383729 (38.37%) |
| 3 | *E. coli* W3110 | **0.718619 (71.86%)** | **0.715905 (71.59%)** | **0.715566 (71.56%)** |
| 4 | *Klebsiella pneumoniae* | 0.027892 (2.79%) | 0.010492 (1.05%) | 0.005348 (0.53%) |
| 5 | *Mycobacterium tuberculosis* | 0.004960 (0.50%) | 0.000036 (0.00%) | 0.000012 (0.00%) |
| 6 | *Pseudomonas aeruginosa* | 0.008145 (0.81%) | 0.000299 (0.03%) | 0.000138 (0.01%) |
| 7 | *Salmonella* Typhimurium | 0.036027 (3.60%) | 0.016009 (1.60%) | 0.008292 (0.83%) |
| 8 | *Shigella flexneri* | 0.419123 (41.91%) | 0.377481 (37.75%) | 0.327059 (32.71%) |
| 9 | *Staphylococcus aureus* | 0.002981 (0.30%) | 0.000076 (0.01%) | 0.000028 (0.00%) |
| 10 | *Yersinia pestis* | 0.008423 (0.84%) | 0.001477 (0.15%) | 0.000729 (0.07%) |

> **Key Takeaway:** Increasing $k$-mer size makes exact matches significantly more restrictive for distantly related species, while closely related strains (*E. coli* W3110) preserve stable similarity scores (~71.5%).

---

### 2. Execution Time Performance Summary

| Metric / Execution Model | Serial Baseline | OpenMP (Best Config) | CUDA GPU (Tesla T4) |
| :--- | :-: | :-: | :-: |
| **Total Wall Time ($k=15$)** | 82.992 s | 29.909 s (16 threads) | **1.101 s** |
| **Total Wall Time ($k=21$)** | 81.364 s | 35.962 s (10 threads) | **1.510 s** |
| **Total Wall Time ($k=31$)** | 76.328 s | 35.022 s (10 threads) | **1.553 s** |
| **Max Speedup vs Serial ($k=15$)** | 1.00× | 2.78× | **73.88×** |
| **Max Speedup vs Serial ($k=21$)** | 1.00× | 2.26× | **53.89×** |
| **Max Speedup vs Serial ($k=31$)** | 1.00× | 2.40× | **52.39×** |
| **Peak RAM / VRAM Usage** | ~2235 MB CPU | ~2884 – 3426 MB CPU | ~14.5 GB Available VRAM |

---

### 3. OpenMP Thread Scaling Analysis

| $k$ Size | Threads | OpenMP Time (s) | Speedup | Parallel Efficiency |
| :-: | :-: | :-: | :-: | :-: |
| **15** | 1 | 65.190 | 1.273× | 127.31% |
| **15** | 4 | 48.057 | 1.727× | 43.17% |
| **15** | 10 | 38.107 | 2.178× | 21.78% |
| **15** | **16** | **29.909** | **2.775×** | **17.34%** |
| **21** | 1 | 58.723 | 1.386× | 138.56% |
| **21** | 4 | 41.193 | 1.975× | 49.38% |
| **21** | **10** | **35.962** | **2.263×** | **22.63%** |
| **21** | 16 | 40.327 | 2.018× | 12.61% |
| **31** | 1 | 84.000 | 0.909× | 90.87% |
| **31** | 4 | 42.034 | 1.816× | 45.40% |
| **31** | 10 | 35.022 | 2.179× | 21.79% |
| **31** | **16** | **31.809** | **2.400×** | **15.00%** |

---

## 🚀 Key Insights & Findings

1. **GPU Superiority**: CUDA execution demonstrated massive performance advantages, achieving up to **73.88× speedup** over the serial baseline at $k=15$, completing all 10 query comparisons in just **1.10 seconds**.
2. **OpenMP Trade-offs**: OpenMP successfully scaled multi-threaded execution on CPU (up to **2.78× speedup**). However, efficiency diminished with higher thread counts due to dynamic thread management overhead, lock-free synchronization, and CPU-RAM bandwidth bottlenecks.
3. **Memory Overhead**: Parallelizing k-mer deduplication across multiple CPU threads or GPU blocks requires allocating thread-local containers, which increased peak memory footprint from ~2.23 GB (Serial) to ~3.42 GB (OpenMP).

---

## 🛠️ Build and Compilation Instructions

### Prerequisites
* GCC / G++ supporting C++17 with OpenMP enabled
* NVIDIA CUDA Toolkit 12.x or later
* CMake (v3.18+)

### Building the Project

```bash
# Clone repository
git clone https://github.com/your-username/kmer-genome-comparison.git
cd kmer-genome-comparison

# Create build directory
mkdir build && cd build

# Configure and compile
cmake ..
make -j4
```

### Running the Executables

```bash
# Run Serial Baseline
./kmer_serial --ref reference.fasta --queries query_list.txt --k 21

# Run OpenMP Implementation
./kmer_openmp --ref reference.fasta --queries query_list.txt --k 21 --threads 16

# Run CUDA GPU Acceleration
./kmer_cuda --ref reference.fasta --queries query_list.txt --k 21
```

---

## 📜 Citation & References

If you use or reference this dataset or implementation in your research, please refer to:
* **Moeckel et al. (2024)** - *A Survey of k-mer Methods and Applications in Bioinformatics*.
* **RapidGKC Authors (2024)** - *GPU-Accelerated K-Mer Counting*, IEEE ICDE.
* **Nisa et al. (2021)** - *Distributed-Memory k-mer Counting on GPUs*, IEEE IPDPS.

---
*Developed for the Computer Organization & Architecture Course Project.*
