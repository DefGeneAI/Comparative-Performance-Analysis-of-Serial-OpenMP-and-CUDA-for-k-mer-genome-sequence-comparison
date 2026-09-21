#!/bin/bash

REF="Reference/Escherichia coli str. K-12 substr. MG1655, complete genome.txt"

QUERIES=(
"Queries/Bacillus subtilis subsp. subtilis str. 168 complete genome.fa"
"Queries/E coli Sakai.txt"
"Queries/Escherichia coli str. K-12 substr. W3110 DNA, complete genome.fa"
"Queries/Klebsiella pneumoniae subsp. pneumoniae MGH 78578, complete sequence.fa"
"Queries/Mycobacterium tuberculosis H37Rv, complete genome.fa"
"Queries/Pseudomonas aeruginosa PAO1, complete genome.fa"
"Queries/Salmonella enterica subsp. enterica serovar Typhimurium str. LT2, complete genome.fa"
"Queries/Shigella flexneri 2a str. 301 chromosome, complete genome.fa"
"Queries/Staphylococcus aureus subsp. aureus N315, complete sequence.fa"
"Queries/Yersinia pestis CO92, complete sequence.fa"
)

THREADS=(1 4 10 16)

mkdir -p results/openmp

for K in 15 21 31
do

    if [ "$K" -eq 15 ]; then
        SERIAL=82.991948
    elif [ "$K" -eq 21 ]; then
        SERIAL=81.363749
    elif [ "$K" -eq 31 ]; then
        SERIAL=76.327713
    fi

    for T in "${THREADS[@]}"
    do
        echo "=========================================="
        echo "Running k=$K with $T threads"
        echo "=========================================="

        OMP_NUM_THREADS=$T ./kmer_openmp \
        "$REF" \
        "${QUERIES[@]}" \
        -k "$K" \
        -r 5 \
        -s "$SERIAL" \
        | tee "results_openmp/k${K}_threads${T}.txt"

        echo ""
        echo "Finished k=$K, threads=$T"
        echo ""
    done

done

echo "=========================================="
echo "ALL OPENMP EXPERIMENTS COMPLETED"
echo "=========================================="
