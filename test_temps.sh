for i in $(seq 1 20);
do
    echo $i " nodes :"
    export OMP_NUM_THREADS=$i
    salloc -N 1 ./sobelf images/original/TimelyHugeGnu.gif images/processed/TimelyHugeGnu-sobel.gif
done
