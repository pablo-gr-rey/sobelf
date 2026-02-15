for i in $(seq 1 10);
do
    echo $i " nodes :"
    export OMP_NUM_THREADS=$i
    salloc -N 1 ./sobelf images/original/Mandelbrot-large.gif images/processed/Mandelbrot-large-sobel.gif
done
