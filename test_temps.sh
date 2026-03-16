for i in $(seq 1 16);
do
    echo $i " nodes :"
    # export OMP_NUM_THREADS=$i
    # salloc -N 1 ./sobelf images/original/Mandelbrot-large.gif images/processed/Mandelbrot-large-sobel.gif
    eval $(python gen_cmd.py ./images/original/mandelbrot1.gif -o /dev/null -n 1 -N 1 -t $i)
done
