# INF560 Final Project - Mathieu Le Louedec & Pablo Rey

## Compiling and running the project

To compile, simply run `make` (after setting the environment to find the CUDA includes, as described in the labs).

To run the project, it is easier to use the script `gen_cmd.py` to automatically generate the full command. It takes as argument the number of threads (`-t`), of processes (`-n`), of nodes (`-N`), the image input and output (`-i` and `-o`) as well as an optional flag `-r` to report the effective bindings. For example, the following command will run the sobelf script with 8 threads per process, 2 processes on 2 nodes on the given files (and report the bindings distribution).

```
eval "$(python gen_cmd.py images/original/mandelbrot2.gif -t 8 -n 2 -N 2 -r -o images/processed/mandelbrot2-sobelf.gif)
```

By default, it starts the processes on the machines of the "bone" room (acromion, apophyse, astragale and cuboide) to have consistent architecture and results. More details are included in the script itself.

An exemple of generated command (with 8 threads, 2 processes, 2 nodes) is:
```
OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close salloc --nodelist acromion,apophyse,astragale,cuboide -N 2 -n 2 --cpus-per-task 8 mpirun --report-bindings --bind-to core --map-by socket:pe=8 ./sobelf input.gif output.gif
```

## Description of the source files

The main code is contained in `main.c`, and contains all the CPU-side implementation.

If `#define KERNEL_BLUR` is present, then the GPU kernel contained in `kernel.cu` will be used for the blur phase. Otherwise, the function `apply_blur_filter` will be used (with the fast sliding window algorithm if `#define FAST_BLUR` is present, and the original slower method otherwise).

## Python utilities

We made several python and bash scripts to help with the development of the project.

- `gen_cmd.py`, as described above, allows easily generating the full launch command with default parameters
- `test_temps.sh` allows automatically running the application with different parameters
- `display_graph.py` allows reading the output of the previous script to display a graph
- `gen_gif.sh` uses the python library `pymandel` (assumed to be installed in a virtual environment `.venv`) to generate Mandelbrot gifs with the chosen resolution and number of frames.