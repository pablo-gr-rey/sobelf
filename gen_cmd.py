"""
Usage:
python test.py [input] [-N N] [-n n] [-t numThreads] [-o output]

run ./sobelf on the given outputgif with the specified number of tasks, nodes and threads.
input defaults to images/original/Mandelbrot-large.gif
output defaults to /dev/null as we are mostly interested in the filtering part.

À utiliser avec eval "$(python gen_cmd.py -t 8)"
"""

import argparse

DEFAULT_INPUT = "images/original/Mandelbrot-large.gif"
DEFAULT_OUTPUT = "/dev/null"

COMMAND = "OMP_NUM_THREADS={t} salloc -N {N} -n {n} mpirun ./sobelf {input} {o}"

parser = argparse.ArgumentParser(prog="test", description="Run sobelf on the given file")

parser.add_argument("input", type=str, nargs="?", default=DEFAULT_INPUT, help="input GIF")
parser.add_argument("-N", type=int, nargs="?", default=1, help="number of nodes")
parser.add_argument("-n", type=int, nargs="?", default=1, help="number of tasks")
parser.add_argument("-t", type=int, nargs="?", default=1, help="number of OMP threads")
parser.add_argument("-o", type=str, nargs="?", default=DEFAULT_OUTPUT, help="output GIF")

args = parser.parse_args()

print (COMMAND.format(**vars(args)))