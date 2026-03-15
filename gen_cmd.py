"""
Usage:
python gen_cmd.py [input] [-N N] [-n n] [-t numThreads] [-o output] [-r]

run ./sobelf on the given outputgif with the specified number of tasks, nodes and threads.
N, n and t default to 1
input defaults to DEFAULT_INPUT
output defaults to DEFAULT_OUTPUT

python gen_cmd.py -h for full option list

À utiliser avec eval "$(python gen_cmd.py -t 8)"
"""

import argparse, sys, math

DEFAULT_INPUT = "images/original/mandelbrot2.gif"
DEFAULT_OUTPUT = "images/processed/mandelbrot2-sobelf.gif"

# --map-by socket:pe={cpu} --bind-to core
COMMAND = "OMP_NUM_THREADS={t} OMP_PLACES=cores OMP_PROC_BIND=close salloc --nodelist {nodes} -N {N} -n {n} --cpus-per-task {t} mpirun {report} --bind-to core --map-by socket:pe={t} ./sobelf {input} {o}"

# each of these hosts has 1 socket, 20 cores per socket, and 2 threads per cores (but that's hyperthreading: we should just use 1 thread per core)
hosts = ["acromion", "apophyse", "astragale", "cuboide"]

parser = argparse.ArgumentParser(prog="gen_cmd", description="Generate full command to run sobelf with the given options")

parser.add_argument("input", type=str, nargs="?", default=DEFAULT_INPUT, help="input GIF")
parser.add_argument("-N", type=int, nargs="?", default=1, help="number of nodes")
parser.add_argument("-n", type=int, nargs="?", default=1, help="number of tasks")
parser.add_argument("-t", type=int, nargs="?", default=1, help="number of OMP threads")
parser.add_argument("-o", type=str, nargs="?", default=DEFAULT_OUTPUT, help="output GIF")
parser.add_argument("-r", help="add --report-bindings", action="store_true")

args = parser.parse_args()

args.report = "--report-bindings" if args.r else ""
args.cpu = args.t * args.n
args.nodes = ','.join(hosts)

if args.N > len(hosts):
    sys.stderr.write(f"WARNING: there are only {len(hosts)} hosts hardcoded, add some to the list (ideally from the bone lab room)")
if (total_threads := int(math.ceil(args.t * args.n / args.N))) > 20:
    sys.stderr.write(f"WARNING: there are 20 threads per node (without hyperthreading), these parameters would need {total_threads}\n")

print (COMMAND.format(**vars(args)))