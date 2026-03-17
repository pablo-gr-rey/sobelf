import matplotlib.pyplot as plt

def parse_output(filename):
    with open(filename) as f:
        lines = f.readlines()

    l1, l2 = [], []
    for line in lines:
        if "duration1 duration2" in line:
            # should be duration1 duration2 (float) (float)
            try:
                f1 = float(line.split()[-2].strip())
                f2 = float(line.split()[-1].strip())
                l1.append(f1)
                l2.append(f2)
            except:
                print ("failed to parse line", line)
    
    return l1, l2

times1, _ = parse_output("temp/output-n1N1.out")
times2og, lagog = parse_output("temp/output-n2N2.out")

# plot only 1/2

xs = list(range(2, len(times1)+1, 2))

times2 = [max(times2og[i], times2og[i+1]) for i in range(0, len(times2og), 2)]
full_time = [times2[i//2] + max(lagog[i], lagog[i+1]) for i in range(0, len(times2og), 2)]

print (times1)
print (times2)
print (full_time)

ax = plt.figure().gca()

plt.plot(xs, times1[1::2], "g--", label="N=1, n=1 (reference, opt.)")
plt.plot(xs, times2, "r", label="N=2, n=2 (not incl. comm, opt.)")
plt.plot(xs, full_time, "b", label="N=2, n=2 (incl. comm, opt.)")

plt.legend()
plt.xlabel("Number of total threads (n*t)")
plt.ylabel("Total time")

plt.xticks(xs)        # show only integers on x axis
ax.set_ylim(bottom=0)

plt.title("N=2, n=2, on mandelbrot1.gif (32 frames, 1920*1080)")
plt.savefig("figures/figure.svg", bbox_inches="tight", format="svg", dpi=1200)
plt.show()