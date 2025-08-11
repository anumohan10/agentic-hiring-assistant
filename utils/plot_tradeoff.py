# utils/plot_tradeoff.py
import json, sys
import matplotlib.pyplot as plt

def load_point(path):
    with open(path) as f:
        d = json.load(f)
    return float(d["mean_service"]), float(d["mean_total_cost"])

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m utils.plot_tradeoff results/eval/*.json")
        sys.exit(1)

    pts = []
    for p in sys.argv[1:]:
        try:
            s, c = load_point(p)
            pts.append((s, c, p))
        except Exception:
            pass

    pts.sort()  # sort by service
    xs = [s for s, _, _ in pts]
    ys = [c for _, c, _ in pts]
    labels = [p.split("/")[-1].replace(".json","") for _, _, p in pts]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    for x,y,l in zip(xs, ys, labels):
        plt.annotate(l, (x,y), textcoords="offset points", xytext=(5,5))
    plt.xlabel("Service Level")
    plt.ylabel("Total Cost per Episode")
    plt.title("Service vs Cost Trade-off")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
