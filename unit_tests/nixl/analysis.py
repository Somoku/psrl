import heapq
import os
import re


def extract_top_times_per_file(log_dir: str, k: int = 20):
    pattern = re.compile(r"time:\s*([0-9.eE+-]+)")
    files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]

    if not files:
        print("No files found in directory.")
        return

    for fname in files:
        fpath = os.path.join(log_dir, fname)
        entries = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    t = float(match.group(1))
                    entries.append((t, line.strip()))

        if not entries:
            print(f"[{fname}] No 'time:' entries found.\n")
            continue

        total_time = sum(t for t, _ in entries)
        top_k = heapq.nlargest(k, entries, key=lambda x: x[0])

        print(f"=== File: {fname} ===")
        print(f"Found {len(entries)} time entries.")
        print(f"Total time sum: {total_time:.6f}s\n")

        print(f"Top {k} longest times:")
        for rank, (t, line) in enumerate(top_k, 1):
            print(f"  [{rank}] {t:.6f}s → {line}")
        print("\n")


if __name__ == "__main__":
    extract_top_times_per_file(log_dir="./log", k=20)
