"""Copy every video into data/renamed/ as bowling_1, bowling_2, ... (sequenced,
grouped by folder). Originals are left untouched. Writes a manifest mapping each
new name back to its original path so nothing is lost.
"""
import csv, os, re, shutil

DATA = "data"
OUT = os.path.join(DATA, "renamed")
EXTS = (".mp4", ".mov", ".avi", ".mkv")
# folder groups in the requested order
GROUPS = [
    "data/raw/own_recordings",
    "data/raw/Pune_pace_academy/JP",
    "data/raw/Pune_pace_academy/Tausif",
    "data/ball_autolabel/qa",
]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main():
    os.makedirs(OUT, exist_ok=True)
    # collect any group dirs that exist, plus a catch-all for anything missed
    seen = set()
    ordered = []
    for g in GROUPS:
        if os.path.isdir(g):
            files = [f for f in os.listdir(g) if f.lower().endswith(EXTS)]
            for f in sorted(files, key=natural_key):
                p = os.path.join(g, f)
                ordered.append(p); seen.add(os.path.normcase(os.path.abspath(p)))
    # catch any videos in other data subdirs not covered above
    extra = []
    for root, _, files in os.walk(DATA):
        if ".venv" in root or os.path.abspath(root).startswith(os.path.abspath(OUT)):
            continue
        for f in files:
            if f.lower().endswith(EXTS):
                p = os.path.join(root, f)
                if os.path.normcase(os.path.abspath(p)) not in seen:
                    extra.append(p)
    for p in sorted(extra, key=lambda x: natural_key(os.path.basename(x))):
        ordered.append(p)

    rows = []
    for i, src in enumerate(ordered, 1):
        ext = os.path.splitext(src)[1].lower()
        new = f"bowling_{i}{ext}"
        shutil.copy2(src, os.path.join(OUT, new))
        rows.append((new, os.path.relpath(src, DATA).replace(os.sep, "/"),
                     os.path.dirname(os.path.relpath(src, DATA)).replace(os.sep, "/")))

    with open(os.path.join(OUT, "_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "original_path", "source_folder"])
        w.writerows(rows)

    print(f"copied {len(rows)} videos -> {OUT}/")
    print("groups:")
    from collections import Counter
    for folder, n in Counter(r[2] for r in rows).most_common():
        idxs = [int(r[0].split('_')[1].split('.')[0]) for r in rows if r[2] == folder]
        print(f"  {folder}: bowling_{min(idxs)}..bowling_{max(idxs)}  ({n})")
    print(f"manifest: {OUT}/_manifest.csv")


if __name__ == "__main__":
    main()
