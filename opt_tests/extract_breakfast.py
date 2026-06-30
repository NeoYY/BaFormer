"""Extract ONLY the data/breakfast/ subtree from the RAM-resident data.zip to
disk, then verify the layout BaFormer expects.
"""
import os
import sys
import time
import zipfile

ZIP = "/dev/shm/data.zip"
DEST = os.path.expanduser("~/baformer_data")  # -> ~/baformer_data/data/breakfast/...
PREFIX = "data/breakfast/"


def main():
    os.makedirs(DEST, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.infolist() if m.filename.startswith(PREFIX)]
        total = len(members)
        total_bytes = sum(m.file_size for m in members)
        print(f"breakfast members: {total}  uncompressed: {total_bytes/1e9:.2f} GB")
        done = 0
        for m in members:
            z.extract(m, DEST)
            done += 1
            if done % 250 == 0 or done == total:
                el = time.time() - t0
                print(f"  {done}/{total} extracted ({el:.0f}s)", flush=True)
    print(f"DONE extract in {time.time()-t0:.0f}s -> {DEST}/{PREFIX}")

    # ---- verify layout ----
    root = os.path.join(DEST, "data")
    bf = os.path.join(root, "breakfast")
    feats = os.path.join(bf, "features")
    gts = os.path.join(bf, "groundTruth")
    splits = os.path.join(bf, "splits")
    mapping = os.path.join(bf, "mapping.txt")
    n_feat = len(os.listdir(feats)) if os.path.isdir(feats) else 0
    n_gt = len(os.listdir(gts)) if os.path.isdir(gts) else 0
    n_classes = 0
    if os.path.isfile(mapping):
        with open(mapping) as f:
            n_classes = sum(1 for line in f if line.strip())
    print("\n=== LAYOUT CHECK ===")
    print("dataset_dir (use this):", root + "/")
    print("features .npy :", n_feat)
    print("groundTruth   :", n_gt)
    print("classes(mapping):", n_classes)
    for s in ["train.split1.bundle", "test.split1.bundle"]:
        p = os.path.join(splits, s)
        n = 0
        if os.path.isfile(p):
            with open(p) as f:
                n = sum(1 for line in f if line.strip())
        print(f"  {s}: {n} videos")


if __name__ == "__main__":
    main()
