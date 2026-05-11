"""
setup_imposters.py
------------------
Creates the imposters/ folder and copies the generated imposter face images
into it from the antigravity artifacts directory.

Run once:
    python setup_imposters.py
"""

import os
import shutil

# Paths
ARTIFACT_DIR = r"C:\Users\bhardwaj\.gemini\antigravity\brain\2a7b37ef-3b48-40d6-abbd-ee8ed90cf1b0"
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMPOSTERS_DIR = os.path.join(PROJECT_DIR, "imposters")

# Mapping: artifact filename → destination filename
IMAGE_MAP = {
    "imposter_1_1777809605416.png": "imposter_1.png",
    "imposter_2_1777809621989.png": "imposter_2.png",
    "imposter_3_1777809637523.png": "imposter_3.png",
}

def main():
    os.makedirs(IMPOSTERS_DIR, exist_ok=True)
    print(f"[Setup] Created folder: {IMPOSTERS_DIR}")

    copied = 0
    for src_name, dst_name in IMAGE_MAP.items():
        src = os.path.join(ARTIFACT_DIR, src_name)
        dst = os.path.join(IMPOSTERS_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Setup] Copied: {dst_name}")
            copied += 1
        else:
            print(f"[Setup] WARNING: Source not found: {src}")

    print(f"\n[Setup] Done — {copied}/{len(IMAGE_MAP)} imposter images ready in imposters/")
    if copied == len(IMAGE_MAP):
        print("[Setup] You can now run:  python main.py --skip_k_analysis")

if __name__ == "__main__":
    main()
