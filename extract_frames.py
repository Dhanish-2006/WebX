"""
Extract 64 WebP frames from public/character.mp4 for the cursor-tracking hero.

Setup:   pip install opencv-python numpy
Step 1:  python extract_frames.py inspect
         -> prints frame count and writes contact_sheet.jpg (every Nth frame, labelled).
         Open it and note: the frame where she looks UP (start), the frame where the
         head has completed the full circle back toward UP (end), and a neutral
         straight-at-camera frame (center). Note the circle must go CLOCKWISE
         (up -> right -> down -> left); use --reverse if your video goes the other way.
Step 2:  python extract_frames.py extract --start 12 --end 180 --center 200 [--reverse]
         -> writes public/frames/frame_00..63.webp and center.webp
"""
import argparse, os
import cv2
import numpy as np

VIDEO = "public/character.mp4"
OUT = "public/frames"
N = 64


def inspect(step):
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames: {total}  FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
    thumbs = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        t = cv2.resize(f, (240, 135))
        cv2.putText(t, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        thumbs.append(t)
    cols = 6
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite("contact_sheet.jpg", np.vstack(rows))
    print("Wrote contact_sheet.jpg")


def extract(start, end, center, reverse):
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    idxs = np.linspace(start, end, N, endpoint=False).round().astype(int)
    if reverse:
        idxs = np.concatenate([[idxs[0]], idxs[:0:-1]])

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {i}")
        return f

    for n, i in enumerate(idxs):
        cv2.imwrite(f"{OUT}/frame_{n:02d}.webp", grab(i), [cv2.IMWRITE_WEBP_QUALITY, 92])
    cv2.imwrite(f"{OUT}/center.webp", grab(center), [cv2.IMWRITE_WEBP_QUALITY, 92])
    print(f"Wrote {N} frames + center.webp to {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect"); a.add_argument("--step", type=int, default=10)
    b = sub.add_parser("extract")
    b.add_argument("--start", type=int, required=True)
    b.add_argument("--end", type=int, required=True)
    b.add_argument("--center", type=int, required=True)
    b.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if args.cmd == "inspect":
        inspect(args.step)
    else:
        extract(args.start, args.end, args.center, args.reverse)
