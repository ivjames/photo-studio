#!/usr/bin/env python3
"""
scan_splitter.py — detect, deskew, and crop individual photos from flatbed scans.

A scan of several photos laid on a scanner bed is split into one cropped,
rotation-corrected image per photo. Pure OpenCV: deterministic, fast, free.

Usage:
    python scan_splitter.py INPUT [INPUT ...] -o OUTPUT_DIR [options]

    INPUT can be image files and/or directories (directories are searched
    for common image extensions).

Examples:
    python scan_splitter.py scans/ -o cropped/
    python scan_splitter.py page1.jpg page2.tif -o out/ --min-area-frac 0.02 --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def estimate_background(img):
    """Estimate the scanner-bed background color from the image border."""
    h, w = img.shape[:2]
    b = max(8, int(0.02 * min(h, w)))  # border thickness to sample
    border = np.concatenate([
        img[:b, :].reshape(-1, 3),
        img[-b:, :].reshape(-1, 3),
        img[:, :b].reshape(-1, 3),
        img[:, -b:].reshape(-1, 3),
    ])
    # Median is robust to a photo touching the edge.
    return np.median(border, axis=0)


def build_foreground_mask(img, bg_color, sensitivity):
    """Mask where the image differs from the background (i.e. is a photo)."""
    diff = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
    # Otsu finds the split automatically; sensitivity nudges the threshold.
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sensitivity != 1.0:
        _, mask = cv2.threshold(diff_u8, max(1, thr * sensitivity), 255, cv2.THRESH_BINARY)

    # Close gaps inside photos (skies, white borders) and remove specks.
    h, w = img.shape[:2]
    k = max(3, int(0.005 * min(h, w))) | 1  # odd kernel scaled to image size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def crop_rotated(img, rect, bbox_aspect):
    """Crop a cv2.minAreaRect region, rotating only to remove tilt (deskew).

    minAreaRect's angle is ambiguous mod 90deg, so we resolve the final
    orientation using the photo's axis-aligned bounding-box aspect ratio
    (its true portrait/landscape-ness) rather than guessing from the angle.
    """
    (cx, cy), (rw, rh), angle = rect
    rw, rh = int(round(rw)), int(round(rh))
    if rw == 0 or rh == 0:
        return None
    box = cv2.boxPoints(rect).astype(np.float32)
    dst = np.array([[0, rh - 1], [0, 0], [rw - 1, 0], [rw - 1, rh - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(img, M, (rw, rh))

    # If the crop's orientation disagrees with the photo's real footprint,
    # rotate 90deg to match. (Does not change which way is "up" within the
    # photo — that can't be inferred from geometry alone.)
    crop_landscape = crop.shape[1] >= crop.shape[0]
    bbox_landscape = bbox_aspect >= 1.0
    if crop_landscape != bbox_landscape:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def find_photos(img, min_area_frac, max_area_frac, sensitivity, deskew, pad):
    """Return a list of cropped photo images found in a scan."""
    bg = estimate_background(img)
    mask = build_foreground_mask(img, bg, sensitivity)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img.shape[0] * img.shape[1]
    crops = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * total or area > max_area_frac * total:
            continue
        if deskew:
            bx, by, bw, bh = cv2.boundingRect(c)
            bbox_aspect = bw / bh if bh else 1.0
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            rect = ((cx, cy), (rw + 2 * pad, rh + 2 * pad), ang)
            crop = crop_rotated(img, rect, bbox_aspect)
        else:
            x, y, w, h = cv2.boundingRect(c)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            crop = img[y0:y1, x0:x1]
        if crop is not None and crop.size > 0:
            # Sort key: top-to-bottom, then left-to-right (reading order).
            M = cv2.moments(c)
            ccx = M["m10"] / M["m00"] if M["m00"] else 0
            ccy = M["m01"] / M["m00"] if M["m00"] else 0
            crops.append((ccy, ccx, crop))

    crops.sort(key=lambda t: (round(t[0] / (img.shape[0] * 0.1)), t[1]))
    return [c for _, _, c in crops]


def iter_inputs(inputs):
    for p in inputs:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.suffix.lower() in IMAGE_EXTS:
                    yield f
        elif path.suffix.lower() in IMAGE_EXTS:
            yield path
        else:
            print(f"  ! skipping (not an image): {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Crop individual photos from flatbed scans.")
    ap.add_argument("inputs", nargs="+", help="Image files and/or directories.")
    ap.add_argument("-o", "--output", required=True, help="Output directory.")
    ap.add_argument("--min-area-frac", type=float, default=0.01,
                    help="Ignore blobs smaller than this fraction of the scan (default 0.01).")
    ap.add_argument("--max-area-frac", type=float, default=0.95,
                    help="Ignore blobs larger than this fraction (filters the whole page).")
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help=">1 keeps less, <1 keeps more vs. the auto threshold (default 1.0).")
    ap.add_argument("--pad", type=int, default=6, help="Pixels of padding around each crop.")
    ap.add_argument("--no-deskew", action="store_true", help="Axis-aligned crops only, no rotation.")
    ap.add_argument("--format", default="jpg", help="Output extension (jpg, png, tif).")
    ap.add_argument("--debug", action="store_true", help="Also save the detection mask.")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    total_found = 0
    for f in iter_inputs(args.inputs):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  ! could not read: {f}", file=sys.stderr)
            continue
        photos = find_photos(
            img,
            min_area_frac=args.min_area_frac,
            max_area_frac=args.max_area_frac,
            sensitivity=args.sensitivity,
            deskew=not args.no_deskew,
            pad=args.pad,
        )
        stem = f.stem
        for i, crop in enumerate(photos, 1):
            name = out / f"{stem}_{i:02d}.{args.format}"
            cv2.imwrite(str(name), crop)
        total_found += len(photos)
        print(f"  {f.name}: {len(photos)} photo(s)")

        if args.debug:
            bg = estimate_background(img)
            mask = build_foreground_mask(img, bg, args.sensitivity)
            cv2.imwrite(str(out / f"{stem}_MASK.png"), mask)

    print(f"\nDone. {total_found} photo(s) written to {out}/")


if __name__ == "__main__":
    main()
