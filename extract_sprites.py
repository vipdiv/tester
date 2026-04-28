#!/usr/bin/env python3
"""Extract sprites from a transparent sprite sheet using alpha connected components."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image


def alpha_mask(image: Image.Image) -> list[list[bool]]:
    """Build a 2D boolean alpha mask where True means non-transparent."""
    alpha = image.getchannel("A")
    w, h = image.size
    values = list(alpha.getdata())
    return [[values[y * w + x] > 0 for x in range(w)] for y in range(h)]


def find_connected_components(mask: list[list[bool]], min_pixels: int) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x_min, y_min, x_max, y_max) for connected opaque regions."""
    h = len(mask)
    w = len(mask[0]) if h else 0
    visited = [[False for _ in range(w)] for _ in range(h)]
    boxes: list[tuple[int, int, int, int]] = []

    # 8-connected neighborhood keeps diagonally touching pixels in the same sprite.
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not mask[y][x] or visited[y][x]:
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True

            count = 0
            min_x = max_x = x
            min_y = max_y = y

            while queue:
                cx, cy = queue.popleft()
                count += 1

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        queue.append((nx, ny))

            if count >= min_pixels:
                boxes.append((min_x, min_y, max_x, max_y))

    # Stable order: top-to-bottom, then left-to-right.
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def extract_sprites(
    input_path: Path,
    output_dir: Path,
    min_pixels: int = 50,
    padding: int = 8,
    scale: int = 4,
) -> int:
    image = Image.open(input_path).convert("RGBA")
    mask = alpha_mask(image)

    boxes = find_connected_components(mask, min_pixels=min_pixels)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        # PIL crop uses right/lower as exclusive.
        sprite = image.crop((x0, y0, x1 + 1, y1 + 1))

        if padding > 0:
            padded = Image.new(
                "RGBA",
                (sprite.width + padding * 2, sprite.height + padding * 2),
                (0, 0, 0, 0),
            )
            padded.paste(sprite, (padding, padding))
            sprite = padded

        upscaled = sprite.resize((sprite.width * scale, sprite.height * scale), Image.Resampling.LANCZOS)
        out_path = output_dir / f"sprite_{idx:02d}.png"
        upscaled.save(out_path)

    return len(boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract alpha-connected sprites, add padding, and export 4x Lanczos upscaled PNGs."
    )
    parser.add_argument("input", type=Path, help="Path to input sprite sheet PNG")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("sprites"), help="Output directory")
    parser.add_argument("--min-pixels", type=int, default=50, help="Ignore components smaller than this size")
    parser.add_argument("--padding", type=int, default=8, help="Transparent padding (in pixels) around each sprite")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_sprites(
        input_path=args.input,
        output_dir=args.output_dir,
        min_pixels=args.min_pixels,
        padding=args.padding,
        scale=args.scale,
    )
    print(f"Extracted {count} sprites to {args.output_dir}")


if __name__ == "__main__":
    main()
