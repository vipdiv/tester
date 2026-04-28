#!/usr/bin/env python3
"""Extract individual sprites from a transparent sprite sheet."""

from __future__ import annotations

import argparse
import math
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Component:
    pixels: int
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max (inclusive)


def alpha_mask(image: Image.Image) -> list[list[bool]]:
    """Build a 2D boolean alpha mask where True means non-transparent."""
    alpha = image.getchannel("A")
    w, h = image.size
    values = list(alpha.getdata())
    return [[values[y * w + x] > 0 for x in range(w)] for y in range(h)]


def detect_components(mask: list[list[bool]]) -> list[Component]:
    """Return connected opaque components with pixel counts and bounding boxes."""
    h = len(mask)
    w = len(mask[0]) if h else 0
    visited = [[False for _ in range(w)] for _ in range(h)]
    components: list[Component] = []

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

            components.append(Component(pixels=count, bbox=(min_x, min_y, max_x, max_y)))

    return components


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Minimum edge-to-edge distance between two inclusive bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    dx = max(0, max(bx0 - ax1 - 1, ax0 - bx1 - 1))
    dy = max(0, max(by0 - ay1 - 1, ay0 - by1 - 1))
    return math.hypot(dx, dy)


def merge_nearby_small_details(
    components: list[Component],
    min_pixels: int,
    detail_distance: int,
) -> list[tuple[int, int, int, int]]:
    """Keep tiny detached components if close to a major sprite by merging their bboxes."""
    major = [c for c in components if c.pixels >= min_pixels]
    minor = [c for c in components if c.pixels < min_pixels]

    if not major:
        return []

    merged = [list(c.bbox) for c in major]

    for tiny in minor:
        nearest_idx = None
        nearest_dist = float("inf")

        for idx, big in enumerate(merged):
            dist = bbox_distance(tuple(big), tiny.bbox)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx

        if nearest_idx is not None and nearest_dist <= detail_distance:
            bx0, by0, bx1, by1 = merged[nearest_idx]
            tx0, ty0, tx1, ty1 = tiny.bbox
            merged[nearest_idx] = [
                min(bx0, tx0),
                min(by0, ty0),
                max(bx1, tx1),
                max(by1, ty1),
            ]

    boxes = [tuple(b) for b in merged]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def find_input_png(root: Path) -> Path:
    """Find the most recently modified PNG in the working directory tree."""
    candidates = [
        p for p in root.rglob("*.png")
        if p.is_file()
        and "exported_sprites" not in p.parts
        and p.name not in {"contact_sheet.png"}
    ]
    if not candidates:
        raise FileNotFoundError("No PNG file found to process.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def create_contact_sheet(sprite_paths: list[Path], output_path: Path) -> None:
    """Create a transparent contact sheet with filename captions."""
    if not sprite_paths:
        return

    sprites = [Image.open(p).convert("RGBA") for p in sprite_paths]
    labels = [p.name for p in sprite_paths]

    max_w = max(img.width for img in sprites)
    max_h = max(img.height for img in sprites)
    font = ImageFont.load_default()

    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    label_h = max(draw.textbbox((0, 0), text, font=font)[3] for text in labels) + 10

    n = len(sprites)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    pad = 24

    cell_w = max_w + pad * 2
    cell_h = max_h + label_h + pad * 2

    sheet = Image.new("RGBA", (cols * cell_w, rows * cell_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for i, (img, label) in enumerate(zip(sprites, labels)):
        row, col = divmod(i, cols)
        x0 = col * cell_w
        y0 = row * cell_h

        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + pad
        sheet.paste(img, (img_x, img_y), img)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x0 + (cell_w - text_w) // 2
        text_y = y0 + pad + max_h + 4
        draw.text((text_x, text_y), label, font=font, fill=(20, 20, 20, 255))

    sheet.save(output_path)


def zip_folder(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(folder.glob("*.png")):
            zf.write(file, arcname=file.name)


def extract_sprites(
    input_path: Path,
    output_dir: Path,
    min_pixels: int = 50,
    padding: int = 32,
    scale: int = 4,
    detail_distance: int = 28,
) -> tuple[int, Path]:
    image = Image.open(input_path).convert("RGBA")
    mask = alpha_mask(image)
    components = detect_components(mask)

    boxes = merge_nearby_small_details(components, min_pixels=min_pixels, detail_distance=detail_distance)
    output_dir.mkdir(parents=True, exist_ok=True)

    sprite_paths: list[Path] = []
    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        sprite = image.crop((x0, y0, x1 + 1, y1 + 1))

        if padding > 0:
            padded = Image.new(
                "RGBA",
                (sprite.width + padding * 2, sprite.height + padding * 2),
                (0, 0, 0, 0),
            )
            padded.paste(sprite, (padding, padding), sprite)
            sprite = padded

        upscaled = sprite.resize((sprite.width * scale, sprite.height * scale), Image.Resampling.LANCZOS)
        out_path = output_dir / f"sprite_{idx:02d}.png"
        upscaled.save(out_path)
        sprite_paths.append(out_path)

    contact_sheet_path = output_dir / "contact_sheet.png"
    create_contact_sheet(sprite_paths, contact_sheet_path)

    zip_path = output_dir.parent / "exported_sprites.zip"
    zip_folder(output_dir, zip_path)
    return len(sprite_paths), zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract alpha-connected sprites into individual upscaled PNGs."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Optional path to input sprite sheet PNG. If omitted, auto-detects latest PNG.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("exported_sprites"),
        help="Output directory for sprite PNG files",
    )
    parser.add_argument("--min-pixels", type=int, default=50, help="Ignore components smaller than this size")
    parser.add_argument("--padding", type=int, default=32, help="Transparent padding in pixels around each sprite")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4)")
    parser.add_argument(
        "--detail-distance",
        type=int,
        default=28,
        help="Attach tiny detached details if within this distance of a major sprite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or find_input_png(Path.cwd())

    count, zip_path = extract_sprites(
        input_path=input_path,
        output_dir=args.output_dir,
        min_pixels=args.min_pixels,
        padding=args.padding,
        scale=args.scale,
        detail_distance=args.detail_distance,
    )

    print(f"Input PNG: {input_path}")
    print(f"Sprites exported: {count}")
    print(f"ZIP created at: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
