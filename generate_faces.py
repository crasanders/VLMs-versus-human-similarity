"""Programmatically generate the 34 schematic faces from Nosofsky (1991) / Reed (1972).

Reads physical specifications from `data/nosofsky1991_table_a1.csv`:
- EH: Eye height (mm from the top of the outline of the face to the center of the eyes)
- ES: Eye separation (mm between the centers of the two eyes)
- NL: Nose length (mm)
- MH: Mouth height (mm from the bottom of the outline of the face to the center of the mouth)

Renders 300x400 px PNG images (at 5 px/mm, 4x supersampled for anti-aliasing)
to `data/34 Faces/face_01.png` ... `face_34.png`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Paths
DATA_CSV = Path("data") / "nosofsky1991_table_a1.csv"
OUTPUT_DIR = Path("data") / "34 Faces"
FIGURE_1_PATH = Path("data") / "nosofsky1991_figure_1.png"

# Canvas settings (5 px/mm at final resolution, 4x supersampling for crisp anti-aliasing)
CANVAS_W = 300
CANVAS_H = 400
SCALE_PX_PER_MM = 5.0
SUPERSAMPLE = 4

# Calibrated fixed physical dimensions (in mm) from Figure 1 of Nosofsky (1991)
HEAD_HEIGHT_MM = 64.0
HEAD_WIDTH_MM = 44.5
HEAD_SPLIT_RATIO = 0.40  # Widest horizontal axis is 40% down from top of head
HEAD_EXP_TOP = 2.50      # Superellipse exponent for cranium (top 40%)
HEAD_EXP_BOT = 1.95      # Superellipse exponent for jaw/chin (bottom 60%)

NOSE_BOTTOM_FROM_CHIN_MM = 22.0  # Bottom of nose is anchored 22.0 mm above bottom of chin (42.0 mm from top)

EYE_WIDTH_MM = 7.4
EYE_HEIGHT_MM = 4.0
IRIS_DIAMETER_MM = 3.5
ES_IRIS_OFFSET_MM = 3.5  # Matches Figure 1 spacing (ES measured between inner iris edges)

MOUTH_WIDTH_MM = 17.0

STROKE_WIDTH_MM = 0.85


def head_contour_points(cx: float, top_y: float, s: float, num_points: int = 360) -> list[tuple[float, float]]:
    """Return polygon vertices for the asymmetric oval head outline."""
    h = HEAD_HEIGHT_MM * s
    a = (HEAD_WIDTH_MM / 2.0) * s
    b_top = HEAD_SPLIT_RATIO * h
    b_bot = (1.0 - HEAD_SPLIT_RATIO) * h
    y_center = top_y + b_top

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = []
    for theta in angles:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if sin_t < 0:  # Top half (cranium)
            p = HEAD_EXP_TOP
            rx = a * (abs(cos_t) ** (2.0 / p)) * np.sign(cos_t)
            ry = b_top * (abs(sin_t) ** (2.0 / p)) * np.sign(sin_t)
        else:          # Bottom half (jaw/chin)
            p = HEAD_EXP_BOT
            rx = a * (abs(cos_t) ** (2.0 / p)) * np.sign(cos_t)
            ry = b_bot * (abs(sin_t) ** (2.0 / p)) * np.sign(sin_t)
        pts.append((cx + rx, y_center + ry))
    return pts


def draw_eye(draw: ImageDraw.ImageDraw, ex: float, ey: float, s: float, lw: int) -> None:
    """Draw an almond-shaped eye with a circular iris ring centered at (ex, ey)."""
    half_w = (EYE_WIDTH_MM / 2.0) * s
    half_h = (EYE_HEIGHT_MM / 2.0) * s

    # Almond contour formed by two sinusoidal arcs meeting at (ex - half_w, ey) and (ex + half_w, ey)
    xs = np.linspace(-1.0, 1.0, 80)
    profile = np.cos(xs * (np.pi / 2.0))
    top_arc = [(ex + u * half_w, ey - profile[i] * half_h) for i, u in enumerate(xs)]
    bot_arc = [(ex + u * half_w, ey + profile[i] * half_h) for i, u in enumerate(xs[::-1])]
    eye_poly = top_arc + bot_arc
    draw.polygon(eye_poly, fill="white", outline=None)
    draw.line(eye_poly + [eye_poly[0]], fill="black", width=lw, joint="curve")

    # Iris ring (circle inside the eye opening, matching Figure 1)
    iris_r = (IRIS_DIAMETER_MM / 2.0) * s
    draw.ellipse(
        [ex - iris_r, ey - iris_r, ex + iris_r, ey + iris_r],
        outline="black",
        width=lw,
    )


def render_face(eh: float, es: float, nl: float, mh: float) -> Image.Image:
    """Render a single schematic face from its 4 physical dimensions (in mm)."""
    s = SCALE_PX_PER_MM * SUPERSAMPLE
    w = CANVAS_W * SUPERSAMPLE
    h = CANVAS_H * SUPERSAMPLE
    lw = max(1, int(round(STROKE_WIDTH_MM * s)))

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    cx = w / 2.0
    head_h = HEAD_HEIGHT_MM * s
    top_y = (h - head_h) / 2.0
    bot_y = top_y + head_h

    # 1. Head outline
    head_pts = head_contour_points(cx, top_y, s)
    draw.line(head_pts + [head_pts[0]], fill="black", width=lw, joint="curve")

    # 2. Eyes (EH mm from top of head outline; ES + ES_IRIS_OFFSET_MM between eye centers)
    eye_y = top_y + eh * s
    center_sep = es + ES_IRIS_OFFSET_MM
    left_eye_x = cx - (center_sep / 2.0) * s
    right_eye_x = cx + (center_sep / 2.0) * s
    draw_eye(draw, left_eye_x, eye_y, s, lw)
    draw_eye(draw, right_eye_x, eye_y, s, lw)

    # 3. Nose (bottom anchored NOSE_BOTTOM_FROM_CHIN_MM from bottom of head outline, extending upward by NL mm)
    nose_bot_y = bot_y - NOSE_BOTTOM_FROM_CHIN_MM * s
    nose_top_y = nose_bot_y - nl * s
    draw.line([(cx, nose_top_y), (cx, nose_bot_y)], fill="black", width=lw)

    # 4. Mouth (straight horizontal line MH mm from bottom of head outline)
    mouth_y = bot_y - mh * s
    half_mw = (MOUTH_WIDTH_MM / 2.0) * s
    draw.line([(cx - half_mw, mouth_y), (cx + half_mw, mouth_y)], fill="black", width=lw)

    return img.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)


def create_figure_1() -> None:
    """Assemble Faces 1-10 into the 2x5 grid of Figure 1 (Category 1 top row, Category 2 bottom row)."""
    # Crop each 300x400 canvas to 268x370 so horizontal/vertical gaps match Figure 1's layout
    tile_w, tile_h = 268, 370
    x0 = (CANVAS_W - tile_w) // 2
    y0 = (CANVAS_H - tile_h) // 2
    pad = 20

    grid = Image.new("RGB", (5 * tile_w + 2 * pad, 2 * tile_h + 2 * pad), "white")
    for idx in range(10):
        face_img = Image.open(OUTPUT_DIR / f"face_{idx + 1:02d}.png")
        tile = face_img.crop((x0, y0, x0 + tile_w, y0 + tile_h))
        row, col = divmod(idx, 5)
        grid.paste(tile, (pad + col * tile_w, pad + row * tile_h))

    grid.save(FIGURE_1_PATH)
    print(f"Saved recreated Figure 1 to '{FIGURE_1_PATH}'.")


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        face_id = int(row["Face"])
        img = render_face(
            eh=float(row["EH"]),
            es=float(row["ES"]),
            nl=float(row["NL"]),
            mh=float(row["MH"]),
        )
        out_path = OUTPUT_DIR / f"face_{face_id:02d}.png"
        img.save(out_path)

    print(f"Generated {len(df)} faces in '{OUTPUT_DIR}'.")
    create_figure_1()


if __name__ == "__main__":
    main()
