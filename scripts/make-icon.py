#!/usr/bin/env python3
"""Draw the jev photos app icon (macOS Big Sur style): a squircle on the 1024 grid with a soft shadow, a deep
violet-to-teal gradient, two tilted photo prints (the front one a sunset over mountains) and a glass magnifier.
Writes assets/jev-photos-1024.png, assets/jev-photos-256.png, assets/jev-photos.ico and assets/jev-photos.svg.
    python3 scripts/make-icon.py        (needs Pillow and numpy)"""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4                      # supersampling
W = 1024 * S
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BODY = 824                 # icon body on the 1024 grid (Apple's template: 100 px margin)
X0 = (1024 - BODY) / 2
N = 5.0                    # superellipse exponent (the Big Sur squircle is close to n = 5)


def squircle_mask(size, n=N):
    t = (np.arange(size) + 0.5) / size * 2 - 1
    xx, yy = np.meshgrid(t, t)
    v = np.abs(xx) ** n + np.abs(yy) ** n
    # soft edge over ~1.5 px for antialiasing at the supersampled size
    edge = 1.5 / size * n
    return np.clip((1 - v) / edge + 0.5, 0, 1)


def gradient(h, w, stops, angle_deg=90):
    """stops: [(pos 0..1, (r,g,b)), ...] along a direction (90 = top to bottom)."""
    a = math.radians(angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a) + 0.5
    d = np.clip(d, 0, 1)
    out = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(d, [p for p, _ in stops], [col[c] for _, col in stops])
    return out


def rgba(arr_rgb, alpha):
    return Image.fromarray(np.dstack([arr_rgb, alpha * 255]).astype(np.uint8), "RGBA")


def shadow(layer, blur, offset, opacity):
    """The layer's soft shadow, on a canvas padded by `pad` on every side (so the blur is never clipped)."""
    pad = int(blur * 3)
    a = np.array(layer.split()[3], np.float32) * opacity
    a = np.pad(a, pad)
    sh = Image.fromarray(np.dstack([np.zeros(a.shape + (3,)), a]).astype(np.uint8), "RGBA")
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    return sh, pad - int(offset[0]), pad - int(offset[1])


def put_shadow(canvas, layer, blur, offset, opacity, at):
    sh, dx, dy = shadow(layer, blur, offset, opacity)
    canvas.alpha_composite(sh, (at[0] - dx, at[1] - dy))


def photo_card(w, h, border, scene):
    """A white print with a rounded corner and the given scene inside."""
    card = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(card)
    r = int(w * 0.035)
    d.rounded_rectangle([0, 0, w - 1, h - 1], r, fill=(250, 250, 252, 255))
    iw, ih = w - 2 * border, h - 2 * border
    card.alpha_composite(scene(iw, ih), (border, border))
    return card


def sunset_scene(w, h):
    sky = gradient(h, w, [(0, (255, 120, 92)), (0.45, (255, 178, 102)), (0.75, (255, 214, 140)), (1, (255, 230, 170))])
    img = Image.fromarray(sky.astype(np.uint8), "RGB").convert("RGBA")
    d = ImageDraw.Draw(img)
    # sun
    cx, cy, rr = w * 0.62, h * 0.50, w * 0.13
    glow = Image.new("RGBA", img.size, (255, 250, 235, 0))  # same colour under the transparent part: no dark fringe when blurred
    ImageDraw.Draw(glow).ellipse([cx - rr * 1.9, cy - rr * 1.9, cx + rr * 1.9, cy + rr * 1.9], fill=(255, 250, 235, 70))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(w * 0.06)))
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 248, 225, 255))
    # mountains: back (violet) and front (deep indigo)
    d.polygon([(0, h * 0.70), (w * 0.22, h * 0.46), (w * 0.40, h * 0.62), (w * 0.58, h * 0.40), (w * 0.80, h * 0.64),
               (w, h * 0.52), (w, h), (0, h)], fill=(150, 86, 160, 255))
    d.polygon([(0, h * 0.84), (w * 0.18, h * 0.66), (w * 0.34, h * 0.78), (w * 0.52, h * 0.60), (w * 0.72, h * 0.82),
               (w * 0.88, h * 0.70), (w, h * 0.80), (w, h), (0, h)], fill=(62, 44, 110, 255))
    return img


def sea_scene(w, h):
    sky = gradient(h, w, [(0, (92, 170, 255)), (0.55, (150, 210, 255)), (0.56, (40, 120, 200)), (1, (20, 80, 160))])
    img = Image.fromarray(sky.astype(np.uint8), "RGB").convert("RGBA")
    return img


def rotate(layer, deg):
    return layer.rotate(deg, resample=Image.BICUBIC, expand=True)


def main():
    canvas = Image.new("RGBA", (W, W))
    body = int(BODY * S)
    x0 = int(X0 * S)
    mask = squircle_mask(body)

    # background: deep violet -> indigo -> teal, lit from the top
    bg = gradient(body, body, [(0, (88, 70, 214)), (0.5, (46, 64, 170)), (1, (14, 150, 170))], angle_deg=70)
    body_img = rgba(bg, mask)
    # soft top light
    light = np.zeros((body, body), np.float32)
    yy, xx = np.mgrid[0:body, 0:body].astype(np.float32)
    light = np.clip(1 - np.hypot((xx - body * 0.35) / (body * 0.75), (yy + body * 0.1) / (body * 0.7)), 0, 1) ** 1.6 * 0.35
    body_img.alpha_composite(rgba(np.full((body, body, 3), 255, np.float32), light * mask))

    # icon drop shadow (macOS: y offset ~ 12, blur ~ 28 on the 1024 grid)
    put_shadow(canvas, body_img, 28 * S, (0, 14 * S), 0.45, (x0, x0))
    canvas.alpha_composite(body_img, (x0, x0))

    # the prints
    cw, ch, border = int(470 * S), int(360 * S), int(22 * S)
    back = rotate(photo_card(cw, ch, border, sea_scene), 12)
    front = rotate(photo_card(cw, ch, border, sunset_scene), -7)
    for layer, (cx, cy) in ((back, (548, 438)), (front, (488, 538))):
        px, py = int(cx * S - layer.width / 2), int(cy * S - layer.height / 2)
        put_shadow(canvas, layer, 18 * S, (0, 10 * S), 0.40, (px, py))
        canvas.alpha_composite(layer, (px, py))

    # glass magnifier at the lower right
    mx, my, mr = 668 * S, 652 * S, 92 * S
    lens = Image.new("RGBA", (W, W))
    d = ImageDraw.Draw(lens)
    hx, hy = mx + mr * 0.70, my + mr * 0.70
    hl = 92 * S
    d.line([(hx, hy), (hx + hl, hy + hl)], fill=(250, 250, 255, 255), width=int(46 * S))  # handle
    d.ellipse([hx + hl - 23 * S, hy + hl - 23 * S, hx + hl + 23 * S, hy + hl + 23 * S], fill=(250, 250, 255, 255))
    d.ellipse([mx - mr - 22 * S, my - mr - 22 * S, mx + mr + 22 * S, my + mr + 22 * S], fill=(250, 250, 255, 255))  # rim
    put_shadow(canvas, lens, 16 * S, (0, 10 * S), 0.40, (0, 0))
    canvas.alpha_composite(lens)
    glass = Image.new("RGBA", (W, W), (120, 205, 235, 0))
    gd = ImageDraw.Draw(glass)
    gd.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(120, 205, 235, 235))
    # highlight on the glass
    gd.ellipse([mx - mr * 0.62, my - mr * 0.70, mx + mr * 0.05, my - mr * 0.12], fill=(255, 255, 255, 120))
    canvas.alpha_composite(glass.filter(ImageFilter.GaussianBlur(1.2 * S)))

    # gloss along the top edge of the squircle
    gloss = np.clip(1 - yy / (body * 0.42), 0, 1) ** 2 * 0.10
    canvas.alpha_composite(rgba(np.full((body, body, 3), 255, np.float32), gloss * mask), (x0, x0))

    big = canvas.resize((1024, 1024), Image.LANCZOS)
    a = os.path.join(ROOT, "assets")
    big.save(os.path.join(a, "jev-photos-1024.png"))
    big.resize((256, 256), Image.LANCZOS).save(os.path.join(a, "jev-photos-256.png"))
    big.save(os.path.join(a, "jev-photos.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    write_svg(os.path.join(a, "jev-photos.svg"))
    print("assets/jev-photos-1024.png, -256.png, .ico, .svg")


def squircle_path(x, y, size, n=N, steps=180):
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        px = abs(c) ** (2 / n) * (1 if c >= 0 else -1)
        py = abs(s) ** (2 / n) * (1 if s >= 0 else -1)
        pts.append((x + size / 2 * (1 + px), y + size / 2 * (1 + py)))
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + " Z"


def write_svg(path):
    sq = squircle_path(X0, X0, BODY)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <linearGradient id="bg" x1="0.2" y1="0" x2="0.8" y2="1">
      <stop offset="0" stop-color="#5846d6"/><stop offset="0.5" stop-color="#2e40aa"/><stop offset="1" stop-color="#0e96aa"/>
    </linearGradient>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ff785c"/><stop offset="0.45" stop-color="#ffb266"/><stop offset="1" stop-color="#ffe6aa"/>
    </linearGradient>
    <linearGradient id="sea" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#5caaff"/><stop offset="0.55" stop-color="#96d2ff"/><stop offset="0.56" stop-color="#2878c8"/><stop offset="1" stop-color="#1450a0"/>
    </linearGradient>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-opacity="0.4"/></filter>
    <clipPath id="sq"><path d="{sq}"/></clipPath>
  </defs>
  <path d="{sq}" fill="url(#bg)" filter="url(#sh)"/>
  <g clip-path="url(#sq)"><rect x="0" y="0" width="1024" height="420" fill="#fff" opacity="0.08"/></g>
  <g transform="translate(548 438) rotate(-12)" filter="url(#sh)">
    <rect x="-235" y="-180" width="470" height="360" rx="16" fill="#fafafc"/>
    <rect x="-213" y="-158" width="426" height="316" fill="url(#sea)"/>
  </g>
  <g transform="translate(488 538) rotate(7)" filter="url(#sh)">
    <rect x="-235" y="-180" width="470" height="360" rx="16" fill="#fafafc"/>
    <rect x="-213" y="-158" width="426" height="316" fill="url(#sky)"/>
    <circle cx="51" cy="0" r="55" fill="#fff8e1"/>
    <path d="M-213,63 L-119,-12 L-43,38 L34,-31 L128,44 L213,6 L213,158 L-213,158 Z" fill="#9656a0"/>
    <path d="M-213,107 L-136,51 L-68,88 L8,32 L94,101 L162,63 L213,95 L213,158 L-213,158 Z" fill="#3e2c6e"/>
  </g>
  <g filter="url(#sh)">
    <line x1="732" y1="716" x2="824" y2="808" stroke="#fafaff" stroke-width="46" stroke-linecap="round"/>
    <circle cx="668" cy="652" r="114" fill="#fafaff"/>
  </g>
  <circle cx="668" cy="652" r="92" fill="#78cdeb"/>
  <ellipse cx="641" cy="614" rx="31" ry="26" fill="#fff" opacity="0.45"/>
</svg>
'''
    with open(path, "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main()
