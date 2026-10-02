#!/usr/bin/env python3
"""Write a few small test photos (PNG, standard library only) for the CI smoke test: gradients and stripes in
different colours, two exact duplicates, names with dates and with CJK characters.
    python3 scripts/smoke-photos.py OUT_DIR"""
import os, struct, sys, zlib


def png(path, w, h, pixel):
    raw = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "smoke-photos"
    os.makedirs(os.path.join(out, "trip 2019"), exist_ok=True)
    colours = [(230, 90, 60), (60, 140, 220), (90, 190, 110), (240, 200, 70), (150, 90, 200), (40, 40, 50)]
    names = ["IMG_20190512_101500.png", "IMG_20190513_093000.png", "Screenshot_2021-11-02.png", "海边 2020-08-01.png",
             "trip 2019/DSC01234.png", "trip 2019/DSC01235.png"]
    for i, (name, (r, g, b)) in enumerate(zip(names, colours)):
        png(os.path.join(out, name), 320, 240,
            lambda x, y, r=r, g=g, b=b, i=i: (min(255, r + x // 6), min(255, g + y // 5), (b + 40 * ((x // (20 + 4 * i)) % 2)) % 256))
    # an exact duplicate
    with open(os.path.join(out, names[0]), "rb") as f:
        dup = f.read()
    with open(os.path.join(out, "trip 2019", "copy of IMG_20190512_101500.png"), "wb") as f:
        f.write(dup)
    print(f"{len(names) + 1} photos in {out}")


if __name__ == "__main__":
    main()
