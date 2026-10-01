#!/usr/bin/env python3
"""Build a messy photo collection for end-to-end tests.

usage: make_fixtures.py OUT_DIR [IMAGE ...]
Images (any JPEG/PNG) supply pixel content; without them flat colour images are used.
Needs Pillow and the exiv2 CLI. Prints the expected decision for every file as JSON lines.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from PIL import Image

out = sys.argv[1]
pool = sys.argv[2:]
shutil.rmtree(out, ignore_errors=True)
src = os.path.join(out, "src")
os.makedirs(src)
counter = [0]


def ts(s):
    return time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


def make(rel, dto=None, mtime="2024-03-01 09:00:00", fmt="JPEG", exiv2=(), expect=None):
    path = os.path.join(src, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if pool:
        im = Image.open(pool[counter[0] % len(pool)]).convert("RGB")
        im.thumbnail((1024, 1024))
    else:
        im = Image.new("RGB", (320, 240), ((counter[0] * 53) % 255, 120, 90))
    counter[0] += 1
    kw = {}
    if fmt == "JPEG":
        ex = Image.Exif()
        ex[0x0110] = "FixtureCam"
        kw = {"exif": ex, "quality": 88}
    if dto:  # Pillow does not reliably write the Exif sub-IFD, so use exiv2
        exiv2 = ("set Exif.Photo.DateTimeOriginal Ascii " + dto,) + tuple(exiv2)
    im.save(path, fmt, **kw)
    if exiv2:
        cmd = os.path.join(out, "cmd.txt")
        with open(cmd, "w") as f:
            f.write("\n".join(exiv2) + "\n")
        subprocess.run(["exiv2", "-q", "-m", cmd, path], check=True)
    t = ts(mtime)
    os.utime(path, (t, t))
    print(json.dumps({"file": rel, **(expect or {})}, ensure_ascii=False))
    return path


# Camera card copied in one go: all mtimes share the copy minute (bulk), EXIF spans several days.
gps = ["set Exif.GPSInfo.GPSLatitudeRef N", "set Exif.GPSInfo.GPSLatitude 48/1 51/1 29/1",
       "set Exif.GPSInfo.GPSLongitudeRef E", "set Exif.GPSInfo.GPSLongitude 2/1 17/1 40/1",
       "set Exif.GPSInfo.GPSDateStamp 2019:05:12", "set Exif.GPSInfo.GPSTimeStamp 12/1 30/1 20/1",
       "set Exif.Photo.OffsetTimeOriginal +02:00"]
make("DCIM/100CANON/IMG_0001.JPG", "2019:05:12 14:30:22", exiv2=gps, expect={"date": "2019-05-12 14:30:22", "src": "exif-original"})
make("DCIM/100CANON/IMG_0002.JPG", "2019:05:13 09:10:00", expect={"date": "2019-05-13 09:10:00"})
make("DCIM/100CANON/IMG_0003.JPG", "2000:01:01 00:00:07", expect={"review": True, "note": "camera clock reset, bulk mtime"})
make("DCIM/100CANON/IMG_0004.JPG", "2019:05:14 19:45:00", expect={"date": "2019-05-14 19:45:00"})
make("DCIM/100CANON/IMG_0005.JPG", "2019:05:15 08:00:00", expect={"date": "2019-05-15 08:00:00"})
make("DCIM/100CANON/IMG_0006.JPG", "2019:05:15 08:00:01", expect={"date": "2019-05-15 08:00:01"})

make("WhatsApp Images/IMG-20200815-WA0003.jpg", mtime="2020-08-15 20:11:05",
     expect={"date": "2020-08-15 20:11:05", "src": "name-date+time:fs-mtime"})
make("2018-07 Hawaii vacation/DSC_0101.jpg", mtime="2023-02-02 10:00:00",
     expect={"date": "2018-07", "src": "dir-month", "location": "Hawaii"})
make("2018-07 Hawaii vacation/DSC_0102.jpg", "2018:07:04 11:00:00",
     exiv2=["set Xmp.dc.subject XmpBag beach", "set Xmp.dc.description LangAlt lang=\"x-default\" Kept by the owner"],
     expect={"date": "2018-07-04 11:00:00", "location": "Hawaii", "keeps": "description + beach keyword"})
make("巴黎旅行/微信图片_20190514123045.jpg", mtime="2022-01-01 00:00:00",
     expect={"date": "2019-05-14 12:30:45", "location": "巴黎"})
make("Screenshots/Screenshot_2021-11-02-08-15-30.png", fmt="PNG", mtime="2021-11-02 08:15:31",
     expect={"date": "2021-11-02 08:15:30"})
make("mmexport1557671422000.jpg", mtime="2023-05-05 05:05:05", expect={"date": "2019-05-12 (local)", "src": "name-unix"})
make("Grandma/scan_0001.jpg", mtime="2021-12-24 18:00:00", expect={"review": True, "location": ""})
make("edits/20200101_120000.jpg", "2018:03:01 10:00:00", mtime="2024-06-01 12:00:00",
     expect={"note": "EXIF vs name conflict -> jev asked", "review": True})
make("misc/IMG_5000.jpg", "2017:10:10 10:10:10", exiv2=["set Xmp.photoshop.City Kyoto"],
     expect={"date": "2017-10-10 10:10:10", "location": "Kyoto", "src_loc": "metadata"})
# exact duplicate of IMG_0001 in a backup folder
dup = os.path.join(src, "backup/IMG_0001.JPG")
os.makedirs(os.path.dirname(dup))
shutil.copy2(os.path.join(src, "DCIM/100CANON/IMG_0001.JPG"), dup)
print(json.dumps({"file": "backup/IMG_0001.JPG", "duplicate": True}))
os.remove(os.path.join(out, "cmd.txt"))

# second exact copy under another name
os.makedirs(os.path.join(src, "shared"))
shutil.copy2(os.path.join(src, "DCIM/100CANON/IMG_0004.JPG"), os.path.join(src, "shared/IMG_0004 (1).JPG"))
print(json.dumps({"file": "shared/IMG_0004 (1).JPG", "duplicate": True}))
# near duplicates: a half-size re-compressed export and a re-saved copy (different bytes, same picture)
os.makedirs(os.path.join(src, "exports"))
im = Image.open(os.path.join(src, "DCIM/100CANON/IMG_0002.JPG"))
im.resize((im.width // 2, im.height // 2)).save(os.path.join(src, "exports/IMG_0002_small.jpg"), quality=60)
print(json.dumps({"file": "exports/IMG_0002_small.jpg", "similar": "DCIM/100CANON/IMG_0002.JPG"}))
Image.open(os.path.join(src, "DCIM/100CANON/IMG_0005.JPG")).save(os.path.join(src, "shared/resaved_0005.jpg"), quality=70)
print(json.dumps({"file": "shared/resaved_0005.jpg", "similar": "DCIM/100CANON/IMG_0005.JPG"}))
