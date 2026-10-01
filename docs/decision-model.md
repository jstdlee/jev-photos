# Decision model

jev-photos makes two decisions per photo: **when** it was taken and **where**. Both are evidence-based. Every
decision is stored with its full evidence (`date_evidence` in the database, the *Date evidence* table in the UI,
`--explain FILE` in the CLI), so any date can be traced back and overridden.

## 1. Capture date

### Evidence sources

| Source id | Where it comes from | Base reliability | Upper bound? |
|---|---|---:|:---:|
| `exif-original` | `Exif.Photo.DateTimeOriginal` | 0.95 | |
| `exif-digitized` | `Exif.Photo.DateTimeDigitized` | 0.90 | |
| `xmp-created` | `Xmp.exif.DateTimeOriginal`, `Xmp.photoshop.DateCreated`, `Xmp.xmp.CreateDate`, IPTC `DateCreated`+`TimeCreated` | 0.85 | |
| `gps` | `GPSDateStamp`+`GPSTimeStamp` (UTC), shifted by `OffsetTimeOriginal` or the local zone | 0.80 | |
| `name-datetime` | `IMG_20190512_143022`, `PXL_…`, `Screenshot_2019-05-12-14-30-22`, `Photo 2019-05-12 at 14.30.22`, `微信图片_20190514123045` | 0.80 | |
| `name-unix` | `mmexport1557671422000`, `wx_camera_…` (10/13-digit epoch, used only if nothing calendar-like matched) | 0.72 | |
| `name-date` | `IMG-20190512-WA0001`, `2019-05-12`, `2019年5月12日` | 0.70 | |
| `neighbor` | a numbered camera sequence: the confident photos just before/after (2nd pass, weak photos only) | 0.60 / 0.40 | |
| `exif-modify` | `Exif.Image.DateTime` (often the edit/export time) | 0.55 | yes |
| `dir-date` / `dir-month` / `dir-year` | folder names: `20190512 生日`, `2019-05 Paris trip`, `May 2019`, `2019`; ×0.9 per level up | 0.55 / 0.45 / 0.30 | |
| `name-month` | `2019-05 xyz.jpg` | 0.40 | |
| `fs-birth`, `fs-mtime` | `statx` birth time, modification time | 0.30 | yes |
| `manual` | set in the UI | 1.00 (always wins) | |

### Sanity rules (weight after base)

* **Rejected:** unparseable values (`0000:00:00 …`), anything in the future, before *Minimum year*, the epoch sentinels
  1970-01-01 / 1980-01-01, and a birth time later than the modify time (a copy creates a new birth time).
* **×0.3 camera reset:** embedded dates on 1 January of a year ≤ 2005, or 1 January 00:00–00:09 of any year.
* **×0.5 later than the file:** a non-file date more than 36 h after the file's own timestamp.
* **×0.15 bulk copy:** an mtime shared by a folder whose files' own evidence spans several days (or 20+ files in one
  minute). This is the copy time, not the capture time.
* **×0.5 implausibly old file time:** filesystem times before 1995.

### Support, winner, confidence

```
agree(a,b):  both ≥ minute precision: ≤2 min → 1.0; whole time-zone offset apart → 0.7; same day 0.5; ±1 day 0.3
             day precision: same day 0.9, ±1 day 0.4    month: same month 0.7    year: same year 0.5
independence: 0.35 when both come from the same family (embedded / name / dir / fs), else 1.0

S_i = w_i + Σ_j w_j · agree(i,j) · independence(i,j)
winner = argmax S (ties: finer precision, then higher base reliability)
```

* A date-only or month-only winner **borrows the finer detail** from the strongest agreeing candidate
  (`name-date+time:fs-mtime`: date from `IMG-20200815-WA0003`, time from the file).
* Upper-bound sources (file times, EXIF modify time) only **contradict** a winner when they are *earlier*
  than it by more than 36 h. A later copy/edit time is consistent, so it never lowers confidence.
* `margin = S_win / (S_win + S_best_contradicting)`, `confidence = margin × min(1, S_win)`.
* Confidence below *Review below* (0.55) marks the photo **needs review**. It is still filed under its best date.

### When jev is asked

If two or more distinct dates compete and `margin < 0.70`, the top (up to 4) cluster anchors go to the jev
`/v1/systemone` API as one `choice` question. Julia 1 in independent mode judges option text only, so every option
is self-describing, e.g.:

```
d0: 2018-03-01 10:00:00 from camera EXIF original time
d1: 2020-01-01 12:00:00 from date and time in the file name
```

The answer is blended, not obeyed: `final_k = (1 − w)·S_k/ΣS + w·p_k` with `w` = *Jev weight (dates)* (0.30). The
decider becomes `rules+jev`, and the confidence is recomputed from the blended margin.

### Neighbour pass

After all photos are decided, any photo still under review that sits in a numbered sequence (`IMG_0003`) takes the
**day** of its confident neighbours with the same prefix within ±20 numbers. Both sides must be within 3 days of each
other (weight 0.60). With a neighbour on only one side the weight is 0.40. This rescues camera-reset clocks and
EXIF-stripped files, and the photo stays flagged for review.

## 2. Location

1. **Existing metadata wins:** `Xmp.iptc.Location`, `Iptc SubLocation`, `Xmp.photoshop.City`, `Iptc City`.
   GPS coordinates are always kept (in the database), never invented.
2. **Folder names** (nearest three, inside the source root) are split into phrases. Dates, numbers, DCF folders
   (`100CANON`), camera prefixes (`IMG`, `DSC`, `PXL`…), generic words (`Camera Roll`, `WhatsApp Images`, `新建文件夹`,
   `截图`…) and trip affixes (`trip`, `vacation`, `旅行`, `之旅`, `游`…) are removed. `2019-05 Paris trip` → `Paris`
   (with a trip cue), and `巴黎旅行` → `巴黎`.
3. **Per-folder vote** (cached in the `dirs` table):

   | Voter | Default weight | Score |
   |---|---:|---|
   | rules | 0.25 | 0.35 nearest folder … 0.20 file name, +0.35 with a trip cue |
   | VL model (text only) | 0.60 | the part it names as a place, times its confidence |
   | jev | 0.15 | choice probability over `"X" is the name of a city, country, region, park or landmark` + a none option |

   The result is the weighted mean over the voters that answered. **jev only breaks ties:** a candidate must first
   be backed (score ≥ 0.5) by the rules or the VL model, because Julia 1 in independent mode almost never picks "none".
   Accepted at ≥ *Accept location at* (0.50).
4. **File name** fallback: a trip-cued phrase in the file name itself (rules only).

Measured on this machine (2026-09-29): Julia 1 picked the place in about half of simple two-name tests and never
chose "none". Qwen3-VL-8B picked it in 7/7. Those results set the default weights; all weights are in Settings.

## 3. Writing metadata (never overwrite)

Only the **library copy** is modified (or an XMP sidecar, or nothing). A key that already has a value is never
changed. Only our own `Xmp.jev.*` namespace is refreshed.

| Field | Written when |
|---|---|
| `Exif.Photo.DateTimeOriginal` | missing, date trusted (not under review) and minute precision or finer |
| `Xmp.photoshop.DateCreated` | missing and date trusted, at the date's real precision (`2018-07`, `2018-07-04`…) |
| `Xmp.dc.description` | missing (VL caption) |
| `Xmp.iptc.Location` | no location field present |
| `Xmp.dc.subject` | keywords appended; existing ones kept, duplicates skipped (case-insensitive) |
| `Iptc.Application2.Keywords` | appended only when the file already uses IPTC keywords |
| `Xmp.jev.ResolvedDate/DateSource/DateConfidence/Caption/Scene/Landmark/Location/LocationSource` | always (provenance) |

If exiv2 cannot write a format, the app falls back to a sidecar. If that also fails, the error is recorded and the
file is left as copied.

## 4. Duplicates

* **Exact:** size → quick hash (XXH3-64 over size + first/last 64 KiB) → full XXH3-128 → optional byte comparison.
  Each tier runs only on the survivors of the previous one. Duplicates get `dup_of = <kept id>` and are listed as
  "duplicate" in the preview; they are never copied or deleted.
* **Kept copy:** pinned > already in the library > richer metadata > does not look like a copy (`(1)`, ` copy`,
  `副本`, `backup/`, `备份/`, `copies/`, `old/`) > older mtime > shorter path > older id.
* **Similar:** dHash (9×8 grayscale difference hash of the oriented image). Two photos match when the Hamming distance
  is ≤ the threshold (6) and their aspect ratios are within 6 % (or swapped for rotation). Flat images (≤ 3 or ≥ 61
  set bits) are ignored. Clusters use union-find, and the largest image represents the cluster. These are
  report-only.
