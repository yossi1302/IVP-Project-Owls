# Clock localization: preprocessing comparison

Measured on the 100 training images with `scripts/compare_preprocessing.py`
(run on 27 Nov 2026). Test images were not used.

## What was compared

Every variant uses the same detector from `src/localization.py` (Hough circle
candidates, radial edge score, "prefer the larger circle" rule). Only the
step before it changes: colour channel, smoothing filter, or the working
resolution the image is shrunk to. Variants run the first pass only, without
the CLAHE fallback, unless the name says "+ fallback". The current method is
gray + median 5 + fallback at 512 px.

## Measures

There are no ground-truth clock positions, so nothing here is an accuracy.

- **found**: images where a circle is returned (a returned circle can be wrong).
- **weak**: returned circles with score below 0.45.
- **half / pad**: stability. The image is shrunk to half size, or a black
  border is added top-left. A result counts as stable if the centre moves by
  at most 5% of the radius and the radius changes by at most 10%, after
  accounting for the known change.
- **ms**: resize + detection time per image on the development machine;
  varies a few percent between runs.
- **differ**: images where the circle is not within those same limits of the
  current method's circle.

## Results

| Variant | found | weak | half | pad | ms | differ |
|---|---:|---:|---:|---:|---:|---:|
| **current method** (gray + median 5 + fallback) | 98 | 10 | 59 | 52 | 522 | – |
| gray + median 5 (first pass only) | 97 | 12 | 60 | 51 | 286 | 7 |
| gray, no blur | 94 | 10 | 63 | 44 | 676 | 52 |
| gray + median 9 | 97 | 17 | 63 | 51 | 153 | 50 |
| gray + median 9 + fallback | 100 | 12 | 66 | 53 | 256 | 47 |
| gray + Gaussian 5 | 97 | 12 | 57 | 51 | 381 | 46 |
| gray + bilateral | 98 | 14 | 57 | 45 | 245 | 48 |
| CLAHE + median 5 | 97 | 20 | 60 | 53 | 546 | 53 |
| HSV saturation + median 5 | 90 | 21 | 52 | 42 | 330 | 61 |
| HSV value + median 5 | 98 | 14 | 63 | 58 | 319 | 50 |
| work size 384 | 98 | 16 | 55 | 55 | 129 | 47 |
| work size 768 | 97 | 21 | 61 | 57 | 1291 | 44 |

## Visual inspection

The "differ" column is large for every variant, so the disagreement images
were inspected side by side (current method left, variant right) for the
three variants with the best numbers. The judgements below are one person's
reading of the overlays, not measurements. Most disagreements are the same
clock with the circle on the inner versus outer edge of the rim (or bezel),
which is the main reason the "differ" counts are so high. Those are counted
as neither better nor worse.

- **HSV value + median 5**: better on 1 image (32, square clock: none instead
  of a wrong circle); worse on 2 (39: circle far too large; 64: wrong circle
  instead of none). About the same as the current first pass otherwise.
- **gray + median 9** (first pass only): better on 2, 20, 32, 40, 58; worse on
  16, 37, 64, 70. 37 is an image the current method only gets right through
  the fallback.
- **gray + median 9 + fallback**: better on 2, 31, 40, 58 (58 only partly) and
  72 (tighter); worse on 16, 37, 64 and slightly on 80.

The other variants were judged on the measures only, not inspected image by
image.

## Findings

1. **No preprocessing option is clearly better than the current one.** The
   differences in stability are a few images out of 100, which is small
   compared with how much the detector already changes its choice on
   difficult images.
2. **Smoothing matters more than the channel.** Without a blur, 3 fewer clocks
   are found than with median 5 (94 vs 97) and padding stability drops
   (44 vs 51). Median 9 is the fastest
   option at 512 px and slightly more stable when shrinking (63 vs 60). Gaussian and
   bilateral filtering gave no gain over median 5.
3. **Colour channels:** HSV saturation is the worst variant on every measure
   (90 found, 21 weak, lowest stability). It only helps when the clock is
   strongly coloured and the background is not, which is rare here. HSV value
   has the best padding stability (58), but visual inspection found no
   overall gain.
4. **CLAHE as a first pass** raises the weak detections from 12 to 20. Local
   contrast enhancement also strengthens background texture. It is more
   useful as the fallback for weak results, which is how it's used now.
5. **Working resolution** (compared with the 512 px first pass): 384 px is
   about 2x faster, less stable when shrinking (55 vs 60), more stable with
   padding (55 vs 51), and has more weak results (16 vs 12). 768 px is about
   4.5x slower and shares the highest number of weak results (21), possibly
   because more fine texture survives. 512 is a reasonable middle.
6. **Median 9 + fallback** is the only variant that looks slightly better
   overall: it returns a circle on all 100 images, is the most stable when
   shrinking (66 vs 59), and is about twice as fast. By inspection it fixes
   about 5 images and makes about 4 worse. That is a small net gain on 100
   images judged by eye, so it is a candidate rather than a clear improvement.

## Decision

The default stays **gray + median 5 + fallback** for now. Switching to median 9
would change the circles on about half of the images, mostly the rim edge.
Hand detection and time reading (Omer's part) were developed on the current
circles, so a switch should be checked with the full pipeline first.

## Reproduce

```sh
.venv/bin/python scripts/compare_preprocessing.py
```

Writes `outputs/localization/preprocessing/summary.csv`, `per_image.csv` and
side-by-side sheets `differ_NN.jpg`, numbered in the order of the variants in
the script. All of these are git-ignored.
