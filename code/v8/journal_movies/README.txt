Journal-style renderer for Supplementary Movies 1, 3 and 4 (23 Oct 2026)

1. python3 prepare.py            -> movie1.json, movie3.json, movie4.json (reads frozen V8 paths and per_realization.csv; set FLY_ROOT if run elsewhere)
2. node driver.cjs movie3.json Movie3_paper.mp4   (needs Playwright + Chromium + ffmpeg; same for movie4, movie1 -> Movie1_flight_chapter.mp4)
3. bash finish.sh                -> appends the existing 18-s readout chapter to Movie 1 (m1_readouts.mp4 = seconds 10-28 of the previous Movie 1), checks decoding, writes contact sheets
Preview single frames: node driver.cjs movie4.json preview.png 0,300,599
No neural or controller simulation is run; positions are the saved paths.
