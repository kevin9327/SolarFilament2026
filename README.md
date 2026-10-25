# Solar Filament Segmentation Challenge 2026 (`CLV Limb-Darkening + Multi-Scale Frangi Tubular Filter + Disjoint Watershed PQ`)

Official open-source repository for **IEEE Big Data Cup 2026 (Challenge 02): Solar Filament Segmentation Challenge 2026** (sponsored by NSF National Solar Observatory).

## Pipeline Overview
1. **Radial Center-to-Limb Variation (`CLV`) Normalization:** Estimates the solar disk center `(cx, cy)` and radius `R` on `2048 x 2048` 8-bit grayscale GONG H-Alpha observations and divides out the radial limb-darkening profile.
2. **Multi-Scale Hessian / Frangi Dark-Tubular Enhancement:** Enhances elongated dark absorption filaments while suppressing circular sunspots.
3. **Morphological Geodesic Reconstruction & Disjoint Instance Partitioning:** Ensures strictly non-overlapping (`overlap == 0`) filament instance masks encoded in column-major RLE (`filament_id,segmentation_rle`) to maximize Panoptic Quality (`PQ = SQ * RQ`).

## Reproducibility
```bash
pip install -r requirements.txt
python solar_filament_pq_solver.py
```
