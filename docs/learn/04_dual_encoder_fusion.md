# 04 Dual-encoder fusion

## Why two encoders
CT and PET are different physical measurements on different grids. Separate encoders have separate weights,
so each learns features suited to its signal and each can be conditioned on its own spacing.

## B2: fusion by level index (resampled)
`DualEncoderUNet`: both volumes resampled to one grid, so level i of PET and CT have the same shape. Fuse =
concat (2C channels) then 1x1x1 conv -> C channels, at each level; decoder shared.

## ACF: fusion by *voxel size*
For a 96 mm box, CT at 1 mm, PET at 4 mm (values measured with `ACF.level_table`):

| CT level | CT shape | CT voxel | PET level | PET shape | PET voxel | action |
|---|---|---|---|---|---|---|
| L1 | 96^3 | 1 mm | | | | CT-only skip |
| L2 | 48^3 | 2 mm | | | | CT-only skip |
| L3 | 24^3 | 4 mm | P1 | 24^3 | 4 mm | fuse |
| L4 | 12^3 | 8 mm | P2 | 12^3 | 8 mm | fuse |
| L5 | 6^3 | 16 mm | P3 | 6^3 | 16 mm | fuse |

`pet_offset = 2 = log2(4 mm / 1 mm)`. Fusing P1 with L1 (what B2 does when grids are equal) would pair a 4 mm
feature with a 1 mm one at a different physical scale; level-matching pairs like with like.

PET depth: three levels because a 24-voxel side halves to 12, 6, and instance norm cannot normalise a single
voxel; 24 -> 3 would already be near the limit.

## Anisotropic cases
CT 1x1x3 mm with PET 4 mm: voxel sizes at L3 are (4,4,12) vs (4,4,4). The shapes differ in z (24 vs 72 for a
288 mm z-extent), so PET features are trilinearly resized onto the CT grid (`F.interpolate`), **after** the
extent check passes. Resizing features is interpolation of learned representations, not resampling of the input
scan; whether this matters is an empirical question (E2).

## Per-fusion code walk (`ACF.forward`)
1. Input checks (CT size divisible by 16; extents agree per sample).
2. Hypernets -> FiLM parameters per block.
3. Encoders -> lists `ct_f` (L1..L5), `pet_f` (P1..P3).
4. For each CT level: PET level j = i - offset; if outside, skip; else project channels, check level extent,
   resize if needed, concat, fuse.
5. Decoder from L5 up, shared; head -> (B, K, X, Y, Z).

## Capacity confound to remember
Configured B2 has 9,481,362 parameters, B3/B4/B5 6,229,986, B1 5,730,274 (measured). B2 carries a full-depth PET
encoder at 1 mm. Differences between B2 and ACF are therefore not purely "design"; report parameter counts next
to every table.
