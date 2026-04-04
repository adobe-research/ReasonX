# Sample Data

Place your scene images here following the naming convention below.
Each scene needs an RGB image and one or more intrinsic modality maps
(all standard 8-bit PNGs at the same resolution):

```
scene_001_rgb.png          # sRGB image
scene_001_depth.png        # grayscale depth (higher intensity = closer)
scene_001_normal.png       # camera-space normals encoded as (n + 1) / 2 * 255
scene_001_albedo.png       # diffuse reflectance (sRGB)
scene_001_irradiance.png   # diffuse illumination (sRGB, optional)
scene_002_rgb.png
...
```

Then run data generation:

```bash
python generate_intrinsic_judgements.py \
    --data_dir examples/sample_data \
    --output_dir data/intrinsic_judgements
```

Not every modality is required for every scene — missing files are skipped
gracefully.

**Note:** 16-bit PNGs are also supported (especially useful for depth maps
where 8-bit quantization may be too coarse). The loading code converts
everything to float32 internally, so higher bit-depth images work
transparently.
