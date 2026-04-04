# ReasonX — Intrinsic Judge

Training, data generation, and evaluation code for the MLLM intrinsic judge from
**[ReasonX: MLLM-Guided Intrinsic Image Decomposition](https://arxiv.org/abs/2512.04222)** (CVPR 2026). The judge is fine-tuned from [InternVL2.5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) via LoRA to make pairwise relative intrinsic comparisons (depth, surface normals, albedo, irradiance) between two marked points on an RGB image. 


The training code under `internvl_chat/` is adapted from [InternVL](https://github.com/OpenGVLab/InternVL).

```
.
├── infer.py                            # single-image inference (no ground truth needed)
├── generate_intrinsic_judgements.py    # reference data pipeline: RGB and intrinsic PNGs -> training JSONL
├── evaluate_intrinsic_judge.py         # evaluation (accuracy & F1)
├── requirements.txt
├── internvl_chat/
│   ├── internvl/                       # InternVL training module
│   ├── merge_lora.py                   # merge LoRA weights into base model
│   └── shell/                          # training scripts & dataset config
├── examples/
│   └── sample_data/                    # 5 sample scenes for testing the pipeline
└── LICENSE
```

## Setup

```bash
git clone https://github.com/adobe-research/ReasonX.git
cd internvl-rgbx-judge
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation   # recommended
```

**Checkpoint link:** [adirik/InternVL2_5-4B-Intrinsic-Judge](https://huggingface.co/adirik/InternVL2_5-4B-Intrinsic-Judge) 

## Inference

You can test the released judge on any RGB image without  ground truth intrinsics:

```bash
# pre-annotated image (red/green markers already drawn):
python infer.py --image annotated.png \
    --question "Which point appears to be closer to the camera - red or green?"

# plain image + point coordinates (markers drawn automatically):
python infer.py --image scene.png --red 120 200 --green 350 180 \
    --question "Which point is more illuminated?"

# ask all four modality questions at once:
python infer.py --image scene.png --red 120 200 --green 350 180 --all --save_annotated output.png
```

Or in Python:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from infer import load_image, draw_points

model = AutoModel.from_pretrained(
    "adirik/InternVL2_5-4B-Intrinsic-Judge",
    torch_dtype=torch.bfloat16,
    use_flash_attn=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "adirik/InternVL2_5-4B-Intrinsic-Judge",
    trust_remote_code=True,
    use_fast=False,
)

pixel_values = load_image("annotated_image.jpg").to(torch.bfloat16).cuda()

question = "<image>\nWhich point appears to be closer to the camera - red or green?"
response = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=128, do_sample=False))
print(response)
```

## Training data

The judge is trained on RGB images from
[HyperSim](https://github.com/apple/ml-hypersim) and
[InteriorVerse](https://interiorverse.github.io/), using the
rendered intrinsic maps (depth, normals, albedo, irradiance) to derive
ground-truth pairwise answers. We provide a simple dataloader
(`generate_intrinsic_judgements.py`) that reads corresponding RGB and
intrinsic images, samples point pairs, and outputs the annotated JSONL
used for LoRA fine-tuning. Follow the instructions of the respective
datasets to download and pre-process the data. Five sample scenes from
HyperSim are included under `examples/sample_data/` for testing the
pipeline.

The dataloader expects corresponding scene images in a flat folder:

```
data_dir/
  scene_001_rgb.png            # sRGB
  scene_001_depth.png          # grayscale, lower intensity = closer (8- or 16-bit)
  scene_001_normal.png         # camera-space normals, (n+1)/2 * 255
  scene_001_albedo.png         # diffuse reflectance (sRGB)
  scene_001_irradiance.png     # optional
  scene_002_rgb.png
  ...
```

Then run:

```bash
python generate_intrinsic_judgements.py \
    --data_dir examples/sample_data \
    --output_dir data/intrinsic_judgements
```

This generates annotated images and train/val JSONL splits under `data/intrinsic_judgements/`.

## Fine-tuning

Before training, update `length` in `internvl_chat/shell/data/intrinsic_judgement_data.json` to match the number of total training samples in your generated JSONL (count the lines in `intrinsic_judgement_train.jsonl`).

```bash
cd internvl_chat
bash shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_lora.sh
```

The training script downloads `OpenGVLab/InternVL2_5-4B` from HuggingFace, reads the dataset config from `shell/data/intrinsic_judgement_data.json` (paths are relative to `internvl_chat/`), and saves checkpoints to `output/`.

After training, merge the LoRA adapter back into the base model:

```bash
python merge_lora.py <path_to_lora_checkpoint> <output_path>
```

## Evaluation
Evaluate on the held-out val split produced by the data generation step:

```bash
python evaluate_intrinsic_judge.py \
    --model_path <merged_model_or_hf_id> \
    --dataset_path data/intrinsic_judgements/intrinsic_judgement_val.jsonl \
    --data_root data/intrinsic_judgements/
```

## Citation

```bibtex
@inproceedings{Dirik2025ReasonXMI,
    title   = {ReasonX: MLLM-Guided Intrinsic Image Decomposition},
    author  = {Dirik, Alara and Wang, Tuanfeng and Ceylan, Duygu and
               Zafeiriou, Stefanos and Fr{\"u}hst{\"u}ck, Anna},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
                 and Pattern Recognition (CVPR)},
    year    = {2026}
}
```

## License

This project is released under the MIT License. The InternVL training code (`internvl_chat/`) is subject to its [original license](https://github.com/OpenGVLab/InternVL/blob/main/LICENSE).
