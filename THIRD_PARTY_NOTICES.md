# Third-party notices

This repository contains no code copied from other projects. It does redistribute small metadata files derived from
third-party data, and test fixtures that include configuration and tokenizer files derived from third-party models;
their sources and licenses are listed here. Models, datasets and simulator assets are not redistributed:
`runtime/fetch_models.py` downloads them from their upstream repositories (see README "Data and licenses").

## LIBERO (hf-libero)

- Source: https://github.com/huggingface/LIBERO (fork of https://github.com/Lifelong-Robot-Learning/LIBERO), commit
  `8561c60eea2fb93096146f240194649df73d8b1e` (`version="0.1.4"`, the PyPI distribution `hf-libero` 0.1.4).
- License: MIT (text below, fetched from `LICENSE` at that commit).
- What is redistributed: `runtime/items-gate/items.jsonl`, `runtime/items-probe/items.jsonl`,
  `runtime/ITEMS-MANIFEST.json` and `tests/fixtures/lerobot/items/*.jsonl` list the LIBERO-Object task names, their
  language instructions and the SHA-256 of LIBERO's packaged initial-state files; the episode logs under
  `tests/fixtures/lerobot/eval2-500/` name the same tasks.

```
MIT License

Copyright (c) 2023 Lifelong Robot Learning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## lerobot/libero (dataset)

- Source: https://huggingface.co/datasets/lerobot/libero, revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.
- License: Apache-2.0 (dataset card `license: apache-2.0`).
- What is redistributed: `runtime/frames/frames_meta.json` (episode and frame index ranges and the SHA-256 of each
  parquet data file) and `runtime/MANIFEST.json` (file names, sizes and SHA-256 of the pinned repositories); no frame,
  state, action or video data.

## SmolVLA base policy and SmolVLM2 configuration and tokenizer files

- Where: `tests/fixtures/lerobot/checkpoint-500/pretrained_model/` (copies of a trained cut's small files, as LeRobot
  0.7.0 wrote them; host paths replaced by a neutral layout; the weights and `tokenizer.json` are not included):
  - `config.json`: the SmolVLA policy configuration, derived from `lerobot/smolvla_base`'s `config.json` (the training
    run's input and output features and settings added);
  - `tokenizer/chat_template.jinja`: the chat template of `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (equal to the
    template in its `chat_template.json`);
  - `tokenizer/tokenizer_config.json`: the tokenizer configuration of `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, as
    re-saved by transformers.
- Sources: `lerobot/smolvla_base` @ `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (https://huggingface.co/lerobot/smolvla_base,
  the LeRobot team, Hugging Face) and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @
  `4cde9d208e19519c36917aa170fffc7e381aa0ad` (https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct,
  Hugging Face).
- License: Apache-2.0 (both model cards declare `license: apache-2.0`; the full text is in this repository's
  `LICENSE`).
- The other files in that directory (the processor descriptions and their normalization statistics, computed from
  `lerobot/libero`, Apache-2.0; `train_config.json`) were written by LeRobot for the training run.
