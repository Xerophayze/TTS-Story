---
license: apache-2.0
language:
- en
- zh
base_model:
- openbmb/MiniCPM4-0.5B
pipeline_tag: text-to-speech
library_name: voxcpm
tags:
- text-to-speech
- speech
- speech generation
- voice cloning
---

## 🎙️ VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning


[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://github.com/OpenBMB/VoxCPM/) [![Technical Report](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2509.24650)[![Live Playground](https://img.shields.io/badge/Live%20PlayGround-Demo-orange)](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://openbmb.github.io/VoxCPM-demopage)

- VoxCPM1.5

 [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OpenBMB-yellow)](https://huggingface.co/openbmb/VoxCPM1.5) [![ModelScope](https://img.shields.io/badge/ModelScope-OpenBMB-purple)](https://modelscope.cn/models/OpenBMB/VoxCPM1.5)  


<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="40%">
</div>

## 🎉 VoxCPM1.5 Updates

**Release Date:** December 5, 2025

VoxCPM1.5 brings improvements in audio quality and efficiency:

| Feature | VoxCPM | VoxCPM1.5 |
|---------|------------|------------|
| **Audio VAE Sampling Rate** | 16kHz | 44.1kHz |
| **LM Token Rate** | 12.5Hz | 6.25Hz |
| **Patch Size** | 2 | 4 |
| **SFT Support** | ✅ | ✅ |
| **LoRA Support** | ✅ | ✅ |

**Key Improvements:**
- 🔊 **Higher Quality**: 44.1kHz sampling rate preserves more high-frequency details for better voice cloning
- ⚡ **More Efficient**: Reduced token rate (6.25Hz) lowers computational cost while maintaining performance
- 🎓 **Fine-tuning Support**: Train personalized voice models with SFT or LoRA

**Note**: Output quality depends on the prompt speech quality. VoxCPM-0.5B remains fully supported with backward compatibility.


## 📚 Model Overview


VoxCPM is a novel tokenizer-free Text-to-Speech (TTS) system that redefines realism in speech synthesis. By modeling speech in a continuous space, it overcomes the limitations of discrete tokenization and enables two flagship capabilities: context-aware speech generation and true-to-life zero-shot voice cloning.

Unlike mainstream approaches that convert speech to discrete tokens, VoxCPM uses an end-to-end diffusion autoregressive architecture that directly generates continuous speech representations from text. Built on [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-0.5B) backbone, it achieves implicit semantic-acoustic decoupling through hierachical language modeling and FSQ constraints, greatly enhancing both expressiveness and generation stability.


<div align="center">
  <img src="assets/voxcpm_model.png" alt="VoxCPM Model Architecture" width="90%">
</div>


###  🚀 Key Features
- **Context-Aware, Expressive Speech Generation** - VoxCPM comprehends text to infer and generate appropriate prosody, delivering speech with remarkable expressiveness and natural flow. It spontaneously adapts speaking style based on content, producing highly fitting vocal expression trained on a massive 1.8 million-hour bilingual corpus.
- **True-to-Life Voice Cloning** - With only a short reference audio clip, VoxCPM performs accurate zero-shot voice cloning, capturing not only the speaker’s timbre but also fine-grained characteristics such as accent, emotional tone, rhythm, and pacing to create a faithful and natural replica.
- **High-Efficiency Synthesis** - VoxCPM supports streaming synthesis with a Real-Time Factor (RTF) as low as 0.17 on a consumer-grade NVIDIA RTX 4090 GPU, making it possible for real-time applications.




##  Quick Start

### 🔧 Install from PyPI
``` sh
pip install voxcpm
```
### 1.  Model Download (Optional)
By default, when you first run the script, the model will be downloaded automatically, but you can also download the model in advance.
- Download VoxCPM1.5
    ```
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/VoxCPM1.5")
    ```

- Or Download VoxCPM-0.5B
    ```
    from huggingface_hub import snapshot_download
    snapshot_download("openbmb/VoxCPM-0.5B")
    ```
- Download ZipEnhancer and SenseVoice-Small. We use ZipEnhancer to enhance speech prompts and SenseVoice-Small for speech prompt ASR in the web demo.
    ```
    from modelscope import snapshot_download
    snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
    snapshot_download('iic/SenseVoiceSmall')
    ```

### 2. Basic Usage
```python
import soundfile as sf
import numpy as np
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# Non-streaming
wav = model.generate(
    text="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
    prompt_wav_path=None,      # optional: path to a prompt speech for voice cloning
    prompt_text=None,          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=False,           # enable external TN tool, but will disable native raw text support
    denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
    retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)

sf.write("output.wav", wav, model.tts_model.sample_rate)
print("saved: output.wav")

# Streaming
chunks = []
for chunk in model.generate_streaming(
    text = "Streaming text to speech is easy with VoxCPM!",
    # supports same args as above
):
    chunks.append(chunk)
wav = np.concatenate(chunks)

sf.write("output_streaming.wav", wav, model.tts_model.sample_rate)
print("saved: output_streaming.wav")
```

### 3. CLI Usage

After installation, the entry point is `voxcpm` (or use `python -m voxcpm.cli`).

```bash
# 1) Direct synthesis (single text)
voxcpm --text "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech." --output out.wav

# 2) Voice cloning (reference audio + transcript)
voxcpm --text "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech." \
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  --output out.wav \
  # --denoise

# (Optinal) Voice cloning (reference audio + transcript file)
voxcpm --text "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech." \
  --prompt-audio path/to/voice.wav \
  --prompt-file "/path/to/text-file" \
  --output out.wav \
  # --denoise

# 3) Batch processing (one text per line)
voxcpm --input examples/input.txt --output-dir outs
# (optional) Batch + cloning
voxcpm --input examples/input.txt --output-dir outs \
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  # --denoise

# 4) Inference parameters (quality/speed)
voxcpm --text "..." --output out.wav \
  --cfg-value 2.0 --inference-timesteps 10 --normalize

# 5) Model loading
# Prefer local path
voxcpm --text "..." --output out.wav --model-path /path/to/VoxCPM_model_dir
# Or from Hugging Face (auto download/cache)
voxcpm --text "..." --output out.wav \
  --hf-model-id openbmb/VoxCPM1.5 --cache-dir ~/.cache/huggingface --local-files-only

# 6) Denoiser control
voxcpm --text "..." --output out.wav \
  --no-denoiser --zipenhancer-path iic/speech_zipenhancer_ans_multiloss_16k_base

# 7) Help
voxcpm --help
python -m voxcpm.cli --help
```

### 4. Start web demo

You can start the UI interface by running `python app.py`, which allows you to perform Voice Cloning and Voice Creation.

### 5. Fine-tuning

VoxCPM1.5 supports both full fine-tuning (SFT) and LoRA fine-tuning, allowing you to train personalized voice models on your own data. See the [Fine-tuning Guide](https://github.com/OpenBMB/VoxCPM/blob/main/docs/finetune.md) for detailed instructions.

**Quick Start:**
```bash
# Full fine-tuning
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml

# LoRA fine-tuning
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml
```


## 👩‍🍳 A Voice Chef's Guide
Welcome to the VoxCPM kitchen! Follow this recipe to cook up perfect generated speech. Let’s begin.

---
### 🥚 Step 1: Prepare Your Base Ingredients (Content)

First, choose how you’d like to input your text:.
1. Regular Text (Classic Mode)
- ✅ Keep "Text Normalization" ON. Type naturally (e.g., "Hello, world! 123"). The system will automatically process numbers, abbreviations, and punctuation using WeTextProcessing library.
2. Phoneme Input (Native Mode)
- ❌ Turn "Text Normalization" OFF. Enter phoneme text like {HH AH0 L OW1} (EN) or {ni3}{hao3} (ZH) for precise pronunciation  control. In this mode, VoxCPM also supports native understanding of other complex non-normalized text—try it out!
- **Phoneme Conversion**: For Chinese, phonemes are converted using pinyin. For English, phonemes are converted using CMUDict. Please refer to the relevant documentation for more details.


---
### 🍳 Step 2: Choose Your Flavor Profile (Voice Style) 

This is the secret sauce that gives your audio its unique sound.

#### 1. Cooking with a Prompt Speech (Following a Famous Recipe)
- A prompt speech provides the desired acoustic characteristics for VoxCPM. The speaker's timbre, speaking style, and even the background sounds and ambiance will be replicated.
- **For a Clean, Denoising Voice:**
  - ✅ Enable "Prompt Speech Enhancement". This acts like a noise filter, removing background hiss and rumble to give you a pure, clean voice clone. However, this will limit the audio sampling rate to 16kHz, restricting the cloning quality ceiling.
- **For High-Quality Audio Cloning (Up to 44.1kHz):**
  - ❌ Disable "Prompt Speech Enhancement" to preserve all original audio information, including background atmosphere, and support audio cloning up to 44.1kHz sampling rate.
    
#### 2. Cooking au Naturel (Letting the Model Improvise)
- If no reference is provided, VoxCPM becomes a creative chef! It will infer a fitting speaking style based on the text itself, thanks to the text-smartness of its foundation model, MiniCPM-4.
- **Pro Tip**: Challenge VoxCPM with any text—poetry, song lyrics, dramatic monologues—it may deliver some interesting results!

---
### 🧂 Step 3: The Final Seasoning (Fine-Tuning Your Results)

You're ready to serve! But for master chefs who want to tweak the flavor, here are two key spices.

#### CFG Value (How Closely to Follow the Recipe)
- **Default**: A great starting point.
- **Voice sounds strained or weird?** Lower this value. It tells the model to be more relaxed and improvisational, great for expressive prompts.
- **Need maximum clarity and adherence to the text?** Raise it slightly to keep the model on a tighter leash.
- **Short sentences?** Consider increasing the CFG value for better clarity and adherence.
- **Long texts?** Consider lowering the CFG value to improve stability and naturalness over extended passages.

#### Inference Timesteps (Simmering Time: Quality vs. Speed)
- **Need a quick snack?** Use a lower number. Perfect for fast drafts and experiments.
- **Cooking a gourmet meal?** Use a higher number. This lets the model "simmer" longer, refining the audio for superior detail and naturalness.

---
Happy creating! 🎉 Start with the default settings and tweak from there to suit your project. The kitchen is yours!


---


## ⚠️ Risks and limitations
- General Model Behavior: While VoxCPM has been trained on a large-scale dataset, it may still produce outputs that are unexpected, biased, or contain artifacts.
- Potential for Misuse of Voice Cloning: VoxCPM's powerful zero-shot voice cloning capability can generate highly realistic synthetic speech. This technology could be misused for creating convincing deepfakes for purposes of impersonation, fraud, or spreading disinformation. Users of this model must not use it to create content that infringes upon the rights of individuals. It is strictly forbidden to use VoxCPM for any illegal or unethical purposes. We strongly recommend that any publicly shared content generated with this model be clearly marked as AI-generated.
- Current Technical Limitations: Although generally stable, the model may occasionally exhibit instability, especially with very long or expressive inputs. Furthermore, the current version offers limited direct control over specific speech attributes like emotion or speaking style.
- Bilingual Model: VoxCPM is trained primarily on Chinese and English data. Performance on other languages is not guaranteed and may result in unpredictable or low-quality audio.
- This model is released for research and development purposes only. We do not recommend its use in production or commercial applications without rigorous testing and safety evaluations. Please use VoxCPM responsibly.



## 📄 License
The VoxCPM model weights and code are open-sourced under the Apache-2.0 license.

