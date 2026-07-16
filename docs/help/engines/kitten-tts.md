# KittenTTS

KittenTTS is TTS-Story's simplest lightweight CPU engine. It provides eight English voices, requires no API key or GPU, and exposes only a few decisions.

## Best for

- Fast setup on a CPU-only computer
- Private English previews and narration
- Users who want named presets rather than reference cloning
- Testing the full TTS-Story workflow before downloading a multi-gigabyte engine

KittenTTS does not clone reference audio in this adapter.

## Requirements and setup

![Engine Settings navigation showing where to choose an engine panel](../../../static/help/screenshots/engine-settings-navigation.png)

*Open Engine Settings and select KittenTTS before choosing its model and default voice.*

The normal setup installs the pinned KittenTTS 0.8.0 wheel and attempts to prefetch the recommended mini model. If prefetch times out or is disabled, the first generation downloads it instead. No separate manual `pip install` is normally required.

Choose a model under [Settings → Engine Settings](app:settings/kitten-tts):

| Model | Approximate upstream size | Use |
|---|---:|---|
| mini | 80 MB | Recommended quality baseline |
| micro | 41 MB | Smaller alternative |
| nano fp32 | 56 MB | Very small model architecture |
| nano int8 | 25 MB | Smallest, experimental quantized option |

The default mini model is not under 25 MB; that description applies only to the nano-int8 option.

## Controls TTS-Story exposes

- **Model ID:** mini, micro, nano fp32, or experimental nano int8
- **Default Voice:** Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, or Leo
- **Chunk Size:** 300 characters by default; roughly 150–350 is a practical CPU range

The Generate page can assign a different one of those eight voices to each speaker. TTS-Story handles final output conversion and speaker FX outside the Kitten model.

## Effective-use tips

1. Begin with mini and Jasper, then compare voices using manuscript-like text.
2. Choose a smaller model for memory or startup reasons, not because smaller always means faster or better on every processor.
3. Use short Quick Tests for all eight voices; their apparent gender or tone may vary with text.
4. Reduce chunk size if long sentences lose rhythm. Increase it cautiously if joins are more distracting than within-chunk prosody.
5. Keep pronunciation fixes in the Alternate Word Registry rather than trying to solve them with a different model size.

## Time, privacy, and limitations

KittenTTS has a much smaller download and memory footprint than the large cloning engines. Setup may already cache the default model, so first use is often short; a newly selected variant still needs its own download and initialization. Actual CPU speed depends on processor and chunk count.

After the model is cached, text and audio remain local and there is no provider cost.

TTS-Story pins the 0.8.0 developer-preview package, supports English, and exposes eight preset voices only. It does not provide voice cloning, multilingual synthesis, model training, or the full feature set of a future KittenTTS release.

## Authoritative reference

- [KittenML KittenTTS official repository](https://github.com/KittenML/KittenTTS)
