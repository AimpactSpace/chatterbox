import argparse
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Chatterbox")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Path to save the generated audio")
    parser.add_argument("-a", "--audio-prompt", dest="audio_prompt", help="Optional reference audio to mimic")
    parser.add_argument("-e", "--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.5 is neutral)")
    parser.add_argument("-c", "--cfg-weight", type=float, default=0.5, help="Classifier free guidance weight")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-d", "--device", default="cuda", help="Device to run the model on")
    args = parser.parse_args()

    model = ChatterboxTTS.from_pretrained(device=args.device)
    wav = model.generate(
        args.text,
        audio_prompt_path=args.audio_prompt,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    ta.save(args.output, wav, model.sr)
    print(f"Saved audio to {args.output}")


if __name__ == "__main__":
    main()
