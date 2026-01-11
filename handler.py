"""
RunPod Serverless Handler for YuE Singing Voice Synthesis

This handler wraps the YuE model for serverless inference on RunPod.
Generates singing audio from lyrics for birthday song generation.

Usage:
    Deploy to RunPod Serverless, then call with:
    {
        "input": {
            "lyrics": "[verse]\nHap-py birth-day to you...",
            "genres": "inspiring female uplifting pop airy vocal",
            "seed": 42  # Optional for reproducibility
        }
    }
"""

import os
import sys
import io
import base64
import time
import random
import logging
import uuid
from typing import Optional, Dict, Any

import numpy as np
import torch

# Add inference directory to path
inference_path = os.path.join(os.path.dirname(__file__), "inference")
sys.path.insert(0, inference_path)
sys.path.insert(0, os.path.join(inference_path, 'xcodec_mini_infer'))
sys.path.insert(0, os.path.join(inference_path, 'xcodec_mini_infer', 'descriptaudiocodec'))

import runpod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Global Model References (loaded once)
# ============================================
model = None
codec_model = None
mmtokenizer = None
codectool = None
device = None


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_models():
    """Load YuE models (called once on cold start)."""
    global model, codec_model, mmtokenizer, codectool, device

    if model is not None:
        logger.info("Models already loaded, skipping...")
        return

    logger.info("Loading YuE models...")
    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Import YuE components
    from transformers import AutoModelForCausalLM
    from omegaconf import OmegaConf
    from mmtokenizer import _MMSentencePieceTokenizer
    from codecmanipulator import CodecManipulator
    from models.soundstream_hubert_new import SoundStream

    # Paths
    base_path = os.path.dirname(__file__)
    tokenizer_path = os.path.join(inference_path, "mm_tokenizer_v0.2_hf", "tokenizer.model")
    codec_config_path = os.path.join(inference_path, "xcodec_mini_infer", "final_ckpt", "config.yaml")
    codec_ckpt_path = os.path.join(inference_path, "xcodec_mini_infer", "final_ckpt", "ckpt_00360000.pth")

    # Model names - use HF_HOME if models are pre-baked
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/app/models")
    stage1_model_name = os.environ.get("STAGE1_MODEL", "m-a-p/YuE-s1-7B-anneal-en-cot")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    mmtokenizer = _MMSentencePieceTokenizer(tokenizer_path)

    # Load codec manipulator
    codectool = CodecManipulator("xcodec", 0, 1)

    # Check if flash attention is available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Flash attention available, using flash_attention_2")
    except ImportError:
        attn_impl = "eager"
        logger.info("Flash attention not available, using eager attention")

    # Load Stage 1 model
    logger.info(f"Loading Stage 1 model: {stage1_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        stage1_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()

    # Load codec model
    logger.info("Loading codec model...")
    model_config = OmegaConf.load(codec_config_path)
    codec_model = SoundStream(**model_config.generator.config).to(device)
    parameter_dict = torch.load(codec_ckpt_path, map_location='cpu', weights_only=False)
    codec_model.load_state_dict(parameter_dict['codec_model'])
    codec_model.eval()

    elapsed = time.time() - start_time
    logger.info(f"Models loaded in {elapsed:.2f}s")


def split_lyrics(lyrics: str) -> list:
    """Split lyrics into segments by section markers."""
    import re
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics if structured_lyrics else [lyrics]


class BlockTokenRangeProcessor:
    """Logits processor to block certain token ranges."""
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def generate_singing(
    lyrics: str,
    genres: str = "inspiring female uplifting pop airy vocal",
    max_new_tokens: int = 3000,
    run_n_segments: int = 2,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate singing audio from lyrics using YuE.
    """
    global model, codec_model, mmtokenizer, codectool, device

    from einops import rearrange
    from transformers import LogitsProcessorList
    import soundfile as sf

    # Set seed if provided
    if seed is not None:
        seed_everything(seed)
    else:
        seed_everything(42)

    start_time = time.time()

    # Parse lyrics into segments
    lyrics_segments = split_lyrics(lyrics)
    logger.info(f"Processing {len(lyrics_segments)} lyric segments")

    # Limit segments
    run_n_segments = min(run_n_segments, len(lyrics_segments))

    # Build prompt following original YuE format
    full_lyrics = "\n".join(lyrics_segments[:run_n_segments])
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics_segments[:run_n_segments]

    # Special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')

    # Generation config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = 1.1

    logger.info("Stage 1: Generating audio codes...")
    stage1_start = time.time()

    raw_output = None

    for i, p in enumerate(prompt_texts[:run_n_segments + 1]):
        if i == 0:
            continue  # Skip the header prompt, it's included in i==1

        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <= 1 else 1.2

        logger.info(f"Processing segment {i}/{run_n_segments}")

        if i == 1:
            # First segment: include full header
            head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            # Subsequent segments
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids

        # Window slicing for context limit
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            logger.info(f'Section {i}: output length {input_ids.shape[-1]} exceeding context, using last {max_context} tokens.')
            input_ids = input_ids[:, -max_context:]

        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([
                    BlockTokenRangeProcessor(0, 32002),
                    BlockTokenRangeProcessor(32016, 32016)
                ]),
                guidance_scale=guidance_scale,
            )

            # Ensure EOA token at end
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)

        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    stage1_time = time.time() - stage1_start
    logger.info(f"Stage 1 completed in {stage1_time:.2f}s")

    # Extract audio codes
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

    if len(soa_idx) != len(eoa_idx):
        raise ValueError(f'Invalid pairs of soa and eoa: {len(soa_idx)} vs {len(eoa_idx)}')

    if len(soa_idx) == 0:
        raise ValueError("No audio codes generated from Stage 1")

    logger.info(f"Found {len(soa_idx)} audio segments")

    # Extract vocals (we only need vocals for birthday song)
    vocals = []
    for i in range(len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1:eoa_idx[i]]
        if len(codec_ids) == 0:
            continue
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        if len(codec_ids) == 0:
            continue
        # Extract vocals (interleaved format: vocal, instrumental, vocal, instrumental...)
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)

    if not vocals:
        raise ValueError("No vocal codes extracted")

    vocals = np.concatenate(vocals, axis=1)
    logger.info(f"Extracted vocal codes shape: {vocals.shape}")

    # Decode vocals to audio
    logger.info("Vocoding: Converting to audio...")
    vocoder_start = time.time()

    with torch.no_grad():
        # Reshape for codec model: [batch, codebooks, time]
        vocals_tensor = torch.from_numpy(vocals).unsqueeze(0).to(device)
        # Decode
        decoded_audio = codec_model.decode(vocals_tensor)
        audio = decoded_audio.squeeze().cpu().numpy()

    vocoder_time = time.time() - vocoder_start
    logger.info(f"Vocoding completed in {vocoder_time:.2f}s")

    # Normalize audio
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max() * 0.95

    # Encode to base64 WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    total_time = time.time() - start_time
    duration = len(audio) / 16000

    logger.info(f"Generation complete: {duration:.2f}s audio in {total_time:.2f}s")

    return {
        "audio_base64": audio_base64,
        "duration": duration,
        "sample_rate": 16000,
        "generation_time": total_time,
        "stage1_time": stage1_time,
        "vocoder_time": vocoder_time,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Input:
        {
            "lyrics": "[verse]\nHap-py birth-day to you...",
            "genres": "inspiring female uplifting pop airy vocal",
            "seed": 42,  # Optional
            "max_new_tokens": 3000,  # Optional
            "run_n_segments": 2  # Optional
        }

    Output:
        {
            "audio_base64": "base64-encoded-wav",
            "duration": 18.5,
            "sample_rate": 16000,
            "status": "success"
        }
    """
    try:
        # Load models on first call (cold start)
        load_models()

        # Parse input
        job_input = job.get("input", {})

        lyrics = job_input.get("lyrics")
        if not lyrics:
            return {"error": "Missing required field: lyrics", "status": "error"}

        genres = job_input.get("genres", "inspiring female uplifting pop airy vocal")
        seed = job_input.get("seed")
        max_new_tokens = job_input.get("max_new_tokens", 3000)
        run_n_segments = job_input.get("run_n_segments", 2)

        # Generate
        result = generate_singing(
            lyrics=lyrics,
            genres=genres,
            max_new_tokens=max_new_tokens,
            run_n_segments=run_n_segments,
            seed=seed,
        )

        result["status"] = "success"
        return result

    except Exception as e:
        logger.exception("Handler error")
        return {
            "error": str(e),
            "status": "error"
        }


# Start RunPod serverless
runpod.serverless.start({"handler": handler})
