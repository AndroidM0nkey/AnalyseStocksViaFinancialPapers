from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


@dataclass(frozen=True)
class HfRuntimeConfig:
    model_name: str
    use_4bit: bool
    device_map: str


_lock = threading.Lock()
_generator: Any | None = None
_tokenizer: Any | None = None
_loaded_key: tuple[str, bool, str] | None = None


def _build_generator(config: HfRuntimeConfig) -> tuple[Any, Any]:
    started_at = time.perf_counter()
    logging.info(
        "HF preload started for model=%s use_4bit=%s device_map=%s",
        config.model_name,
        config.use_4bit,
        config.device_map,
    )
    bnb_config = None
    if config.use_4bit:
        logging.info("Preparing 4-bit quantization config")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    logging.info("Loading tokenizer for %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    logging.info("Loading model weights for %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config,
    )
    model.eval()
    logging.info("Building transformers pipeline for %s", config.model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    logging.info(
        "Loaded HF model %s (4bit=%s, cuda=%s) in %.2fs",
        config.model_name,
        config.use_4bit,
        torch.cuda.is_available(),
        time.perf_counter() - started_at,
    )
    return tokenizer, generator


def get_generator(config: HfRuntimeConfig) -> tuple[Any, Any]:
    global _generator, _tokenizer, _loaded_key

    key = (config.model_name, config.use_4bit, config.device_map)
    if _generator is not None and _tokenizer is not None and _loaded_key == key:
        return _tokenizer, _generator

    with _lock:
        if _generator is not None and _tokenizer is not None and _loaded_key == key:
            return _tokenizer, _generator
        _tokenizer, _generator = _build_generator(config)
        _loaded_key = key
        return _tokenizer, _generator


def run_llm_json(config: HfRuntimeConfig, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    tokenizer, generator = get_generator(config)
    messages = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = generator(
        formatted_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    raw_text = output[0]["generated_text"][len(formatted_prompt) :]
    return {
        "prompt": prompt,
        "raw_response": raw_text,
    }


def preload_model(config: HfRuntimeConfig) -> None:
    get_generator(config)
