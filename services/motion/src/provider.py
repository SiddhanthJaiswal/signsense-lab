"""A single persistent RoboSTAR instance, loaded without duplicating the 7 GB model."""
import json
import os
import sys
from pathlib import Path
import time

from .paths import ROOT, MODEL_ROOT
from .schema import MODEL_REVISION

sys.path.insert(0, str(ROOT / "vendor" / "robostar"))

class RoboSTARProvider:
    def __init__(self):
        self.generator = None
        self.device = None
        self.load_seconds = None

    def load(self):
        if self.generator is not None:
            return
        import numpy as np
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from robostar.config import GeneratorConfig
        from robostar.generator import RoboSTARGenerator
        from robostar.vocabulary import RoboSTARVocabulary
        from robostar.tokenizer import RoboSTARTokenizer
        folder = MODEL_ROOT / "generator"
        if not (MODEL_ROOT / "revision.json").exists():
            raise RuntimeError("Model download is incomplete. Run python3 scripts/download_model.py.")
        revision = json.loads((MODEL_ROOT / "revision.json").read_text())
        if revision.get("revision") != MODEL_REVISION:
            raise RuntimeError("Unexpected model revision. Re-run the pinned model download.")
        started = time.monotonic()
        base = os.environ.get("ROBOSTAR_BASE_MODEL", str(ROOT / "models" / "mt5-large"))
        values = json.loads((folder / "config.json").read_text())
        values["model_path"] = base
        cfg = GeneratorConfig.from_mapping(values)
        vocab = RoboSTARVocabulary(base, cfg.body_codes, cfg.hand_codes, cfg.use_compact_map)
        expected = json.loads((folder / "vocab.json").read_text())
        for key, actual in [("vocab_size", vocab.vocab_size), ("en_asl_id", vocab.en_asl_id)]:
            if int(expected[key]) != actual:
                raise RuntimeError(f"Tokenizer vocabulary mismatch: {key}")
        # Parameters initialize on meta; buffers stay on CPU because the upstream
        # constructor uses their values to build masks. Safetensors are memory-mapped.
        with init_empty_weights(include_buffers=False):
            generator = RoboSTARGenerator(cfg, vocab, backbone_init="config_only")
        index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
        state = {}
        for filename in sorted(set(index.values())):
            state.update(load_file(str(folder / filename), device="cpu"))
        generator.load_state_dict(state, strict=True, assign=True)
        del state
        # The shared embedding and LM head must remain shared after assign=True.
        generator.main_lm.tie_weights()
        requested = os.environ.get("MOTION_DEVICE", "auto")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(requested)
        # Float32 preserves the released weights and avoids fp16 mT5 overflows.
        generator = generator.to(self.device).eval()
        tokenizer = RoboSTARTokenizer.from_pretrained(MODEL_ROOT, device=self.device).eval()
        mean = np.load(MODEL_ROOT / "stats" / "mean133.npy").astype(np.float32)
        std = np.load(MODEL_ROOT / "stats" / "std133.npy").astype(np.float32)
        dictionary = json.loads((MODEL_ROOT / "retrieval" / "word2code.json").read_text())
        self.generator, self.vocab, self.tokenizer = generator, vocab, tokenizer
        self.mean, self.std, self.dictionary = mean, std, dictionary
        self.load_seconds = round(time.monotonic() - started, 3)

    def generate(self, text: str, seed: int):
        import numpy as np
        import torch
        from robostar.retrieval import extract_keywords
        from robostar.representation import denormalize_motion
        self.load()
        torch.manual_seed(seed)
        keywords = extract_keywords(text, self.dictionary, 3)
        prompt = self.vocab.retrieval_prompt(text, keywords, self.dictionary)
        encoded = self.vocab.encode_text([prompt], self.generator.cfg.max_source_length)
        started = time.monotonic()
        with torch.inference_mode():
            result = self.generator.generate(encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device),
                                             None, temperature=1.0, top_k=0, do_sample=False)
            length = int(result["full_lengths"][0].item())
            codes = {part: result["final_codes"][part][0:1, :length] for part in ["body", "left", "right"]}
            normalized = self.tokenizer.decode(codes)[0].float().cpu().numpy()
        motion = denormalize_motion(normalized, self.mean, self.std)
        if motion.ndim != 2 or motion.shape[1] != 133 or not np.isfinite(motion).all():
            raise RuntimeError("Model produced invalid motion; playback was stopped.")
        return motion, {"generationSeconds": round(time.monotonic() - started, 3), "device": str(self.device),
                        "retrievalKeywords": keywords, "seed": seed, "sampling": False,
                        "model": "Ivystream/RoboSTAR", "revision": MODEL_REVISION, "fps": 20}
