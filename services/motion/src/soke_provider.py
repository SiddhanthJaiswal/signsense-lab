"""SOKE inference adapter: compact mBART vocabulary + matching learned VQ decoders.

Algorithm reference: https://github.com/2000ZRL/SOKE (mgpt_mbart/lm_multihead).
The language model, three codebooks, normalization and retrieval memory must be
loaded as one versioned bundle; motion-code IDs are not portable across models.
"""
import json
import os
import re
import sys
import time
import numpy as np
from .paths import ROOT
from .phrases import phrase_plan
from .rotations import to_quaternion, to_rotvec

SOKE_ROOT = ROOT / 'models/soke'
SOKE_REVISION = '1385f54f5324c3cb9abaf0a3b8bfcab97b2a0ef2'
sys.path.insert(0, str(ROOT / 'vendor/robostar'))


class SOKEProvider:
    def __init__(self):
        self.generator = None
        self.device = None

    def load(self):
        if self.generator is not None:
            return
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import MBartConfig, MBartForConditionalGeneration, MBartTokenizer
        from robostar.temporal import TemporalDecoder
        state = load_file(str(SOKE_ROOT / 'model.safetensors'))
        tokenizer = MBartTokenizer(vocab_file=str(SOKE_ROOT / 'sentencepiece.bpe.model'), src_lang='en_XX')
        languages = ['en_ASL', 'zh_CSL', 'de_DGS']
        streams = [('body', 'motion', 96, 'vae', 43), ('left', 'hand', 192, 'hand_vae', 45), ('right', 'rhand', 192, 'rhand_vae', 45)]
        tokenizer.add_tokens(languages, special_tokens=True)
        symbols = {part: [f'<{prefix}_id_{i}>' for i in range(size + 3)] for part, prefix, size, _, _ in streams}
        additions = languages + sum([symbols[part] for part, *_ in streams], [])
        tokenizer.add_tokens(additions[len(languages):])
        mapping = {int(k): int(v) for k, v in json.loads((SOKE_ROOT / 'map_ids.json').read_text()).items()}
        for token in additions:
            mapping[tokenizer.convert_tokens_to_ids(token)] = len(mapping)
        prefix = 'lm.language_model.main_lm.'
        lm_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        vocab_size = lm_state['model.shared.weight'].shape[0]
        if len(mapping) != vocab_size:
            raise ValueError(f'SOKE vocabulary mismatch: {len(mapping)} vs {vocab_size}')
        config = MBartConfig.from_pretrained(SOKE_ROOT, vocab_size=vocab_size, attn_implementation='eager')
        with init_empty_weights(include_buffers=False):
            generator = MBartForConditionalGeneration(config)
        generator.load_state_dict(lm_state, strict=True, assign=True)
        generator.tie_weights()
        requested = os.environ.get('MOTION_DEVICE', 'auto')
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(requested)
        self.decoders, self.codebooks, self.code_ids, self.allowed_ids = {}, {}, {}, {}
        special = [0, 1, 2, 3] + [mapping[tokenizer.convert_tokens_to_ids(t)] for t in ['en_XX','zh_CN','de_DE','<mask>',*languages]]
        for part, _, size, vae, nfeats in streams:
            decoder = TemporalDecoder(nfeats)
            decoder.load_state_dict({k[len(vae+'.decoder.'):]: v for k, v in state.items() if k.startswith(vae+'.decoder.')}, strict=True)
            self.decoders[part] = decoder.to(self.device).eval()
            self.codebooks[part] = state[vae+'.quantizer.codebook'].to(self.device)
            ids = [mapping[tokenizer.convert_tokens_to_ids(t)] for t in symbols[part]]
            self.code_ids[part] = ids[:size]
            self.allowed_ids[part] = torch.tensor(sorted(set(ids + special)), device=self.device)
        self.tokenizer, self.mapping = tokenizer, mapping
        self.start = mapping[tokenizer.convert_tokens_to_ids('en_ASL')]
        self.eos = mapping[tokenizer.eos_token_id]
        self.mean = np.load(SOKE_ROOT / 'mean133.npy')
        self.std = np.load(SOKE_ROOT / 'std133.npy')
        self.dictionary = json.loads((SOKE_ROOT / 'word2code.json').read_text())
        self.generator = generator.to(self.device).eval()

    def source_text(self, text):
        # Punctuation remains text; retrieval tokenizes words rather than spaces.
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        aliases = {'hi': 'hello', 'hey': 'hello'}
        stop = {'a', 'an', 'the', 'am', 'is', 'are', 'was', 'were', 'to', 'of'}
        keywords = []
        for word in words:
            word = aliases.get(word, word)
            if word not in stop and word in self.dictionary and word not in keywords:
                keywords.append(word)
        keywords = keywords[:3]
        prompt = text
        for i, word in enumerate(keywords):
            entry = self.dictionary[word]
            n = len(entry['body'])
            indices = np.linspace(0, n-1, min(n, 10), dtype=int)
            prompt += f' Key word {i+1}, '
            for j in indices:
                prompt += f"<motion_id_{entry['body'][j]}><hand_id_{entry['lhand'][j]}><rhand_id_{entry['rhand'][j]}>"
        return prompt, keywords

    def generate(self, text, seed=20260703):
        import torch
        import torch.nn.functional as F
        self.load()
        started = time.monotonic()
        plan = phrase_plan(text)
        if plan:
            return self.generate_phrase(plan, seed, started)
        prompt, keywords = self.source_text(text)
        source = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
        raw_ids = source['input_ids'][0].tolist()
        unknown = self.mapping[self.tokenizer.unk_token_id]
        ids = torch.tensor([[self.mapping.get(i, unknown) for i in raw_ids]], device=self.device)
        attention = source['attention_mask'].to(self.device)
        sequences = {p: torch.tensor([[self.start]], device=self.device) for p in self.code_ids}
        with torch.inference_mode():
            encoded = self.generator.get_encoder()(input_ids=ids, attention_mask=attention).last_hidden_state
            decoder = self.generator.get_decoder()
            for _ in range(100):
                emb = decoder.get_input_embeddings()
                mixed = 0.2 * emb(sequences['body']) + 0.4 * emb(sequences['left']) + 0.4 * emb(sequences['right'])
                hidden = decoder(inputs_embeds=mixed, encoder_hidden_states=encoded, encoder_attention_mask=attention,
                                 use_cache=False).last_hidden_state[:, -1]
                logits = self.generator.lm_head(hidden) + self.generator.final_logits_bias
                next_ids = {}
                for part, allowed in self.allowed_ids.items():
                    next_id = allowed[logits.index_select(1, allowed).argmax(dim=-1)].reshape(1, 1)
                    sequences[part] = torch.cat([sequences[part], next_id], dim=1)
                    next_ids[part] = int(next_id.item())
                if next_ids['body'] == self.eos:
                    break
            codes, decoded = {}, {}
            for part, sequence in sequences.items():
                lookup = {token: i for i, token in enumerate(self.code_ids[part])}
                codes[part] = [lookup[t] for t in sequence[0].tolist() if t in lookup]
                if len(codes[part]) < 2:
                    raise RuntimeError(f'SOKE returned too few {part} motion tokens')
                indices = torch.tensor(codes[part], device=self.device)
                latent = self.codebooks[part][indices].T[None]
                decoded[part] = self.decoders[part](latent).transpose(1, 2)[0]
            length = max(len(x) for x in decoded.values())
            for part, x in decoded.items():
                if len(x) < length:
                    decoded[part] = torch.cat([x, x[-1:].expand(length-len(x), -1)])
            b = decoded['body']
            normalized = torch.cat([b[:, :30], decoded['left'], decoded['right'], b[:, 30:]], dim=-1).cpu().numpy()
        motion = (normalized * self.std + self.mean).astype(np.float32)
        if not np.isfinite(motion).all():
            raise RuntimeError('SOKE produced nonfinite motion')
        return motion, {'generationSeconds': round(time.monotonic()-started, 3), 'device': str(self.device),
                        'retrievalKeywords': keywords, 'seed': seed, 'sampling': False, 'model': 'SOKE',
                        'revision': SOKE_REVISION, 'fps': 20, 'motionTokens': codes, 'inputTokenCount': len(raw_ids), 'mode': 'research'}

    def generate_phrase(self, plan, seed, started):
        """Decode matching SOKE codebooks, preserving each sign's learned motion.

        These token streams come from SOKE's published retrieval memory, not
        RoboSTAR's unrelated FSQ code IDs. Phrase planning never emits ARE.
        """
        import torch
        from .body import BodyModel
        from .phrase_constraints import point_at_addressee
        body = BodyModel()
        clips, segments = [], []
        cursor = 0
        for word in plan['glosses']:
            entry = self.dictionary[word]
            decoded = {}
            with torch.inference_mode():
                for part, key in [('body', 'body'), ('left', 'lhand'), ('right', 'rhand')]:
                    indices = torch.tensor(entry[key], device=self.device)
                    latent = self.codebooks[part][indices].T[None]
                    decoded[part] = self.decoders[part](latent).transpose(1, 2)[0]
                b = decoded['body']
                normalized = torch.cat([b[:, :30], decoded['left'], decoded['right'], b[:, 30:]], -1).cpu().numpy()
            clip = (normalized * self.std + self.mean).astype(np.float32)
            if word == 'you':
                clip = point_at_addressee(clip, body)
            if clips:
                # A short transition joins the learned clips. Smoothstep
                # avoids a velocity jump; rotations remain native axis angles.
                transition = blend_motion(clips[-1][-1], clip[0], 8)
                clips.append(transition)
                cursor += len(transition)
            segments.append({'gloss': word.upper(), 'start': cursor / 20, 'end': (cursor + len(clip)) / 20})
            clips.append(clip)
            cursor += len(clip)
        motion = np.concatenate(clips)
        return motion, {'generationSeconds': round(time.monotonic()-started, 3), 'device': str(self.device),
                        'retrievalKeywords': plan['glosses'], 'seed': seed, 'sampling': False,
                        'model': 'SOKE', 'revision': SOKE_REVISION, 'fps': 20, 'mode': 'phrase',
                        'glosses': [w.upper() for w in plan['glosses']], 'segments': segments,
                        'constraints': ['YOU: aim right index toward front-facing addressee'] if 'you' in plan['glosses'] else [],
                        'planSource': plan['source'], 'motionSource': 'SOKE published retrieval memory + learned VQ decoder'}


def blend_motion(first, last, frames):
    """Interpolate the full native rotations, including the non-flat hand means."""
    from .body import BodyModel
    body = BodyModel(); body.load()
    ends = np.stack([first, last]).copy()
    ends[:, 30:75] += body.left_mean
    ends[:, 75:120] += body.right_mean
    vectors = ends[:, :123].reshape(2, 41, 3).astype(np.float64)
    q = to_quaternion(vectors)
    q[1] *= np.where((q[0]*q[1]).sum(-1, keepdims=True) < 0, -1, 1)
    t = np.linspace(0, 1, frames+2)[1:-1]
    t = (t*t*(3-2*t))[:, None, None]
    dot = np.clip((q[0]*q[1]).sum(-1), -1, 1)
    theta = np.arccos(dot)[None, :, None]
    sin = np.sin(theta)
    a = np.divide(np.sin((1-t)*theta), sin, out=np.broadcast_to(1-t, (frames,41,1)).copy(), where=sin>1e-6)
    b = np.divide(np.sin(t*theta), sin, out=np.broadcast_to(t, (frames,41,1)).copy(), where=sin>1e-6)
    blended = a*q[0]+b*q[1]
    rotations = to_rotvec(blended).reshape(frames, 123)
    result = np.empty((frames,133),dtype=np.float32)
    result[:,:123] = rotations
    result[:,30:75] -= body.left_mean
    result[:,75:120] -= body.right_mean
    result[:,123:] = first[123:]*(1-t[:,0])+last[123:]*t[:,0]
    return result
