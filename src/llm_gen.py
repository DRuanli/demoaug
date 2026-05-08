"""Toy LLMs with hard token-level banning.

ToyClassConditionalLLM: bigram per (G,Y) cell — INHERITS group bias.
                        Used by the LLM-Unconstrained baseline.

ToyNeutralizingLLM:     bigram per Y class only — group-agnostic by design.
                        Stand-in for what a real LLM does with few-shot
                        prompting from BOTH groups + logit-bias on V_disc.
                        Used inside DEMOAUG.
"""
import random
from collections import defaultdict, Counter
from config import GEN_TEMPERATURE, SEED


def _sample_next(counter, banned, temperature, rng):
    if not counter:
        return None
    items = [(w, c) for w, c in counter.items() if w not in banned]
    if not items:
        items = list(counter.items())
    words, counts = zip(*items)
    log_p = [c ** (1.0 / max(temperature, 1e-3)) for c in counts]
    total = sum(log_p)
    probs = [p / total for p in log_p]
    return rng.choices(words, weights=probs, k=1)[0]


def _generate(bg, banned, max_tokens, n, temperature, seed, cell_hash=0):
    outputs = []
    for k in range(n):
        local_seed = (int(seed) + k * 1009 + cell_hash) & 0x7FFFFFFF
        local_rng = random.Random(local_seed)
        tokens = []
        prev = "<S>"
        for _ in range(max_tokens):
            if prev not in bg:
                break
            nxt = _sample_next(bg[prev], banned, temperature, local_rng)
            if nxt is None or nxt == "<E>":
                break
            tokens.append(nxt)
            prev = nxt
        outputs.append(" ".join(tokens))
    return outputs


class ToyClassConditionalLLM:
    def __init__(self):
        self.bigram_models = {}
        self._fitted = False

    def fit(self, df):
        for (g, y), group in df.groupby(["G", "Y"]):
            bg = defaultdict(Counter)
            for text in group["text"]:
                tokens = ["<S>"] + text.split() + ["<E>"]
                for prev, nxt in zip(tokens, tokens[1:]):
                    bg[prev][nxt] += 1
            self.bigram_models[(int(g), int(y))] = dict(bg)
        self._fitted = True

    def generate(self, cell, banned_tokens, max_tokens=120, n=1,
                 temperature=GEN_TEMPERATURE, seed=SEED):
        assert self._fitted
        bg = self.bigram_models.get((int(cell[0]), int(cell[1])), {})
        return _generate(bg, banned_tokens, max_tokens, n, temperature, seed,
                         cell_hash=abs(hash(cell)) % 1000003)


class ToyNeutralizingLLM:
    def __init__(self):
        self.bigram_per_y = {}
        self._fitted = False

    def fit(self, df):
        for y, group in df.groupby("Y"):
            bg = defaultdict(Counter)
            for text in group["text"]:
                tokens = ["<S>"] + text.split() + ["<E>"]
                for prev, nxt in zip(tokens, tokens[1:]):
                    bg[prev][nxt] += 1
            self.bigram_per_y[int(y)] = dict(bg)
        self._fitted = True

    def generate(self, cell, banned_tokens, max_tokens=120, n=1,
                 temperature=GEN_TEMPERATURE, seed=SEED):
        assert self._fitted
        bg = self.bigram_per_y.get(int(cell[1]), {})
        return _generate(bg, banned_tokens, max_tokens, n, temperature, seed,
                         cell_hash=abs(hash(cell)) % 1000003)


def get_unconstrained_llm():
    return ToyClassConditionalLLM()


def get_neutral_llm():
    return ToyNeutralizingLLM()


def get_llm():
    return ToyClassConditionalLLM()


# ============================================================================
# Real-data integration: MistralLLM
# ============================================================================
# This class is the production implementation. It is gated behind
# USE_REAL_MODELS=True in config.py. The interface matches the toy LLMs
# above — same generate(cell, banned_tokens, max_tokens, n) signature.
#
# IMPORTANT: This class has NOT been integration-tested against a live
# Mistral-7B-Instruct download in this prototype environment (no GPU
# available). It is structurally complete and follows the documented
# vLLM / HuggingFace logits_processor API, but a real-data run is the
# first place to verify correctness. We mark it explicitly here rather
# than claiming "drop-in replacement" as the prior README did.
# ============================================================================

class MistralLLM:
    """Real Mistral-7B-Instruct wrapper with token-level hard banning.

    Usage:
        from llm_gen import MistralLLM
        llm = MistralLLM(model_name="mistralai/Mistral-7B-Instruct-v0.3")
        llm.fit(df_train)  # caches few-shot examples per (G, Y) cell
        outputs = llm.generate(cell=(1, 1), banned_tokens={"moreover", "very"},
                                max_tokens=120, n=10)

    Hard banning is implemented via HuggingFace LogitsProcessor that sets
    logit = -inf for any token id in the expanded subword set of banned
    words. This is strictly stronger than instruction-level negative
    prompting (which has ~30-40% violation rate on 7B-class models).
    """

    FEW_SHOT_PROMPT_TEMPLATE = (
        "You are generating educational forum posts for an academic course. "
        "Below are real example posts from students with various backgrounds. "
        "Generate ONE more post in the same style and topic.\n\n"
        "{examples}\n\n"
        "New post:"
    )

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 device: str = "auto", load_in_4bit: bool = True,
                 n_few_shot: int = 5):
        # Lazy imports — module loadable without transformers installed
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(
                "MistralLLM requires `transformers` and `torch`. Install with:\n"
                "  pip install torch transformers accelerate bitsandbytes\n"
                f"Original error: {e}"
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": device}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                pass  # fall back to fp16 if bitsandbytes not available
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.few_shot_per_cell = {}
        self.n_few_shot = n_few_shot
        self._fitted = False

    def fit(self, df):
        """Cache few-shot examples per (G, Y) cell."""
        for (g, y), group in df.groupby(["G", "Y"]):
            if len(group) >= self.n_few_shot:
                examples = group["text"].sample(self.n_few_shot,
                                                 random_state=int(g) * 7 + int(y)).tolist()
            else:
                examples = group["text"].tolist()
            self.few_shot_per_cell[(int(g), int(y))] = examples
        self._fitted = True
        return self

    def _make_logit_bias_processor(self, banned_token_ids):
        """Return a LogitsProcessor that sets logits[banned] = -inf."""
        from transformers import LogitsProcessor
        torch = self._torch
        banned = list(banned_token_ids)

        class HardBan(LogitsProcessor):
            def __init__(self_inner):
                self_inner.banned = torch.tensor(banned, dtype=torch.long)

            def __call__(self_inner, input_ids, scores):
                if len(self_inner.banned) == 0:
                    return scores
                scores = scores.clone()
                if scores.is_cuda:
                    self_inner.banned = self_inner.banned.to(scores.device)
                scores[:, self_inner.banned] = float("-inf")
                return scores

        return HardBan()

    def _expand_to_subwords(self, banned_words):
        """Expand banned words to all their subword token ids."""
        banned_ids = set()
        for w in banned_words:
            for variant in [w, " " + w, w.capitalize(), " " + w.capitalize()]:
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                banned_ids.update(ids)
        return banned_ids

    def generate(self, cell, banned_tokens, max_tokens=120, n=1,
                 temperature=0.85, seed=42):
        from transformers import LogitsProcessorList
        torch = self._torch
        assert self._fitted

        examples = self.few_shot_per_cell.get(tuple(cell), [])
        examples_str = "\n---\n".join(f'"{e}"' for e in examples)
        prompt = self.FEW_SHOT_PROMPT_TEMPLATE.format(examples=examples_str)

        banned_ids = self._expand_to_subwords(banned_tokens)
        processor = LogitsProcessorList([self._make_logit_bias_processor(banned_ids)])

        torch.manual_seed(seed)
        outputs = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        for k in range(n):
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    logits_processor=processor,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                out_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            outputs.append(text)
        return outputs


def get_real_llm():
    """Factory for real Mistral. Raises ImportError if transformers missing."""
    return MistralLLM()
