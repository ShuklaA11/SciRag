"""NLI classifier for SciFact-style claim verification.

Wraps a HuggingFace MNLI-style sequence-classification model and maps
its 3-way ENTAILMENT / NEUTRAL / CONTRADICTION output onto SciFact's
SUPPORT / NEI / CONTRADICT labels.

Design notes:
  - Default checkpoint is ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``
    (~750MB). FEVER training data is closer to scientific claim
    verification than vanilla MNLI, so it is the right zero-shot anchor
    for Week 9 PLAN.md target (~72% zero-shot baseline).
  - HF MNLI label conventions vary across checkpoints. We read
    ``model.config.id2label`` at load time and build the mapping
    dynamically rather than hardcoding indices.
  - SciFact has a NEI class but MNLI models output entailment +
    contradiction + neutral. We map ``NEUTRAL -> NEI`` directly, and
    additionally route low-confidence entailment/contradiction
    predictions (max prob < ``nei_threshold``) to NEI. The threshold
    is a constructor arg so SB9.3 can tune it on the train split.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORT = "SUPPORT"
CONTRADICT = "CONTRADICT"
NEI = "NEI"
SCIFACT_LABELS = (SUPPORT, CONTRADICT, NEI)

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_NEI_THRESHOLD = 0.5
DEFAULT_MAX_LENGTH = 512


@dataclass(frozen=True)
class NLIPrediction:
    label: str
    support_prob: float
    contradict_prob: float
    nei_prob: float

    @property
    def max_directional_prob(self) -> float:
        """Max of SUPPORT/CONTRADICT probabilities (used for NEI gating)."""
        return max(self.support_prob, self.contradict_prob)


def _build_label_map(id2label: dict[int, str]) -> dict[int, str]:
    """Map HF MNLI label strings -> SciFact labels.

    Accepts the common variants (ENTAILMENT/NEUTRAL/CONTRADICTION,
    entailment/neutral/contradiction, ENTAIL/NEUTRAL/CONTRADICT).
    Raises ValueError if the checkpoint exposes labels we don't know
    how to interpret — better to fail loudly than to silently mis-map.
    """
    out: dict[int, str] = {}
    for idx, name in id2label.items():
        norm = name.strip().lower()
        if norm.startswith("entail"):
            out[int(idx)] = SUPPORT
        elif norm.startswith("contradict"):
            out[int(idx)] = CONTRADICT
        elif norm.startswith("neutral"):
            out[int(idx)] = NEI
        else:
            raise ValueError(
                f"Unrecognised MNLI label '{name}' at index {idx}; "
                "extend _build_label_map to handle this checkpoint."
            )
    if set(out.values()) != set(SCIFACT_LABELS):
        raise ValueError(
            f"Checkpoint id2label does not cover all three MNLI classes: {id2label}"
        )
    return out


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class NLIClassifier:
    """Zero-shot NLI classifier mapping MNLI -> SciFact labels.

    The hypothesis is the claim; the premise is the evidence text. This
    matches the standard SciFact framing (Wadden 2020).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        nei_threshold: float = DEFAULT_NEI_THRESHOLD,
        batch_size: int = 8,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = _pick_device(device)
        self.max_length = max_length
        self.nei_threshold = nei_threshold
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = _build_label_map(self.model.config.id2label)

    @classmethod
    def _from_parts(
        cls,
        model,
        tokenizer,
        label_map: dict[int, str],
        *,
        device: str = "cpu",
        max_length: int = DEFAULT_MAX_LENGTH,
        nei_threshold: float = DEFAULT_NEI_THRESHOLD,
        batch_size: int = 8,
    ) -> "NLIClassifier":
        """Test-only constructor that skips the HF load."""
        inst = cls.__new__(cls)
        inst.model_name = "test-stub"
        inst.device = device
        inst.max_length = max_length
        inst.nei_threshold = nei_threshold
        inst.batch_size = batch_size
        inst.tokenizer = tokenizer
        inst.model = model
        inst.label_map = label_map
        return inst

    def _decode_probs(self, probs_row) -> tuple[float, float, float]:
        """Read a single softmax row and pull out (sup, con, nei) probs."""
        sup = con = nei = 0.0
        for idx, label in self.label_map.items():
            p = float(probs_row[idx])
            if label == SUPPORT:
                sup = p
            elif label == CONTRADICT:
                con = p
            else:
                nei = p
        return sup, con, nei

    def _apply_threshold(self, sup: float, con: float, nei: float) -> str:
        """Pick the SciFact label given the three softmax probabilities."""
        if max(sup, con) < self.nei_threshold:
            return NEI
        # Argmax over all three classes (NEI can still win if the model
        # is genuinely confident the evidence is neutral).
        triples = ((SUPPORT, sup), (CONTRADICT, con), (NEI, nei))
        return max(triples, key=lambda kv: kv[1])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        """Classify (claim, evidence) pairs.

        ``claim`` is the hypothesis, ``evidence`` the premise — order
        matters: HF NLI pipelines expect ``premise [SEP] hypothesis``.
        """
        import torch

        if not pairs:
            return []

        outputs: list[NLIPrediction] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            premises = [evidence for _, evidence in chunk]
            hypotheses = [claim for claim, _ in chunk]
            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for row in probs:
                sup, con, nei = self._decode_probs(row)
                label = self._apply_threshold(sup, con, nei)
                outputs.append(
                    NLIPrediction(
                        label=label,
                        support_prob=sup,
                        contradict_prob=con,
                        nei_prob=nei,
                    )
                )
        return outputs

    def predict(self, claim: str, evidence: str) -> NLIPrediction:
        return self.predict_batch([(claim, evidence)])[0]
