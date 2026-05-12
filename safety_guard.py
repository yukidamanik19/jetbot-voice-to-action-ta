import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from difflib import SequenceMatcher


@dataclass
class GuardDecision:
    allowed: bool
    decision: str            # allow | reject | ask_clarification
    reason: str
    normalized_target: Optional[str] = None
    rate_limited: bool = False


class SafetyGuard:
    def __init__(self, config: Dict[str, Any]):
        self.allowed_intents: List[str] = list(config["allowed_intents"])
        self.poi_aliases: Dict[str, List[str]] = dict(config["poi_aliases"])
        self.min_confidence: float = float(config.get("min_confidence", 0.55))
        self.cooldown_seconds: float = float(config.get("cooldown_seconds", 1.0))
        self.max_text_len: int = int(config.get("max_text_len", 200))

        # rate-limit state
        self._last_allowed_ts = 0.0

        # precompute alias->canonical
        self._alias_to_canonical = {}
        for canonical, aliases in self.poi_aliases.items():
            for a in aliases:
                self._alias_to_canonical[a.strip().lower()] = canonical

    def _rate_limit(self) -> bool:
        now = time.time()
        if now - self._last_allowed_ts < self.cooldown_seconds:
            return True
        self._last_allowed_ts = now
        return False
    
    def _similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def normalize_target(self, target: str) -> Optional[str]:
        if not target:
            return None

        t = target.strip().lower()
        t = "".join(ch for ch in t if ch.isalnum() or ch.isspace()).strip()

        # 1) exact alias
        if t in self._alias_to_canonical:
            return self._alias_to_canonical[t]

        # 2) substring alias
        for alias, canonical in self._alias_to_canonical.items():
            if alias and alias in t:
                return canonical

        # 3) fuzzy match (token pendek seperti "klas" -> "kelas")
        #    cek 1-2 token terakhir (sering target di akhir kalimat)
        toks = t.split()
        candidates = []
        if toks:
            candidates.append(toks[-1])
        if len(toks) >= 2:
            candidates.append(" ".join(toks[-2:]))

        best_score = 0.0
        best_canonical = None
        for cand in candidates:
            for alias, canonical in self._alias_to_canonical.items():
                s = self._similar(cand, alias)
                if s > best_score:
                    best_score = s
                    best_canonical = canonical

        # threshold bisa kamu tuning
        if best_score >= 0.80:
            return best_canonical

        return None

    def validate(self, raw_text: str, llm: Dict[str, Any]) -> GuardDecision:
        raw_text = (raw_text or "").strip()
        if len(raw_text) == 0:
            return GuardDecision(False, "reject", "empty_input")

        if len(raw_text) > self.max_text_len:
            return GuardDecision(False, "reject", "input_too_long")

        intent = str(llm.get("intent", "unknown"))
        conf = float(llm.get("confidence", 0.0))
        target = llm.get("target", None)
        target = None if target is None else str(target)

        # Intent whitelist
        if intent not in self.allowed_intents:
            # unknown juga kita treat sebagai reject/clarify
            if intent == "unknown":
                return GuardDecision(False, "ask_clarification", "intent_unknown")
            return GuardDecision(False, "reject", "intent_not_allowed")

        # confidence gate
        if conf < self.min_confidence:
            return GuardDecision(False, "ask_clarification", f"low_confidence:{conf:.2f}")

        # intent rules
        if intent in ("stop", "return_home", "cancel_goal"):
            # rate limit: stop/return/cancel biasanya boleh, tapi tetap aman batasi spam
            rl = self._rate_limit()
            if rl:
                return GuardDecision(False, "reject", "rate_limited", rate_limited=True)
            return GuardDecision(True, "allow", "intent_ok")

        if intent == "go_to":
            norm = self.normalize_target(target or "")
            if not norm:
                return GuardDecision(
                    False,
                    "ask_clarification",
                    "target_not_in_poi_whitelist",
                    normalized_target=None
                )
            rl = self._rate_limit()
            if rl:
                return GuardDecision(False, "reject", "rate_limited", normalized_target=norm, rate_limited=True)

            return GuardDecision(True, "allow", "intent+target_valid", normalized_target=norm)

        return GuardDecision(False, "reject", "unhandled_intent")