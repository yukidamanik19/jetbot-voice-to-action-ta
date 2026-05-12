import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    intent: str
    target: Optional[str]
    confidence: float
    normalized_transcript: str = ""
    notes: str = ""


class BaseLLMClient:
    def infer(self, text: str) -> LLMResult:
        raise NotImplementedError


class OpenAIIntentClient(BaseLLMClient):
    """
    OpenAI API (official) + Structured Outputs (json_schema strict).
    Requires env var: OPENAI_API_KEY
    Default model dibuat hemat.
    """
    def __init__(self, model: str = "gpt-5.4-nano", api_key: Optional[str] = None):
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY belum di-set. Set env var OPENAI_API_KEY terlebih dahulu.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

        self.schema = {
            "name": "robot_intent",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "normalized_transcript": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "enum": ["go_to", "stop", "return_home", "cancel_goal", "unknown"]
                    },
                    "target": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "notes": {"type": "string"},
                },
                "required": [
                    "normalized_transcript",
                    "intent",
                    "target",
                    "confidence",
                    "notes",
                ],
            },
            "strict": True,
        }

        self.system = (
            "Kamu adalah parser intent untuk robot JetBot dalam Bahasa Indonesia. "
            "Tugasmu memperbaiki transkrip command pendek agar lebih rapi TANPA mengubah makna, "
            "lalu ekstrak intent dan target. "
            "Keluaran HARUS JSON sesuai schema. "
            "Target/POI valid: kelas, lab, pintu, home. "
            "Intent valid: go_to, stop, return_home, cancel_goal, unknown. "
            "Aturan: "
            "'pergi ke kelas/lab/pintu/home' => go_to dengan target yang sesuai. "
            "'kembali' atau 'pulang' => return_home. "
            "'kembali ke home' => return_home. "
            "'kembali ke kelas/lab/pintu' => go_to target tersebut. "
            "'berhenti' atau 'hentikan' => stop. "
            "'batal' atau 'batalkan' => cancel_goal. "
            "Jika tidak yakin, pakai unknown atau target=null dan turunkan confidence. "
            "normalized_transcript harus berupa transkrip yang lebih rapi dan natural. "
            "Jangan keluarkan teks di luar JSON."
        )

    def infer(self, text: str) -> LLMResult:
        # retry sederhana agar lebih tahan kalau kena rate limit sesaat
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system},
                        {"role": "user", "content": "pergi ke kelas"},
                        {"role": "assistant", "content": json.dumps({
                            "normalized_transcript": "pergi ke kelas",
                            "intent": "go_to",
                            "target": "kelas",
                            "confidence": 0.98,
                            "notes": ""
                        }, ensure_ascii=False)},
                        {"role": "user", "content": "kembali ke lab"},
                        {"role": "assistant", "content": json.dumps({
                            "normalized_transcript": "kembali ke lab",
                            "intent": "go_to",
                            "target": "lab",
                            "confidence": 0.95,
                            "notes": ""
                        }, ensure_ascii=False)},
                        {"role": "user", "content": "pulang"},
                        {"role": "assistant", "content": json.dumps({
                            "normalized_transcript": "pulang",
                            "intent": "return_home",
                            "target": None,
                            "confidence": 0.99,
                            "notes": ""
                        }, ensure_ascii=False)},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                    response_format={"type": "json_schema", "json_schema": self.schema},
                )
                content = resp.choices[0].message.content
                obj = json.loads(content)

                normalized_transcript = str(obj.get("normalized_transcript", text)).strip()
                intent = str(obj.get("intent", "unknown"))
                target = obj.get("target", None)
                if target is not None:
                    target = str(target)
                confidence = float(obj.get("confidence", 0.0))
                notes = str(obj.get("notes", ""))

                confidence = max(0.0, min(1.0, confidence))

                return LLMResult(
                    intent=intent,
                    target=target,
                    confidence=confidence,
                    normalized_transcript=normalized_transcript,
                    notes=notes,
                )
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    raise last_err


class MockLLMClient(BaseLLMClient):
    """
    Offline testing. Tetap isi normalized_transcript agar interface konsisten.
    """
    def infer(self, text: str) -> LLMResult:
        t = (text or "").lower().strip()

        t_norm = re.sub(r"[^a-z0-9\s]", " ", t)
        t_norm = re.sub(r"\s+", " ", t_norm).strip()
        tokens = t_norm.split()

        def similar(a: str, b: str) -> float:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()

        stop_keywords = ["berhenti", "hentikan", "stop", "halt", "diam", "diem"]
        if any(k in t_norm for k in stop_keywords):
            return LLMResult(intent="stop", target=None, confidence=0.90, normalized_transcript=t_norm)

        for tok in tokens:
            if similar(tok, "hentikan") >= 0.78 or similar(tok, "berhenti") >= 0.78:
                return LLMResult(intent="stop", target=None, confidence=0.80,
                                 normalized_transcript=t_norm, notes="fuzzy_stop")
            if similar(tok, "diem") >= 0.78 or similar(tok, "diam") >= 0.78:
                return LLMResult(intent="stop", target=None, confidence=0.75,
                                 normalized_transcript=t_norm, notes="fuzzy_diam")

        if re.search(r"\b(pulang|kembali|return home|home)\b", t_norm):
            return LLMResult(intent="return_home", target=None, confidence=0.85, normalized_transcript=t_norm)

        if re.search(r"\b(batal|cancel|batalkan)\b", t_norm):
            return LLMResult(intent="cancel_goal", target=None, confidence=0.85, normalized_transcript=t_norm)

        m = re.search(r"\b(pergi|menuju|go to)\b\s*(ke)?\s+(.+)$", t_norm)
        if m:
            target = m.group(3).strip()
            return LLMResult(intent="go_to", target=target, confidence=0.75, normalized_transcript=t_norm)

        m_any = re.search(r"\bke\s+([a-z0-9\s]+)$", t_norm)
        if m_any:
            target = m_any.group(1).strip()
            return LLMResult(intent="go_to", target=target, confidence=0.70,
                             normalized_transcript=t_norm, notes="anywhere_ke")

        m2 = re.search(r"\b(pergi|menuju)\s*(ka|ke)\s*([a-z]+)\b", t_norm)
        if m2:
            target = m2.group(3).strip()
            return LLMResult(intent="go_to", target=target, confidence=0.65,
                             normalized_transcript=t_norm, notes="joined_ke")

        return LLMResult(
            intent="unknown",
            target=None,
            confidence=0.2,
            normalized_transcript=t_norm,
            notes="unmatched",
        )


class OpenAICompatibleJSONClient(BaseLLMClient):
    """
    Generic OpenAI-compatible server (local vLLM/LM Studio/Ollama gateway etc).
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def infer(self, text: str) -> LLMResult:
        import requests

        system = (
            "Kamu adalah parser intent untuk robot. "
            "Keluarkan JSON SAJA sesuai schema: "
            "{normalized_transcript: string, intent: string, target: string|null, confidence: number 0..1, notes: string}. "
            "Intent hanya salah satu dari: go_to, stop, return_home, cancel_goal, unknown. "
            "POI valid: kelas, lab, pintu, home. "
            "normalized_transcript harus berupa versi yang lebih rapi dari transkrip mentah."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

        obj = _extract_json(content)

        normalized_transcript = str(obj.get("normalized_transcript", text)).strip()
        intent = str(obj.get("intent", "unknown"))
        target = obj.get("target", None)
        if target is not None:
            target = str(target)
        conf = float(obj.get("confidence", 0.0))
        notes = str(obj.get("notes", ""))

        conf = max(0.0, min(1.0, conf))

        return LLMResult(
            intent=intent,
            target=target,
            confidence=conf,
            normalized_transcript=normalized_transcript,
            notes=notes,
        )


def _extract_json(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])

    raise ValueError("LLM tidak mengembalikan JSON yang valid.")