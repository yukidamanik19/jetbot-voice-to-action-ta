import argparse
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
import time
import threading

from pynput import keyboard
from RealtimeSTT import AudioToTextRecorder

from llm_client import MockLLMClient, OpenAIIntentClient, OpenAICompatibleJSONClient
from safety_guard import SafetyGuard

import requests

JETBOT_URL = "http://10.10.164.170:5000/command"


def send_command_to_jetbot(command: dict, url: str) -> dict:
    try:
        r = requests.post(url, json=command, timeout=2.0)
        return {
            "ok": r.ok,
            "status_code": r.status_code,
            "text": r.text[:300]
        }
    except Exception as e:
        return {
            "ok": False,
            "status_code": None,
            "text": str(e)
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Wake word + smalltalk
# =========================

WAKE_ALIASES = [
    "jetbot",
    "jet bot",
    "jetboard",
    "jet board",
    "jinbot",
]

SMALLTALK = {
    "halo", "hai", "hello", "hey",
    "tes", "test", "testing", "uji coba",
    "cek", "cek mic", "mic test",
    "permisi",
    "speak now",
    "terima kasih", "makasih",
}


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_smalltalk(text: str) -> bool:
    return _norm(text) in SMALLTALK


def detect_wake(text: str, threshold: float = 0.78, window_tokens: int = 4):
    """
    Wake word boleh muncul di awal atau setelah sapaan (dalam window_tokens pertama).
    Return: (is_wake, stripped_command, matched_alias, score)
    """
    raw = (text or "").strip()
    t = _norm(raw)
    if not t:
        return False, "", None, 0.0

    toks = t.split()
    raw_toks = re.sub(r"\s+", " ", raw).strip().split()

    best = (0.0, None, None)  # score, alias, start_idx
    max_i = min(len(toks), window_tokens)

    for start in range(max_i):
        head1 = toks[start]
        head2 = " ".join(toks[start:start + 2]) if start + 1 < len(toks) else head1

        for alias in WAKE_ALIASES:
            a = _norm(alias)
            if not a:
                continue

            if head1 == a or head2 == a:
                score = 1.0
            else:
                score = max(_similar(head1, a), _similar(head2, a))

            if score > best[0]:
                best = (score, alias, start)

    score, alias, start_idx = best
    if alias and score >= threshold:
        n_alias = len(_norm(alias).split())
        after = raw_toks[start_idx + n_alias:]
        stripped = " ".join(after).strip()

        if stripped == "":
            return True, "", alias, score
        return True, stripped, alias, score

    return False, raw, None, score


def parse_args():
    p = argparse.ArgumentParser(description="Live Voice-to-Action: STT -> LLM -> SafetyGuard -> JSONL")

    # STT params
    p.add_argument("--model", default="small", help="Whisper model size/path (tiny/base/small/...)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for STT model")
    p.add_argument("--language", default="id", help="Language code (empty for auto-detect)")
    p.add_argument("--out", default=os.path.join("outputs", "v2a_log.jsonl"), help="Output JSONL path")

    p.add_argument("--post-speech-silence", type=float, default=1.2)
    p.add_argument("--min-length", type=float, default=1.0)
    p.add_argument("--min-gap", type=float, default=0.4)

    # realtime (default OFF for stability)
    p.add_argument("--enable-realtime", action="store_true", default=False)
    p.add_argument("--realtime-model", default="tiny")
    p.add_argument("--use-main-model-for-realtime", action="store_true")
    p.add_argument("--demo", action="store_true", default=False, help="Mode demo: output terminal dirapikan dan spinner dimatikan otomatis.")
    p.add_argument("--no-spinner", action="store_true")

    # LLM mode
    p.add_argument("--llm", choices=["mock", "openai", "openai_compat"], default="mock")
    p.add_argument("--openai-model", default="gpt-5.4-nano")

    # for openai_compat (local servers)
    p.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "local-model"))

    # Guard config
    p.add_argument("--poi-config", default="config_poi.json")

    # Wake policy
    p.add_argument("--require-wake", action="store_true", default=False,
                   help="Jika diaktifkan, perintah wajib ada wake word (fuzzy).")
    p.add_argument("--wake-threshold", type=float, default=0.78,
                   help="Ambang fuzzy wake detection (0..1).")

    # JetBot command sender
    p.add_argument("--jetbot-url", default="http://10.10.164.170:5000/command",
                   help="HTTP endpoint JetBot untuk menerima command")
    p.add_argument("--send-to-jetbot", action="store_true", default=False,
                   help="Jika aktif, command final akan dikirim ke JetBot")

    return p.parse_args()


def apply_return_target_rule(text: str, llm_payload: dict, guard_cfg: dict) -> dict:
    """
    Aturan opsi 2:
    - 'kembali' / 'kembali ke home' => return_home
    - 'kembali ke <poi selain home>' => go_to(<poi>)
    """
    t = _norm(text)

    poi_aliases = guard_cfg.get("poi_aliases", {}) or {}

    alias_to_key = {}
    for canonical, aliases in poi_aliases.items():
        alias_to_key[_norm(canonical)] = canonical
        for a in aliases:
            alias_to_key[_norm(a)] = canonical

    m = re.search(r"\bkembali\s+ke\s+(.+)$", t)
    if not m:
        if re.search(r"\bkembali\b", t):
            llm_payload["intent"] = "return_home"
            llm_payload["target"] = None
            llm_payload["notes"] = f'{llm_payload.get("notes", "")}|rule:return_only=>return_home'.strip("|")
        return llm_payload

    raw_target = m.group(1).strip()
    canonical_target = alias_to_key.get(raw_target)

    if canonical_target is None:
        best_key = None
        best_score = 0.0
        for alias_norm, key in alias_to_key.items():
            score = _similar(raw_target, alias_norm)
            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= 0.80:
            canonical_target = best_key

    if canonical_target:
        if canonical_target == "home":
            llm_payload["intent"] = "return_home"
            llm_payload["target"] = None
            llm_payload["notes"] = f'{llm_payload.get("notes", "")}|rule:return_to_home=>return_home'.strip("|")
        else:
            llm_payload["intent"] = "go_to"
            llm_payload["target"] = canonical_target
            llm_payload["notes"] = f'{llm_payload.get("notes", "")}|rule:return_to_target=>go_to'.strip("|")
    else:
        llm_payload["intent"] = "go_to"
        llm_payload["target"] = raw_target
        llm_payload["notes"] = f'{llm_payload.get("notes", "")}|rule:return_to_unknown=>go_to'.strip("|")

    return llm_payload


def pretty_block(title: str, value: str) -> None:
    print(f"\n[{title}] {value}")


def pretty_command(command: dict) -> None:
    if command.get("intent") == "go_to":
        print(f"[CMD] go_to -> {command.get('target')}")
    else:
        print(f"[CMD] {command.get('intent')}")


def main():
    args = parse_args()
    if args.demo:
        args.no_spinner = True
    ensure_parent(args.out)

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load guard config
    guard_cfg = load_json(args.poi_config)
    guard = SafetyGuard(guard_cfg)

    # LLM client
    if args.llm == "mock":
        llm_client = MockLLMClient()
    elif args.llm == "openai":
        llm_client = OpenAIIntentClient(model=args.openai_model)
    else:
        llm_client = OpenAICompatibleJSONClient(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
        )

    def write_event(payload: dict):
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    write_event({
        "type": "session_start",
        "ts": now_iso(),
        "session_id": session_id,
        "stt_model": args.model,
        "stt_device": args.device,
        "language": args.language,
        "llm_mode": args.llm,
        "openai_model": args.openai_model if args.llm == "openai" else None,
        "wake": {
            "require_wake": bool(args.require_wake),
            "threshold": float(args.wake_threshold),
            "aliases": WAKE_ALIASES,
        },
        "guard": {
            "allowed_intents": guard_cfg.get("allowed_intents", []),
            "poi_keys": list((guard_cfg.get("poi_aliases", {}) or {}).keys())
        }
    })

    def on_final_text(text: str):
        text = (text or "").strip()
        if not text:
            return

        pretty_block("STT", text)

        if is_smalltalk(text):
            write_event({
                "type": "ignored_utterance",
                "ts": now_iso(),
                "session_id": session_id,
                "raw_text": text,
                "reason": "smalltalk_ignored"
            })
            print("[IGNORED] smalltalk")
            return

        is_wake, stripped, matched_alias, score = detect_wake(text, threshold=float(args.wake_threshold))

        if is_wake and stripped == "":
            write_event({
                "type": "ignored_utterance",
                "ts": now_iso(),
                "session_id": session_id,
                "raw_text": text,
                "reason": "wake_only_ignored",
                "wake": {"matched": matched_alias, "score": float(score)}
            })
            print(f"[WAKE] detected ({matched_alias}, score={score:.2f}) — no command yet")
            return

        if args.require_wake and not is_wake:
            write_event({
                "type": "ignored_utterance",
                "ts": now_iso(),
                "session_id": session_id,
                "raw_text": text,
                "reason": "no_wake_word"
            })
            print("[IGNORED] no wake word")
            return

        processed_text = stripped if (is_wake and stripped) else text

        try:
            res = llm_client.infer(processed_text)
            normalized_text = (res.normalized_transcript or processed_text).strip()

            llm_payload = {
                "normalized_transcript": normalized_text,
                "intent": res.intent,
                "target": res.target,
                "confidence": float(res.confidence),
                "notes": res.notes
            }
        except Exception as e:
            normalized_text = processed_text
            llm_payload = {
                "normalized_transcript": normalized_text,
                "intent": "unknown",
                "target": None,
                "confidence": 0.0,
                "notes": f"llm_error:{type(e).__name__}:{str(e)}"
            }

        pretty_block("RAW", processed_text)
        pretty_block("NORMALIZED", normalized_text)

        llm_payload = apply_return_target_rule(normalized_text, llm_payload, guard_cfg)
        gd = guard.validate(normalized_text, llm_payload)

        command = None
        assistant_reply = None

        if gd.allowed:
            if llm_payload["intent"] == "go_to":
                command = {"intent": "go_to", "target": gd.normalized_target}
            else:
                command = {"intent": llm_payload["intent"]}
        else:
            if gd.decision == "ask_clarification":
                if gd.reason == "target_not_in_poi_whitelist":
                    poi_list = ", ".join(list(guard_cfg.get("poi_aliases", {}).keys()))
                    assistant_reply = f"Aku belum punya lokasi itu. Tujuan yang tersedia: {poi_list}. Mau ke yang mana?"
                elif gd.reason.startswith("low_confidence"):
                    assistant_reply = "Aku kurang yakin. Bisa ulangi perintahnya dengan lebih jelas?"
                else:
                    assistant_reply = "Aku butuh klarifikasi. Maksudmu ingin robot melakukan apa?"
            else:
                assistant_reply = "Perintah ditolak oleh safety guard."

        event = {
            "type": "robot_command",
            "ts": now_iso(),
            "session_id": session_id,
            "raw_text": text,
            "processed_text": processed_text,
            "normalized_text": normalized_text,
            "wake": {
                "used": bool(is_wake),
                "matched": matched_alias,
                "score": float(score),
                "require_wake": bool(args.require_wake)
            },
            "llm": llm_payload,
            "guard": {
                "allowed": bool(gd.allowed),
                "decision": gd.decision,
                "reason": gd.reason,
                "normalized_target": gd.normalized_target,
                "rate_limited": bool(gd.rate_limited)
            },
            "command": command,
            "assistant_reply": assistant_reply
        }
        write_event(event)

        if assistant_reply:
            pretty_block("ASSIST", assistant_reply)
        if command:
            pretty_command(command)

            if args.send_to_jetbot:
                send_result = send_command_to_jetbot(command, args.jetbot_url)
                print("\n[SEND TO JETBOT]")
                print(json.dumps(send_result, ensure_ascii=False, indent=2))

    recorder = AudioToTextRecorder(
        model=args.model,
        device=args.device,
        language=args.language if args.language is not None else "",
        spinner=(not args.no_spinner),
        post_speech_silence_duration=args.post_speech_silence,
        min_length_of_recording=args.min_length,
        min_gap_between_recordings=args.min_gap,
        enable_realtime_transcription=bool(args.enable_realtime),
        use_main_model_for_realtime=bool(args.use_main_model_for_realtime),
        realtime_model_type=args.realtime_model,
    )

    print(f"Session: {session_id}")
    print("Hold 'r' to talk, release 'r' to process. Ctrl+C to stop.")

    is_recording = {"value": False}
    r_is_down = {"value": False}
    lock = threading.Lock()

    def start_recording():
        with lock:
            if is_recording["value"]:
                return
            is_recording["value"] = True
            print("\n[PTT] recording...")
            try:
                recorder.start()
            except Exception as e:
                print(f"[PTT] start error: {e}")
                is_recording["value"] = False

    def stop_recording():
        with lock:
            if not is_recording["value"]:
                return
            is_recording["value"] = False
            print("\n[PTT] processing...")
            try:
                recorder.stop()
                text = recorder.text()
                if text:
                    on_final_text(text)
                else:
                    print("[PTT] no text captured")
            except Exception as e:
                print(f"[PTT] stop/process error: {e}")

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char == "r":
                if not r_is_down["value"]:
                    r_is_down["value"] = True
                    start_recording()
        except Exception:
            pass

    def on_release(key):
        try:
            if hasattr(key, "char") and key.char == "r":
                if r_is_down["value"]:
                    r_is_down["value"] = False
                    stop_recording()
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        try:
            recorder.shutdown()
        except Exception:
            pass
        write_event({"type": "session_end", "ts": now_iso(), "session_id": session_id})


if __name__ == "__main__":
    main()