"""Fill missing i18n keys in locale files by translating them with a chat model.

Reads en.json as the source of truth, finds keys missing from each other locale,
batches them, and asks an OpenAI-compatible chat model for translations. Existing
keys are never touched. Usage:

    python scripts/translate_i18n.py            # all locales
    python scripts/translate_i18n.py fr ja      # subset
"""

import io
import json
import os
import sys
import time

import requests

API_URL = "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions"
API_KEY = os.environ.get("ARK_API_KEY", "")
MODEL = "doubao-seed-2.0-mini"

LANG_NAMES = {
    "zh_TW": "Traditional Chinese (Taiwan)",
    "fr": "French",
    "ar": "Arabic",
    "ru": "Russian",
    "ja": "Japanese",
    "de": "German",
    "uk": "Ukrainian",
    "ko": "Korean",
    "vi": "Vietnamese",
}

BATCH = 40


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def unflatten(flat_dict):
    root = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return root


def translate_batch(target_lang, items):
    """items: list of (key, english_text). Returns dict key -> translation."""
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    payload_lines = "\n".join(f"{k} = {v}" for k, v in items)
    prompt = (
        f"Translate each UI string below from English into {lang_name}.\n"
        "Rules:\n"
        "- Keep placeholders like {{name}} or {{count}} exactly as they are.\n"
        "- Keep the tone concise, natural for software UI.\n"
        "- Output ONLY a JSON object mapping each key to the translated string.\n"
        "- Do not add comments or extra text.\n\n"
        f"{payload_lines}"
    )
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0.3,
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def main():
    if not API_KEY:
        sys.exit("Set the ARK_API_KEY environment variable to use this script.")
    targets = sys.argv[1:] or list(LANG_NAMES)
    here = os.path.dirname(os.path.abspath(__file__))
    locale_dir = os.path.join(here, "..", "lightrag_webui", "src", "locales")
    en = json.load(io.open(os.path.join(locale_dir, "en.json"), encoding="utf-8"))
    en_flat = flat(en)

    for lang in targets:
        path = os.path.join(locale_dir, f"{lang}.json")
        data = json.load(io.open(path, encoding="utf-8"))
        have = flat(data)
        missing = [(k, en_flat[k]) for k in en_flat if k not in have]
        if not missing:
            print(f"{lang}: up to date")
            continue

        print(f"{lang}: {len(missing)} keys to translate")
        translated = {}
        for i in range(0, len(missing), BATCH):
            chunk = missing[i : i + BATCH]
            for attempt in range(3):
                try:
                    translated.update(translate_batch(lang, chunk))
                    break
                except Exception as e:
                    print(f"  batch {i//BATCH} attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            else:
                print(f"  batch {i//BATCH} skipped, keeping English fallback")
            print(f"  batch {i//BATCH + 1}/{(len(missing) + BATCH - 1)//BATCH} done")

        # merge: existing keys stay, missing keys get translation or English
        merged = dict(en_flat)
        merged.update(have)
        for k, _ in missing:
            value = translated.get(k)
            if not value or not isinstance(value, str):
                value = en_flat[k]
            merged[k] = value

        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(unflatten(merged), f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"{lang}: written")


if __name__ == "__main__":
    main()
