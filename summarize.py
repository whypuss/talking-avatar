#!/usr/bin/env python3
"""Optional helper: convert a news article into a concise spoken broadcast script."""

import argparse
import os
import sys
import requests


def summarize_article(text: str, language: str = "Cantonese", max_chars: int = 120) -> str:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY (or OPENAI_API_KEY) environment variable.")

    prompt = (
        f"請將以下文章改寫成適合新聞主播口播的簡短講稿（語言：{language}，約 {max_chars} 字以內），"
        "只需輸出純口播文字，不要包含任何標題、括號動作提示或標記：\n\n"
        f"{text}"
    )

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an article into a spoken broadcast script.")
    parser.add_argument("--input", required=True, help="Path to input article text file.")
    parser.add_argument("--output", required=True, help="Path to output script text file.")
    parser.add_argument("--language", default="Cantonese", help="Target spoken language (default: Cantonese).")
    parser.add_argument("--max-chars", type=int, default=120, help="Approximate max character count.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        article = f.read().strip()

    script = summarize_article(article, language=args.language, max_chars=args.max_chars)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(script + "\n")
    print(f"Saved broadcast script to {args.output}:\n{script}")


if __name__ == "__main__":
    main()
