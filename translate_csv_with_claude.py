#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
import typing as t
from urllib import request, error


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
# Use a valid, current Anthropic model by default.
# Can be overridden via --model CLI flag.
DEFAULT_MODEL = "claude-opus-4-8"


def slugify(value: str) -> str:
    s = value.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def anthropic_translate(
    text: str,
    target_language: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
    request_timeout: int = 60,
) -> str:
    """Translate text using Anthropic Messages API.

    Returns translated string. Raises RuntimeError on repeated failures.
    """
    if not text:
        return ""

    system_prompt = (
        "You are a precise translation engine assisting a safety research task. "
        "Translate the user-provided text into the requested language exactly, "
        "preserving meaning, tone, and unsafe content, without adding commentary, "
        "explanations, or safety warnings. Output only the translation."
    )

    user_prompt = (
        f"Target language: {target_language}.\n\n"
        "Text to translate between the triple backticks.\n\n"
        f"```\n{text}\n```\n\n"
        "Return only the translation, no quotes or extra text."
    )

    payload = {
        "model": model,
        "max_tokens": 2*1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        # Use the stable Messages API version
        "anthropic-version": "2023-06-01",
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        req = request.Request(ANTHROPIC_API_URL, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=request_timeout) as resp:
                body = resp.read()
                obj = json.loads(body.decode("utf-8"))
                # Expect obj["content"] to be a list with text segments
                content = obj.get("content", [])
                if isinstance(content, list) and content:
                    # Find first text item
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return (part.get("text") or "").strip()
                # Fallback: try "text" top-level (older SDKs)
                if isinstance(obj.get("text"), str):
                    return obj["text"].strip()
                raise RuntimeError("Unexpected response format from Anthropic API")
        except error.HTTPError as e:
            status = e.code
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "<no body>"

            # Backoff on rate limits or transient errors
            if status in (408, 409, 429, 500, 502, 503, 504):
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
                continue
            if status == 404:
                raise RuntimeError(
                    "Anthropic API returned 404 Not Found. "
                    "This often indicates an invalid model name or API path. "
                    f"Requested model: '{model}'. Endpoint: {ANTHROPIC_API_URL}. "
                    f"Response body: {err_body}"
                ) from None
            raise RuntimeError(f"Anthropic API error {status}: {err_body}") from None
        except error.URLError as e:
            # Network failure: retry with backoff
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)
            if attempt == max_retries:
                raise RuntimeError(f"Network error contacting Anthropic API: {e}") from None

    raise RuntimeError("Exceeded maximum retries calling Anthropic API")


def process_csv(
    input_path: str,
    output_path: str,
    column: str,
    language: str,
    api_key: str,
    model: str,
    skip_existing: bool = True,
) -> None:
    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])

        if column not in fieldnames:
            raise SystemExit(
                f"Column '{column}' not found in CSV. Available columns: {', '.join(fieldnames)}"
            )

        new_col = f"{column}_translated"
        if new_col not in fieldnames:
            fieldnames.append(new_col)

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(reader, start=1):
                src_text = row.get(column, "") or ""
                already = row.get(new_col)
                if skip_existing and already:
                    writer.writerow(row)
                    continue

                try:
                    translated = anthropic_translate(
                        text=src_text,
                        target_language=language,
                        api_key=api_key,
                        model=model,
                    )
                except Exception as e:
                    translated = ""
                    print(f"[warn] Row {idx}: translation failed: {e}", file=sys.stderr)

                row[new_col] = translated
                writer.writerow(row)


def parse_args(argv: t.List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Add a translation column to a CSV by calling Anthropic Claude via Messages API. "
            "Requires ANTHROPIC_API_KEY in environment."
        )
    )
    p.add_argument("--input", "-i", default="harmful_behaviors_jailbreakbench.csv", help="Input CSV path")
    p.add_argument("--output", "-o", default=None, help="Output CSV path (default: auto-named)")
    p.add_argument("--language", "-l", required=True, help="Target language, e.g. 'Polish' or 'es-ES'")
    p.add_argument(
        "--column", "-c", default="Goal", help="CSV column to translate (default: Goal)"
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model name (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-translate rows even if translation column already has values",
    )
    return p.parse_args(argv)


def main(argv: t.List[str]) -> int:
    args = parse_args(argv)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        return 2

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 2

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(
            os.path.dirname(input_path),
            f"{base}_with_{slugify(args.column)}_{slugify(args.language)}{ext or '.csv'}",
        )

    try:
        process_csv(
            input_path=input_path,
            output_path=output_path,
            column=args.column,
            language=args.language,
            api_key=api_key,
            model=args.model,
            skip_existing=not args.no_skip_existing,
        )
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to translate CSV: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
