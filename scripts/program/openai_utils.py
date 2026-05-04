from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os

from pathlib import Path

from openai import AsyncOpenAI, OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_PATH = REPO_ROOT / ".tmp_openai_key"
DEFAULT_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_VLM_MODEL = "gpt-5.5"


def load_api_key(key_path: Path | None = None) -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    path = key_path or DEFAULT_KEY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing OpenAI API key file: {path}")
    key = path.read_text().strip()
    if not key:
        raise ValueError(f"OpenAI API key file is empty: {path}")
    return key


def make_client(key_path: Path | None = None) -> OpenAI:
    return OpenAI(api_key=load_api_key(key_path))


def make_async_client(key_path: Path | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=load_api_key(key_path))


def response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def llm_call(
    prompt: str,
    *,
    model: str = DEFAULT_LLM_MODEL,
    instructions: str | None = None,
    max_output_tokens: int = 256,
    key_path: Path | None = None,
) -> str:
    client = make_client(key_path)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return response_text(response)


def llm_json_call(
    prompt: str,
    *,
    schema: dict[str, object],
    schema_name: str,
    model: str = DEFAULT_LLM_MODEL,
    instructions: str | None = None,
    max_output_tokens: int = 4096,
    key_path: Path | None = None,
) -> dict[str, object]:
    client = make_client(key_path)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    )
    text = response_text(response).strip()
    if not text:
        raise ValueError("LLM returned empty JSON output")
    return json.loads(text)


def image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def vlm_call(
    prompt: str,
    image_path: Path,
    *,
    model: str = DEFAULT_VLM_MODEL,
    instructions: str | None = None,
    max_output_tokens: int = 256,
    key_path: Path | None = None,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    client = make_client(key_path)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url(image_path)},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )
    return response_text(response)


async def async_vlm_json_call(
    prompt: str,
    image_path: Path,
    *,
    schema: dict[str, object],
    schema_name: str,
    model: str = DEFAULT_VLM_MODEL,
    instructions: str | None = None,
    max_output_tokens: int = 256,
    key_path: Path | None = None,
    client: AsyncOpenAI | None = None,
    max_attempts: int = 2,
) -> dict[str, object]:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    openai_client = client or make_async_client(key_path)
    try:
        last_error = ""
        for _ in range(max(1, int(max_attempts))):
            response = await openai_client.responses.create(
                model=model,
                instructions=instructions,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url(image_path)},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            text = response_text(response).strip()
            if not text:
                last_error = "empty output"
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON output: {exc}"
                continue
    finally:
        if client is None:
            await openai_client.close()
    raise ValueError(f"VLM returned unusable JSON for {image_path}: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small OpenAI Responses API helper for LLM/VLM calls.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    text_parser = subparsers.add_parser("llm", help="Run a text-only LLM call.")
    text_parser.add_argument("--prompt", required=True)
    text_parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    text_parser.add_argument("--instructions", default=None)
    text_parser.add_argument("--max-output-tokens", type=int, default=256)
    text_parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)

    vision_parser = subparsers.add_parser("vlm", help="Run an image+text VLM call.")
    vision_parser.add_argument("--prompt", required=True)
    vision_parser.add_argument("--image", required=True, type=Path)
    vision_parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    vision_parser.add_argument("--instructions", default=None)
    vision_parser.add_argument("--max-output-tokens", type=int, default=256)
    vision_parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "llm":
        print(
            llm_call(
                args.prompt,
                model=args.model,
                instructions=args.instructions,
                max_output_tokens=args.max_output_tokens,
                key_path=args.key_path,
            )
        )
    elif args.mode == "vlm":
        print(
            vlm_call(
                args.prompt,
                args.image,
                model=args.model,
                instructions=args.instructions,
                max_output_tokens=args.max_output_tokens,
                key_path=args.key_path,
            )
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
