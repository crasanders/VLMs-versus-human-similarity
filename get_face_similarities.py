"""Obtain pairwise visual similarity ratings for the 32 unique schematic faces
from Nosofsky (1991) using GPT-4o with logprob-weighted average ratings (1-10 scale),
following get_similarities.py and its ENCOURAGE_MIDDLE_PROMPT.

Usage:
    python get_face_similarities.py [--model gpt-4o] [--output-folder output/gpt4o] [--test]
"""

import argparse
import asyncio
import base64
from datetime import datetime
import io
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import openai
from openai import AsyncOpenAI
import pandas as pd
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

IMAGE_FOLDER = Path("data") / "34 Faces"
DUPLICATE_FACES = {"face_26", "face_30"}  # Identical to face_11 and face_18 in Nosofsky (1991)
NUMBER_MAP_FILE = Path("number_map.json")
DEFAULT_OUTPUT_FOLDER = Path("output") / "gpt4o"
DEFAULT_CONCURRENCY = 12
DEFAULT_TEST_N = 4

BASE_PROMPT = (
    "You are assisting in a study in which you are shown pairs of schematic faces and "
    "rate how visually similar they are on a scale from 1 to 10, with 1 being most dissimilar, "
    "5 or 6 being moderately similar, and 10 being most similar. "
    "You only respond with a single number from 1 to 10, without explaining your reasoning. "
)

ENCOURAGE_MIDDLE_PROMPT = BASE_PROMPT + (
    "You use the full range of the 1-10 scale and err on the side of using the middle of the scale (5 or 6) when you are unsure. "
    "Faces that are very similar in almost all respects should be rated as highly similar (9 or 10). "
    "You only use a 1 or 2 when the faces are truly different in every meaningful visual way. "
    "Most ratings should fall somewhere in the middle of the scale. "
)

QUESTION_PROMPT = "From 1-10, how visually similar are these two faces?"

_encoded_images: dict[str, str] = {}


def load_api_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = Path("openai_key.txt")
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    raise RuntimeError("No OpenAI API key found in OPENAI_API_KEY or openai_key.txt")


def load_number_map() -> dict[str, int]:
    number_map: dict[str, int] = {}
    if NUMBER_MAP_FILE.exists():
        with open(NUMBER_MAP_FILE, "r", encoding="utf-8") as f:
            number_map.update(json.load(f))
    # Add 10 mappings for the 1-10 scale
    number_map.update(
        {
            "10": 10,
            "ten": 10,
            "diez": 10,
            "dix": 10,
            "zehn": 10,
            "dieci": 10,
            "dez": 10,
            "tien": 10,
            "十": 10,
            "じゅう": 10,
            "десять": 10,
            "열": 10,
        }
    )
    return number_map


def encode_image(face: str) -> str:
    if face not in _encoded_images:
        image_path = IMAGE_FOLDER / f"{face}.png"
        with Image.open(image_path) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=95)
            _encoded_images[face] = base64.b64encode(buf.getvalue()).decode("utf-8")
    return _encoded_images[face]


def create_messages(face1: str, face2: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(face1)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(face2)}"},
                },
            ],
        },
    ]


def get_average_rating(logprobs, number_map: dict[str, int]) -> tuple[float, np.ndarray]:
    """Compute logprob-weighted average rating on the 1-10 scale."""
    ratings = np.arange(1, 11, dtype=float)
    weights = np.zeros(10, dtype=float)
    for lp in logprobs:
        token = lp.token.strip().lower()
        mapped = number_map.get(token, token)
        try:
            idx = int(mapped) - 1
            if 0 <= idx < 10:
                weights[idx] += float(np.exp(lp.logprob))
        except Exception:
            continue
    total_w = float(weights.sum())
    if total_w > 0:
        probs = weights / total_w
        av = float(np.average(ratings, weights=weights))
    else:
        probs = np.zeros(10, dtype=float)
        av = np.nan
    return av, probs


@retry(
    wait=wait_random_exponential(min=1, max=45),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)),
)
async def completion_with_backoff(client: AsyncOpenAI, **kwargs):
    return await client.chat.completions.create(**kwargs)


async def rate_pair(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pair_idx: int,
    total_pairs: int,
    pair: dict,
    system_prompt: str,
    number_map: dict[str, int],
    model: str,
    seed: int,
) -> tuple[dict, dict]:
    face1, face2 = pair["Face1"], pair["Face2"]
    messages = create_messages(face1, face2, system_prompt)
    async with semaphore:
        response = await completion_with_backoff(
            client,
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=20,
            seed=seed,
            max_completion_tokens=1,
        )

    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    avg_rating, probs = get_average_rating(logprobs, number_map)
    top_token = response.choices[0].message.content.strip()

    row = {
        "index": pair_idx,
        "Face1": face1,
        "Face2": face2,
        "Face1 Index": pair["Face1 Index"],
        "Face2 Index": pair["Face2 Index"],
        "Model Rating": avg_rating,
        "ChatGPT Rating": avg_rating,
        "Top Token": top_token,
    }
    for level in range(10):
        row[f"P({level + 1})"] = float(probs[level])
    row["Model"] = response.model

    raw_record = {
        "index": pair_idx,
        "Face1": face1,
        "Face2": face2,
        "response": response.model_dump() if hasattr(response, "model_dump") else str(response),
    }

    if (pair_idx + 1) % 25 == 0 or (pair_idx + 1) == total_pairs:
        print(f"Processed {pair_idx + 1}/{total_pairs} pairs (latest: {face1} vs {face2} -> {avg_rating:.3f})")

    return row, raw_record


async def run_similarities(args) -> None:
    all_faces = sorted(path.stem for path in IMAGE_FOLDER.glob("face_*.png"))
    unique_faces = [face for face in all_faces if face not in DUPLICATE_FACES]
    faces = unique_faces[: args.test_n] if args.test else unique_faces

    pairs = [
        {
            "Face1": face1,
            "Face2": face2,
            "Face1 Index": unique_faces.index(face1),
            "Face2 Index": unique_faces.index(face2),
        }
        for face1, face2 in combinations(faces, 2)
    ]

    output_folder = args.output_folder / "test" if args.test else args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    system_prompt = ENCOURAGE_MIDDLE_PROMPT if args.prompt_type == "encourage_middle" else BASE_PROMPT

    metadata = {
        "timestamp": run_timestamp,
        "model": args.model,
        "prompt_type": args.prompt_type,
        "system_prompt": system_prompt,
        "question_prompt": QUESTION_PROMPT,
        "scale": "1-10",
        "seed": args.seed,
        "n_faces": len(faces),
        "n_pairs": len(pairs),
    }
    metadata_path = output_folder / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    number_map = load_number_map()
    api_key = load_api_key()
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"Rating {len(pairs)} face pairs with {args.model} (prompt_type={args.prompt_type}, concurrency={args.concurrency})...")
    async with AsyncOpenAI(api_key=api_key, max_retries=5) as client:
        tasks = [
            rate_pair(
                client=client,
                semaphore=semaphore,
                pair_idx=idx,
                total_pairs=len(pairs),
                pair=pair,
                system_prompt=system_prompt,
                number_map=number_map,
                model=args.model,
                seed=args.seed,
            )
            for idx, pair in enumerate(pairs)
        ]
        completed = await asyncio.gather(*tasks)

    rows = [r for r, _ in completed]
    raw_records = [raw for _, raw in completed]

    results = pd.DataFrame(rows).set_index("index")
    similarities_path = output_folder / "face_similarities.csv"
    results.to_csv(similarities_path)

    matrix = pd.DataFrame(np.nan, index=faces, columns=faces)
    for _, row in results.iterrows():
        matrix.loc[row["Face1"], row["Face2"]] = row["Model Rating"]
        matrix.loc[row["Face2"], row["Face1"]] = row["Model Rating"]
    matrix_path = output_folder / "face_similarity_matrix.csv"
    matrix.to_csv(matrix_path, na_rep="NA")

    raw_path = output_folder / "raw_responses.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec) + "\n")

    metadata["completion_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"\nSaved {len(results)} ratings to {similarities_path}\n"
        f"Rating range: [{results['Model Rating'].min():.3f}, {results['Model Rating'].max():.3f}], "
        f"mean = {results['Model Rating'].mean():.3f}, std = {results['Model Rating'].std():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument(
        "--prompt-type",
        choices=["encourage_middle", "base"],
        default="encourage_middle",
        help="Prompt variant to use (default: encourage_middle)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for OpenAI completion (default: 123)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent API requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument("--test", action="store_true", help="Only rate pairs among the first --test-n faces")
    parser.add_argument("--test-n", type=int, default=DEFAULT_TEST_N)
    asyncio.run(run_similarities(parser.parse_args()))


if __name__ == "__main__":
    main()

