"""Rate the visual similarity of pairs of schematic faces with TypeSafe's Jev or OpenAI's Decisions API.

Usage:
    python fast_model_face_similarities.py [--test] [--api jev|openai] [--pairs-per-request N]

Requires the TYPESAFE_API_KEY environment variable (or OPENAI_API_KEY for --api openai).
"""

import argparse
import asyncio
import base64
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from typesafe_sdk import AsyncTypeSafeClient, Image, Score

MODEL = "jev-latest"
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
MAX_CONCURRENT_REQUESTS = 8

IMAGE_FOLDER = Path("data") / "34 Faces"
DUPLICATE_FACES = {"face_26", "face_30"}  # Identical to face_11 and face_18 in Nosofsky (1991)
JEV_OUTPUT_FOLDER = Path("output") / "jev"
OPENAI_MODEL = "gpt-6-luna"
OPENAI_OUTPUT_FOLDER = Path("output") / "openai_decisions"

# Jev requests with 16 images (8 pairs without anchors) stay well within token limits.
MAX_IMAGES_PER_REQUEST = 16
DEFAULT_TEST_N = 4

SIMILARITY_QUESTION = "From 1-10, how visually similar are these two faces?"
PAIR_LABEL = "Rate these faces."
SIMILARITY_LEVELS = [
    "most dissimilar",
    "very dissimilar",
    "dissimilar",
    "somewhat dissimilar",
    "slightly dissimilar",
    "slightly similar",
    "somewhat similar",
    "similar",
    "very similar",
    "most similar",
]

_images: dict[str, Image] = {}


def load_image(face: str) -> Image:
    """Reuse one Image per face so each file is read only once."""
    if face not in _images:
        _images[face] = Image(IMAGE_FOLDER / f"{face}.png")
    return _images[face]


def image_data_url(face: str) -> str:
    data = base64.b64encode((IMAGE_FOLDER / f"{face}.png").read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def load_pairs(args):
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
    return faces, pairs


def score_result(pair, score, confidence, probabilities, model):
    result = dict(pair)
    result.update(
        {
            "Model Score": score,
            "Model Rating": score + 1,  # Match the 1-10 human scale in Nosofsky (1991)
            "Confidence": confidence,
        }
    )
    result.update({f"P({level + 1})": probabilities.get(level, 0.0) for level in range(len(SIMILARITY_LEVELS))})
    result["Model"] = model
    return result


async def rate_pairs_jev(client, semaphore, pairs):
    questions = {
        f"pair_{i}": Score(
            instructions=[SIMILARITY_QUESTION, load_image(pair["Face1"]), load_image(pair["Face2"])],
            criteria=SIMILARITY_LEVELS,
        )
        for i, pair in enumerate(pairs)
    }
    async with semaphore:
        response = await client.system_one(
            state=[],
            questions=questions,
            extra_headers=IMAGE_HEADERS,
        )
    print(f"Rated {len(pairs)} pairs ({response.usage.input_tokens} input tokens)")
    results = []
    for i, pair in enumerate(pairs):
        answer = response.answers[f"pair_{i}"]
        results.append(score_result(pair, answer.score, answer.confidence, answer.probabilities, response.model))
    return results, response.usage.input_tokens or 0


def pair_content(face1: str, face2: str):
    return [
        {"type": "input_image", "image_url": image_data_url(face1)},
        {"type": "input_image", "image_url": image_data_url(face2)},
    ]


async def rate_pair_openai(client, semaphore, pair):
    request = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": PAIR_LABEL}]
                + pair_content(pair["Face1"], pair["Face2"]),
            }
        ],
        "questions": [
            {
                "type": "score",
                "name": "similarity",
                "instructions": SIMILARITY_QUESTION,
                "levels": [{"label": level} for level in SIMILARITY_LEVELS],
            }
        ],
    }
    async with semaphore:
        response = await client.decisions.create(**request)
    answer = response.answers[0]
    if answer.type == "refusal":
        print(f"Refused: {pair['Face1']} vs {pair['Face2']}")
        result = score_result(pair, np.nan, np.nan, {}, response.model)
    else:
        probabilities = {p.value: p.probability for p in answer.probabilities}
        result = score_result(pair, answer.score, answer.confidence, probabilities, response.model)
    return [result], response.usage.input_tokens


async def run_similarities(args):
    faces, pairs = load_pairs(args)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    if args.api == "jev":
        pairs_per_request = args.pairs_per_request or (MAX_IMAGES_PER_REQUEST // 2)
        async with AsyncTypeSafeClient(model=MODEL) as client:
            batches = await asyncio.gather(
                *(
                    rate_pairs_jev(client, semaphore, pairs[i : i + pairs_per_request])
                    for i in range(0, len(pairs), pairs_per_request)
                )
            )
        output_folder = JEV_OUTPUT_FOLDER
    else:
        async with AsyncOpenAI(max_retries=5) as client:
            batches = await asyncio.gather(*(rate_pair_openai(client, semaphore, pair) for pair in pairs))
        output_folder = OPENAI_OUTPUT_FOLDER

    results = pd.DataFrame([row for rows, _ in batches for row in rows])
    results.index.name = "index"

    matrix = pd.DataFrame(np.nan, index=faces, columns=faces)
    for _, row in results.iterrows():
        matrix.loc[row["Face1"], row["Face2"]] = row["Model Rating"]
        matrix.loc[row["Face2"], row["Face1"]] = row["Model Rating"]

    if args.test:
        output_folder = output_folder / "test"
    output_folder.mkdir(parents=True, exist_ok=True)
    similarities_path = output_folder / "face_similarities.csv"
    matrix_path = output_folder / "face_similarity_matrix.csv"
    results.to_csv(similarities_path)
    matrix.to_csv(matrix_path, na_rep="NA")
    print(
        f"Saved {len(results)} ratings to {similarities_path} "
        f"({sum(t for _, t in batches)} input tokens total, "
        f"{results['Model Rating'].isna().sum()} refusals)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", choices=["jev", "openai"], default="jev")
    parser.add_argument("--test", action="store_true", help="Only rate pairs among the first --test-n faces")
    parser.add_argument("--test-n", type=int, default=DEFAULT_TEST_N)
    parser.add_argument("--pairs-per-request", type=int, help="Pairs per Jev request")
    asyncio.run(run_similarities(parser.parse_args()))


if __name__ == "__main__":
    main()

