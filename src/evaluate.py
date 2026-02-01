# src/evaluate.py
"""
evaluate.py
Threshold tuning / evaluation using enrollment crops (aligned 112x112).
Assumptions:
- Enrollment crops exist under: data/enroll/<name>/*.jpg
- Crops are aligned (112x112) already (as saved by enroll.py / haar_5pt pipeline)
- Uses ArcFaceEmbedderONNX from embed.py (your working embedder)
Outputs:
- Prints summary stats for genuine/impostor cosine distances
- Suggests a threshold based on a target FAR
Run:
python -m src.evaluate
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX

# -------------------------
# Config
# -------------------------

@dataclass
class EvalConfig:
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80  # cap for speed
    target_far: float = 0.01  # 1% FAR target
    thresholds: Tuple[float, float, float] = (0.10, 1.20, 0.01)  # start, end, step

    # Optional sanity constraints
    require_size: Tuple[int, int] = (112, 112)

# -------------------------
# Math
# -------------------------

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = vector_a.reshape(-1).astype(np.float32)
    vector_b = vector_b.reshape(-1).astype(np.float32)
    # embeddings are already L2-normalized in embed, so dot is cosine
    return float(np.dot(vector_a, vector_b))

def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    # distance = 1 - cosine similarity
    return 1.0 - cosine_similarity(vector_a, vector_b)

# -------------------------
# IO
# -------------------------

def list_people(config: EvalConfig) -> List[Path]:
    if not config.enroll_dir.exists():
        raise FileNotFoundError(
            f"Enroll dir not found: {config.enroll_dir}. Run enroll.py first."
        )
    return sorted([p for p in config.enroll_dir.iterdir() if p.is_dir()])

def _is_aligned_crop(image: np.ndarray, required_size: Tuple[int, int]) -> bool:
    height, width = image.shape[:2]
    return (width, height) == (int(required_size[0]), int(required_size[1]))

def load_embeddings_for_person(
    embedding_model: ArcFaceEmbedderONNX,
    person_directory: Path,
    config: EvalConfig,
) -> List[np.ndarray]:
    images = sorted(list(person_directory.glob("*.jpg")))[: config.max_imgs_per_person]
    embeddings: List[np.ndarray] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        # Skip non-aligned crops to keep eval clean
        if config.require_size is not None and not _is_aligned_crop(image, config.require_size):
            continue

        embedding_result = embedding_model.embed(image)
        embeddings.append(embedding_result.embedding_vector)

    return embeddings

# -------------------------
# Eval
# -------------------------

def pairwise_distances(
    embeddings_a: List[np.ndarray],
    embeddings_b: List[np.ndarray],
    same_person: bool,
) -> List[float]:
    distances: List[float] = []

    if same_person:
        for i in range(len(embeddings_a)):
            for j in range(i + 1, len(embeddings_a)):
                distances.append(cosine_distance(embeddings_a[i], embeddings_a[j]))
    else:
        for embedding_a in embeddings_a:
            for embedding_b in embeddings_b:
                distances.append(cosine_distance(embedding_a, embedding_b))

    return distances

def sweep_thresholds(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    config: EvalConfig,
):
    threshold_start, threshold_end, threshold_step = config.thresholds
    threshold_values = np.arange(threshold_start, threshold_end + 1e-9, threshold_step, dtype=np.float32)

    # FAR: impostor accepted => dist <= thr
    # FRR: genuine rejected  => dist > thr
    evaluation_results = []
    for threshold in threshold_values:
        false_acceptance_rate = float(np.mean(impostor_distances <= threshold)) if impostor_distances.size else 0.0
        false_rejection_rate = float(np.mean(genuine_distances > threshold)) if genuine_distances.size else 0.0
        evaluation_results.append((float(threshold), false_acceptance_rate, false_rejection_rate))

    return evaluation_results

def describe(array: np.ndarray) -> str:
    if array.size == 0:
        return "n=0"
    return (
        f"n={array.size} mean={array.mean():.3f} std={array.std():.3f} "
        f"p05={np.percentile(array, 5):.3f} "
        f"p50={np.percentile(array, 50):.3f} "
        f"p95={np.percentile(array, 95):.3f}"
    )

# -------------------------
# Main
# -------------------------

def main():
    config = EvalConfig()

    embedding_model = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    people_directories = list_people(config)
    if len(people_directories) < 1:
        print("No enrolled people found.")
        return

    # Load embeddings per person
    embeddings_per_person: Dict[str, List[np.ndarray]] = {}

    for person_directory in people_directories:
        person_name = person_directory.name
        embeddings = load_embeddings_for_person(embedding_model, person_directory, config)

        if len(embeddings) >= config.min_imgs_per_person:
            embeddings_per_person[person_name] = embeddings
        else:
            print(
                f"Skipping {person_name}: only {len(embeddings)} valid aligned crops "
                f"(need >= {config.min_imgs_per_person})."
            )

    person_names = sorted(embeddings_per_person.keys())
    if len(person_names) < 1:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    # Genuine (same identity)
    genuine_distances_list: List[float] = []
    for person_name in person_names:
        genuine_distances_list.extend(
            pairwise_distances(embeddings_per_person[person_name], embeddings_per_person[person_name], same_person=True)
        )

    # Impostor (different identities)
    impostor_distances_list: List[float] = []
    for i in range(len(person_names)):
        for j in range(i + 1, len(person_names)):
            impostor_distances_list.extend(
                pairwise_distances(
                    embeddings_per_person[person_names[i]],
                    embeddings_per_person[person_names[j]],
                    same_person=False,
                )
            )

    genuine_distances = np.array(genuine_distances_list, dtype=np.float32)
    impostor_distances = np.array(impostor_distances_list, dtype=np.float32)

    print("\n=== Distance Distributions (cosine distance = 1 - cosine similarity) ===")
    print(f"Genuine (same person): {describe(genuine_distances)}")
    print(f"Impostor (diff persons): {describe(impostor_distances)}")

    evaluation_results = sweep_thresholds(genuine_distances, impostor_distances, config)

    # Choose threshold with FAR <= target_far and minimal FRR
    best_threshold = None
    for threshold, false_acceptance_rate, false_rejection_rate in evaluation_results:
        if false_acceptance_rate <= config.target_far:
            if best_threshold is None or false_rejection_rate < best_threshold[2]:
                best_threshold = (threshold, false_acceptance_rate, false_rejection_rate)

    print("\n=== Threshold Sweep ===")
    print_stride = max(1, len(evaluation_results) // 10)
    for threshold, false_acceptance_rate, false_rejection_rate in evaluation_results[::print_stride]:
        print(f"thr={threshold:.2f} FAR={false_acceptance_rate*100:5.2f}% FRR={false_rejection_rate*100:5.2f}%")

    if best_threshold is not None:
        threshold, false_acceptance_rate, false_rejection_rate = best_threshold
        print(
            f"\nSuggested threshold (target FAR {config.target_far*100:.1f}%): "
            f"thr={threshold:.2f} FAR={false_acceptance_rate*100:.2f}% FRR={false_rejection_rate*100:.2f}%"
        )
    else:
        print(
            f"\nNo threshold in range met FAR <= {config.target_far*100:.1f}%. "
            "Try widening threshold sweep range or collecting more varied samples."
        )

    # Extra: similarity-style threshold
    if best_threshold is not None:
        similarity_threshold = 1.0 - best_threshold[0]
        print(
            f"\n(Equivalent cosine similarity threshold ~ {similarity_threshold:.3f}, "
            "since sim = 1 - dist)"
        )

    print()

if __name__ == "__main__":
    main()
