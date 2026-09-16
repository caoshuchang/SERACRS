"""Build configurable hard-tag matrices and leakage-safe scene memories.

The default tag backend clusters the existing MSCRS text embeddings.  It is a
fully reproducible fallback when explicit metadata is unavailable.  A
paper-ready run should prefer ``--tag_backend categories`` after collecting
DBpedia/curated category labels; both backends emit the same tensor contract.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/autodl-tmp/MSCRS/rec_data/redial")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/MSCRS/MSCRS-improved/assets")
    parser.add_argument("--text_encoder", default="/root/autodl-tmp/MSCRS/model/roberta-base")
    parser.add_argument("--num_tags", default="8,12,16,24")
    parser.add_argument("--tag_backend", choices=["clusters", "categories"], default="clusters")
    parser.add_argument("--category_file", default=None)
    parser.add_argument("--preference_file", default=None)
    parser.add_argument("--top_tags_per_item", type=int, default=2)
    parser.add_argument("--scene_batch_size", type=int, default=128)
    parser.add_argument("--scene_max_length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _load_rows(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _conversation_number(row: dict) -> int:
    if "conv_id" in row:
        return int(row["conv_id"])
    return int(str(row["identity"]).split("/", 1)[0])


def _preference_for(preferences: Dict[str, dict], conversation_id: int) -> dict:
    return preferences.get(f"train:{conversation_id}", preferences.get(str(conversation_id), {}))


def _entity_title(uri: str) -> str:
    title = str(uri).rstrip('>').rsplit('/', 1)[-1]
    title = re.sub(r'_\([^)]*\)$', '', title)
    return title.replace('_', ' ')


@torch.no_grad()
def _encode_missing_item_titles(
    data_dir: str,
    missing: List[int],
    encoder_path: str,
    device: str,
    batch_size: int = 128,
) -> Dict[int, np.ndarray]:
    entity2id = _load_json(os.path.join(data_dir, 'entity2id.json'))
    id2entity = {int(identity): entity for entity, identity in entity2id.items()}
    titles = [_entity_title(id2entity.get(item, str(item))) for item in missing]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path).to(device).eval()
    outputs = []
    for start in range(0, len(titles), batch_size):
        batch = tokenizer(
            titles[start : start + batch_size], padding=True, truncation=True,
            max_length=64, return_tensors='pt',
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu().numpy())
    matrix = np.concatenate(outputs, axis=0)
    return {item: vector for item, vector in zip(missing, matrix)}


def _load_item_embeddings(
    data_dir: str,
    item_ids: List[int],
    encoder_path: str,
    device: str,
) -> Tuple[np.ndarray, dict]:
    embeddings = _load_json(os.path.join(data_dir, "id_embeddings_text.json"))
    missing = [item for item in item_ids if str(item) not in embeddings]
    if missing:
        fallback = _encode_missing_item_titles(
            data_dir, missing, encoder_path, device
        )
        embeddings.update({str(item): vector.tolist() for item, vector in fallback.items()})
    matrix = np.asarray([embeddings[str(item)] for item in item_ids], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    coverage = {
        "native_text_embeddings": len(item_ids) - len(missing),
        "title_fallback_embeddings": len(missing),
        "coverage": 1.0,
    }
    return matrix, coverage


def _cluster_tags(
    embeddings: np.ndarray,
    num_tags: int,
    top_tags_per_item: int,
    seed: int,
) -> Tuple[torch.Tensor, dict]:
    model = MiniBatchKMeans(
        n_clusters=num_tags,
        random_state=seed,
        batch_size=1024,
        n_init=10,
        max_iter=300,
    ).fit(embeddings)
    centers = model.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    similarities = embeddings @ centers.T
    k = min(top_tags_per_item, num_tags)
    top = np.argpartition(similarities, -k, axis=1)[:, -k:]
    tags = np.zeros_like(similarities, dtype=np.float32)
    rows = np.arange(tags.shape[0])[:, None]
    # Hard multi-hot labels: content similarity only selects the top groups.
    tags[rows, top] = 1.0
    sample_size = min(4000, embeddings.shape[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(embeddings.shape[0], sample_size, replace=False)
    metrics = {
        "backend": "clusters",
        "num_tags": num_tags,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(embeddings[sample], model.labels_[sample], metric="cosine")),
        "cluster_sizes": np.bincount(model.labels_, minlength=num_tags).tolist(),
        "top_tags_per_item": k,
    }
    return torch.from_numpy(tags), metrics


def _normalize_category(category: str) -> str:
    category = category.rsplit("Category:", 1)[-1]
    category = re.sub(r"^\d{4}s?_", "", category)
    category = re.sub(r"^(American|British|Canadian|French|German|Indian|Japanese|English-language)_", "", category)
    category = category.replace("_films", "").replace("_film", "").replace("_movies", "")
    return category.lower()


def _category_tags(
    item_ids: List[int], category_file: str, num_tags: int
) -> Tuple[torch.Tensor, dict]:
    raw = _load_json(category_file)
    normalized: Dict[int, List[str]] = {}
    frequency = Counter()
    for item in item_ids:
        labels = {_normalize_category(label) for label in raw.get(str(item), [])}
        labels = {label for label in labels if label and not label.isdigit()}
        normalized[item] = sorted(labels)
        frequency.update(labels)
    vocabulary = [label for label, _ in frequency.most_common(num_tags)]
    tag_to_id = {tag: index for index, tag in enumerate(vocabulary)}
    tags = torch.zeros(len(item_ids), len(vocabulary))
    for row, item in enumerate(item_ids):
        for label in normalized[item]:
            if label in tag_to_id:
                tags[row, tag_to_id[label]] = 1
    metrics = {
        "backend": "categories",
        "num_tags": len(vocabulary),
        "vocabulary": vocabulary,
        "tag_frequency": [int(frequency[tag]) for tag in vocabulary],
        "item_coverage": float(tags.any(dim=1).float().mean()),
    }
    return tags, metrics


def _build_scenes(
    train_file: str,
    item_to_local: Dict[int, int],
    preferences: Dict[str, dict],
) -> List[dict]:
    conversations = defaultdict(lambda: {"texts": [], "movies": set(), "entities": set()})
    for row in _load_rows(train_file):
        conversation_id = _conversation_number(row)
        scene = conversations[conversation_id]
        if "context_tokens" in row:
            text = " ".join(row.get("context_tokens", []))
            targets = [row["items"]]
        else:
            text = " ".join(utterance for utterance in row.get("context", []) if utterance)
            targets = row.get("rec", [])
        scene["texts"].append(text)
        scene["movies"].update(
            int(movie) for movie in targets if int(movie) in item_to_local
        )
        scene["movies"].update(
            int(movie) for movie in row.get("all_movies", []) if int(movie) in item_to_local
        )
        entities = row.get("context_entities", row.get("entity", []))
        scene["entities"].update(int(entity) for entity in entities)

    scenes = []
    for conversation_id, scene in conversations.items():
        positive_movies = {
            int(movie)
            for movie in _preference_for(preferences, conversation_id).get("positive", [])
        }
        # A historical scene is positive evidence only.  Movies marked disliked
        # or unknown never contribute an incidence edge or a tag profile.
        scene["movies"].intersection_update(positive_movies)
        if len(scene["movies"]) < 2:
            continue
        # The longest processed prefix is the most complete history for a dialogue.
        text = max(scene["texts"], key=len)
        scenes.append(
            {
                "conversation_id": conversation_id,
                "text": text,
                "movie_ids": sorted(scene["movies"]),
                "entity_ids": sorted(scene["entities"]),
            }
        )
    return scenes


@torch.no_grad()
def _encode_scene_texts(
    scenes: List[dict],
    encoder_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path).to(device).eval()
    outputs = []
    for start in range(0, len(scenes), batch_size):
        texts = [scene["text"] for scene in scenes[start : start + batch_size]]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs)


def _scene_assets(
    scenes: List[dict],
    scene_embeddings: torch.Tensor,
    item_ids: List[int],
    item_tags: torch.Tensor,
) -> dict:
    item_to_local = {item: index for index, item in enumerate(item_ids)}
    incidence_rows = []
    incidence_cols = []
    profiles = torch.zeros(len(scenes), item_tags.shape[1])
    conversation_ids = torch.tensor([scene["conversation_id"] for scene in scenes])
    for index, scene in enumerate(scenes):
        local_movies = [item_to_local[item] for item in scene["movie_ids"]]
        incidence_rows.extend([index] * len(local_movies))
        incidence_cols.extend(local_movies)
        profiles[index] = item_tags[local_movies].amax(dim=0)
    indices = torch.tensor([incidence_rows, incidence_cols], dtype=torch.long)
    incidence = torch.sparse_coo_tensor(
        indices, torch.ones(len(incidence_rows)), size=(len(scenes), len(item_ids))
    ).coalesce()
    return {
        "scene_embeddings": scene_embeddings,
        "scene_tag_profiles": profiles,
        "scene_movie_incidence": incidence,
        "scene_conversation_ids": conversation_ids,
        "scene_metadata": scenes,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    item_ids = [int(item) for item in _load_json(os.path.join(args.data_dir, "item_ids.json"))]
    item_to_local = {item: index for index, item in enumerate(item_ids)}
    tag_counts = sorted({int(value) for value in args.num_tags.split(",") if value.strip()})
    if not tag_counts:
        raise ValueError("--num_tags must contain at least one positive integer")

    item_embeddings = None
    embedding_coverage = None
    if args.tag_backend == "clusters":
        item_embeddings, embedding_coverage = _load_item_embeddings(
            args.data_dir, item_ids, args.text_encoder, args.device
        )
    elif not args.category_file:
        raise ValueError("--category_file is required by the categories backend")

    tag_assets = {}
    for num_tags in tag_counts:
        if args.tag_backend == "clusters":
            item_tags, metrics = _cluster_tags(
                item_embeddings, num_tags, args.top_tags_per_item, args.seed
            )
            metrics["embedding_coverage"] = embedding_coverage
        else:
            item_tags, metrics = _category_tags(item_ids, args.category_file, num_tags)
        tag_assets[num_tags] = (item_tags, metrics)
        torch.save(
            {"item_ids": torch.tensor(item_ids), "item_tags": item_tags, "metrics": metrics},
            os.path.join(args.output_dir, f"item_tags_r{num_tags}.pt"),
        )

    preference_file = args.preference_file or os.path.join(
        args.data_dir, "conversation_preferences.json"
    )
    if not os.path.isfile(preference_file):
        raise FileNotFoundError(
            f"positive historical scenes require preference labels: {preference_file}"
        )
    preferences = _load_json(preference_file).get("preferences", {})
    train_file = os.path.join(args.data_dir, "train_data.jsonl")
    if not os.path.isfile(train_file):
        train_file = os.path.join(args.data_dir, "train_data_train.jsonl")
    scenes = _build_scenes(train_file, item_to_local, preferences)
    if not scenes:
        raise RuntimeError(
            "no positive multi-movie training scenes were built; inspect preference coverage"
        )
    embeddings = _encode_scene_texts(
        scenes,
        args.text_encoder,
        args.scene_batch_size,
        args.scene_max_length,
        args.device,
    )
    for num_tags, (item_tags, metrics) in tag_assets.items():
        assets = _scene_assets(scenes, embeddings, item_ids, item_tags)
        assets["item_ids"] = torch.tensor(item_ids)
        assets["tag_metrics"] = metrics
        torch.save(assets, os.path.join(args.output_dir, f"scene_memory_r{num_tags}.pt"))

    summary = {
        "tag_backend": args.tag_backend,
        "tag_counts": tag_counts,
        "num_items": len(item_ids),
        "num_scenes": len(scenes),
        "embedding_coverage": embedding_coverage,
        "selection_rule": "choose R on validation metrics only; never use test metrics",
    }
    with open(os.path.join(args.output_dir, "asset_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
