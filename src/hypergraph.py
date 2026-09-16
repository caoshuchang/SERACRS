"""Session-level hypergraph construction extracted from the ReDial main flow."""

import json
import os
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm


class HyperGraph:
    """Build a sparse entity-by-dialogue incidence matrix for ReDial."""

    def __init__(
        self,
        dataset: str = "redial",
        split: str = "train",
        debug: bool = False,
        n_entity: Optional[int] = None,
        pad_entity_id: Optional[int] = None,
        entity_max_length: int = 32,
        max_edges: int = 80000,
        min_edge_size: int = 2,
        data_root: str = "/root/autodl-tmp/MSCRS/rec_data",
        cache_dir: Optional[str] = None,
    ):
        if n_entity is None or pad_entity_id is None:
            raise ValueError("n_entity and pad_entity_id must be provided")
        self.dataset = dataset
        self.split = split
        self.debug = debug
        self.n_entity = int(n_entity)
        self.pad_entity_id = int(pad_entity_id)
        self.entity_max_length = int(entity_max_length)
        self.max_edges = int(max_edges)
        self.min_edge_size = int(min_edge_size)
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, f"hyper_H_{split}.pt")

    def get_entity_hyper_info(self) -> Dict[str, torch.Tensor]:
        if self.cache_path is not None and os.path.exists(self.cache_path):
            incidence = torch.load(self.cache_path, map_location="cpu")
            if not isinstance(incidence, torch.Tensor) or not incidence.is_sparse:
                raise ValueError(f"{self.cache_path} is not a sparse tensor")
            return {"hyper_H": incidence.coalesce()}

        incidence = self._build_hypergraph_incidence()
        if self.cache_path is not None:
            torch.save(incidence, self.cache_path)
        return {"hyper_H": incidence}

    def _load_jsonl(self) -> List[dict]:
        dataset_dir = os.path.join(self.data_root, self.dataset)
        data_file = os.path.join(dataset_dir, f"{self.split}_data.jsonl")
        if not os.path.exists(data_file):
            data_file = os.path.join(dataset_dir, f"{self.split}_data_train.jsonl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"No recommendation data found for {self.dataset}/{self.split}")
        with open(data_file, "r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        return rows[: min(2000, len(rows))] if self.debug else rows

    @staticmethod
    def _downsample_indices(size: int, max_keep: int) -> List[int]:
        if size <= max_keep:
            return list(range(size))
        step = max(1, size // max_keep)
        return list(range(0, size, step))[:max_keep]

    def _build_hypergraph_incidence(self) -> torch.Tensor:
        dialogs = self._load_jsonl()
        keep = self._downsample_indices(len(dialogs), self.max_edges)
        indices: List[List[int]] = []
        values: List[float] = []
        edge_id = 0

        for index in tqdm(keep, desc=f"Building hypergraph H ({self.dataset}/{self.split})"):
            dialog = dialogs[index]
            entity_ids = dialog.get("context_entities", dialog.get("entity", []))
            entity_ids = entity_ids[-self.entity_max_length :] if isinstance(entity_ids, list) else []
            entity_ids = [
                int(entity_id)
                for entity_id in set(entity_ids)
                if int(entity_id) != self.pad_entity_id
                and 0 <= int(entity_id) < self.n_entity
            ]
            if len(entity_ids) < self.min_edge_size:
                continue

            weight = 1.0 / float(len(entity_ids))
            for entity_id in entity_ids:
                indices.append([entity_id, edge_id])
                values.append(weight)
            edge_id += 1

        if edge_id == 0:
            raise RuntimeError("No hyperedges were constructed")
        index_tensor = torch.tensor(indices, dtype=torch.long).t().contiguous()
        value_tensor = torch.tensor(values, dtype=torch.float)
        return torch.sparse_coo_tensor(
            index_tensor,
            value_tensor,
            size=(self.n_entity, edge_id),
        ).coalesce()
