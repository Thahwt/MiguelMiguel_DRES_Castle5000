"""
Image-to-Image retrieval module for Castle5000.
Adapted from DS314_CuoiKi retrieval pipelines.

Supports:
- Single or multiple reference query image IDs
- Configurable top-k
- Multi-model hybrid fusion (SigLIP2 + DINOv2 via Reciprocal Rank Fusion)
- Output reporting with Image ID, SigLIP2 Score, DINOv2 Score, and Hybrid Score
- 2-Round Retrieval with Relevance Feedback
"""

import os
from typing import List, Optional, Union, Tuple, Dict, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ==============================================================================
# 1. Similarity & Ranking Functions
# ==============================================================================

def cosine_similarity_matrix(query_emb: np.ndarray, gallery_emb: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between normalized query and gallery embeddings.
    (Q, D) @ (N, D).T -> (Q, N)
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    if gallery_emb.ndim == 1:
        gallery_emb = gallery_emb.reshape(1, -1)
    return query_emb @ gallery_emb.T


def mean_query_embedding(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute mean of multiple embeddings and L2-normalize.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    mean_emb = np.mean(embeddings, axis=0, keepdims=True)
    norm = np.linalg.norm(mean_emb, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return mean_emb / norm


def rank_by_similarity(
    similarities: np.ndarray,
    top_k: int = 20,
    exclude_indices: Optional[List[int]] = None,
) -> List[List[dict]]:
    """
    Rank gallery images by similarity scores descending.
    """
    if similarities.ndim == 1:
        similarities = similarities.reshape(1, -1)

    results = []
    exclude_set = set(exclude_indices) if exclude_indices else set()

    for q_idx in range(similarities.shape[0]):
        scores = similarities[q_idx].copy()
        if exclude_set:
            for idx in exclude_set:
                if 0 <= idx < len(scores):
                    scores[idx] = -np.inf

        sorted_indices = np.argsort(scores)[::-1][:top_k]

        ranked = []
        for rank, idx in enumerate(sorted_indices, 1):
            if scores[idx] == -np.inf:
                break
            ranked.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "rank": rank,
            })
        results.append(ranked)

    return results


# ==============================================================================
# 2. Reciprocal Rank Fusion (RRF)
# ==============================================================================

def model_fusion_rrf(
    siglip_ranking: List[dict],
    dinov2_ranking: List[dict],
    k: int = 60,
    top_k: int = 20,
    exclude_indices: Optional[Set[int]] = None,
) -> List[dict]:
    """
    Fuse SigLIP2 and DINOv2 rankings using Reciprocal Rank Fusion.
    Score = 1 / (k + rank_siglip) + 1 / (k + rank_dinov2)
    """
    if exclude_indices is None:
        exclude_indices = set()

    rrf_scores = defaultdict(float)
    siglip_ranks = {}
    dinov2_ranks = {}

    for item in siglip_ranking:
        idx = item["index"]
        if idx in exclude_indices:
            continue
        rrf_scores[idx] += 1.0 / (k + item["rank"])
        siglip_ranks[idx] = item["rank"]

    for item in dinov2_ranking:
        idx = item["index"]
        if idx in exclude_indices:
            continue
        rrf_scores[idx] += 1.0 / (k + item["rank"])
        dinov2_ranks[idx] = item["rank"]

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for final_rank, (idx, score) in enumerate(sorted_items[:top_k], 1):
        results.append({
            "index": idx,
            "siglip_rank": siglip_ranks.get(idx),
            "dinov2_rank": dinov2_ranks.get(idx),
            "fusion_score": float(score),
            "final_rank": final_rank,
        })

    return results


def multi_reference_rrf(
    per_ref_rankings: List[List[dict]],
    k: int = 60,
    top_k: int = 20,
    exclude_indices: Optional[Set[int]] = None,
) -> List[dict]:
    """
    Fuse rankings from multiple reference images using RRF.
    """
    if exclude_indices is None:
        exclude_indices = set()

    rrf_scores = defaultdict(float)
    for ranking in per_ref_rankings:
        for item in ranking:
            idx = item["index"]
            if idx in exclude_indices:
                continue
            rrf_scores[idx] += 1.0 / (k + item["rank"])

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for final_rank, (idx, score) in enumerate(sorted_items[:top_k], 1):
        results.append({
            "index": idx,
            "fusion_score": float(score),
            "final_rank": final_rank,
        })
    return results


# ==============================================================================
# 3. Interactive Image Retriever Class
# ==============================================================================

class ImageToImageRetriever:
    """
    Image-to-Image retrieval engine using SigLIP2 and DINOv2 embeddings.
    Supports single/multi query images, configurable top-k,
    exact individual and hybrid score reporting, and 2-round retrieval.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        siglip_embeddings: Optional[np.ndarray] = None,
        dinov2_embeddings: Optional[np.ndarray] = None,
        rrf_k: int = 60,
        default_top_k: int = 20,
        exclude_query_images: bool = True,
    ):
        self.metadata = metadata.copy()
        self.siglip_embeddings = siglip_embeddings
        self.dinov2_embeddings = dinov2_embeddings
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k
        self.exclude_query_images = exclude_query_images

        # Build mapping for image ID lookups (supports '08_527', 'day1_Kitchen_08_527', '08_527.jpg')
        self.id_to_idx = {}
        for idx, row in self.metadata.iterrows():
            img_id = str(row["image_id"]).strip()
            self.id_to_idx[img_id] = idx

            # Add variants for robustness
            if img_id.startswith("day1_Kitchen_"):
                stem = img_id.replace("day1_Kitchen_", "")
                self.id_to_idx[stem] = idx
            else:
                self.id_to_idx[f"day1_Kitchen_{img_id}"] = idx

            if "file_name" in row and pd.notna(row["file_name"]):
                self.id_to_idx[str(row["file_name"]).strip()] = idx

    def _resolve_indices(self, image_ids: Union[str, List[str]]) -> List[int]:
        """Convert input image ID(s) to metadata indices."""
        if isinstance(image_ids, str):
            image_ids = [image_ids]

        indices = []
        for raw_id in image_ids:
            clean_id = str(raw_id).strip()
            if clean_id in self.id_to_idx:
                indices.append(self.id_to_idx[clean_id])
            else:
                raise KeyError(
                    f"Image ID '{raw_id}' not found in metadata. "
                    f"Example valid IDs: {list(self.metadata['image_id'][:5])}"
                )
        return indices

    def retrieve(
        self,
        reference_image_ids: Union[str, List[str]],
        top_k: Optional[int] = None,
        strategy: str = "mean_embedding",
    ) -> List[dict]:
        """
        Perform hybrid image retrieval.

        Args:
            reference_image_ids: 1 or multiple image IDs
            top_k: Number of retrieved results
            strategy: 'mean_embedding' or 'per_reference_rrf'

        Returns:
            List of result dictionaries with:
            - final_rank
            - image_id
            - siglip_score
            - dinov2_score
            - hybrid_score
            - file_name
            - image_path
        """
        if self.siglip_embeddings is None or self.dinov2_embeddings is None:
            raise RuntimeError("Both SigLIP2 and DINOv2 embeddings are required for hybrid retrieval.")

        top_k = top_k or self.default_top_k
        if isinstance(reference_image_ids, str):
            reference_image_ids = [reference_image_ids]

        ref_indices = self._resolve_indices(reference_image_ids)
        exclude_set = set(ref_indices) if self.exclude_query_images else set()
        exclude_list = list(exclude_set) if exclude_set else None

        # Compute query representation
        if len(ref_indices) == 1:
            q_siglip = self.siglip_embeddings[ref_indices[0]:ref_indices[0] + 1]
            q_dinov2 = self.dinov2_embeddings[ref_indices[0]:ref_indices[0] + 1]
        else:
            q_siglip = mean_query_embedding(self.siglip_embeddings[ref_indices])
            q_dinov2 = mean_query_embedding(self.dinov2_embeddings[ref_indices])

        # Exact cosine similarities across entire dataset
        siglip_sims = cosine_similarity_matrix(q_siglip, self.siglip_embeddings)[0]
        dinov2_sims = cosine_similarity_matrix(q_dinov2, self.dinov2_embeddings)[0]

        # Hybrid fusion ranking
        if len(ref_indices) == 1 or strategy == "mean_embedding":
            extended_k = min(top_k * 5, len(self.metadata))
            siglip_ranking = rank_by_similarity(
                siglip_sims, top_k=extended_k, exclude_indices=exclude_list
            )[0]
            dinov2_ranking = rank_by_similarity(
                dinov2_sims, top_k=extended_k, exclude_indices=exclude_list
            )[0]

            fused = model_fusion_rrf(
                siglip_ranking,
                dinov2_ranking,
                k=self.rrf_k,
                top_k=top_k,
                exclude_indices=exclude_set,
            )
        else:
            # per_reference_rrf
            extended_k = min(top_k * 5, len(self.metadata))
            siglip_per_ref = []
            dinov2_per_ref = []

            for ref_idx in ref_indices:
                s_emb = self.siglip_embeddings[ref_idx:ref_idx + 1]
                s_sim = cosine_similarity_matrix(s_emb, self.siglip_embeddings)
                siglip_per_ref.append(
                    rank_by_similarity(s_sim, top_k=extended_k, exclude_indices=exclude_list)[0]
                )

                d_emb = self.dinov2_embeddings[ref_idx:ref_idx + 1]
                d_sim = cosine_similarity_matrix(d_emb, self.dinov2_embeddings)
                dinov2_per_ref.append(
                    rank_by_similarity(d_sim, top_k=extended_k, exclude_indices=exclude_list)[0]
                )

            siglip_fused = multi_reference_rrf(
                siglip_per_ref, k=self.rrf_k, top_k=extended_k, exclude_indices=exclude_set
            )
            dinov2_fused = multi_reference_rrf(
                dinov2_per_ref, k=self.rrf_k, top_k=extended_k, exclude_indices=exclude_set
            )

            siglip_ranking = [
                {"index": item["index"], "rank": item["final_rank"], "score": item["fusion_score"]}
                for item in siglip_fused
            ]
            dinov2_ranking = [
                {"index": item["index"], "rank": item["final_rank"], "score": item["fusion_score"]}
                for item in dinov2_fused
            ]

            fused = model_fusion_rrf(
                siglip_ranking,
                dinov2_ranking,
                k=self.rrf_k,
                top_k=top_k,
                exclude_indices=exclude_set,
            )

        # Build output results with guaranteed scores
        results = []
        for item in fused[:top_k]:
            idx = item["index"]
            row = self.metadata.iloc[idx]
            results.append({
                "final_rank": item["final_rank"],
                "image_id": row["image_id"],
                "siglip_score": float(siglip_sims[idx]),
                "dinov2_score": float(dinov2_sims[idx]),
                "hybrid_score": float(item["fusion_score"]),
                "siglip_rank": item.get("siglip_rank"),
                "dinov2_rank": item.get("dinov2_rank"),
                "file_name": row.get("file_name", ""),
                "image_path": row.get("image_path", ""),
            })

        return results

    def to_dataframe(self, results: List[dict]) -> pd.DataFrame:
        """Format results as a neat pandas DataFrame."""
        rows = []
        for r in results:
            rows.append({
                "Rank": r["final_rank"],
                "Image ID": r["image_id"],
                "SigLIP2 Score": round(r["siglip_score"], 4),
                "DINOv2 Score": round(r["dinov2_score"], 4),
                "Hybrid Score": round(r["hybrid_score"], 6),
                "File Name": r.get("file_name", ""),
            })
        return pd.DataFrame(rows)

    def run_round1(
        self,
        query_ids: Union[str, List[str]],
        top_k: int = 10,
        strategy: str = "mean_embedding",
    ) -> Tuple[List[str], List[dict], pd.DataFrame]:
        """
        Execute Round 1 retrieval.
        Returns: (query_ids, results, df)
        """
        if isinstance(query_ids, str):
            query_ids = [query_ids]

        results = self.retrieve(query_ids, top_k=top_k, strategy=strategy)
        df = self.to_dataframe(results)
        return query_ids, results, df

    def run_round2(
        self,
        round1_query_ids: List[str],
        additional_relevant_ids: List[str],
        top_k: int = 10,
        strategy: str = "mean_embedding",
    ) -> Tuple[List[str], List[dict], pd.DataFrame]:
        """
        Execute Round 2 relevance feedback retrieval.
        Expands initial query IDs with newly selected relevant IDs.
        Returns: (combined_query_ids, results, df)
        """
        combined = list(round1_query_ids)
        for img_id in additional_relevant_ids:
            if img_id not in combined:
                combined.append(img_id)

        results = self.retrieve(combined, top_k=top_k, strategy=strategy)
        df = self.to_dataframe(results)
        return combined, results, df


# ==============================================================================
# 4. Visualization Utilities
# ==============================================================================

def get_image_path(metadata: pd.DataFrame, image_id: str, base_dir: str = ".") -> str:
    """Resolve image path from metadata row or search directory."""
    clean_id = str(image_id).strip()
    id_col = "image_id" if "image_id" in metadata.columns else "imageID"
    matches = metadata[metadata[id_col] == clean_id]

    if len(matches) > 0:
        row = matches.iloc[0]
        path = row.get("image_path") or row.get("file_path")
        if path and os.path.exists(path):
            return path
        # Try relative to base_dir
        if path:
            candidate = os.path.join(base_dir, path)
            if os.path.exists(candidate):
                return candidate
            candidate2 = os.path.join(base_dir, "data", path)
            if os.path.exists(candidate2):
                return candidate2

    # Fallback scan in data folder
    for batch in ["Batch1", "Batch2", "Batch3"]:
        candidate = os.path.join(base_dir, "data", batch, f"{clean_id}.jpg")
        if os.path.exists(candidate):
            return candidate
    return ""


def show_query_images(
    metadata: pd.DataFrame,
    query_ids: List[str],
    title: str = "Query Image(s)",
    figsize_per_img: Tuple[float, float] = (3.2, 3.2),
):
    """Display query image(s) with blue border highlights."""
    n = len(query_ids)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(figsize_per_img[0] * n, figsize_per_img[1]))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{title} (N={n})", fontsize=13, fontweight="bold", color="#1f77b4", y=1.05)

    for i, q_id in enumerate(query_ids):
        path = get_image_path(metadata, q_id)
        if path and os.path.exists(path):
            img = Image.open(path).convert("RGB")
            axes[i].imshow(img)
        else:
            axes[i].text(0.5, 0.5, f"Not found:\n{q_id}", ha="center", va="center")

        axes[i].set_title(f"REF #{i+1}: {q_id}", fontsize=10, fontweight="bold", color="#1f77b4")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#1f77b4")
            spine.set_linewidth(3)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()


def show_retrieval_results_with_scores(
    results: List[dict],
    title: str = "Retrieval Results",
    ncols: int = 5,
    figsize_per_image: Tuple[float, float] = (3.0, 3.3),
):
    """
    Display retrieved images in a grid with:
    - #{Rank}: {Image ID}
    - SigLIP: {siglip_score:.4f}
    - DINO:   {dinov2_score:.4f}
    - Hybrid: {hybrid_score:.5f}
    """
    n = len(results)
    if n == 0:
        print("No results to display.")
        return

    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    figsize = (ncols * figsize_per_image[0], nrows * figsize_per_image[1])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"{title} — Top-{n}", fontsize=14, fontweight="bold", y=1.01)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        if i < n:
            item = results[i]
            img_path = item.get("image_path", "")
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Image:\n{item['image_id']}", ha="center", va="center")

            title_text = (
                f"#{item['final_rank']}: {item['image_id']}\n"
                f"SigLIP: {item['siglip_score']:.4f}\n"
                f"DINO: {item['dinov2_score']:.4f}\n"
                f"Hybrid: {item['hybrid_score']:.5f}"
            )
            ax.set_title(title_text, fontsize=8.5, linespacing=1.2)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
