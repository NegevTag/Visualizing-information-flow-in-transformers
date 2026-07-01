import torch


def consecutive_cosine_similarities(self, vectors: list[torch.Tensor]) -> list[float]:  # len(vectors) -> len(vectors) - 1
    """Cosine similarity between each consecutive pair of vectors.
    Given vectors [v0, v1, ..., v_{n-1}], returns [cos(v0,v1), cos(v1,v2), ..., cos(v_{n-2}, v_{n-1})],
    a list of length n - 1.
    CLAUDE_WRITTEN
    """
    return [torch.nn.functional.cosine_similarity(a, b, dim=0).item() for a, b in zip(vectors, vectors[1:])]
