import torch


@torch.no_grad()
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    if outputs.ndim == 1:
        # Binary classification or regression thresholded
        preds = (outputs > 0).long()
    else:
        # Multi-class classification
        preds = outputs.argmax(dim=-1)

    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_calibration_stats(probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    # Basic calibration diagnostics
    preds = probs.argmax(dim=-1)
    max_probs = probs.max(dim=-1).values

    accuracy = (preds == targets).float().mean().item()
    confidence = max_probs.mean().item()

    return {
        "accuracy": accuracy,
        "mean_confidence": confidence,
        "calibration_gap": abs(accuracy - confidence),
    }
