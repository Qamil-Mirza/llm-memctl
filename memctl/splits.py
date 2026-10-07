"""Dataset splits shared by environments (which play them) and analysis (which reads results by them)."""

from __future__ import annotations


def fold_indices(instances: list[dict], k: int, fold: int, part: str = "test") -> list[int]:
    """Indices of one fold of a k-fold split stratified by question type (abstention questions apart).

    The file is sorted by question type, so a contiguous range is not a fair sample. Questions are
    listed stratum by stratum (file order within each) and dealt to folds in turn, so every fold gets
    each stratum to within one question and the folds differ in size by at most one. `part` "test"
    is fold `fold`; "train" is every other fold. Returned in file order.
    """
    if not 0 <= fold < k or part not in ("test", "train"):
        raise ValueError(f"bad fold {fold} of {k} or part {part!r}")
    strata: dict[tuple[str, bool], list[int]] = {}
    for index, instance in enumerate(instances):
        strata.setdefault((instance["question_type"], str(instance["question_id"]).endswith("_abs")), []).append(index)
    dealt = [index for members in strata.values() for index in members]
    return sorted(index for place, index in enumerate(dealt) if (place % k == fold) == (part == "test"))
