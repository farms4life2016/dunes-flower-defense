#!/usr/bin/env python3
"""Validate and draw monotone rectilinear planar 3SAT YAML instances."""

from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import yaml


@dataclass(frozen=True)
class Clause:
    label: str
    variables: tuple[int, int, int]
    sign: str

    @property
    def left(self) -> int:
        return min(self.variables)

    @property
    def right(self) -> int:
        return max(self.variables)


@dataclass(frozen=True)
class Instance:
    n: int
    m: int
    positive: tuple[Clause, ...]
    negative: tuple[Clause, ...]

    @property
    def clauses(self) -> tuple[Clause, ...]:
        return self.positive + self.negative


class ValidationError(Exception):
    """Raised when the input is not a valid MRP3SAT instance."""


def load_instance(path: Path) -> Instance:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ValidationError(f"Input file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"YAML parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValidationError("Top-level YAML value must be a mapping.")
    raw_instance = data.get("instance")
    if not isinstance(raw_instance, dict):
        raise ValidationError("Missing required mapping: instance.")

    n = _required_int(raw_instance, "n")
    m = _required_int(raw_instance, "m")
    if n <= 0:
        raise ValidationError("instance.n must be a positive integer.")
    if m < 0:
        raise ValidationError("instance.m must be a nonnegative integer.")

    raw_positive = _required_clause_list(raw_instance, "positive")
    raw_negative = _required_clause_list(raw_instance, "negative")
    if len(raw_positive) + len(raw_negative) != m:
        raise ValidationError(
            "instance.m must equal len(instance.positive) + len(instance.negative)."
        )

    positive = tuple(
        _parse_clause(raw_clause, f"C{index}", "positive", n)
        for index, raw_clause in enumerate(raw_positive, start=1)
    )
    negative = tuple(
        _parse_clause(raw_clause, f"C{index}", "negative", n)
        for index, raw_clause in enumerate(raw_negative, start=len(positive) + 1)
    )

    return Instance(n=n, m=m, positive=positive, negative=negative)


def _required_int(mapping: dict, key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ValidationError(f"instance.{key} must be an integer.")
    return value


def _required_clause_list(mapping: dict, key: str) -> list:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValidationError(f"instance.{key} must be a list of clauses.")
    return value


def _parse_clause(raw_clause: object, label: str, sign: str, n: int) -> Clause:
    if not isinstance(raw_clause, list):
        raise ValidationError(f"{label} must be a list of exactly 3 variable IDs.")
    if len(raw_clause) != 3:
        raise ValidationError(f"{label} must contain exactly 3 variables.")
    if any(type(variable) is not int for variable in raw_clause):
        raise ValidationError(f"{label} must contain only integer variable IDs.")

    variables = tuple(raw_clause)
    if len(set(variables)) != 3:
        raise ValidationError(f"{label} must contain 3 distinct variables.")
    out_of_range = [variable for variable in variables if variable < 1 or variable > n]
    if out_of_range:
        bad = ", ".join(str(variable) for variable in out_of_range)
        raise ValidationError(f"{label} contains variable IDs outside 1..{n}: {bad}.")

    return Clause(label=label, variables=variables, sign=sign)


def compute_clause_levels(clauses: Iterable[Clause], side_name: str) -> dict[str, int]:
    clauses = tuple(clauses)
    by_label = {clause.label: clause for clause in clauses}
    edges: dict[str, set[str]] = {clause.label: set() for clause in clauses}
    indegree = {clause.label: 0 for clause in clauses}

    for outer in clauses:
        outer_variables = set(outer.variables)
        for inner in clauses:
            if outer == inner:
                continue
            has_blocked_connection = any(
                outer.left < variable < outer.right and variable not in outer_variables
                for variable in inner.variables
            )
            if not has_blocked_connection:
                continue
            if outer.label not in edges[inner.label]:
                edges[inner.label].add(outer.label)
                indegree[outer.label] += 1

    queue = deque(
        sorted(
            (label for label, degree in indegree.items() if degree == 0),
            key=_label_number,
        )
    )
    order: list[str] = []
    while queue:
        label = queue.popleft()
        order.append(label)
        for neighbor in sorted(edges[label], key=_label_number):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(clauses):
        cyclic = ", ".join(
            sorted((label for label, degree in indegree.items() if degree > 0), key=_label_number)
        )
        raise ValidationError(
            f"{side_name} clauses cannot be drawn without crossings; "
            f"ordering constraints contain a cycle involving: {cyclic}."
        )

    return {label: level for level, label in enumerate(order, start=1) if label in by_label}


def _label_number(label: str) -> int:
    return int(label[1:])


def validate_instance(instance: Instance) -> tuple[dict[str, int], dict[str, int]]:
    positive_levels = compute_clause_levels(instance.positive, "Positive")
    negative_levels = compute_clause_levels(instance.negative, "Negative")
    return positive_levels, negative_levels


def draw_instance(
    instance: Instance,
    positive_levels: dict[str, int],
    negative_levels: dict[str, int],
    output: Path | None,
    show: bool,
) -> None:
    max_positive = max(positive_levels.values(), default=0)
    max_negative = max(negative_levels.values(), default=0)
    max_level = max(max_positive, max_negative, 1)

    fig_width = max(7.0, instance.n * 1.2)
    fig_height = max(4.5, (max_positive + max_negative + 2) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    axis.axhline(0, color="#2f2f2f", linewidth=1.2)
    axis.scatter(range(1, instance.n + 1), [0] * instance.n, color="#222222", zorder=4)
    for variable in range(1, instance.n + 1):
        axis.text(variable, -0.18, f"x{variable}", ha="center", va="top", fontsize=10)

    _draw_side(axis, instance.positive, positive_levels, direction=1, color="#2166ac")
    _draw_side(axis, instance.negative, negative_levels, direction=-1, color="#b2182b")

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.5, instance.n + 0.5)
    axis.set_ylim(-(max_negative + 1), max_positive + 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Monotone Rectilinear Planar 3SAT Instance")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved drawing to {output}")
    if show:
        plt.show()
    plt.close(fig)


def _draw_side(axis, clauses: Iterable[Clause], levels: dict[str, int], direction: int, color: str) -> None:
    for clause in clauses:
        y = direction * levels[clause.label]
        axis.plot([clause.left, clause.right], [y, y], color=color, linewidth=2.4)
        for variable in sorted(clause.variables):
            axis.plot([variable, variable], [0, y], color=color, linewidth=1.2, alpha=0.8)
            axis.scatter([variable], [y], color=color, s=18, zorder=3)
        label_y = y + (0.16 * direction)
        vertical_alignment = "bottom" if direction > 0 else "top"
        axis.text(
            (clause.left + clause.right) / 2,
            label_y,
            clause.label,
            ha="center",
            va=vertical_alignment,
            fontsize=10,
            color=color,
            fontweight="bold",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and draw monotone rectilinear planar 3SAT YAML instances."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path(__file__).with_name("input.yaml"),
        type=Path,
        help="YAML file to validate. Defaults to input.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the drawing to this path. Extension controls format, e.g. .png or .svg.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window with the drawing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        instance = load_instance(args.input)
        positive_levels, negative_levels = validate_instance(instance)
    except ValidationError as exc:
        print(f"Invalid MRP3SAT instance: {exc}")
        return 1

    print(
        f"Valid MRP3SAT instance: {instance.n} variables, "
        f"{instance.m} clauses ({len(instance.positive)} positive, "
        f"{len(instance.negative)} negative)."
    )

    if args.output is not None or args.show:
        draw_instance(instance, positive_levels, negative_levels, args.output, args.show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
