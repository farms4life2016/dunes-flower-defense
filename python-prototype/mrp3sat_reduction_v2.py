# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solving Monotone Planar (Rectilinear) 3SAT using Bloons TD
#
# A quick prototype in Python to assess the feasibility of our Unity project.
#
# ## Special Definitions
# See `definitions.tex` LaTeX file for formal definitions.
#
# ## Important Assessments
#
# - validate MRP3SAT instances, ensuring instances are planar and monotone
# - given a valid instance, find a rectilinear embedding/drawing
# - align that drawing to a square and triangle grid (SGA-MPR3SAT and TGA-MPR3SAT)
# - convert that drawing to planar vertex cover on a triangle grid (TGA-PVC)
# - convert the triangle-grid-aligned instance into a triangle-grid embedded instance (TGE-PVC)
# - convert the triangle-grid-embedded instance to a Bloons TD instance (BTD) that can be played in a video game
#
# ## Implementation Notes
#
# - MRP3SAT is only NP-complete if we allow duplicate variables within clauses
#   (or allow clauses with *at most* three variables instead of *exactly* 3). (Theorem 1 of Thai paper)
#   - Otherwise, every such instance is satisfiable. A solution can be found in polytime. (Theorem 11 of Pilz paper)
#   - We should store variables as multisets, sorted arrays of size 3, or some custom and iterable data structure;
#     at minimum, we need to be able to loop through all variables while being able to easily query
#     the left-most, middle, and right-most variable in the clause (L, M, R)
# - The input format for MRP3SAT should include a certificate of satisfiablilty or state that none exists.
# - Our program should pad clauses until they contain exactly 3 variables.
#   I recommend repeating the variable with smallest index.
# - The grid-like data structures for graphs should be stored as integers.
#   - I propose a map of int to int array: the keys are the y-levels (0 being the variable row),
#     and the int array representing x-values.
#   - For triangle-style grids, 
#     odd y-levels would be drawn +0.5 units to the right during rendering,
#     and all y-coordinates would be scaled by a factor of root(3)/2 during rendering
#
# ## Reduction Notes
#
# - each problem (except BTD) is in NP because they are just restricted versions of 3SAT and VC, which are in NP.
# - the only odd numbers we should see is the three vertices in the triangle of the triple-OR gadget for VC.
#   everything else (wire gadget, variable gadget) should have an **even** number of vertices.
#   **This is crucial for the reduction!!!**
#   - the gap between variables should be zero or some even number on the square/triangle grid
#   - the number of vertices between the variable vertex and the triple-OR gadget should be an even number (not counting either endpoint)
# - evenness is important for bipartite colouring: both colourings are the same size, so we use up the same number of cover vertices (from k) no matter which bipartition we pick.
# - also useful for calculating the limit "k", k = ((total number of vertices) - (3m)) / 2 + (2m)
#   - there are m number of triple-OR gadgets, which are all 3-vertex triangles
#   - after subtracting, a bunch of even bipartite trees remain. colouring each takes exactly half of the remainging vertices thanks to evenness.
#   - then add back 2 vertices per triangle since it takes at minimum 2 vertices to cover a triangle
# - to convert MRP3SAT yes-certificates into BTD yes-certificates, follow the same steps as converting 3SAT certificates into VC certificates.
#   Just run an additional BFS to propagate the bipartite colouring from the variable row to the triple-OR gadget.
# - the resulting TGE-PVC instance will have max degree of 3. Not sure if this is useful or easily provable.
#   We are also trivially guarenteed a max degree of 6 from the triangle grid.
#

# %% [markdown]
# ## How To Run
#
# From `python-prototype/`:
#
# ```bash
# source venv/bin/activate
# pip install -r requirements.txt
# jupyter notebook mrp3sat_reduction.ipynb
# ```
#
# Or, without activating the venv:
#
# ```bash
# venv/bin/python -m pip install -r requirements.txt
# venv/bin/jupyter notebook mrp3sat_reduction.ipynb
# ```
#
# In VS Code, select the interpreter/kernel from `python-prototype/venv` and run cells top to bottom.

# %% [markdown]
# ## Input Validation

# %%
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, RegularPolygon
from PIL import Image as PILImage
import yaml

# %matplotlib inline

INPUT_PATH = "input-unsatisfiable.yaml" # unsatisfiable input-varied.yaml

print("All OK!")


# %% [markdown]
# ### YAML input format
#
#
# - Every instance provides integers $n > 0$ and $m \geq 0$.
# - There are two list of clauses: one list for positive and another list for negative.
# - The clauses themselves don't store whether they are positive or negative.
# - Clauses are stored as positive integers which can range from $1, 2, \dots, n$
# - Clauses contain at most 3 elements, and duplicates are allowed. Order is also not enforced. (Our program will normalize all clauses anyways.)
# - There is an boolean representing the instance's possibility to be satisfied
# - There is always a certificate of size $n$, representing a truth valuation of all $n$ variables.
# - The certificate is either an array of `true`s and `false`s, or an array of `1`s and `0`s
# - There should be no redundant information in the data structure, apart from $m$ being the size of the list of positive clauses plus the size of the list of negative clauses
#

# %%
# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class Clause:
    label: int
    variables: tuple[int, int, int]

    @property
    def left(self) -> int:
        return self.variables[0]

    @property
    def middle(self) -> int:
        return self.variables[1]

    @property
    def right(self) -> int:
        return self.variables[2]


@dataclass(frozen=True)
class MRP3SATInstance:
    n: int
    m: int

    positive: tuple[Clause, ...]
    negative: tuple[Clause, ...]

    satisfiable: bool
    certificate: tuple[bool, ...]

    @property
    def clauses(self) -> tuple[Clause, ...]:
        return self.positive + self.negative


class ValidationError(Exception):
    """Raised when the input is not a valid MRP3SAT instance."""


# %%
# ============================================================
# YAML Loading
# ============================================================

def load_instance(path: Path | str) -> MRP3SATInstance:
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    except FileNotFoundError as exc:
        raise ValidationError(f"Input file not found: {path}") from exc

    except yaml.YAMLError as exc:
        raise ValidationError(f"YAML parse error: {exc}") from exc

    return parse_instance_data(data)


# %%
# ============================================================
# Parsing Helpers
# ============================================================

def _required_int(mapping: dict, key: str) -> int:
    value = mapping.get(key)

    if type(value) is not int:
        raise ValidationError(f"instance.{key} must be an integer.")

    return value


def _required_clause_list(mapping: dict, key: str) -> list:
    value = mapping.get(key)

    if not isinstance(value, list):
        raise ValidationError(f"instance.{key} must be a list.")

    return value


def _required_bool(mapping: dict, key: str) -> bool:
    value = mapping.get(key)

    if type(value) is not bool:
        raise ValidationError(f"instance.{key} must be true or false.")

    return value



# %%
def _parse_certificate(raw_certificate: object, n: int) -> tuple[bool, ...]:

    if not isinstance(raw_certificate, list):
        raise ValidationError("instance.certificate must be a list.")

    if len(raw_certificate) != n:
        raise ValidationError(
            f"instance.certificate must contain exactly {n} entries."
        )

    result: list[bool] = []

    for index, value in enumerate(raw_certificate, start=1):

        if value in (0, False):
            result.append(False)

        elif value in (1, True):
            result.append(True)

        else:
            raise ValidationError(
                f"certificate entry {index} must be true/false or 0/1."
            )

    return tuple(result)


# %%
def _label_str(label: int) -> str:
    return "C" + str(label)

def _parse_clause(
    raw_clause: object,
    label: int,
    n: int,
) -> Clause:

    if not isinstance(raw_clause, list):
        raise ValidationError(
            f"{_label_str(label)} must be a list containing 1–3 variable IDs."
        )

    if not (1 <= len(raw_clause) <= 3):
        raise ValidationError(
            f"{_label_str(label)} must contain between 1 and 3 variables."
        )

    if any(type(v) is not int for v in raw_clause):
        raise ValidationError(
            f"{_label_str(label)} must contain only integer variable IDs."
        )

    for variable in raw_clause:
        if variable < 1 or variable > n:
            raise ValidationError(
                f"{_label_str(label)} contains variable {variable} outside 1..{n}."
            )

    # --------------------------------------------------------
    # Canonicalization
    #
    # Sort first.
    # Then pad to length 3 by repeating the smallest variable.
    #
    # [4]       -> [4,4,4]
    # [2,7]     -> [2,2,7]
    # [4,4,1]   -> [1,4,4]
    # --------------------------------------------------------

    variables = sorted(raw_clause)

    while len(variables) < 3:
        variables.insert(0, variables[0])

    return Clause(
        label=label,
        variables=tuple(variables),
    )


# %%
# actual parsing?
def parse_instance_data(data: object) -> MRP3SATInstance:

    if not isinstance(data, dict):
        raise ValidationError("Top-level YAML value must be a mapping.")

    raw_instance = data.get("instance")

    if not isinstance(raw_instance, dict):
        raise ValidationError("Missing required mapping: instance.")

    n = _required_int(raw_instance, "n")
    m = _required_int(raw_instance, "m")

    if n <= 0:
        raise ValidationError("instance.n must be positive.")

    if m <= 0:
        raise ValidationError("instance.m must be nonnegative.")

    raw_positive = _required_clause_list(raw_instance, "positive")
    raw_negative = _required_clause_list(raw_instance, "negative")

    if len(raw_positive) + len(raw_negative) != m:
        raise ValidationError(
            "instance.m must equal "
            "len(positive) + len(negative)."
        )

    positive = tuple(
        _parse_clause(clause, i, n)
        for i, clause in enumerate(raw_positive, start=1)
    )

    negative = tuple(
        _parse_clause(clause, i, n)
        for i, clause in enumerate(
            raw_negative,
            start=len(positive) + 1,
        )
    )

    satisfiable = _required_bool(raw_instance, "satisfiable")

    certificate = _parse_certificate(
        raw_instance.get("certificate"),
        n,
    )

    return MRP3SATInstance(
        n=n,
        m=m,
        positive=positive,
        negative=negative,
        satisfiable=satisfiable,
        certificate=certificate,
    )


# %%
# certificate validation
def evaluate_clause(
    clause: Clause,
    certificate: tuple[bool, ...],
    is_positive: bool
) -> bool:

    if is_positive:
        return any(
            certificate[variable - 1]
            for variable in clause.variables
        )

    return any(
        not certificate[variable - 1]
        for variable in clause.variables
    )

def evaluate_formula(
    instance: MRP3SATInstance,
) -> bool:

    neg = all(
        evaluate_clause(
            clause,
            instance.certificate,
            False
        )
        for clause in instance.negative
    )

    return neg and all(
        evaluate_clause(
            clause,
            instance.certificate,
            True
        )
        for clause in instance.positive
    )


# %%
def verify_certificate(
    instance: MRP3SATInstance,
) -> None:

    actual_result = evaluate_formula(instance)

    if actual_result != instance.satisfiable:

        raise ValidationError(
            "Certificate does not match satisfiable flag. "
            f"Expected satisfiable={instance.satisfiable}, "
            f"but certificate evaluates to {actual_result}."
        )


# %%
# ============================================================
# Clause Level Computation
# ============================================================

def compute_clause_levels(
    clauses: Iterable[Clause]
) -> dict[int, int]:

    clauses = tuple(clauses)

    by_label: dict[int, Clause] = {
        clause.label: clause
        for clause in clauses
    }

    edges: dict[int, set[int]] = {
        clause.label: set()
        for clause in clauses
    }

    indegree: dict[int, int] = {
        clause.label: 0
        for clause in clauses
    }

    for outer in clauses:
        for inner in clauses:
            if outer == inner:
                continue

            has_blocked_connection = any(
                outer.left < variable < outer.right
                for variable in inner.variables
            )

            if not has_blocked_connection:
                continue

            if outer.label not in edges[inner.label]:
                edges[inner.label].add(outer.label)
                indegree[outer.label] += 1

    queue = deque(
        sorted(
            label
            for label, degree in indegree.items()
            if degree == 0
        )
    )

    order: list[int] = []

    while queue:
        label = queue.popleft()
        order.append(label)

        for neighbor in sorted(edges[label]):
            indegree[neighbor] -= 1

            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(clauses):
        cyclic = ", ".join(
            _label_str(label)
            for label in sorted(
                label
                for label, degree in indegree.items()
                if degree > 0
            )
        )

        raise ValidationError(
            f"Clauses cannot be drawn "
            f"without crossings. Cycle: {cyclic}"
        )

    return {
        label: level
        for level, label in enumerate(order, start=1)
        if label in by_label
    }


# %%
def validate_instance(
    instance: MRP3SATInstance,
) -> tuple[dict[int, int], dict[int, int]]:

    verify_certificate(instance)

    positive_levels = compute_clause_levels(
        instance.positive
    )

    negative_levels = compute_clause_levels(
        instance.negative
    )

    return positive_levels, negative_levels


# %%
def summarize_instance(
    instance: MRP3SATInstance,
) -> None:

    print(
        f"Valid MRP3SAT instance: "
        f"{instance.n} variables, "
        f"{instance.m} clauses "
        f"({len(instance.positive)} positive, "
        f"{len(instance.negative)} negative)."
    )

    for clause in instance.positive:

        literals = ", ".join(
            f"x{v}"
            for v in clause.variables
        )

        print(f"{_label_str(clause.label)}: {literals}")

    for clause in instance.negative:

        literals = ", ".join(
            f"¬x{v}"
            for v in clause.variables
        )

        print(f"{_label_str(clause.label)}: {literals}")

    print(
        "Certificate:",
        list(instance.certificate),
    )

    print(
        "Satisfiable:",
        instance.satisfiable,
    )


# %%
input_path = Path(INPUT_PATH)

if not input_path.exists():
    input_path = Path("python-prototype/" + INPUT_PATH)

mrp3sat_instance = load_instance(input_path)

positive_levels, negative_levels = validate_instance(
    mrp3sat_instance
)

summarize_instance(mrp3sat_instance)

print("Positive levels:", positive_levels)
print("Negative levels:", negative_levels)

print("All OK!")

# %% [markdown]
# ## SGA-MRP3SAT

# %%
# ============================================================
# Data Structures — Grid-Aligned MRP3SAT Graph
# ============================================================

from dataclasses import dataclass
from pathlib import Path

# common data structs, may be reused or extended later
@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass(frozen=True)
class UnitSegment:
    start: Point
    end: Point

@dataclass(frozen=True)
class GridEdge:
    segments: tuple[UnitSegment, ...]

    @property
    def start(self) -> Point:
        return self.segments[0].start

    @property
    def end(self) -> Point:
        return self.segments[-1].end


LEFT = 0
MIDDLE = 1
RIGHT = 2


@dataclass(frozen=True)
class ConnectorMetadata:
    clause_label: int
    variable: int
    slot: int  # 0 for left, 1 for middle, 2 for right. same index as ClauseSegment.variables

@dataclass(frozen=True)
class ConnectorEdge:
    metadata: ConnectorMetadata
    edge: GridEdge

    @property
    def clause_label(self) -> int:
        return self.metadata.clause_label

    @property
    def variable(self) -> int:
        return self.metadata.variable

    @property
    def slot(self) -> int:
        return self.metadata.slot

    @property
    def segments(self) -> tuple[UnitSegment, ...]:
        return self.edge.segments

    @property
    def start(self) -> Point:
        return self.edge.start

    @property
    def end(self) -> Point:
        return self.edge.end

# PR3SAT data structs
@dataclass(frozen=True)
class VariableSegment:
    var: int
    x_start: int
    x_end: int
    y: int = 0  # this field feels useless xD

@dataclass(frozen=True)
class ClauseSegment:
    label: int
    x_start: int
    x_end: int
    y: int
    variables: tuple[int, int, int]
    # connector_xs: tuple[int, int, int]
    # i know it's inefficient for straight lines, but we should use GridEdge

    @property
    def left(self) -> int:
        return self.variables[0]

    @property
    def middle(self) -> int:
        return self.variables[1]

    @property
    def right(self) -> int:
        return self.variables[2]

@dataclass(frozen=True)
class MRP3SATGraph:
    variables: dict[int, VariableSegment]
    clauses: dict[int, ClauseSegment]
    edges: tuple[ConnectorEdge, ...]



# %%
# ============================================================
# Layout Computation
# ============================================================

def _build_queues(
    clauses: tuple[Clause, ...],
    levels: dict[int, int],
) -> dict[int, list[ConnectorMetadata]]:
    sorted_clauses = sorted(clauses, key=lambda clause: (levels[clause.label], clause.label))

    all_vars = {variable for clause in clauses for variable in clause.variables}
    lefty = {variable: [] for variable in all_vars}
    middle = {variable: [] for variable in all_vars}
    righty = {variable: [] for variable in all_vars}

    for clause in sorted_clauses:
        L, M, R = sorted(clause.variables)

        if L == R:
            # all three variables are the same
            lefty[R].append(ConnectorMetadata(clause.label, R, LEFT))
            lefty[R].append(ConnectorMetadata(clause.label, R, MIDDLE))
            lefty[R].append(ConnectorMetadata(clause.label, R, RIGHT))

        elif L == M:
            # first two variables are the same
            righty[L].insert(0, ConnectorMetadata(clause.label, L, MIDDLE))
            righty[L].insert(0, ConnectorMetadata(clause.label, L, LEFT))
            lefty[R].append(ConnectorMetadata(clause.label, R, RIGHT))

        elif M == R:
            # last two variables are the same
            righty[L].insert(0, ConnectorMetadata(clause.label, L, LEFT))
            lefty[R].append(ConnectorMetadata(clause.label, R, MIDDLE))
            lefty[R].append(ConnectorMetadata(clause.label, R, RIGHT))

        else:
            # all variables are distinct
            righty[L].insert(0, ConnectorMetadata(clause.label, L, LEFT))
            middle[M].append(ConnectorMetadata(clause.label, M, MIDDLE))
            lefty[R].append(ConnectorMetadata(clause.label, R, RIGHT))

    return {
        variable: lefty[variable] + middle[variable] + righty[variable]
        for variable in all_vars
    }
    
def _add_edges_from_queues(
    queues: dict[int, list[ConnectorMetadata]],
    x_start: dict[int, int],
    levels: dict[int, int],
    direction: int,
    connector_xs_by_clause: dict[int, list[int]],
    edges: list[ConnectorEdge],
) -> None:
    for variable, queue in queues.items():
        for idx, metadata in enumerate(queue):
            if metadata.variable != variable:
                raise ValueError(
                    f"Connector for x{metadata.variable} found in x{variable} queue."
                )
            x = x_start[variable] + idx
            y = direction * levels[metadata.clause_label]

            connector_xs_by_clause.setdefault(metadata.clause_label, []).append(x)

            start = Point(x, 0)
            end = Point(x, y)
            edges.append(ConnectorEdge(
                metadata=metadata,
                edge=GridEdge(segments=_unit_segments_between(start, end)),
            ))

def _unit_segments_between(start: Point, end: Point) -> tuple[UnitSegment, ...]:
    """Expand one horizontal or vertical grid path into unit segments."""
    if start.x != end.x and start.y != end.y:
        raise ValueError("SGA edge segments must be horizontal or vertical.")

    if start == end:
        return ()

    segments: list[UnitSegment] = []

    if start.x == end.x:
        step = 1 if end.y > start.y else -1
        for y in range(start.y, end.y, step):
            segments.append(UnitSegment(Point(start.x, y), Point(start.x, y + step)))
    else:
        step = 1 if end.x > start.x else -1
        for x in range(start.x, end.x, step):
            segments.append(UnitSegment(Point(x, start.y), Point(x + step, start.y)))

    return tuple(segments)



# %%
def build_sga_graph(
    instance: MRP3SATInstance,
    positive_levels: dict[int, int],
    negative_levels: dict[int, int],
) -> MRP3SATGraph:
    pos_queues = _build_queues(instance.positive, positive_levels)
    neg_queues = _build_queues(instance.negative, negative_levels)

    segment_width = {
        variable: max(
            1,
            len(pos_queues.get(variable, [])),
            len(neg_queues.get(variable, [])),
        )
        for variable in range(1, instance.n + 1)
    }

    x_start: dict[int, int] = {}
    cursor = 0

    for variable in range(1, instance.n + 1):
        x_start[variable] = cursor
        cursor += segment_width[variable]

    variables = {
        variable: VariableSegment(
            var=variable,
            x_start=x_start[variable],
            x_end=x_start[variable] + segment_width[variable] - 1,
        )
        for variable in range(1, instance.n + 1)
    }

    connector_xs_by_clause: dict[int, list[int]] = {}
    edges: list[ConnectorEdge] = []

    _add_edges_from_queues(
        pos_queues,
        x_start,
        positive_levels,
        direction=1,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    _add_edges_from_queues(
        neg_queues,
        x_start,
        negative_levels,
        direction=-1,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    clauses: dict[int, ClauseSegment] = {}

    for clause in instance.positive:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=positive_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    for clause in instance.negative:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=-negative_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    return MRP3SATGraph(
        variables=variables,
        clauses=clauses,
        edges=tuple(edges),
    )

# %%
# ============================================================
# Matplotlib Renderer
# ============================================================


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_BOX_H = 0.36
_VARROW_COLOUR = "#32cd32"
_POS_COLOUR = "#2166ac"
_NEG_COLOUR = "#b2182b"
_GRIDLINE_COLOUR = "#aaaaaa"

def _graph_bounds(graph: MRP3SATGraph) -> tuple[int, int, int, int]:
    points = []
    for variable in graph.variables.values():
        points.extend([Point(variable.x_start, variable.y), Point(variable.x_end, variable.y)])
    for clause in graph.clauses.values():
        points.extend([Point(clause.x_start, clause.y), Point(clause.x_end, clause.y)])
    for edge in graph.edges:
        points.extend([edge.start, edge.end])
        for segment in edge.segments:
            points.extend([segment.start, segment.end])

    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
    )


def draw_sga_graph(graph: MRP3SATGraph, output: Path | str | None = None):
    """Render an MRP3SATGraph on the square grid. Returns (fig, axis)."""
    min_x, max_x, min_y, max_y = _graph_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + 2) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    for variable in graph.variables.values():
        bx = variable.x_start - 0.4
        bw = variable.x_end - variable.x_start + 0.8
    
        axis.add_patch(FancyBboxPatch(
            (bx, variable.y - _BOX_H / 2),
            bw,
            _BOX_H,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#222222",
            facecolor="white",
            zorder=5,
        ))
    
        axis.text(
            bx + bw / 2,
            variable.y,
            f"x{variable.var}",
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
            zorder=6,
        )

    for clause in graph.clauses.values():
        color = _POS_COLOUR if clause.y > 0 else _NEG_COLOUR
        axis.plot(
            [clause.x_start, clause.x_end],
            [clause.y, clause.y],
            color=color,
            linewidth=2.4,
            zorder=3,
        )
        axis.text(
            (clause.x_start + clause.x_end) / 2,
            clause.y + (0.16 if clause.y > 0 else -0.16),
            _label_str(clause.label),
            ha="center",
            va=("bottom" if clause.y > 0 else "top"),
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    for edge in graph.edges:
        color = _POS_COLOUR if edge.end.y > 0 else _NEG_COLOUR
        for segment in edge.segments:
            axis.plot(
                [segment.start.x, segment.end.x],
                [segment.start.y, segment.end.y],
                color=color,
                linewidth=1.2,
                alpha=0.8,
                zorder=2,
            )
        axis.scatter([edge.end.x], [edge.end.y], color=color, s=18, zorder=4)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x - 1, max_x + 1)
    axis.set_ylim(min_y - 1, max_y + 1)
    axis.set_xticks(range(min_x - 1, max_x + 2))
    axis.set_yticks(range(min_y - 1, max_y + 2))
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.grid(True, color=_GRIDLINE_COLOUR, linewidth=0.6, linestyle="--", zorder=0)
    axis.set_title("Square-Grid-Aligned Monotone Rectilinear Planar 3SAT")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
sga_graph = build_sga_graph(mrp3sat_instance, positive_levels, negative_levels)
fig, axis = draw_sga_graph(sga_graph)
plt.show()


# %% [markdown]
# ## Alternating SGA-MRP3SAT

# %%
def _add_alternating_edges_from_queues(
    queues: dict[int, list[ConnectorMetadata]],
    x_start: dict[int, int],
    levels: dict[int, int],
    direction: int,
    parity_offset: int,
    connector_xs_by_clause: dict[int, list[int]],
    edges: list[ConnectorEdge],
) -> None:
    for variable, queue in queues.items():
        for idx, metadata in enumerate(queue):
            if metadata.variable != variable:
                raise ValueError(
                    f"Connector for x{metadata.variable} found in x{variable} queue."
                )
            x = x_start[variable] + 2 * idx + parity_offset
            y = direction * levels[metadata.clause_label]

            connector_xs_by_clause.setdefault(metadata.clause_label, []).append(x)

            start = Point(x, 0)
            end = Point(x, y)
            edges.append(ConnectorEdge(
                metadata=metadata,
                edge=GridEdge(segments=_unit_segments_between(start, end)),
            ))



# %%
def build_alternating_sga_graph(
    instance: MRP3SATInstance,
    positive_levels: dict[int, int],
    negative_levels: dict[int, int],
) -> MRP3SATGraph:
    pos_queues = _build_queues(instance.positive, positive_levels)
    neg_queues = _build_queues(instance.negative, negative_levels)

    slot_count = {
        variable: max(
            1,
            len(pos_queues.get(variable, [])),
            len(neg_queues.get(variable, [])),
        )
        for variable in range(1, instance.n + 1)
    }

    segment_width = {
        variable: 2 * slot_count[variable]
        for variable in range(1, instance.n + 1)
    }

    x_start: dict[int, int] = {}
    cursor = 0

    for variable in range(1, instance.n + 1):
        x_start[variable] = cursor
        cursor += segment_width[variable]

    variables = {
        variable: VariableSegment(
            var=variable,
            x_start=x_start[variable],
            x_end=x_start[variable] + segment_width[variable] - 1,
        )
        for variable in range(1, instance.n + 1)
    }

    connector_xs_by_clause: dict[int, list[int]] = {}
    edges: list[ConnectorEdge] = []

    _add_alternating_edges_from_queues(
        pos_queues,
        x_start,
        positive_levels,
        direction=1,
        parity_offset=0,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    _add_alternating_edges_from_queues(
        neg_queues,
        x_start,
        negative_levels,
        direction=-1,
        parity_offset=1,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    clauses: dict[int, ClauseSegment] = {}

    for clause in instance.positive:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=positive_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    for clause in instance.negative:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=-negative_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    return MRP3SATGraph(
        variables=variables,
        clauses=clauses,
        edges=tuple(edges),
    )


# %%
alternating_sga_graph = build_alternating_sga_graph(
    mrp3sat_instance,
    positive_levels,
    negative_levels,
)
fig, axis = draw_sga_graph(alternating_sga_graph)
axis.set_title("Alternating Square-Grid-Aligned Monotone Rectilinear Planar 3SAT")
plt.show()

# %% [markdown]
# ## (Alternating) TGA-3SAT

# %%
_TRIANGLE_GRID_STEP = math.sqrt(3) / 2


def _triangle_x(point: Point) -> float:
    return point.x + (0.5 if point.y % 2 else 0.0)


def _triangle_y(point: Point) -> float:
    return point.y * _TRIANGLE_GRID_STEP


def _triangle_xy(point: Point) -> tuple[float, float]:
    return _triangle_x(point), _triangle_y(point)


def _add_tga_edges_from_queues(
    queues: dict[int, list[ConnectorMetadata]],
    x_start: dict[int, int],
    levels: dict[int, int],
    direction: int,
    parity_offset: int,
    connector_xs_by_clause: dict[int, list[int]],
    edges: list[ConnectorEdge],
) -> None:
    for variable, queue in queues.items():
        for idx, metadata in enumerate(queue):
            if metadata.variable != variable:
                raise ValueError(
                    f"Connector for x{metadata.variable} found in x{variable} queue."
                )
            x = x_start[variable] + 2 * idx + parity_offset
            y = direction * 2 * levels[metadata.clause_label]

            connector_xs_by_clause.setdefault(metadata.clause_label, []).append(x)

            step = 1 if y > 0 else -1
            segments = tuple(
                UnitSegment(Point(x, row), Point(x, row + step))
                for row in range(0, y, step)
            )
            edges.append(ConnectorEdge(
                metadata=metadata,
                edge=GridEdge(segments=segments),
            ))


# %%
def build_tga_graph(
    instance: MRP3SATInstance,
    positive_levels: dict[int, int],
    negative_levels: dict[int, int],
) -> MRP3SATGraph:
    pos_queues = _build_queues(instance.positive, positive_levels)
    neg_queues = _build_queues(instance.negative, negative_levels)

    slot_count = {
        variable: max(
            1,
            len(pos_queues.get(variable, [])),
            len(neg_queues.get(variable, [])),
        )
        for variable in range(1, instance.n + 1)
    }

    segment_width = {
        variable: 2 * slot_count[variable]
        for variable in range(1, instance.n + 1)
    }

    x_start: dict[int, int] = {}
    cursor = 0

    for variable in range(1, instance.n + 1):
        x_start[variable] = cursor
        cursor += segment_width[variable]

    variables = {
        variable: VariableSegment(
            var=variable,
            x_start=x_start[variable],
            x_end=x_start[variable] + segment_width[variable] - 1,
        )
        for variable in range(1, instance.n + 1)
    }

    connector_xs_by_clause: dict[int, list[int]] = {}
    edges: list[ConnectorEdge] = []

    _add_tga_edges_from_queues(
        pos_queues,
        x_start,
        positive_levels,
        direction=1,
        parity_offset=0,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    _add_tga_edges_from_queues(
        neg_queues,
        x_start,
        negative_levels,
        direction=-1,
        parity_offset=1,
        connector_xs_by_clause=connector_xs_by_clause,
        edges=edges,
    )

    clauses: dict[int, ClauseSegment] = {}

    for clause in instance.positive:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=2 * positive_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    for clause in instance.negative:
        xs = connector_xs_by_clause[clause.label]

        clauses[clause.label] = ClauseSegment(
            label=clause.label,
            x_start=min(xs),
            x_end=max(xs),
            y=-2 * negative_levels[clause.label],
            variables=tuple(sorted(clause.variables)),
        )

    return MRP3SATGraph(
        variables=variables,
        clauses=clauses,
        edges=tuple(edges),
    )


# %%
def _triangle_graph_bounds(graph: MRP3SATGraph) -> tuple[float, float, float, float, int, int]:
    min_x, max_x, min_y, max_y = _graph_bounds(graph)
    row_min = min_y - 1
    row_max = max_y + 1

    points = []
    for x in range(min_x - 1, max_x + 2):
        for y in range(row_min, row_max + 1):
            points.append(_triangle_xy(Point(x, y)))

    return (
        min(x for x, _ in points),
        max(x for x, _ in points),
        min(y for _, y in points),
        max(y for _, y in points),
        row_min,
        row_max,
    )


def _draw_triangle_grid(
    axis,
    x_min: float,
    x_max: float,
    row_min: int,
    row_max: int,
    color: str = _GRIDLINE_COLOUR,
    linewidth: float = 0.6,
    linestyle: str = "--",
) -> None:
    """
    Draw the three parallel line families of the triangular grid.

    The rendered grid has side length 1:
    - horizontal rows at y = row * sqrt(3) / 2
    - slope +sqrt(3) lines where x = a + row / 2
    - slope -sqrt(3) lines where x = b - row / 2
    """
    y_bot = row_min * _TRIANGLE_GRID_STEP
    y_top = row_max * _TRIANGLE_GRID_STEP
    kw = dict(color=color, linewidth=linewidth, linestyle=linestyle, zorder=0)

    for row in range(row_min, row_max + 1):
        axis.axhline(row * _TRIANGLE_GRID_STEP, **kw)

    a_min = math.floor(x_min - row_max / 2) - 1
    a_max = math.ceil(x_max - row_min / 2) + 1
    for a in range(a_min, a_max + 1):
        axis.plot(
            [a + row_min / 2, a + row_max / 2],
            [y_bot, y_top],
            **kw,
        )

    b_min = math.floor(x_min + row_min / 2) - 1
    b_max = math.ceil(x_max + row_max / 2) + 1
    for b in range(b_min, b_max + 1):
        axis.plot(
            [b - row_min / 2, b - row_max / 2],
            [y_bot, y_top],
            **kw,
        )


def draw_tga_graph(graph: MRP3SATGraph, output: Path | str | None = None):
    """Render an MRP3SATGraph on the triangle grid. Returns (fig, axis)."""
    min_i, max_i, _, _ = _graph_bounds(graph)
    min_x, max_x, min_y, max_y, row_min, row_max = _triangle_graph_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + _TRIANGLE_GRID_STEP) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    _draw_triangle_grid(axis, min_x, max_x, row_min, row_max)

    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    for variable in graph.variables.values():
        center_y = _triangle_y(Point(0, variable.y))
        bx = _triangle_x(Point(variable.x_start, variable.y)) - 0.4
        bw = variable.x_end - variable.x_start + 0.8

        axis.add_patch(FancyBboxPatch(
            (bx, center_y - _BOX_H / 2),
            bw,
            _BOX_H,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#222222",
            facecolor="white",
            zorder=5,
        ))

        axis.text(
            bx + bw / 2,
            center_y,
            f"x{variable.var}",
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
            zorder=6,
        )

    for clause in graph.clauses.values():
        color = _POS_COLOUR if clause.y > 0 else _NEG_COLOUR
        start = _triangle_xy(Point(clause.x_start, clause.y))
        end = _triangle_xy(Point(clause.x_end, clause.y))
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2.4,
            zorder=3,
        )
        axis.text(
            (start[0] + end[0]) / 2,
            start[1] + (0.16 if clause.y > 0 else -0.16),
            _label_str(clause.label),
            ha="center",
            va=("bottom" if clause.y > 0 else "top"),
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    for edge in graph.edges:
        color = _POS_COLOUR if edge.end.y > 0 else _NEG_COLOUR
        for segment in edge.segments:
            start = _triangle_xy(segment.start)
            end = _triangle_xy(segment.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=1.2,
                alpha=0.8,
                zorder=2,
            )
        end = _triangle_xy(edge.end)
        axis.scatter([end[0]], [end[1]], color=color, s=18, zorder=4)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_yticks([row * _TRIANGLE_GRID_STEP for row in range(row_min, row_max + 1)])
    axis.set_yticklabels([str(row) for row in range(row_min, row_max + 1)], fontsize=7)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.set_title("Triangle-Grid-Aligned Monotone Rectilinear Planar 3SAT")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
tga_graph = build_tga_graph(
    mrp3sat_instance,
    positive_levels,
    negative_levels,
)
fig, axis = draw_tga_graph(tga_graph)
plt.show()

# %% [markdown]
# ## Rounded TGA-MRP3SAT

# %%
def _copy_grid_edge(edge: GridEdge) -> GridEdge:
    return GridEdge(
        segments=tuple(
            UnitSegment(
                Point(segment.start.x, segment.start.y),
                Point(segment.end.x, segment.end.y),
            )
            for segment in edge.segments
        ),
    )


def _copy_connector_metadata(metadata: ConnectorMetadata) -> ConnectorMetadata:
    return ConnectorMetadata(
        clause_label=metadata.clause_label,
        variable=metadata.variable,
        slot=metadata.slot,
    )


def _copy_connector_edge(connector: ConnectorEdge) -> ConnectorEdge:
    return ConnectorEdge(
        metadata=_copy_connector_metadata(connector.metadata),
        edge=_copy_grid_edge(connector.edge),
    )


def _round_left_connector(connector: ConnectorEdge) -> ConnectorEdge:
    if connector.slot != LEFT:
        return _copy_connector_edge(connector)

    if not connector.segments:
        raise ValueError(
            f"Left connector for {_label_str(connector.clause_label)} has no segments."
        )

    segments = list(_copy_grid_edge(connector.edge).segments)
    last_segment = segments[-1]
    segments[-1] = UnitSegment(
        start=last_segment.start,
        end=Point(last_segment.end.x + 1, last_segment.end.y),
    )

    return ConnectorEdge(
        metadata=_copy_connector_metadata(connector.metadata),
        edge=GridEdge(segments=tuple(segments)),
    )



# %%
def build_rounded_tga_graph(graph: MRP3SATGraph) -> MRP3SATGraph:
    variables = {
        variable: VariableSegment(
            var=segment.var,
            x_start=segment.x_start,
            x_end=segment.x_end,
            y=segment.y,
        )
        for variable, segment in graph.variables.items()
    }

    clauses = {
        label: ClauseSegment(
            label=clause.label,
            x_start=clause.x_start + 1,
            x_end=clause.x_end,
            y=clause.y,
            variables=clause.variables,
        )
        for label, clause in graph.clauses.items()
    }

    edges = tuple(
        _round_left_connector(edge)
        for edge in graph.edges
    )

    return MRP3SATGraph(
        variables=variables,
        clauses=clauses,
        edges=edges,
    )



# %%
rounded_tga_graph = build_rounded_tga_graph(tga_graph)
fig, axis = draw_tga_graph(rounded_tga_graph)
axis.set_title("Rounded Triangle-Grid-Aligned Monotone Rectilinear Planar 3SAT")
plt.show()

# %% [markdown]
# ## TGA-PVC
# with max degree of 3
#

# %%
# GRAPH DATA STRUCTURES

@dataclass(frozen=True)
class VariableGadget:
    variable: int
    verticies: tuple[Point, ...]
    edges: tuple[GridEdge, ...]


@dataclass(frozen=True)
class ClauseGadget:
    label: int
    left: Point
    middle: Point
    right: Point
    edges: tuple[GridEdge, ...]


@dataclass(frozen=True)
class TGAConnectorGadget:
    clause_label: int
    variable: int
    slot: int
    edge: GridEdge


@dataclass(frozen=True)
class TGAPVCGraph:
    variable_gadgets: dict[int, VariableGadget]
    clause_gadgets: dict[int, ClauseGadget]
    connector_gadgets: tuple[TGAConnectorGadget, ...]
    k: int  # can we find a vertex cover of size k?



# %%

def _single_segment_edge(start: Point, end: Point) -> GridEdge:
    return GridEdge(segments=(UnitSegment(start, end),))


def _path_points(edge: GridEdge) -> tuple[Point, ...]:
    if not edge.segments:
        return ()

    points = [edge.segments[0].start]
    for segment in edge.segments:
        points.append(segment.end)
    return tuple(points)


def _concat_segments(*segment_groups: tuple[UnitSegment, ...]) -> tuple[UnitSegment, ...]:
    segments: list[UnitSegment] = []

    for group in segment_groups:
        if not group:
            continue
        if segments and segments[-1].end != group[0].start:
            raise ValueError(
                f"Cannot join grid paths at {segments[-1].end} and {group[0].start}."
            )
        segments.extend(group)

    return tuple(segments)


def _connector_edge_to_corner(
    occurrence_edge: GridEdge,
    corner: Point,
) -> GridEdge:
    if occurrence_edge.end == corner:
        return occurrence_edge

    tail = _unit_segments_between(occurrence_edge.end, corner)
    return GridEdge(
        segments=_concat_segments(occurrence_edge.segments, tail),
    )

def _connectors_by_clause_and_slot(
    graph: MRP3SATGraph,
) -> dict[int, dict[int, ConnectorEdge]]:
    connectors_by_clause = {
        label: {}
        for label in graph.clauses
    }

    for connector in graph.edges:
        if connector.clause_label not in connectors_by_clause:
            raise ValueError(
                f"Connector references unknown clause {_label_str(connector.clause_label)}."
            )
        if connector.slot not in (LEFT, MIDDLE, RIGHT):
            raise ValueError(
                f"Connector for {_label_str(connector.clause_label)} has invalid slot {connector.slot}."
            )

        clause_connectors = connectors_by_clause[connector.clause_label]
        if connector.slot in clause_connectors:
            raise ValueError(
                f"Clause {_label_str(connector.clause_label)} has multiple slot {connector.slot} connectors."
            )
        clause_connectors[connector.slot] = connector

    expected_slots = {LEFT, MIDDLE, RIGHT}
    for clause_label, clause_connectors in connectors_by_clause.items():
        if set(clause_connectors) != expected_slots:
            roles = ", ".join(
                str(slot)
                for slot in sorted(clause_connectors)
            )
            raise ValueError(
                f"Clause {_label_str(clause_label)} has connector slots {{{roles}}}, expected 0/1/2."
            )

    return connectors_by_clause



# %%
# CONVERTER

def convert_rounded_tga_to_pvc(graph: MRP3SATGraph) -> TGAPVCGraph:
    variable_gadgets: dict[int, VariableGadget] = {}

    for variable in graph.variables.values():
        vertices = tuple(
            Point(x, variable.y)
            for x in range(variable.x_start, variable.x_end + 1)
        )
        edges = tuple(
            _single_segment_edge(vertices[i], vertices[i + 1])
            for i in range(len(vertices) - 1)
        )
        variable_gadgets[variable.var] = VariableGadget(
            variable=variable.var,
            verticies=vertices,
            edges=edges,
        )

    connectors_by_clause = _connectors_by_clause_and_slot(graph)

    clause_gadgets: dict[int, ClauseGadget] = {}
    connector_gadgets: list[TGAConnectorGadget] = []

    for clause_label, occurrence_by_slot in sorted(connectors_by_clause.items()):
        left_occurrence = occurrence_by_slot[LEFT]
        middle_occurrence = occurrence_by_slot[MIDDLE]
        right_occurrence = occurrence_by_slot[RIGHT]

        left_corner = middle_occurrence.end
        middle_corner = middle_occurrence.segments[-1].start
        right_corner = Point(left_corner.x + 1, left_corner.y)

        clause_gadgets[clause_label] = ClauseGadget(
            label=clause_label,
            left=left_corner,
            middle=middle_corner,
            right=right_corner,
            edges=(
                _single_segment_edge(left_corner, middle_corner),
                _single_segment_edge(left_corner, right_corner),
                _single_segment_edge(middle_corner, right_corner),
            ),
        )

        connector_specs = (
            (left_occurrence, left_corner),
            (middle_occurrence, middle_corner),
            (right_occurrence, right_corner),
        )

        for occurrence, corner in connector_specs:
            if occurrence.slot == MIDDLE:
                connector_edge = GridEdge(
                    segments=occurrence.segments[:-1],
                )
            else:
                connector_edge = _connector_edge_to_corner(occurrence.edge, corner)

            connector_gadgets.append(TGAConnectorGadget(
                clause_label=occurrence.clause_label,
                variable=occurrence.variable,
                slot=occurrence.slot,
                edge=connector_edge,
            ))

    variable_vertex_count = sum(
        len(gadget.verticies)
        for gadget in variable_gadgets.values()
    )
    k = variable_vertex_count // 2 + 2 * len(clause_gadgets)

    return TGAPVCGraph(
        variable_gadgets=variable_gadgets,
        clause_gadgets=clause_gadgets,
        connector_gadgets=tuple(connector_gadgets),
        k=k,
    )



# %%
# TEXT OUTPUT


def summarize_tga_pvc_graph(graph: TGAPVCGraph) -> None:
    variable_vertex_count = sum(
        len(gadget.verticies)
        for gadget in graph.variable_gadgets.values()
    )
    variable_edge_count = sum(
        len(gadget.edges)
        for gadget in graph.variable_gadgets.values()
    )
    clause_edge_count = sum(
        len(gadget.edges)
        for gadget in graph.clause_gadgets.values()
    )

    warnings: list[str] = []

    odd_variables = [
        gadget.variable
        for gadget in graph.variable_gadgets.values()
        if len(gadget.verticies) % 2 != 0
    ]
    if odd_variables:
        warnings.append(f"odd variable gadget vertex counts: {odd_variables}")

    malformed_clauses = [
        _label_str(gadget.label)
        for gadget in graph.clause_gadgets.values()
        if len({gadget.left, gadget.middle, gadget.right}) != 3
        or len(gadget.edges) != 3
    ]
    if malformed_clauses:
        warnings.append(f"malformed clause triangles: {malformed_clauses}")

    odd_connectors = [
        f"{_label_str(gadget.clause_label)}:{gadget.slot}:x{gadget.variable}"
        for gadget in graph.connector_gadgets
        if max(0, len(_path_points(gadget.edge)) - 2) % 2 != 0
    ]
    if odd_connectors:
        warnings.append(f"odd connector interior point counts: {odd_connectors}")

    print(
        "TGA-PVC: "
        f"{len(graph.variable_gadgets)} variable gadgets, "
        f"{variable_vertex_count} variable vertices, "
        f"{variable_edge_count} variable edges, "
        f"{len(graph.clause_gadgets)} clause gadgets, "
        f"{clause_edge_count} clause edges, "
        f"{len(graph.connector_gadgets)} connector gadgets, "
        f"k={graph.k}"
    )

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No TGA-PVC warnings.")



# %%
# DRAWING

def _tga_pvc_bounds(graph: TGAPVCGraph) -> tuple[float, float, float, float, int, int]:
    points: list[Point] = []

    for gadget in graph.variable_gadgets.values():
        points.extend(gadget.verticies)

    for gadget in graph.clause_gadgets.values():
        points.extend([gadget.left, gadget.middle, gadget.right])

    for gadget in graph.connector_gadgets:
        for segment in gadget.edge.segments:
            points.extend([segment.start, segment.end])

    xs, ys = zip(*(_triangle_xy(point) for point in points))
    rows = [point.y for point in points]
    return (
        min(xs) - 1.0,
        max(xs) + 1.0,
        min(ys) - _TRIANGLE_GRID_STEP,
        max(ys) + _TRIANGLE_GRID_STEP,
        min(rows) - 1,
        max(rows) + 1,
    )


def _draw_pvc_vertex(
    axis,
    point: Point,
    *,
    color: str,
    filled: bool,
    size: float,
    zorder: int,
) -> None:
    x, y = _triangle_xy(point)

    if filled:
        axis.scatter(
            [x], [y],
            s=size,
            color=color,
            linewidths=1.2,
            zorder=zorder,
        )
    else:
        axis.scatter(
            [x], [y],
            s=size,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=zorder,
        )


def draw_tga_pvc_graph(graph: TGAPVCGraph, output: Path | str | None = None):
    min_x, max_x, min_y, max_y, row_min, row_max = _tga_pvc_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + _TRIANGLE_GRID_STEP) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    _draw_triangle_grid(axis, min_x, max_x, row_min, row_max)
    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    for connector in graph.connector_gadgets:
        color = _POS_COLOUR if connector.edge.end.y > 0 else _NEG_COLOUR
        for segment in connector.edge.segments:
            start = _triangle_xy(segment.start)
            end = _triangle_xy(segment.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=1.2,
                alpha=0.65,
                zorder=2,
            )

    for gadget in graph.variable_gadgets.values():
        for edge in gadget.edges:
            start = _triangle_xy(edge.start)
            end = _triangle_xy(edge.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#222222",
                linewidth=1.6,
                zorder=3,
            )

        for index, vertex in enumerate(gadget.verticies):
            _draw_pvc_vertex(
                axis,
                vertex,
                color="#222222",
                filled=(index % 2 == 0),
                size=42,
                zorder=5,
            )

        variable_start_x, variable_start_y = _triangle_xy(gadget.verticies[0])
        variable_end_x, variable_end_y = _triangle_xy(gadget.verticies[-1])
        axis.text(
            (variable_start_x + variable_end_x) / 2,
            (variable_start_y + variable_end_y) / 2 - 0.28,
            f"x{gadget.variable}",
            ha="center",
            va="top",
            fontsize=8,
            color="#222222",
            zorder=6,
        )

    for gadget in graph.clause_gadgets.values():
        color = _POS_COLOUR if gadget.left.y > 0 else _NEG_COLOUR
        for edge in gadget.edges:
            start = _triangle_xy(edge.start)
            end = _triangle_xy(edge.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=2.0,
                zorder=4,
            )

        corners = (gadget.left, gadget.middle, gadget.right)
        for corner in corners:
            _draw_pvc_vertex(
                axis,
                corner,
                color=color,
                filled=(gadget.left.y < 0),
                size=46,
                zorder=6,
            )

        left_x, left_y = _triangle_xy(gadget.left)
        right_x, right_y = _triangle_xy(gadget.right)
        axis.text(
            (left_x + right_x) / 2,
            (left_y + right_y) / 2 + (0.18 if gadget.left.y > 0 else -0.18),
            _label_str(gadget.label),
            ha="center",
            va=("bottom" if gadget.left.y > 0 else "top"),
            fontsize=10,
            color=color,
            fontweight="bold",
            zorder=7,
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_yticks([row * _TRIANGLE_GRID_STEP for row in range(row_min, row_max + 1)])
    axis.set_yticklabels([str(row) for row in range(row_min, row_max + 1)], fontsize=7)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.set_title(f"Triangle-Grid-Aligned Planar Vertex Cover (k = {graph.k})")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
tga_pvc_graph = convert_rounded_tga_to_pvc(rounded_tga_graph)
summarize_tga_pvc_graph(tga_pvc_graph)
fig, axis = draw_tga_pvc_graph(tga_pvc_graph)
plt.show()

# %% [markdown]
# ## TGE-PVC
# with max degree 3

# %%
# GRAPH DATA STRUCTURES

@dataclass(frozen=True)
class TGEConnectorGadget:
    clause_label: int
    variable: int
    slot: int
    verticies: tuple[Point, ...]
    edges: tuple[GridEdge, ...]


@dataclass(frozen=True)
class TGEPVCGraph:
    variable_gadgets: dict[int, VariableGadget]
    clause_gadgets: dict[int, ClauseGadget]
    connector_gadgets: tuple[TGEConnectorGadget, ...]
    k: int  # can we find a vertex cover of size k?



# %%

def _copy_point(point: Point) -> Point:
    return Point(point.x, point.y)


def _copy_unit_segment(segment: UnitSegment) -> UnitSegment:
    return UnitSegment(
        start=_copy_point(segment.start),
        end=_copy_point(segment.end),
    )


def _copy_variable_gadget(gadget: VariableGadget) -> VariableGadget:
    return VariableGadget(
        variable=gadget.variable,
        verticies=tuple(_copy_point(point) for point in gadget.verticies),
        edges=tuple(_copy_grid_edge(edge) for edge in gadget.edges),
    )


def _copy_clause_gadget(gadget: ClauseGadget) -> ClauseGadget:
    return ClauseGadget(
        label=gadget.label,
        left=_copy_point(gadget.left),
        middle=_copy_point(gadget.middle),
        right=_copy_point(gadget.right),
        edges=tuple(_copy_grid_edge(edge) for edge in gadget.edges),
    )


def _subdivide_connector_gadget(gadget: TGAConnectorGadget) -> TGEConnectorGadget:
    points = _path_points(gadget.edge)
    interior_verticies = tuple(
        _copy_point(point)
        for point in points[1:-1]
    )
    unit_edges = tuple(
        GridEdge(segments=(_copy_unit_segment(segment),))
        for segment in gadget.edge.segments
    )

    return TGEConnectorGadget(
        clause_label=gadget.clause_label,
        variable=gadget.variable,
        slot=gadget.slot,
        verticies=interior_verticies,
        edges=unit_edges,
    )



# %%
# CONVERTER

def convert_tga_pvc_to_tge_pvc(graph: TGAPVCGraph) -> TGEPVCGraph:
    variable_gadgets = {
        variable: _copy_variable_gadget(gadget)
        for variable, gadget in graph.variable_gadgets.items()
    }
    clause_gadgets = {
        label: _copy_clause_gadget(gadget)
        for label, gadget in graph.clause_gadgets.items()
    }
    connector_gadgets = tuple(
        _subdivide_connector_gadget(gadget)
        for gadget in graph.connector_gadgets
    )

    connector_vertex_count = sum(
        len(gadget.verticies)
        for gadget in connector_gadgets
    )
    k = graph.k + connector_vertex_count // 2

    return TGEPVCGraph(
        variable_gadgets=variable_gadgets,
        clause_gadgets=clause_gadgets,
        connector_gadgets=connector_gadgets,
        k=k,
    )



# %%
# TEXT OUTPUT

def summarize_tge_pvc_graph(graph: TGEPVCGraph) -> None:
    variable_vertex_count = sum(
        len(gadget.verticies)
        for gadget in graph.variable_gadgets.values()
    )
    variable_edge_count = sum(
        len(gadget.edges)
        for gadget in graph.variable_gadgets.values()
    )
    clause_vertex_count = 3 * len(graph.clause_gadgets)
    clause_edge_count = sum(
        len(gadget.edges)
        for gadget in graph.clause_gadgets.values()
    )
    connector_vertex_count = sum(
        len(gadget.verticies)
        for gadget in graph.connector_gadgets
    )
    connector_edge_count = sum(
        len(gadget.edges)
        for gadget in graph.connector_gadgets
    )

    warnings: list[str] = []

    odd_connectors = [
        f"{_label_str(gadget.clause_label)}:{gadget.slot}:x{gadget.variable}"
        for gadget in graph.connector_gadgets
        if len(gadget.verticies) % 2 != 0
    ]
    if odd_connectors:
        warnings.append(f"odd connector vertex counts: {odd_connectors}")

    non_unit_edges = [
        f"x{gadget.variable}"
        for gadget in graph.variable_gadgets.values()
        for edge in gadget.edges
        if len(edge.segments) != 1
    ]
    non_unit_edges.extend(
        _label_str(gadget.label)
        for gadget in graph.clause_gadgets.values()
        for edge in gadget.edges
        if len(edge.segments) != 1
    )
    non_unit_edges.extend(
        f"{_label_str(gadget.clause_label)}:{gadget.slot}:x{gadget.variable}"
        for gadget in graph.connector_gadgets
        for edge in gadget.edges
        if len(edge.segments) != 1
    )
    if non_unit_edges:
        warnings.append(f"non-unit edges: {non_unit_edges}")

    print(
        "TGE-PVC: "
        f"{len(graph.variable_gadgets)} variable gadgets, "
        f"{variable_vertex_count} variable vertices, "
        f"{variable_edge_count} variable edges, "
        f"{len(graph.clause_gadgets)} clause gadgets, "
        f"{clause_vertex_count} clause vertices, "
        f"{clause_edge_count} clause edges, "
        f"{len(graph.connector_gadgets)} connector gadgets, "
        f"{connector_vertex_count} connector vertices, "
        f"{connector_edge_count} connector edges, "
        f"k={graph.k}"
    )

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No TGE-PVC warnings.")



# %%
# DRAWING

def _tge_pvc_bounds(graph: TGEPVCGraph) -> tuple[float, float, float, float, int, int]:
    points: list[Point] = []

    for gadget in graph.variable_gadgets.values():
        points.extend(gadget.verticies)

    for gadget in graph.clause_gadgets.values():
        points.extend([gadget.left, gadget.middle, gadget.right])

    for gadget in graph.connector_gadgets:
        points.extend(gadget.verticies)
        for edge in gadget.edges:
            for segment in edge.segments:
                points.extend([segment.start, segment.end])

    xs, ys = zip(*(_triangle_xy(point) for point in points))
    rows = [point.y for point in points]
    return (
        min(xs) - 1.0,
        max(xs) + 1.0,
        min(ys) - _TRIANGLE_GRID_STEP,
        max(ys) + _TRIANGLE_GRID_STEP,
        min(rows) - 1,
        max(rows) + 1,
    )

def draw_tge_pvc_graph(graph: TGEPVCGraph, output: Path | str | None = None):
    min_x, max_x, min_y, max_y, row_min, row_max = _tge_pvc_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + _TRIANGLE_GRID_STEP) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    _draw_triangle_grid(axis, min_x, max_x, row_min, row_max)
    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    for connector in graph.connector_gadgets:
        first_edge = connector.edges[0]
        color = _POS_COLOUR if first_edge.end.y > 0 else _NEG_COLOUR
        for edge in connector.edges:
            segment = edge.segments[0]
            start = _triangle_xy(segment.start)
            end = _triangle_xy(segment.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=1.2,
                alpha=0.65,
                zorder=2,
            )

        for index, vertex in enumerate(connector.verticies):
            # bipartite drawing:
            # if y > 0, then connector connects from solid circle on variable bar, so start with hollow fill
            # if y < 0, then connector connects from hollow circle on variable bar, so start with solid fill
            if vertex.y > 0:
                is_filled = (index % 2 == 1)
            elif vertex.y < 0:
                is_filled = (index % 2 == 0)
            else:
                raise ValueError("Connector vertex cannot be on the variable row.")
            _draw_pvc_vertex(
                axis,
                vertex,
                color=color,
                filled=is_filled,
                size=32,
                zorder=5,
            )

    for gadget in graph.variable_gadgets.values():
        for edge in gadget.edges:
            start = _triangle_xy(edge.start)
            end = _triangle_xy(edge.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#222222",
                linewidth=1.6,
                zorder=3,
            )

        for index, vertex in enumerate(gadget.verticies):
            _draw_pvc_vertex(
                axis,
                vertex,
                color="#222222",
                filled=(index % 2 == 0),
                size=42,
                zorder=6,
            )

        variable_start_x, variable_start_y = _triangle_xy(gadget.verticies[0])
        variable_end_x, variable_end_y = _triangle_xy(gadget.verticies[-1])
        axis.text(
            (variable_start_x + variable_end_x) / 2,
            (variable_start_y + variable_end_y) / 2 - 0.28,
            f"x{gadget.variable}",
            ha="center",
            va="top",
            fontsize=8,
            color="#222222",
            zorder=7,
        )

    for gadget in graph.clause_gadgets.values():
        color = _POS_COLOUR if gadget.left.y > 0 else _NEG_COLOUR
        for edge in gadget.edges:
            start = _triangle_xy(edge.start)
            end = _triangle_xy(edge.end)
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=2.0,
                zorder=4,
            )

        corners = (gadget.left, gadget.middle, gadget.right)
        for corner in corners:
            _draw_pvc_vertex(
                axis,
                corner,
                color=color,
                filled=(gadget.left.y < 0),
                size=46,
                zorder=6,
            )

        left_x, left_y = _triangle_xy(gadget.left)
        right_x, right_y = _triangle_xy(gadget.right)
        axis.text(
            (left_x + right_x) / 2,
            (left_y + right_y) / 2 + (0.18 if gadget.left.y > 0 else -0.18),
            _label_str(gadget.label),
            ha="center",
            va=("bottom" if gadget.left.y > 0 else "top"),
            fontsize=10,
            color=color,
            fontweight="bold",
            zorder=7,
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_yticks([row * _TRIANGLE_GRID_STEP for row in range(row_min, row_max + 1)])
    axis.set_yticklabels([str(row) for row in range(row_min, row_max + 1)], fontsize=7)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.set_title(f"Triangle-Grid-Embedded Planar Vertex Cover (k = {graph.k})")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
tge_pvc_graph = convert_tga_pvc_to_tge_pvc(tga_pvc_graph)
summarize_tge_pvc_graph(tge_pvc_graph)
fig, axis = draw_tge_pvc_graph(tge_pvc_graph)
plt.show()

# %% [markdown]
# ## BTD

# %%
# GRAPH DATA STRUCTURES

@dataclass(frozen=True)
class TrackSegment:
    start: Point
    end: Point
    start_degree: int
    end_degree: int


@dataclass(frozen=True)
class BTDVariableGadget:
    variable: int
    verticies: tuple[Point, ...]
    tracks: tuple[TrackSegment, ...]


@dataclass(frozen=True)
class BTDClauseGadget:
    label: int
    left: Point
    middle: Point
    right: Point
    tracks: tuple[TrackSegment, ...]


@dataclass(frozen=True)
class BTDConnectorGadget:
    clause_label: int
    variable: int
    slot: int
    verticies: tuple[Point, ...]
    tracks: tuple[TrackSegment, ...]


@dataclass(frozen=True)
class BTDGraph:
    variable_gadgets: dict[int, BTDVariableGadget]
    clause_gadgets: dict[int, BTDClauseGadget]
    connector_gadgets: tuple[BTDConnectorGadget, ...]
    k: int



# %%
# CONVERTER

def _segments_from_grid_edges(edges: Iterable[GridEdge]) -> tuple[UnitSegment, ...]:
    return tuple(
        segment
        for edge in edges
        for segment in edge.segments
    )


def _track_from_segment(
    segment: UnitSegment,
    degree: dict[Point, int],
) -> TrackSegment:
    return TrackSegment(
        start=_copy_point(segment.start),
        end=_copy_point(segment.end),
        start_degree=degree.get(segment.start, 0),
        end_degree=degree.get(segment.end, 0),
    )


def _tracks_from_grid_edges(
    edges: Iterable[GridEdge],
    degree: dict[Point, int],
) -> tuple[TrackSegment, ...]:
    return tuple(
        _track_from_segment(segment, degree)
        for segment in _segments_from_grid_edges(edges)
    )


def _tge_pvc_degree(graph: TGEPVCGraph) -> dict[Point, int]:
    degree: dict[Point, int] = {}

    edge_groups = []
    edge_groups.extend(gadget.edges for gadget in graph.variable_gadgets.values())
    edge_groups.extend(gadget.edges for gadget in graph.clause_gadgets.values())
    edge_groups.extend(gadget.edges for gadget in graph.connector_gadgets)

    for edges in edge_groups:
        for segment in _segments_from_grid_edges(edges):
            degree[segment.start] = degree.get(segment.start, 0) + 1
            degree[segment.end] = degree.get(segment.end, 0) + 1

    return degree


def convert_tge_pvc_to_btd(graph: TGEPVCGraph) -> BTDGraph:
    degree = _tge_pvc_degree(graph)

    variable_gadgets = {
        variable: BTDVariableGadget(
            variable=gadget.variable,
            verticies=tuple(_copy_point(vertex) for vertex in gadget.verticies),
            tracks=_tracks_from_grid_edges(gadget.edges, degree),
        )
        for variable, gadget in graph.variable_gadgets.items()
    }

    clause_gadgets = {
        label: BTDClauseGadget(
            label=gadget.label,
            left=_copy_point(gadget.left),
            middle=_copy_point(gadget.middle),
            right=_copy_point(gadget.right),
            tracks=_tracks_from_grid_edges(gadget.edges, degree),
        )
        for label, gadget in graph.clause_gadgets.items()
    }

    connector_gadgets = tuple(
        BTDConnectorGadget(
            clause_label=gadget.clause_label,
            variable=gadget.variable,
            slot=gadget.slot,
            verticies=tuple(_copy_point(vertex) for vertex in gadget.verticies),
            tracks=_tracks_from_grid_edges(gadget.edges, degree),
        )
        for gadget in graph.connector_gadgets
    )

    return BTDGraph(
        variable_gadgets=variable_gadgets,
        clause_gadgets=clause_gadgets,
        connector_gadgets=connector_gadgets,
        k=graph.k,
    )



# %%
# TEXT OUTPUT

def _all_btd_tracks(graph: BTDGraph) -> tuple[TrackSegment, ...]:
    tracks: list[TrackSegment] = []
    for gadget in graph.variable_gadgets.values():
        tracks.extend(gadget.tracks)
    for gadget in graph.clause_gadgets.values():
        tracks.extend(gadget.tracks)
    for gadget in graph.connector_gadgets:
        tracks.extend(gadget.tracks)
    return tuple(tracks)


def _all_btd_vertices(graph: BTDGraph) -> tuple[Point, ...]:
    vertices = {
        vertex
        for gadget in graph.variable_gadgets.values()
        for vertex in gadget.verticies
    }
    vertices.update(
        corner
        for gadget in graph.clause_gadgets.values()
        for corner in (gadget.left, gadget.middle, gadget.right)
    )
    vertices.update(
        vertex
        for gadget in graph.connector_gadgets
        for vertex in gadget.verticies
    )
    return tuple(sorted(vertices, key=lambda point: (point.y, point.x)))


def summarize_btd_graph(graph: BTDGraph) -> None:
    tracks = _all_btd_tracks(graph)
    vertices = _all_btd_vertices(graph)
    degree_distribution: dict[int, int] = {}

    for vertex in vertices:
        degree = sum(
            1
            for track in tracks
            if track.start == vertex or track.end == vertex
        )
        degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

    connector_vertex_count = sum(
        len(gadget.verticies)
        for gadget in graph.connector_gadgets
    )

    print(
        "BTD: "
        f"{len(graph.variable_gadgets)} variable gadgets, "
        f"{len(graph.clause_gadgets)} clause gadgets, "
        f"{len(graph.connector_gadgets)} connector gadgets, "
        f"{len(vertices)} towers, "
        f"{len(tracks)} track segments, "
        f"{connector_vertex_count} connector vertices, "
        f"k={graph.k}"
    )
    print(f"Degree distribution: {dict(sorted(degree_distribution.items()))}")



# %%
# DRAWING

_BTD_TOWER_RANGE = 0.45
_BTD_DART_MONKEY_PATH = "BTD5_dart_monke.png"
_BTD_RED_BLOON_PATH = "BTD5_red_bloon.png"
_BTD_ICON_HALF_MONKEY = 0.22
_BTD_ICON_HALF_BLOON = 0.15
_BTD_RANGE_FILL = "#00d4ff"
_BTD_RANGE_EDGE = "#0099cc"
_BTD_RANGE_ALPHA = 0.13
_BTD_MONKEY_FILL = "#8b6914"
_BTD_MONKEY_EDGE = "#4a3800"
_BTD_BLOON_FILL = "#ff2222"
_BTD_BLOON_EDGE = "#990000"
_BTD_BLOON_HILITE = "#ff9999"
_BTD_DEGREE_COLOURS = {
    1: "#ff00ff",
    2: "#16db65",
    3: "#0d2818",
}
_BTD_TRACK_LINEWIDTH = 3.0

_btd_img_cache: dict[str, np.ndarray | None] = {}


def _load_square_rgba(path: str) -> np.ndarray:
    img = PILImage.open(path).convert("RGBA")
    width, height = img.size
    side = max(width, height)
    square = PILImage.new("RGBA", (side, side), (0, 0, 0, 0))
    x = (side - width) // 2
    y = (side - height) // 2
    square.paste(img, (x, y))
    return np.array(square)


def _try_load_btd_img(path: str) -> np.ndarray | None:
    if path not in _btd_img_cache:
        try:
            _btd_img_cache[path] = _load_square_rgba(path)
            print(f"[BTD] loaded {path}")
        except Exception as exc:
            print(f"[BTD] image unavailable ({exc}), using fallback")
            _btd_img_cache[path] = None

    return _btd_img_cache[path]


def _fallback_dart_monkey(axis, x: float, y: float, half: float) -> None:
    axis.add_patch(RegularPolygon(
        (x, y),
        numVertices=6,
        radius=half,
        orientation=math.pi / 6,
        facecolor=_BTD_MONKEY_FILL,
        edgecolor=_BTD_MONKEY_EDGE,
        linewidth=0.9,
        zorder=7,
    ))
    axis.annotate(
        "",
        xy=(x + half * 0.95, y),
        xytext=(x + half * 0.15, y),
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
        zorder=8,
    )


def _fallback_red_bloon(axis, x: float, y: float, half: float) -> None:
    axis.add_patch(Ellipse(
        (x, y - half * 0.05),
        width=half * 1.7,
        height=half * 2.0,
        facecolor=_BTD_BLOON_FILL,
        edgecolor=_BTD_BLOON_EDGE,
        linewidth=0.7,
        zorder=7,
    ))
    axis.add_patch(Ellipse(
        (x - half * 0.28, y + half * 0.45),
        width=half * 0.32,
        height=half * 0.42,
        facecolor=_BTD_BLOON_HILITE,
        edgecolor="none",
        alpha=0.9,
        zorder=8,
    ))


def _draw_btd_icon(
    axis,
    x: float,
    y: float,
    path: str,
    fallback_fn,
    half: float,
) -> None:
    img = _try_load_btd_img(path)

    if img is not None:
        axis.imshow(
            img,
            extent=[x - half, x + half, y - half, y + half],
            aspect="auto",
            zorder=7,
            interpolation="antialiased",
        )
    else:
        fallback_fn(axis, x, y, half)


def _degree_colour(degree: int) -> str:
    return _BTD_DEGREE_COLOURS.get(degree, "#777777")


def _btd_bounds(graph: BTDGraph) -> tuple[float, float, float, float, int, int]:
    points = set(_all_btd_vertices(graph))
    for track in _all_btd_tracks(graph):
        points.add(track.start)
        points.add(track.end)

    xs, ys = zip(*(_triangle_xy(point) for point in points))
    rows = [point.y for point in points]
    return (
        min(xs) - 1.0,
        max(xs) + 1.0,
        min(ys) - _TRIANGLE_GRID_STEP,
        max(ys) + _TRIANGLE_GRID_STEP,
        min(rows) - 1,
        max(rows) + 1,
    )


def draw_btd_graph(
    graph: BTDGraph,
    tower_range: float = _BTD_TOWER_RANGE,
    output: Path | str | None = None,
):
    min_x, max_x, min_y, max_y, row_min, row_max = _btd_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + _TRIANGLE_GRID_STEP) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    _draw_triangle_grid(axis, min_x, max_x, row_min, row_max)
    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    tracks = _all_btd_tracks(graph)
    vertices = _all_btd_vertices(graph)

    for track in tracks:
        x1, y1 = _triangle_xy(track.start)
        x2, y2 = _triangle_xy(track.end)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        axis.plot(
            [x1, mid_x],
            [y1, mid_y],
            color=_degree_colour(track.start_degree),
            linewidth=_BTD_TRACK_LINEWIDTH,
            zorder=2,
            solid_capstyle="butt",
        )
        axis.plot(
            [mid_x, x2],
            [mid_y, y2],
            color=_degree_colour(track.end_degree),
            linewidth=_BTD_TRACK_LINEWIDTH,
            zorder=2,
            solid_capstyle="butt",
        )

    for vertex in vertices:
        x, y = _triangle_xy(vertex)
        axis.add_patch(Circle(
            (x, y),
            tower_range,
            facecolor=_BTD_RANGE_FILL,
            edgecolor=_BTD_RANGE_EDGE,
            linewidth=0.8,
            alpha=_BTD_RANGE_ALPHA,
            zorder=3,
        ))

    for vertex in vertices:
        x, y = _triangle_xy(vertex)
        _draw_btd_icon(
            axis,
            x,
            y,
            _BTD_DART_MONKEY_PATH,
            _fallback_dart_monkey,
            _BTD_ICON_HALF_MONKEY,
        )

    for track in tracks:
        x1, y1 = _triangle_xy(track.start)
        x2, y2 = _triangle_xy(track.end)
        _draw_btd_icon(
            axis,
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            _BTD_RED_BLOON_PATH,
            _fallback_red_bloon,
            _BTD_ICON_HALF_BLOON,
        )

    for degree, colour in sorted(_BTD_DEGREE_COLOURS.items()):
        axis.plot([], [], color=colour, linewidth=2.5, label=f"degree {degree}")
    axis.legend(loc="upper right", fontsize=7, framealpha=0.7)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x, max_x + 1) # add 1 cuz legend box
    axis.set_ylim(min_y, max_y)
    axis.set_yticks([row * _TRIANGLE_GRID_STEP for row in range(row_min, row_max + 1)])
    axis.set_yticklabels([str(row) for row in range(row_min, row_max + 1)], fontsize=7)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.set_title(
        f"Bloons TD Graph (k = {graph.k}, range = {tower_range})"
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
btd_graph = convert_tge_pvc_to_btd(tge_pvc_graph)
summarize_btd_graph(btd_graph)
fig, axis = draw_btd_graph(btd_graph)
plt.show()

# %% [markdown]
# ## Direct Conversion and Flattening BTDGraph
#
# TODO: later...but essentially we just want to verify these tasks:
#
# + we can flatten BTDGraph into BTDInstance, a minimal data structure with a set of verticies and tracks
# + we can convert directly from MRP3SATInstance to BTDGraph without any NP-Complete problems in-between
# + ditto but convert MRP3SATInstance into BTDInstance
#
# I'm still unsure if the game engine should use BTDGraph so it's easier to show the solution, or if it should use BTDInstance because it's easier to store and render.

# %%
# this cell is intentional empty so we can add code here later

# %% [markdown]
# ## Yes-instance to Yes-instance

# %% [markdown]
# ### MRP3SAT Certificate Visualizer

# %%
_CERT_TRUE_FILL = "#d9f0d3"
_CERT_TRUE_EDGE = "#1b7837"
_CERT_FALSE_FILL = "#fddbc7"
_CERT_FALSE_EDGE = "#b2182b"
_CERT_SATISFIED_COLOUR = "#ffb000"
_CERT_UNSATISFIED_COLOUR = "#555555"


def _clause_signs_by_label(instance: MRP3SATInstance) -> dict[int, bool]:
    signs: dict[int, bool] = {}

    for clause in instance.positive:
        signs[clause.label] = True

    for clause in instance.negative:
        signs[clause.label] = False

    return signs


def _connector_is_satisfied(
    instance: MRP3SATInstance,
    connector: ConnectorEdge,
    positive_by_label: dict[int, bool],
) -> bool:
    is_positive = positive_by_label[connector.clause_label]
    value = instance.certificate[connector.variable - 1]

    return value if is_positive else not value


def _satisfied_connectors_by_clause(
    instance: MRP3SATInstance,
    graph: MRP3SATGraph,
) -> dict[int, tuple[ConnectorEdge, ...]]:
    positive_by_label = _clause_signs_by_label(instance)
    satisfied: dict[int, list[ConnectorEdge]] = {
        clause.label: []
        for clause in instance.clauses
    }

    for connector in graph.edges:
        if _connector_is_satisfied(instance, connector, positive_by_label):
            satisfied[connector.clause_label].append(connector)

    return {
        label: tuple(sorted(
            connectors,
            key=lambda connector: (connector.slot, connector.variable),
        ))
        for label, connectors in satisfied.items()
    }


def _certificate_truth_text(value: bool) -> str:
    return "T" if value else "F"


def _certificate_clause_text(
    clause: ClauseSegment,
    satisfied_connectors: tuple[ConnectorEdge, ...],
) -> str:
    if not satisfied_connectors:
        return f"{_label_str(clause.label)} UNSAT"

    occurrences = ", ".join(
        f"x{connector.variable}:s{connector.slot}"
        for connector in satisfied_connectors
    )
    return f"{_label_str(clause.label)} SAT: {occurrences}"


# %%
def draw_sga_certificate(
    instance: MRP3SATInstance,
    graph: MRP3SATGraph,
    output: Path | str | None = None,
):
    """Render the provided MRP3SAT certificate on the square-grid graph."""
    min_x, max_x, min_y, max_y = _graph_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + 2) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    positive_by_label = _clause_signs_by_label(instance)
    satisfied_by_clause = _satisfied_connectors_by_clause(instance, graph)
    satisfied_connectors = {
        connector
        for connectors in satisfied_by_clause.values()
        for connector in connectors
    }

    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    for variable in graph.variables.values():
        value = instance.certificate[variable.var - 1]
        bx = variable.x_start - 0.4
        bw = variable.x_end - variable.x_start + 0.8
        fill = _CERT_TRUE_FILL if value else _CERT_FALSE_FILL
        edge = _CERT_TRUE_EDGE if value else _CERT_FALSE_EDGE

        axis.add_patch(FancyBboxPatch(
            (bx, variable.y - _BOX_H / 2),
            bw,
            _BOX_H,
            boxstyle="round,pad=0.02",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
            zorder=5,
        ))

        axis.text(
            bx + bw / 2,
            variable.y,
            f"x{variable.var} = {_certificate_truth_text(value)}",
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
            fontweight="bold",
            zorder=6,
        )

    for clause in graph.clauses.values():
        is_positive = positive_by_label[clause.label]
        color = _POS_COLOUR if is_positive else _NEG_COLOUR
        clause_is_satisfied = bool(satisfied_by_clause[clause.label])
        label_color = color if clause_is_satisfied else _CERT_UNSATISFIED_COLOUR

        axis.plot(
            [clause.x_start, clause.x_end],
            [clause.y, clause.y],
            color=color,
            linewidth=2.4,
            zorder=3,
        )
        axis.text(
            (clause.x_start + clause.x_end) / 2,
            clause.y + (0.16 if clause.y > 0 else -0.16),
            _certificate_clause_text(clause, satisfied_by_clause[clause.label]),
            ha="center",
            va=("bottom" if clause.y > 0 else "top"),
            fontsize=9,
            color=label_color,
            fontweight="bold",
            zorder=7,
        )

    for connector in graph.edges:
        is_positive = positive_by_label[connector.clause_label]
        color = _POS_COLOUR if is_positive else _NEG_COLOUR
        is_satisfied = connector in satisfied_connectors

        for segment in connector.segments:
            axis.plot(
                [segment.start.x, segment.end.x],
                [segment.start.y, segment.end.y],
                color=color,
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
            )

            if is_satisfied:
                axis.plot(
                    [segment.start.x, segment.end.x],
                    [segment.start.y, segment.end.y],
                    color=_CERT_SATISFIED_COLOUR,
                    linewidth=3.0,
                    alpha=0.85,
                    zorder=4,
                )

        endpoint_size = 54 if is_satisfied else 18
        endpoint_color = _CERT_SATISFIED_COLOUR if is_satisfied else color
        axis.scatter(
            [connector.end.x],
            [connector.end.y],
            color=endpoint_color,
            edgecolors="#222222" if is_satisfied else "none",
            linewidths=0.6 if is_satisfied else 0,
            s=endpoint_size,
            zorder=6,
        )

    axis.plot(
        [],
        [],
        color=_CERT_SATISFIED_COLOUR,
        linewidth=3.0,
        label="satisfied literal occurrence",
    )
    axis.legend(loc="upper right", fontsize=7, framealpha=0.7)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x - 1, max_x + 1)
    axis.set_ylim(min_y - 1, max_y + 1)
    axis.set_xticks(range(min_x - 1, max_x + 2))
    axis.set_yticks(range(min_y - 1, max_y + 2))
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.grid(True, color=_GRIDLINE_COLOUR, linewidth=0.6, linestyle="--", zorder=0)
    axis.set_title("MRP3SAT Certificate on SGA Graph")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
fig, axis = draw_sga_certificate(mrp3sat_instance, sga_graph)
plt.show()


# %% [markdown]
# ### BTD Certificate Transformer and Visualizer

# %%
def _btd_clause_corner_by_slot(gadget: BTDClauseGadget) -> dict[int, Point]:
    return {
        LEFT: gadget.left,
        MIDDLE: gadget.middle,
        RIGHT: gadget.right,
    }


def _btd_connector_is_satisfied(
    instance: MRP3SATInstance,
    connector: BTDConnectorGadget,
    positive_by_label: dict[int, bool],
) -> bool:
    is_positive = positive_by_label[connector.clause_label]
    value = instance.certificate[connector.variable - 1]

    return value if is_positive else not value


def _btd_connector_path(connector: BTDConnectorGadget) -> tuple[Point, ...]:
    if not connector.tracks:
        raise ValueError(
            f"Connector {_label_str(connector.clause_label)}:"
            f"{connector.slot}:x{connector.variable} has no tracks."
        )

    first_track = connector.tracks[0]
    if first_track.start.y == 0:
        current = first_track.start
    elif first_track.end.y == 0:
        current = first_track.end
    else:
        raise ValueError(
            f"Connector {_label_str(connector.clause_label)}:"
            f"{connector.slot}:x{connector.variable} does not start on the variable row."
        )

    points = [current]

    for track in connector.tracks:
        if track.start == current:
            current = track.end
        elif track.end == current:
            current = track.start
        else:
            raise ValueError(
                f"Connector {_label_str(connector.clause_label)}:"
                f"{connector.slot}:x{connector.variable} has non-adjacent tracks."
            )
        points.append(current)

    path = tuple(points)
    if path[0].y != 0 or path[-1].y == 0:
        raise ValueError(
            f"Connector {_label_str(connector.clause_label)}:"
            f"{connector.slot}:x{connector.variable} path endpoints look wrong."
        )

    if path[1:-1] != connector.verticies:
        raise ValueError(
            f"Connector {_label_str(connector.clause_label)}:"
            f"{connector.slot}:x{connector.variable} path does not match stored vertices."
        )

    return path


def _btd_satisfied_connectors_by_clause(
    instance: MRP3SATInstance,
    graph: BTDGraph,
) -> dict[int, tuple[BTDConnectorGadget, ...]]:
    positive_by_label = _clause_signs_by_label(instance)
    satisfied: dict[int, list[BTDConnectorGadget]] = {
        label: []
        for label in graph.clause_gadgets
    }

    for connector in graph.connector_gadgets:
        if _btd_connector_is_satisfied(instance, connector, positive_by_label):
            satisfied[connector.clause_label].append(connector)

    return {
        label: tuple(sorted(
            connectors,
            key=lambda connector: (connector.slot, connector.variable),
        ))
        for label, connectors in satisfied.items()
    }


def _selected_btd_variable_vertices(
    instance: MRP3SATInstance,
    graph: BTDGraph,
) -> set[Point]:
    selected: set[Point] = set()

    for variable, gadget in graph.variable_gadgets.items():
        use_even_x = instance.certificate[variable - 1]
        for vertex in gadget.verticies:
            if (vertex.x % 2 == 0) == use_even_x:
                selected.add(vertex)

    return selected


# %%
def build_btd_certificate(
    instance: MRP3SATInstance,
    graph: BTDGraph,
) -> tuple[Point, ...]:
    """Convert the MRP3SAT certificate into raw BTD monkey placements."""
    selected = _selected_btd_variable_vertices(instance, graph)

    for connector in graph.connector_gadgets:
        path = _btd_connector_path(connector)
        variable_is_selected = path[0] in selected

        for index, vertex in enumerate(path[1:], start=1):
            should_select = (
                variable_is_selected
                if index % 2 == 0
                else not variable_is_selected
            )
            if should_select:
                selected.add(vertex)

    # Clause-triangle repair is intentionally disabled for now so the drawing
    # shows where raw connector propagation leaves the BTD graph.
    repair_clause_triangles = False
    if repair_clause_triangles:
        satisfied_by_clause = _btd_satisfied_connectors_by_clause(instance, graph)

        for label, clause_gadget in graph.clause_gadgets.items():
            satisfied_connectors = satisfied_by_clause[label]
            if not satisfied_connectors:
                selected.update(_btd_clause_corner_by_slot(clause_gadget).values())
                continue

            omitted_connector = satisfied_connectors[0]
            omitted_corner = _btd_clause_corner_by_slot(clause_gadget)[
                omitted_connector.slot
            ]

            for corner in _btd_clause_corner_by_slot(clause_gadget).values():
                if corner != omitted_corner:
                    selected.add(corner)

    certificate = tuple(sorted(selected, key=lambda point: (point.y, point.x)))

    return certificate


def _btd_uncovered_tracks(
    graph: BTDGraph,
    certificate_points: tuple[Point, ...],
) -> tuple[TrackSegment, ...]:
    selected = set(certificate_points)

    return tuple(
        track
        for track in _all_btd_tracks(graph)
        if track.start not in selected and track.end not in selected
    )


def verify_btd_certificate(
    graph: BTDGraph,
    certificate_points: tuple[Point, ...],
) -> dict[str, int]:
    selected = set(certificate_points)

    if len(selected) != len(certificate_points):
        raise ValueError("BTD certificate contains duplicate points.")

    vertices = set(_all_btd_vertices(graph))
    invalid_points = sorted(
        selected - vertices,
        key=lambda point: (point.y, point.x),
    )
    if invalid_points:
        raise ValueError(f"BTD certificate contains non-vertices: {invalid_points}")

    uncovered_tracks = _btd_uncovered_tracks(graph, certificate_points)
    over_budget = max(0, len(certificate_points) - graph.k)
    under_budget = max(0, graph.k - len(certificate_points))

    print(
        "BTD certificate: "
        f"{len(certificate_points)} monkeys, "
        f"k={graph.k}, "
        f"over_budget={over_budget}, "
        f"under_budget={under_budget}, "
        f"uncovered_tracks={len(uncovered_tracks)}"
    )

    return {
        "monkeys": len(certificate_points),
        "k": graph.k,
        "over_budget": over_budget,
        "under_budget": under_budget,
        "uncovered_tracks": len(uncovered_tracks),
    }


# %%
def draw_btd_certificate(
    graph: BTDGraph,
    certificate_points: tuple[Point, ...],
    tower_range: float = _BTD_TOWER_RANGE,
    output: Path | str | None = None,
):
    selected = set(certificate_points)
    uncovered_tracks = set(_btd_uncovered_tracks(graph, certificate_points))
    min_x, max_x, min_y, max_y, row_min, row_max = _btd_bounds(graph)
    fig_width = max(7.0, (max_x - min_x + 1) * 0.5)
    fig_height = max(4.5, (max_y - min_y + _TRIANGLE_GRID_STEP) * 0.9)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    _draw_triangle_grid(axis, min_x, max_x, row_min, row_max)
    axis.axhline(0, color=_VARROW_COLOUR, linewidth=1.2, zorder=1)

    tracks = _all_btd_tracks(graph)
    vertices = _all_btd_vertices(graph)

    for track in tracks:
        x1, y1 = _triangle_xy(track.start)
        x2, y2 = _triangle_xy(track.end)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        track_is_uncovered = track in uncovered_tracks
        line_zorder = 5 if track_is_uncovered else 2
        axis.plot(
            [x1, mid_x],
            [y1, mid_y],
            color=("#ff3333" if track_is_uncovered else _degree_colour(track.start_degree)),
            linewidth=(_BTD_TRACK_LINEWIDTH + 1.4 if track_is_uncovered else _BTD_TRACK_LINEWIDTH),
            zorder=line_zorder,
            solid_capstyle="butt",
        )
        axis.plot(
            [mid_x, x2],
            [mid_y, y2],
            color=("#ff3333" if track_is_uncovered else _degree_colour(track.end_degree)),
            linewidth=(_BTD_TRACK_LINEWIDTH + 1.4 if track_is_uncovered else _BTD_TRACK_LINEWIDTH),
            zorder=line_zorder,
            solid_capstyle="butt",
        )

    for vertex in vertices:
        x, y = _triangle_xy(vertex)
        if vertex in selected:
            axis.add_patch(Circle(
                (x, y),
                tower_range,
                facecolor=_BTD_RANGE_FILL,
                edgecolor=_BTD_RANGE_EDGE,
                linewidth=0.8,
                alpha=_BTD_RANGE_ALPHA,
                zorder=3,
            ))
        else:
            axis.scatter(
                [x], [y],
                s=22,
                facecolors="white",
                edgecolors="#999999",
                linewidths=0.7,
                alpha=0.65,
                zorder=4,
            )

    for vertex in vertices:
        if vertex not in selected:
            continue

        x, y = _triangle_xy(vertex)
        _draw_btd_icon(
            axis,
            x,
            y,
            _BTD_DART_MONKEY_PATH,
            _fallback_dart_monkey,
            _BTD_ICON_HALF_MONKEY,
        )

    for track in tracks:
        x1, y1 = _triangle_xy(track.start)
        x2, y2 = _triangle_xy(track.end)
        _draw_btd_icon(
            axis,
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            _BTD_RED_BLOON_PATH,
            _fallback_red_bloon,
            _BTD_ICON_HALF_BLOON,
        )

    for degree, colour in sorted(_BTD_DEGREE_COLOURS.items()):
        axis.plot([], [], color=colour, linewidth=2.5, label=f"degree {degree}")
    axis.plot([], [], color="#ff3333", linewidth=3.5, label="uncovered track")
    axis.scatter(
        [], [],
        s=22,
        facecolors="white",
        edgecolors="#999999",
        linewidths=0.7,
        label="empty tower vertex",
    )
    axis.legend(loc="upper right", fontsize=7, framealpha=0.7)

    budget_delta = len(certificate_points) - graph.k
    if budget_delta > 0:
        budget_line = f"budget delta: +{budget_delta}"
    elif budget_delta < 0:
        budget_line = f"budget delta: {budget_delta}"
    else:
        budget_line = "budget delta: 0"
    stats_text = "\n".join((
        f"monkeys used: {len(certificate_points)} / {graph.k}",
        budget_line,
        f"uncovered tracks: {len(uncovered_tracks)}",
    ))
    axis.text(
        0.98,
        0.02,
        stats_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#999999",
            alpha=0.78,
        ),
        zorder=20,
    )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(min_x, max_x + 2) # extend by 2 because text is longer
    axis.set_ylim(min_y, max_y)
    axis.set_yticks([row * _TRIANGLE_GRID_STEP for row in range(row_min, row_max + 1)])
    axis.set_yticklabels([str(row) for row in range(row_min, row_max + 1)], fontsize=7)
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", left=False, length=0, labelleft=True)
    axis.set_title(f"Bloons TD Certificate (k = {graph.k})")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    if output is not None:
        fig.savefig(Path(output), bbox_inches="tight")
        print(f"Saved to {output}")

    return fig, axis


# %%
btd_certificate = build_btd_certificate(mrp3sat_instance, btd_graph)
verify_btd_certificate(btd_graph, btd_certificate)
fig, axis = draw_btd_certificate(btd_graph, btd_certificate)
plt.show()

# %%
