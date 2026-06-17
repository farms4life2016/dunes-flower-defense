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
# - the only odd numbers we should see is the three nodes in the triangle of the triple-OR gadget for VC.
#   everything else (wire gadget, variable gadget) should have an **even** number of nodes.
#   **This is crucial for the reduction!!!**
#   - the gap between variables should be zero or some even number on the square/triangle grid
#   - the number of nodes between the variable node and the triple-OR gadget should be an even number (not counting either endpoint)
# - evenness is important for bipartite colouring: both colourings are the same size, so we use up the same number of cover nodes (from k) no matter which bipartition we pick.
# - also useful for calculating the limit "k", k = ((total number of nodes) - (3m)) / 2 + (2m)
#   - there are m number of triple-OR gadgets, which are all 3-node triangles
#   - after subtracting, a bunch of even bipartite trees remain. colouring each takes exactly half of the remainging nodes thanks to evenness.
#   - then add back 2 nodes per triangle since it takes at minimum 2 nodes to cover a triangle
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
from matplotlib.patches import FancyBboxPatch
import yaml

# %matplotlib inline

INPUT_PATH = "input-unsatisfiable.yaml"

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
    clauses: Iterable[Clause],
    side_name: str,
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
        outer_variables = set(outer.variables)

        for inner in clauses:
            if outer == inner:
                continue

            has_blocked_connection = any(
                outer.left < variable < outer.right
                and variable not in outer_variables
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
            f"{side_name} clauses cannot be drawn "
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
        instance.positive,
        "Positive",
    )

    negative_levels = compute_clause_levels(
        instance.negative,
        "Negative",
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

# %%
