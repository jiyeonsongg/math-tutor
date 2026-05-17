"""Structured diagram specs (single flat schema for OpenAI structured output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DiagramType = Literal[
    "none",
    "graph",
    "polygon",
    "right_triangle",
    "rectangle",
    "circle",
    "number_line",
]


class GraphPoint(BaseModel):
    x: float
    y: float
    label: str = ""


class GraphLine(BaseModel):
    slope: float | None = None
    intercept: float | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    label: str = ""


class PolygonVertex(BaseModel):
    x: float
    y: float
    label: str = ""


class NumberLinePoint(BaseModel):
    value: float
    label: str = ""


class QuestionDiagram(BaseModel):
    """One object schema — set `type` and fill only the fields for that figure."""

    type: DiagramType = "none"
    title: str = ""
    show_grid: bool = True
    # graph
    x_min: float = -10
    x_max: float = 10
    y_min: float = -10
    y_max: float = 10
    lines: list[GraphLine] = Field(default_factory=list)
    graph_points: list[GraphPoint] = Field(default_factory=list)
    # polygon
    vertices: list[PolygonVertex] = Field(default_factory=list)
    # right_triangle
    leg_x: float = 3
    leg_y: float = 4
    label_horizontal: str = ""
    label_vertical: str = ""
    label_hypotenuse: str = ""
    # rectangle
    width: float = 4
    height: float = 3
    label_width: str = ""
    label_height: str = ""
    # circle
    center_x: float = 0
    center_y: float = 0
    radius: float = 2
    # number_line
    number_line_min: float = -5
    number_line_max: float = 5
    number_line_points: list[NumberLinePoint] = Field(default_factory=list)
