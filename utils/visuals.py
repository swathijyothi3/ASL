"""
Plotly figures for ASL Vision.

The 3D hand view is the reason MediaPipe's depth channel is worth keeping:
the ANN classifies on 63 numbers, 21 of which are depth, and a flat 2D
overlay gives no sense of them at all. Rotating the skeleton makes it
obvious why, say, M and N are easy to confuse and D and I are not.
"""

import numpy as np
import plotly.graph_objects as go

from utils.predictor import FINGERS, FINGER_COLORS_HEX, LANDMARK_NAMES


# MediaPipe returns x to the right, y downwards and z towards the camera.
# Plotly draws z upwards, so the axes are remapped to put the hand the
# right way up with depth running into the screen.

def _to_plot_space(landmarks):
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    z = landmarks[:, 2]

    return x, z, -y


def _skeleton_traces(landmarks, colors, name, width=6, opacity=1.0, dashed=False):
    """One line trace per finger, plus a marker trace for the joints."""

    px, py, pz = _to_plot_space(landmarks)

    traces = []

    for finger, bones in FINGERS.items():
        xs, ys, zs = [], [], []

        for start, end in bones:
            xs += [px[start], px[end], None]
            ys += [py[start], py[end], None]
            zs += [pz[start], pz[end], None]

        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(
                    color=colors[finger],
                    width=width,
                    dash="dash" if dashed else "solid",
                ),
                opacity=opacity,
                name=f"{name} — {finger}",
                legendgroup=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    traces.append(
        go.Scatter3d(
            x=px, y=py, z=pz,
            mode="markers",
            marker=dict(
                size=5,
                color=colors["palm"],
                line=dict(color="#0b1120", width=1),
            ),
            opacity=opacity,
            name=name,
            legendgroup=name,
            showlegend=True,
            text=LANDMARK_NAMES,
            hovertemplate="<b>%{text}</b><br>depth %{y:.3f}<extra></extra>",
        )
    )

    return traces


def _scene(show_axes=False):
    axis = dict(
        showgrid=show_axes,
        zeroline=False,
        showticklabels=False,
        title="",
        showbackground=False,
        visible=show_axes,
    )

    return dict(
        xaxis=axis,
        yaxis=axis,
        zaxis=axis,
        # "data" keeps the real proportions of the hand instead of
        # stretching it to fill a cube.
        aspectmode="data",
        camera=dict(eye=dict(x=0.1, y=-2.0, z=0.35)),
    )


def hand_3d(landmarks, height=430, show_axes=False):
    """Interactive 3D skeleton of one detected hand."""

    figure = go.Figure(
        data=_skeleton_traces(landmarks, FINGER_COLORS_HEX, "Your hand")
    )

    figure.update_layout(
        scene=_scene(show_axes),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        dragmode="orbit",
    )

    return figure


# The reference hand is drawn in a muted single colour so the user's own
# skeleton (drawn in full finger colours) stays the focus.
_REFERENCE_COLORS = {finger: "#64748b" for finger in FINGERS}


def hand_3d_comparison(landmarks, reference, letter, height=430, show_axes=False):
    """The user's hand overlaid on the dataset's average hand for a letter."""

    traces = []

    if reference is not None:
        traces += _skeleton_traces(
            _align(reference, landmarks),
            _REFERENCE_COLORS,
            f"Reference {letter}",
            width=10,
            opacity=0.35,
            dashed=True,
        )

    traces += _skeleton_traces(landmarks, FINGER_COLORS_HEX, "Your hand")

    figure = go.Figure(data=traces)

    figure.update_layout(
        scene=_scene(show_axes),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.0,
            x=0.0,
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="orbit",
    )

    return figure


def _align(reference, landmarks):
    """
    Move the reference hand onto the user's hand.

    Both sets are normalised to their own frame, so without this the two
    skeletons sit in different corners of the plot and the comparison is
    unreadable. Only translation and uniform scale are applied — the
    pose itself, which is the thing being compared, is left alone.
    """

    def spread(points):
        return float(np.linalg.norm(points - points.mean(axis=0), axis=1).mean())

    reference_spread = spread(reference)
    target_spread = spread(landmarks)

    if reference_spread < 1e-6:
        return reference

    scale = target_spread / reference_spread

    centred = (reference - reference.mean(axis=0)) * scale

    return centred + landmarks.mean(axis=0)


def confidence_bars(ranked, height=240):
    """Horizontal bar chart of the most likely letters."""

    letters = [letter for letter, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]

    # The winner is highlighted; the runners-up stay muted so the chart
    # reads at a glance.
    colors = ["#334155"] * len(values)
    colors[-1] = "#38bdf8"

    figure = go.Figure(
        go.Bar(
            x=values,
            y=letters,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13),
            hovertemplate="<b>%{y}</b>  %{x:.2f}%<extra></extra>",
            cliponaxis=False,
        )
    )

    figure.update_layout(
        height=height,
        margin=dict(l=0, r=44, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 108],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#e2e8f0", size=17, family="monospace"),
        ),
        bargap=0.35,
        showlegend=False,
    )

    return figure


def class_distribution(counts, height=300):
    """Training samples per letter."""

    letters = list(counts.keys())
    values = list(counts.values())

    figure = go.Figure(
        go.Bar(
            x=letters,
            y=values,
            marker=dict(color="#38bdf8", line=dict(width=0)),
            hovertemplate="<b>%{x}</b>  %{y} samples<extra></extra>",
        )
    )

    figure.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(color="#e2e8f0", size=13, family="monospace"),
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(148,163,184,0.15)",
        ),
        showlegend=False,
    )

    return figure
