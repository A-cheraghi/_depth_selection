#!/usr/bin/env python3
"""Show a KITTI image and its saved NPZ depth map, one above the other."""
from pathlib import Path
import base64
import io

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_figure(depth_path, image_path):
    """The heatmap z values are the unmodified depth array from the NPZ."""
    depth_path = Path(depth_path)
    image_path = Path(image_path)
    if depth_path.stem != image_path.stem:
        raise ValueError('The NPZ and image must have the same image ID.')

    with np.load(depth_path, allow_pickle=False) as data:
        if 'depth' not in data:
            raise ValueError("The NPZ does not contain a 'depth' array.")
        depth_map = data['depth']
    if depth_map.ndim != 2 or not np.issubdtype(depth_map.dtype, np.floating):
        raise ValueError(f'Expected a 2D float depth map, got {depth_map.shape}, {depth_map.dtype}')
    if not np.isfinite(depth_map).all():
        raise ValueError('The depth map contains NaN or infinity.')

    h, w = depth_map.shape
    with Image.open(image_path) as im:
        rgb = im.convert('RGB')
        if rgb.size != (w, h):
            raise ValueError(f'Image size {rgb.size} does not match depth size {(w, h)}.')
        # The original image is encoded as PNG to avoid transmitting a large
        # RGB array through the notebook. The depth array remains unchanged.
        buf = io.BytesIO()
        rgb.save(buf, format='PNG')
    image_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

    vmin, vmax = float(depth_map.min()), float(depth_map.max())
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                        subplot_titles=('Original image', 'Metric depth map - Gray'))
    fig.add_trace(go.Image(source=image_uri, hoverinfo='skip'), row=1, col=1)
    # Exactly the same Plotly Heatmap and Gray color scale as the supplied code.
    fig.add_trace(go.Heatmap(
        z=depth_map,
        colorscale='Gray',
        zmin=vmin,
        zmax=vmax,
        zsmooth=False,
        colorbar=dict(title='Depth (meters)', y=0.23, len=0.42),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Depth: %{z:.17g} meters<extra></extra>',
    ), row=2, col=1)
    for row in (1, 2):
        fig.update_xaxes(title_text='X (pixel)', range=[-0.5, w - 0.5], row=row, col=1)
        fig.update_yaxes(title_text='Y (pixel)', range=[h - 0.5, -0.5],
                         scaleanchor='x' if row == 1 else 'x2', scaleratio=1,
                         constrain='domain', row=row, col=1)
    fig.update_layout(title=f'{depth_path.stem} | depth from NPZ', width=1100, height=900,
                      margin=dict(l=70, r=90, t=90, b=60))
    return fig


def show_depth(depth_path, image_path):
    fig = make_figure(depth_path, image_path)
    fig.show()
