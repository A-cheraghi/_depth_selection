#!/usr/bin/env python3
"""Interactive KITTI depth-map viewer for notebook or standalone HTML.

The NPZ's `depth` values are used directly for both plotting and hover. Color
scaling affects only the appearance, not the displayed numerical values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_depth(depth_path):
    """Return the original 2D depth array without normalization or resizing."""
    path = Path(depth_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Depth NPZ not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        if 'depth' not in data:
            raise ValueError(f"File does not contain a 'depth' array: {path}")
        depth = data['depth']
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"Expected a 2D floating-point 'depth' array; got {depth.shape}, {depth.dtype}")
    return depth


def make_figure(depth_path, image_path=None, colorscale='Turbo'):
    """Build a Plotly figure. Hovering over either panel shows original depth.

    depth_path: NPZ produced by build_depth.py.
    image_path: optional corresponding RGB image (must have the same size).
    colorscale: Plotly color scale, e.g. 'Turbo' or 'Gray'.
    """
    depth = load_depth(depth_path)
    h, w = depth.shape
    rgb = None
    if image_path is not None:
        image_path = Path(image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f'Image not found: {image_path}')
        if image_path.stem != Path(depth_path).stem:
            raise ValueError('Image and NPZ names do not match. Select the image with the same ID.')
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert('RGB'))
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f'Image size {rgb.shape[:2]} does not match depth size {depth.shape}.')

    # 17 significant digits are enough to round-trip the float value from the NPZ.
    # No normalization, grayscale conversion, interpolation or resampling occurs.
    header = 'Pixel: x=%{x}, y=%{y}<br>Depth: '
    fig = make_subplots(rows=1, cols=2 if rgb is not None else 1,
                        subplot_titles=('Image', 'Depth map') if rgb is not None else ('Depth map',),
                        horizontal_spacing=0.05)
    depth_col = 2 if rgb is not None else 1
    if rgb is not None:
        fig.add_trace(go.Image(z=rgb, customdata=depth,
                               hovertemplate=header + '%{customdata:.17g} m<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Heatmap(z=depth, colorscale=colorscale, zsmooth=False,
                             colorbar=dict(title='Depth<br>(m)'),
                             hovertemplate=header + '%{z:.17g} m<extra></extra>'), row=1, col=depth_col)
    for col in range(1, depth_col + 1):
        x_name = 'x' if col == 1 else f'x{col}'
        fig.update_xaxes(title_text='x (pixel)', range=[-0.5, w - 0.5], constrain='domain', row=1, col=col)
        fig.update_yaxes(title_text='y (pixel)', range=[h - 0.5, -0.5],
                         scaleanchor=x_name, scaleratio=1, constrain='domain', row=1, col=col)
    fig.update_layout(title=f'{Path(depth_path).stem} | {w} × {h} | {depth.dtype}',
                      width=1400 if rgb is not None else 900, height=550,
                      margin=dict(l=60, r=80, t=80, b=55), hovermode='closest')
    return fig


def show_depth(depth_path, image_path=None, colorscale='Turbo', renderer='colab'):
    """Display inside Colab. Use make_figure() to customize or save the figure."""
    fig = make_figure(depth_path, image_path, colorscale)
    fig.show(renderer=renderer)


def main():
    p = argparse.ArgumentParser(description='Show NPZ depth values interactively in a standalone HTML file.')
    p.add_argument('--npz', required=True, help='Path to depth NPZ')
    p.add_argument('--image', help='Optional original image with same ID and dimensions')
    p.add_argument('--html', required=True, help='Output HTML file')
    p.add_argument('--colorscale', default='Turbo', help='Plotly color scale, e.g. Turbo or Gray')
    args = p.parse_args()
    fig = make_figure(args.npz, args.image, args.colorscale)
    output = Path(args.html).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, full_html=True)
    print(f'Saved interactive HTML: {output}')


if __name__ == '__main__':
    main()
