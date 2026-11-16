#!/usr/bin/env python3
"""Build metric depth maps from KITTI image_2, one NPZ per image.

V2 uses the official metric-depth repo and VKITTI weights. V3 uses the DA3
API and camera focal correction. All models must already exist locally.
CLI arguments override the YAML depth section.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')),
                   help='YAML config (default: config.yaml next to this script)')
    p.add_argument('--version', choices=['v2', 'v3'])
    p.add_argument('--image-dir', help='Directory containing KITTI images')
    p.add_argument('--calib-dir', help='Directory containing matching KITTI calibration files (V3)')
    p.add_argument('--output-dir', help='Directory for NPZ files; use separate directories per run')
    p.add_argument('--input-size', type=int, help='V2: short side; V3: long side, in pixels')
    p.add_argument('--batch-size', type=int, help='Independent images per model forward pass')
    p.add_argument('--v2-model', choices=['small', 'base', 'large'])
    p.add_argument('--v2-repo', help='Root of official Depth-Anything-V2 git repository')
    p.add_argument('--v2-weights-dir', help='Directory with official V2 metric VKITTI weights')
    p.add_argument('--v3-model-dir', help='Local DA3METRIC-LARGE model directory')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--image-id', help='Process one image stem, e.g. 000004')
    p.add_argument('--limit', type=int, help='Process at most this many sorted images')
    p.add_argument('--overwrite', action='store_true', help='Overwrite matching files (default: error)')
    p.add_argument('--dry-run', action='store_true', help='Check paths and list images; do not load models')
    return p.parse_args()


def read_config(args):
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError('PyYAML is missing. Install with: pip install pyyaml') from e
    path = Path(args.config).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Config not found: {path}')
    with path.open(encoding='utf-8') as f:
        full = yaml.safe_load(f) or {}
    if not isinstance(full, dict) or not isinstance(full.get('depth', {}), dict):
        raise ValueError('The YAML must contain a depth: mapping.')
    d = full.get('depth', {})
    v2, v3 = d.get('v2', {}), d.get('v3', {})
    if not isinstance(v2, dict) or not isinstance(v3, dict):
        raise ValueError('depth.v2 and depth.v3 must be YAML mappings.')
    version = args.version or d.get('version', 'v3')
    if version not in ('v2', 'v3'):
        raise ValueError('depth.version must be v2 or v3')
    out = args.output_dir if args.output_dir is not None else d.get('output_dir')
    if not out:
        raise ValueError('Specify --output-dir or set depth.output_dir in YAML.')
    model_dir = args.v3_model_dir or v3.get('model_dir', '/content/depth_models/DA3METRIC-LARGE')
    # The focal correction below is only for the single-branch DA3 metric model.
    if version == 'v3' and Path(str(model_dir)).name != 'DA3METRIC-LARGE':
        raise ValueError('V3 model directory must be named DA3METRIC-LARGE.')
    size = args.input_size if args.input_size is not None else (v2.get('input_size', 518) if version == 'v2' else v3.get('input_size', 504))
    if isinstance(size, bool) or not isinstance(size, int) or size < 14:
        raise ValueError('input_size must be an integer >= 14.')
    batch_size = args.batch_size if args.batch_size is not None else d.get('batch_size', 2)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be an integer >= 1.')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be >= 1')
    model = args.v2_model or v2.get('model', 'small')
    if model not in ('small', 'base', 'large'):
        raise ValueError('v2.model must be small, base, or large')
    device = args.device or d.get('device', 'auto')
    if device not in ('auto', 'cuda', 'cpu'):
        raise ValueError('device must be auto, cuda, or cpu')
    return dict(version=version, image_dir=args.image_dir or d.get('image_dir'),
                calib_dir=args.calib_dir or d.get('calib_dir'), output_dir=out,
                input_size=size, batch_size=batch_size, device=device, v2_model=model,
                v2_repo=args.v2_repo or v2.get('repo_dir', '/content/Depth-Anything-V2'),
                v2_weights_dir=args.v2_weights_dir or v2.get('weights_dir', '/content/depth_models/v2'), v3_model_dir=model_dir)


def image_paths(cfg, args):
    if not cfg['image_dir']:
        raise ValueError('Specify image_dir in YAML or --image-dir.')
    folder = Path(cfg['image_dir']).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f'Image directory not found: {folder}')
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    if args.image_id:
        files = [p for p in files if p.stem == args.image_id]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError(f'No matching images in {folder}')
    names = [p.stem for p in files]
    if len(names) != len(set(names)):
        raise ValueError('Different images have the same stem and would overwrite an NPZ.')
    outdir = Path(cfg['output_dir']).expanduser()
    if outdir.resolve() == folder.resolve():
        raise ValueError('Output directory must be different from the image directory.')
    if not args.overwrite:
        existing = [outdir / (p.stem + '.npz') for p in files if (outdir / (p.stem + '.npz')).exists()]
        if existing:
            raise FileExistsError(f'{len(existing)} outputs already exist (e.g. {existing[0]}). Use another output directory or --overwrite.')
    if cfg['version'] == 'v3':
        if not cfg['calib_dir'] or not Path(cfg['calib_dir']).expanduser().is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {cfg['calib_dir']}")
        missing = [p.stem for p in files if not (Path(cfg['calib_dir']).expanduser() / (p.stem + '.txt')).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} calibration files missing; first: {missing[0]}.txt')
    return files


def check_output_manifest(cfg, create=False):
    """Prevent accidentally mixing different model settings in one output folder."""
    out = Path(cfg['output_dir']).expanduser()
    path = out / 'depth_run.json'
    identity = {'schema': 1, 'version': cfg['version'], 'input_size': cfg['input_size'],
                'resize_method': 'lower_bound' if cfg['version'] == 'v2' else 'upper_bound_resize',
                'model': cfg['v2_model'] if cfg['version'] == 'v2' else 'DA3METRIC-LARGE',
                'model_dir': str(Path(cfg['v2_weights_dir'] if cfg['version'] == 'v2' else cfg['v3_model_dir']).expanduser().resolve()),
                'image_dir': str(Path(cfg['image_dir']).expanduser().resolve()),
                'calib_dir': str(Path(cfg['calib_dir']).expanduser().resolve()) if cfg['version'] == 'v3' else None}
    if path.is_file():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != identity:
            raise ValueError(f'Output folder has different run settings: {path}. Choose a separate --output-dir.')
    elif out.is_dir() and any(out.glob('*.npz')):
        raise ValueError(f'Output directory already has NPZ files but no depth_run.json: {out}. Choose an empty output directory.')
    elif create:
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=out, prefix='.manifest_', delete=False) as f:
            tmp = f.name
            json.dump(identity, f, indent=2, ensure_ascii=False)
        try:
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def read_p2(path):
    import numpy as np
    raw = path.read_text(encoding='utf-8')
    matches = [line.split(':', 1)[1].strip() for line in raw.splitlines() if line.split(':', 1)[0].strip() in ('P2', 'P_rect_02') and ':' in line]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one P2 (or P_rect_02) in {path}, found {len(matches)}')
    numbers = [float(x) for x in matches[0].split()]
    if len(numbers) != 12 or not np.isfinite(numbers).all():
        raise ValueError(f'P2 must contain 12 finite numbers: {path}')
    p2 = np.asarray(numbers, dtype=np.float64).reshape(3, 4)
    if p2[0, 0] <= 0 or p2[1, 1] <= 0:
        raise ValueError(f'Invalid focal lengths in {path}')
    return p2, hashlib.sha256(raw.encode('utf-8')).hexdigest()


def load_model(cfg):
    import torch
    dev = cfg['device']
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false.')
    if cfg['version'] == 'v3':
        from depth_anything_3.api import DepthAnything3
        model_dir = Path(cfg['v3_model_dir']).expanduser()
        for filename in ('config.json', 'model.safetensors'):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f'V3 model file not found: {model_dir / filename}. Run the installation cell first.')
        model = DepthAnything3.from_pretrained(str(model_dir.resolve()), local_files_only=True).to(dev).eval()
        return model, dev

    repo = Path(cfg['v2_repo'] or '').expanduser()
    metric_dir = repo / 'metric_depth'
    if not (metric_dir / 'depth_anything_v2' / 'dpt.py').is_file():
        raise FileNotFoundError(f'Official V2 metric repository not found at {metric_dir}; see README.md or --v2-repo.')
    sys.path.insert(0, str(metric_dir.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2
    variants = {
        'small': ('vits', 64, [48, 96, 192, 384], 'Small'),
        'base': ('vitb', 128, [96, 192, 384, 768], 'Base'),
        'large': ('vitl', 256, [256, 512, 1024, 1024], 'Large'),
    }
    enc, features, channels, label = variants[cfg['v2_model']]
    if not cfg['v2_weights_dir']:
        raise ValueError('Set v2.weights_dir in config.yaml or use --v2-weights-dir.')
    checkpoint = Path(cfg['v2_weights_dir']).expanduser() / f'depth_anything_v2_metric_vkitti_{enc}.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'V2 weight file not found: {checkpoint}. Run the installation cell first.')
    model = DepthAnythingV2(encoder=enc, features=features, out_channels=channels, max_depth=80)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(dev).eval(), dev


def make_v2_transforms(size):
    import cv2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    return (
        Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    )


def prepare_one(image_path, cfg, model, v2_transforms):
    """Preprocess one image independently; never crop to match other images."""
    import cv2
    import numpy as np
    import torch
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Could not read image: {image_path}')
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    data = {'path': image_path, 'height': h, 'width': w}
    if cfg['version'] == 'v2':
        sample = {'image': rgb / 255.0}
        for transform in v2_transforms:
            sample = transform(sample)
        tensor = torch.from_numpy(sample['image'])
    else:
        p2, digest = read_p2(Path(cfg['calib_dir']).expanduser() / (image_path.stem + '.txt'))
        # Call DA3's official preprocessor for each image individually. Its public
        # multi-image API center-crops differing sizes, which is not suitable here.
        tensor_batch, _, _ = model.input_processor(
            [rgb], process_res=cfg['input_size'], process_res_method='upper_bound_resize',
            num_workers=1, print_progress=False, sequential=True,
        )
        if tensor_batch.ndim != 4 or tensor_batch.shape[0] != 1:
            raise ValueError(f'Unexpected DA3 preprocessed shape: {tuple(tensor_batch.shape)}')
        tensor = tensor_batch[0]
        data['p2'] = p2
        data['calib_sha256'] = digest
    data['tensor'] = tensor
    return data


def install_independent_v3_sky(model):
    """Apply official DA3 sky postprocessing separately to each batch item.

    The official model computes a sky depth quantile over its entire input batch.
    Running it separately ensures one image's sky value cannot depend on another.
    """
    import torch
    from addict import Dict
    net = model.model
    official = net._process_mono_sky_estimation

    def per_image(output):
        if 'sky' not in output or output['depth'].shape[0] <= 1:
            return official(output)
        batch = output['depth'].shape[0]
        if output['sky'].shape[0] != batch:
            raise ValueError('Unexpected DA3 sky/depth batch dimensions.')
        processed = []
        for i in range(batch):
            single = Dict(depth=output['depth'][i:i+1], sky=output['sky'][i:i+1])
            processed.append(official(single).depth)
        output.depth = torch.cat(processed, dim=0)
        return output

    net._process_mono_sky_estimation = per_image


def predict_batch(model, dev, cfg, records):
    """True independent-image batch. All records have the same processed shape."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    inputs = torch.stack([r['tensor'] for r in records]).to(dev)
    ph, pw = inputs.shape[-2:]
    with torch.inference_mode():
        if cfg['version'] == 'v2':
            depth_batch = model(inputs)
        else:
            # First axis is batch B, second is views N=1. Do not pass a list of
            # unrelated frames to inference(): it treats them as multiple views.
            # The official inference() passes an empty list. The forward()
            # default is None, which the backbone cannot iterate over.
            output = model.forward(inputs[:, None].float(), export_feat_layers=[])
            depth_batch = output['depth']
            if depth_batch.ndim != 4 or depth_batch.shape[1] != 1:
                raise ValueError(f'Unexpected DA3 depth shape: {tuple(depth_batch.shape)}')
            depth_batch = depth_batch[:, 0]
        if depth_batch.ndim != 3 or depth_batch.shape[0] != len(records):
            raise ValueError(f'Unexpected depth batch shape: {tuple(depth_batch.shape)}')
        if tuple(depth_batch.shape[-2:]) != (ph, pw):
            raise ValueError(f'Unexpected depth spatial size: {tuple(depth_batch.shape[-2:])}, expected {(ph, pw)}')
        # Match previous code: V3 interpolates on CPU; V2 on model device.
        if cfg['version'] == 'v3':
            depth_batch = depth_batch.float().cpu()
        results = []
        for i, rec in enumerate(records):
            h, w = rec['height'], rec['width']
            depth = F.interpolate(depth_batch[i:i+1, None], size=(h, w), mode='bilinear',
                                  align_corners=(cfg['version'] == 'v2'))[0, 0].float().cpu().numpy()
            if cfg['version'] == 'v2':
                extra = {'model_input_hw': [ph, pw], 'max_depth_m': 80.0}
            else:
                p2 = rec['p2']
                fx = float(p2[0, 0]) * pw / w
                fy = float(p2[1, 1]) * ph / h
                focal = (fx + fy) / 2.0
                depth = depth * (focal / 300.0)
                extra = {'model_input_hw': [ph, pw], 'fx_original': float(p2[0, 0]),
                         'fy_original': float(p2[1, 1]), 'fx_processed': fx,
                         'fy_processed': fy, 'focal_processed': focal,
                         'focal_scale': focal / 300.0, 'calib_sha256': rec['calib_sha256']}
            results.append((np.asarray(depth, dtype=np.float32), extra))
    return results


def save_atomic(path, depth, metadata):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.depth_', suffix='.tmp', delete=False) as f:
            name = f.name
            np.savez_compressed(f, depth=depth, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True)))
        os.replace(name, path)
        name = None
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def main():
    args = arguments()
    cfg = read_config(args)
    files = image_paths(cfg, args)
    check_output_manifest(cfg)
    print(f"Model: {cfg['version']} | input-size: {cfg['input_size']} | batch-size: {cfg['batch_size']} | images: {len(files)}", flush=True)
    print(f"Image dir: {cfg['image_dir']}\nOutput dir: {cfg['output_dir']}", flush=True)
    if cfg['version'] == 'v2':
        print(f"V2 metric VKITTI: {cfg['v2_model']} | resize shorter side", flush=True)
    else:
        print(f"V3: {cfg['v3_model_dir']} | resize longer side | calib: {cfg['calib_dir']}", flush=True)
    if args.dry_run:
        print(f'Dry run: no model loaded; first: {files[0].name}, last: {files[-1].name}')
        return 0
    # DA3's internal info messages would otherwise break the progress line.
    os.environ.setdefault('DA3_LOG_LEVEL', 'ERROR')
    import numpy as np
    import torch
    model, dev = load_model(cfg)
    if cfg['version'] == 'v3':
        install_independent_v3_sky(model)
        v2_transforms = None
    else:
        v2_transforms = make_v2_transforms(cfg['input_size'])
    check_output_manifest(cfg, create=True)
    print(f'Device: {dev}', flush=True)

    failures = []
    saved = 0
    handled = 0
    started = time.monotonic()
    stopped = False
    line_visible = False

    def progress():
        nonlocal line_visible
        elapsed = time.monotonic() - started
        text = f'Progress: {handled}/{len(files)} | saved: {saved} | errors: {len(failures)} | elapsed: {elapsed:.0f}s'
        # Carriage return updates the same output line in Colab / terminals.
        print('\r' + text + ' ' * 15, end='', flush=True)
        line_visible = True

    def error(path, exc):
        nonlocal handled, line_visible
        if line_visible:
            print('', flush=True)
            line_visible = False
        first_error = not failures
        failures.append((path.name, f'{type(exc).__name__}: {exc}'))
        handled += 1
        print(f'ERROR {path.name}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        if first_error:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    for start in range(0, len(files), cfg['batch_size']):
        records = []
        for path in files[start:start + cfg['batch_size']]:
            try:
                records.append(prepare_one(path, cfg, model, v2_transforms))
            except Exception as exc:
                error(path, exc)
        if not records:
            print('No image in this batch could be prepared; stopping to avoid repeated errors.', file=sys.stderr, flush=True)
            progress()
            break
        # Different original sizes can give different model tensor sizes. Group
        # only equal tensors; never stretch/crop one image to match another.
        groups = {}
        for rec in records:
            groups.setdefault(tuple(rec['tensor'].shape), []).append(rec)
        for group in groups.values():
            try:
                outputs = predict_batch(model, dev, cfg, group)
            except Exception as exc:
                for rec in group:
                    error(rec['path'], exc)
                if 'out of memory' in str(exc).lower():
                    print('GPU memory exhausted. Reduce batch_size to 1 or lower input_size.', file=sys.stderr, flush=True)
                else:
                    print('Model inference failed; stopping to avoid repeating the same error.', file=sys.stderr, flush=True)
                stopped = True
                break
            for rec, (depth, extra) in zip(group, outputs):
                try:
                    path = rec['path']
                    h, w = rec['height'], rec['width']
                    if depth.shape != (h, w) or not np.isfinite(depth).all() or (depth < 0).any():
                        raise ValueError(f'Invalid depth output: shape={depth.shape}, expected={(h,w)}, finite={np.isfinite(depth).all()}')
                    meta = {'schema': 1, 'image_id': path.stem, 'image_name': path.name,
                            'image_hw': [h, w], 'version': cfg['version'], 'input_size': cfg['input_size'],
                            'v2_model': cfg['v2_model'] if cfg['version'] == 'v2' else None,
                            'model_id': 'DA3METRIC-LARGE' if cfg['version'] == 'v3' else f"Depth-Anything-V2-Metric-VKITTI-{cfg['v2_model']}",
                            'units': 'meters', 'device': dev, 'resize_method': 'lower_bound' if cfg['version'] == 'v2' else 'upper_bound_resize',
                            'torch_version': torch.__version__, 'batch_size_requested': cfg['batch_size'],
                            'batch_size_used': len(group), **extra}
                    target = Path(cfg['output_dir']).expanduser() / (path.stem + '.npz')
                    save_atomic(target, depth, meta)
                    saved += 1
                    handled += 1
                except Exception as exc:
                    error(rec['path'], exc)
        progress()
        if stopped:
            break
    if line_visible:
        print('', flush=True)
    print(f'Finished. Saved: {saved}/{len(files)}. Errors: {len(failures)}. Unprocessed: {len(files) - handled}.', flush=True)
    return 1 if failures or handled != len(files) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
