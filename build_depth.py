"""Build metric depth maps from KITTI image_2, one NPZ per image.

V2 uses the official metric-depth repo and VKITTI weights. V3 uses the DA3
API and camera focal correction. CLI arguments override the YAML depth section.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
    p.add_argument('--v2-model', choices=['small', 'base', 'large'])
    p.add_argument('--v2-repo', help='Root of official Depth-Anything-V2 git repository')
    p.add_argument('--v2-checkpoint', help='Optional local official VKITTI .pth checkpoint')
    p.add_argument('--v3-model-id', help='DA3METRIC-LARGE Hub ID or local model directory')
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
    model_id = args.v3_model_id or v3.get('model_id', 'depth-anything/DA3METRIC-LARGE')
    # Formula below applies to the single-branch DA3 metric model, not relative/nested models.
    if version == 'v3' and Path(str(model_id)).name != 'DA3METRIC-LARGE':
        raise ValueError('V3 must be DA3METRIC-LARGE, or a local directory named DA3METRIC-LARGE; other checkpoints are not supported.')
    size = args.input_size if args.input_size is not None else (v2.get('input_size', 518) if version == 'v2' else v3.get('input_size', 504))
    if isinstance(size, bool) or not isinstance(size, int) or size < 14:
        raise ValueError('input_size must be an integer >= 14.')
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
                input_size=size, device=device, v2_model=model,
                v2_repo=args.v2_repo or v2.get('repo_dir'),
                v2_checkpoint=args.v2_checkpoint or v2.get('checkpoint'), v3_model_id=model_id)


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
                'model': cfg['v2_model'] if cfg['version'] == 'v2' else cfg['v3_model_id'],
                'v2_checkpoint': str(Path(cfg['v2_checkpoint']).expanduser().resolve()) if cfg['version'] == 'v2' and cfg['v2_checkpoint'] else None,
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
        model = DepthAnything3.from_pretrained(cfg['v3_model_id']).to(dev).eval()
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
    checkpoint = cfg['v2_checkpoint']
    if checkpoint:
        if not Path(checkpoint).expanduser().is_file():
            raise FileNotFoundError(f'V2 checkpoint not found: {checkpoint}')
        checkpoint = str(Path(checkpoint).expanduser())
    else:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(repo_id=f'depth-anything/Depth-Anything-V2-Metric-VKITTI-{label}',
                                     filename=f'depth_anything_v2_metric_vkitti_{enc}.pth')
    model = DepthAnythingV2(encoder=enc, features=features, out_channels=channels, max_depth=80)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(dev).eval(), dev


def predict_v2(model, dev, bgr, size):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    # Same transform and interpolation as the official metric infer_image(),
    # but put the tensor on the explicitly selected device.
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    sample = {'image': rgb}
    for transform in (Resize(width=size, height=size, resize_target=False, keep_aspect_ratio=True,
                             ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
                      NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()):
        sample = transform(sample)
    tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(dev)
    ph, pw = tensor.shape[-2:]
    with torch.inference_mode():
        pred = model(tensor)
        pred = F.interpolate(pred[:, None], (h, w), mode='bilinear', align_corners=True)[0, 0]
    return pred.float().cpu().numpy(), {'model_input_hw': [ph, pw], 'max_depth_m': 80.0}


def predict_v3(model, bgr, p2, size):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # DA3's metric branch returns canonical depth: apply known focal correction once.
    pred = model.inference([rgb], process_res=size, process_res_method='upper_bound_resize')
    raw = np.asarray(pred.depth[0], dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError(f'Unexpected DA3 output shape: {raw.shape}')
    ph, pw = raw.shape
    fx = float(p2[0, 0]) * pw / w
    fy = float(p2[1, 1]) * ph / h
    focal = (fx + fy) / 2.0
    depth = torch.from_numpy(raw.copy())[None, None]
    depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)[0, 0].numpy()
    depth = depth * (focal / 300.0)
    return depth, {'model_input_hw': [ph, pw], 'fx_original': float(p2[0, 0]),
                   'fy_original': float(p2[1, 1]), 'fx_processed': fx,
                   'fy_processed': fy, 'focal_processed': focal, 'focal_scale': focal / 300.0}


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
    print(f"Model: {cfg['version']} | input-size: {cfg['input_size']} | images: {len(files)}", flush=True)
    print(f"Image dir: {cfg['image_dir']}\nOutput dir: {cfg['output_dir']}", flush=True)
    if cfg['version'] == 'v2':
        print(f"V2 metric VKITTI: {cfg['v2_model']} | resize shorter side", flush=True)
    else:
        print(f"V3: {cfg['v3_model_id']} | resize longer side | calib: {cfg['calib_dir']}", flush=True)
    if args.dry_run:
        print(f'Dry run: no model loaded; first: {files[0].name}, last: {files[-1].name}')
        return 0
    import cv2
    import numpy as np
    import torch
    model, dev = load_model(cfg)
    check_output_manifest(cfg, create=True)
    print(f'Device: {dev}', flush=True)
    failures = []
    for i, image_path in enumerate(files, 1):
        try:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'Could not read image: {image_path}')
            h, w = bgr.shape[:2]
            extra = {}
            if cfg['version'] == 'v2':
                depth, extra = predict_v2(model, dev, bgr, cfg['input_size'])
            else:
                calib = Path(cfg['calib_dir']).expanduser() / (image_path.stem + '.txt')
                p2, digest = read_p2(calib)
                depth, extra = predict_v3(model, bgr, p2, cfg['input_size'])
                extra['calib_sha256'] = digest
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != (h, w) or not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError(f'Invalid depth output: shape={depth.shape}, expected={(h,w)}, finite={np.isfinite(depth).all()}')
            meta = {'schema': 1, 'image_id': image_path.stem, 'image_name': image_path.name,
                    'image_hw': [h, w], 'version': cfg['version'], 'input_size': cfg['input_size'],
                    'v2_model': cfg['v2_model'] if cfg['version'] == 'v2' else None,
                    'model_id': cfg['v3_model_id'] if cfg['version'] == 'v3' else f"Depth-Anything-V2-Metric-VKITTI-{cfg['v2_model']}",
                    'units': 'meters', 'device': dev, 'resize_method': 'lower_bound' if cfg['version'] == 'v2' else 'upper_bound_resize',
                    'torch_version': torch.__version__, **extra}
            output = Path(cfg['output_dir']).expanduser() / (image_path.stem + '.npz')
            save_atomic(output, depth, meta)
            print(f'[{i}/{len(files)}] {image_path.name} -> {output.name} | input {extra["model_input_hw"]} | output {h}x{w}', flush=True)
        except Exception as exc:
            failures.append((image_path.name, str(exc)))
            print(f'ERROR [{i}/{len(files)}] {image_path.name}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            if 'out of memory' in str(exc).lower():
                print('GPU memory exhausted. Reduce --input-size or try a smaller V2 model.', file=sys.stderr)
                break
    print(f'Finished. Saved: {i - len(failures)}/{len(files)}. Errors: {len(failures)}.', flush=True)
    if failures:
        for name, message in failures:
            print(f'  {name}: {message}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
