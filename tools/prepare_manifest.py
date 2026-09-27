"""Resolve a dataset's portable compatibility manifest for historical readers."""
import argparse
import json
from pathlib import Path


def resolve_manifest(dataset_root, manifest):
    root = Path(dataset_root).resolve()
    data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    for entries in data['splits'].values():
        for entry in entries:
            for key in ['image_path', 'clean_gt_path']:
                relative = Path(entry[key])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Manifest paths must be relative and within the dataset')
                path = (root / relative).resolve()
                path.relative_to(root)
                if not path.is_file():
                    raise FileNotFoundError(path)
                entry[key] = str(path)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = resolve_manifest(a.dataset_root, a.manifest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print('Resolved paths for', sum(map(len, data['splits'].values())), 'images')


if __name__ == '__main__': main()
