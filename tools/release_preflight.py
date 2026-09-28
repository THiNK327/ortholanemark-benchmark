"""Verify package integrity, source provenance, and component metadata offline."""
import argparse
import hashlib
import json
from pathlib import Path

from verify_sources import verify as verify_sources
from verify_integrity import main as verify_integrity
from prepare_manifest import resolve_manifest


def inspect(root, companions=False):
    meta = json.loads((root / 'provenance/release_metadata.json').read_text(encoding='utf-8'))
    required = [
        'README.md', 'LICENSE', 'CITATION.cff', 'LICENSES/manifest.json',
        'provenance/README.md', 'assets/annotation_examples.png',
        'THIRD_PARTY_NOTICES.md',
    ]
    for relative in required:
        assert (root / relative).is_file(), f'Missing package file: {relative}'
    licenses = json.loads((root / 'LICENSES/manifest.json').read_text(encoding='utf-8'))
    for row in licenses['files']:
        path = root / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
    for method in ['scnn', 'ufldv2', 'polylanenet', 'laneatt', 'clrnet']:
        assert (root / f'ortholanemark/literature/{method}_faithful/LICENSE_upstream').is_file()
    assert meta['code_license'] == 'MIT'
    assert meta['data_license'] == meta['results_license'] == 'CC-BY-4.0'
    assert meta['weight_license_scope'], 'Checkpoint component scope is missing'
    ignore_rules = (root / '.gitignore').read_text().splitlines()
    assert '!artifacts/**/*.npz' in ignore_rules, 'Saved predictions are ignored'
    assert '/data/' in ignore_rules and 'data/' not in ignore_rules, 'Source data/ directory would be ignored'
    assert '/weights/' in ignore_rules and 'weights/' not in ignore_rules, 'Weights ignore should apply only at repository root'
    assert '* -text' in (root / '.gitattributes').read_text().splitlines(), 'Exact-byte Git checkout rule is missing'
    total, notices = verify_sources(root)
    report = {'package_verification': 'pass', 'source_files': total,
              'retained_attribution_prefixes': notices, 'component_licenses': len(licenses['files'])}
    if companions:
        parent = root.parent
        checkpoints = json.loads((parent / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
        assert meta['creators'] == checkpoints['creators'], 'Creator lists disagree'
        assert meta['funding'] == checkpoints['funding'], 'Funding metadata disagrees'
        assert meta['weight_license'] == checkpoints['license'], 'Checkpoint license disagrees'
        assert meta['corresponding_contact'] == checkpoints['corresponding_contact'], 'Contacts disagree'
        assert (parent / 'checkpoints/LICENSE').is_file(), 'Missing checkpoint license'

        dataset_root = (parent / 'dataset').resolve()
        dataset_readme = (dataset_root / 'README.md').read_text(encoding='utf-8')
        assert 'creativecommons.org/licenses/by/4.0' in dataset_readme, 'Missing dataset license statement'
        assert (dataset_root / 'preview.png').is_file(), 'Missing dataset preview'
        dataset = resolve_manifest(dataset_root, dataset_root / 'annotations/splits.json')
        counts = {split: len(entries) for split, entries in dataset['splits'].items()}
        expected = {key: meta['splits'][key] for key in ['train', 'val', 'test']}
        expected['buffer_excluded'] = meta['splits']['excluded_buffer']
        assert counts == expected, 'Dataset split counts disagree'
        ids, annotations, images = set(), set(), set()
        for entries in dataset['splits'].values():
            for entry in entries:
                key = (entry['project'], entry['stem'])
                assert key not in ids, f'Duplicate dataset membership: {key}'
                ids.add(key)
                intensity = Path(entry['image_path'])
                relative = intensity.relative_to(dataset_root / 'images/intensity')
                assert relative == Path(entry['project']) / (entry['stem'] + '.png'), key
                paired_range = dataset_root / 'images/range' / relative
                assert paired_range.is_file(), f'Missing range image: {key}'
                images.update([intensity, paired_range])
                annotation = Path(entry['clean_gt_path'])
                assert annotation == dataset_root / 'annotations' / entry['project'] / (entry['stem'] + '.json'), key
                annotations.add(annotation)
        assert images == set((dataset_root / 'images').rglob('*.png')), 'Image pairs do not match split membership'
        assert annotations == set((dataset_root / 'annotations').glob('section_*/*.json')), 'Annotations do not match split membership'
        report['dataset_layout'] = {'image_pairs': len(ids), 'annotations': len(annotations), 'splits': counts}
        report['companion_metadata'] = 'pass'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companions', action='store_true', help='Also check metadata of sibling dataset/ and checkpoints/ packages')
    parser.add_argument('--output', type=Path, help='Optional JSON report, e.g. work/package_check.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_integrity()
    report = inspect(root, companions=args.companions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
