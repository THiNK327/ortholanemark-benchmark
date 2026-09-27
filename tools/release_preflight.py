"""Verify package integrity, source provenance, and component metadata offline."""
import argparse
import hashlib
import json
from pathlib import Path

from verify_sources import verify as verify_sources
from verify_integrity import main as verify_integrity


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
        dataset = json.loads((parent / 'dataset/manifests/publication_metadata.json').read_text(encoding='utf-8'))
        checkpoints = json.loads((parent / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
        assert meta['creators'] == dataset['creators'] == checkpoints['creators'], 'Creator lists disagree'
        assert meta['funding'] == dataset['funding'] == checkpoints['funding'], 'Funding metadata disagrees'
        assert meta['data_license'] == dataset['license'], 'Dataset license disagrees'
        assert meta['weight_license'] == checkpoints['license'], 'Checkpoint license disagrees'
        assert meta['corresponding_contact'] == dataset['corresponding_contact'] == checkpoints['corresponding_contact'], 'Contacts disagree'
        for package in ['dataset', 'checkpoints']:
            assert (parent / package / 'LICENSE').is_file(), f'Missing {package} license'
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
