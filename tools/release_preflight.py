"""Check local release preparation; --publication also requires final release decisions.

This offline command never uploads, creates a repository, or grants a license.
"""
import argparse
import json
from pathlib import Path
import sys

from verify_sources import verify as verify_sources
from verify_integrity import main as verify_integrity


def inspect(root):
    meta = json.loads((root / 'RELEASE_METADATA.json').read_text(encoding='utf-8'))
    companion = root.parent
    dataset = json.loads((companion / 'dataset/manifests/publication_metadata.draft.json').read_text(encoding='utf-8'))
    checkpoints = json.loads((companion / 'checkpoints/manifest.json').read_text(encoding='utf-8'))
    assert meta['creators'] == dataset['creators'] == checkpoints['creators'], 'Creator lists disagree'
    assert meta['funding'] == dataset['funding'] == checkpoints['funding'] == [], 'Funding metadata disagrees'
    assert meta['repository_owner'] == dataset['repository_owner'] == 'THiNK327', 'Repository owner disagrees'
    assert meta['data_license'] == dataset['license'], 'Dataset license disagrees'
    assert meta['weight_license'] == checkpoints['license'], 'Checkpoint license disagrees'
    assert meta['corresponding_contact'] == dataset['corresponding_contact'] == checkpoints['corresponding_contact'], 'Contacts disagree'
    required = [
        'LICENSES/Kornia-Apache-2.0.txt', 'LICENSES/Kornia-COPYRIGHT.txt',
        'LICENSES/Idelbayev-ResNet-BSD-2-Clause.txt',
        'LICENSES/LaneATT-NMS-BSD-3-Clause-and-MIT.txt',
        'LICENSES/Torchvision-BSD-3-Clause.txt', 'LICENSES/MMDetection-Apache-2.0.txt',
        'docs/METHOD_ADAPTATIONS.md', 'docs/CHECKPOINT_RIGHTS.md', 'docs/SOURCE_PROVENANCE.md',
        'docs/UPLOAD_GUIDE.md', 'THIRD_PARTY_NOTICES.md',
    ]
    for rel in required:
        assert (root / rel).is_file(), f'Missing release record: {rel}'
    for method in ['scnn', 'ufldv2', 'polylanenet', 'laneatt', 'clrnet']:
        assert (root / f'lcms_lane_benchmark/literature/{method}_faithful/LICENSE_upstream').is_file()
    ignore_rules = (root / '.gitignore').read_text().splitlines()
    assert '!artifacts/**/*.npz' in ignore_rules, 'Saved predictions still ignored'
    assert '/data/' in ignore_rules and 'data/' not in ignore_rules, 'Nested source data/ directory would be ignored'
    assert '/weights/' in ignore_rules and 'weights/' not in ignore_rules, 'Weights ignore should apply only at repository root'
    total, notices = verify_sources(root)
    pending = []
    if not meta.get('public_redistribution_approved'):
        pending.append('Agreement of all authors and authority to release the included material')
    if not meta.get('copyright_holder'):
        pending.append('Confirmed copyright holder for benchmark-authored material')
    effective_licenses = {
        'code_license': root / 'LICENSE',
        'data_license': companion / 'dataset/LICENSE',
        'weight_license': companion / 'checkpoints/LICENSE',
    }
    for key, label in [('code_license', 'code'), ('data_license', 'dataset'), ('weight_license', 'checkpoint')]:
        if not meta.get(key):
            pending.append(f'Final {label} license and matching effective license file')
        elif not effective_licenses[key].is_file():
            pending.append(f'Effective {label} LICENSE file matching the recorded terms')
        else:
            license_text = effective_licenses[key].read_text(encoding='utf-8')
            if ('NOT AN EFFECTIVE LICENSE GRANT' in license_text
                    or 'TO BE CONFIRMED' in license_text or len(license_text.strip()) < 40):
                pending.append(f'Replace the provisional {label} LICENSE with the approved terms')
    if meta.get('checkpoint_redistribution_status') != 'cleared_for_planned_release':
        pending.append('Checkpoint rights determination, including pretrained initialization terms')
    if not meta.get('repository_url'):
        pending.append('Actual GitHub repository URL')
    if not meta.get('code_doi') or not meta.get('dataset_doi'):
        pending.append('Actual reserved benchmark and dataset DOIs')
    if not meta.get('article_type'):
        pending.append('Submission category: Technical Paper or Data Paper')
    return {'technical_preparation': 'pass', 'evaluated_source_files': total,
            'comment_only_source_notices': notices, 'publication_ready': not pending,
            'pending_publication_items': pending, 'uploads_performed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', action='store_true', help='Fail if final rights/identifiers/author decisions remain unset')
    p.add_argument('--output', type=Path, help='Optional report path outside the tracked release files')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_integrity()
    report = inspect(root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.publication and not report['publication_ready']:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
