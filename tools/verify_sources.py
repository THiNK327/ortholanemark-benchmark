"""Verify evaluated source hashes and any comment-only release notice prefixes."""
import ast
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'provenance/source_manifest.json').read_text(encoding='utf-8'))
    records = json.loads((root / 'provenance/source_notice_changes.json').read_text(encoding='utf-8'))
    notices = {row['path']: row for row in records['files']}
    assert len(notices) == len(records['files']), 'Duplicate source notice record'
    expected = {row['path'] for row in manifest['files']}
    assert set(notices) <= expected, 'Notice for a file outside evaluated source manifest'
    for row in manifest['files']:
        rel = row['path']
        data = (root / rel).read_bytes()
        original = data
        if rel in notices:
            notice = notices[rel]
            assert notice['insertion'] == 'utf8_file_prefix', rel
            prefix = notice['prefix_utf8'].encode('utf-8')
            assert prefix and all(not line.strip() or line.startswith('#') for line in notice['prefix_utf8'].splitlines()), rel
            assert data.startswith(prefix), rel
            assert hashlib.sha256(data).hexdigest() == notice['packaged_sha256'], rel
            original = data[len(prefix):]
            assert notice['evaluated_sha256'] == row['sha256'], rel
            assert len(data) == notice['packaged_bytes'], rel
            assert len(original) == notice['evaluated_bytes'], rel
            assert ast.dump(ast.parse(data), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False), rel
        assert hashlib.sha256(original).hexdigest() == row['sha256'], rel
        assert len(original) == row['bytes'], rel
    return len(manifest['files']), len(notices)


def main():
    total, notices = verify(Path(__file__).resolve().parents[1])
    print(f'PASS: {total} evaluated-source hashes; {notices} comment-only notices with identical ASTs.')


if __name__ == '__main__':
    main()
