"""Verify distributed source integrity and retained attribution comment prefixes."""
import ast
import hashlib
import json
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repository_path(root, relative):
    """Accept canonical relative manifest paths that stay inside the repository."""
    require(isinstance(relative, str) and relative, 'Empty or invalid manifest path')
    require('\\' not in relative and ':' not in relative, f'Invalid manifest path: {relative}')
    require(all(part not in ('', '.', '..') for part in relative.split('/')),
            f'Invalid manifest path: {relative}')
    require(relative.startswith('ortholanemark/'), f'Unsupported source path: {relative}')
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root / 'ortholanemark'),
            f'Path outside source package: {relative}')
    return resolved


def index_records(root, rows, label):
    require(isinstance(rows, list), f'Invalid {label} records')
    result = {}
    seen_paths = set()
    for row in rows:
        relative = row['path']
        repository_path(root, relative)
        require(relative.casefold() not in seen_paths, f'Duplicate {label}: {relative}')
        seen_paths.add(relative.casefold())
        result[relative] = row
    return result


def check_reference(digest, size, label):
    require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
            f'Invalid SHA-256: {label}')
    require(type(size) is int and size >= 0, f'Invalid byte count: {label}')


def check_bytes(data, digest, size, label):
    check_reference(digest, size, label)
    require(len(data) == size, f'Byte count mismatch: {label}')
    require(hashlib.sha256(data).hexdigest() == digest, f'SHA-256 mismatch: {label}')


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'provenance/source_manifest.json').read_text(encoding='utf-8'))
    notice_manifest = json.loads((root / 'provenance/source_notices.json').read_text(encoding='utf-8'))
    require(type(manifest['format_version']) is int and manifest['format_version'] == 2,
            'Unsupported source manifest version')
    sources = index_records(root, manifest['files'], 'source path')
    notices = index_records(root, notice_manifest['files'], 'source notice path')
    require(len(sources) == 84, 'Expected 84 distributed source and resource files')
    require(len(notices) == 9, 'Expected nine attribution comment prefixes')
    require(set(notices) <= set(sources), 'Notice for a file outside the source manifest')

    actual = {
        path.relative_to(root).as_posix()
        for path in (root / 'ortholanemark').rglob('*')
        if path.is_file()
        and not {'__pycache__', '.pytest_cache'}.intersection(path.relative_to(root).parts)
        and path.suffix not in ('.pyc', '.pyo')
    }
    require(actual == set(sources),
            f'Source coverage mismatch: unlisted={sorted(actual - set(sources))}; '
            f'missing={sorted(set(sources) - actual)}')

    for relative, row in sources.items():
        current = repository_path(root, relative).read_bytes()
        check_bytes(current, row['sha256'], row['bytes'], relative)
        # Historical values are audit references, not claims of reconstruction.
        check_reference(row['evaluated_sha256'], row['evaluated_bytes'],
                        f'{relative} historical reference')
        if relative in notices:
            require(relative.endswith('.py'), f'Attribution prefix on a non-Python file: {relative}')
            prefix_text = notices[relative]['prefix_utf8']
            require(isinstance(prefix_text, str) and prefix_text.endswith('\n')
                    and any(line.startswith('#') for line in prefix_text.splitlines())
                    and all(not line.strip() or line.startswith('#')
                            for line in prefix_text.splitlines()),
                    f'Attribution prefix contains non-comment text: {relative}')
            prefix = prefix_text.encode('utf-8')
            require(current.startswith(prefix), f'Attribution prefix missing: {relative}')
            require(ast.dump(ast.parse(current), include_attributes=False)
                    == ast.dump(ast.parse(current[len(prefix):]), include_attributes=False),
                    f'Attribution prefix changes the source AST: {relative}')
    return len(sources), len(notices)


def main():
    total, notices = verify(Path(__file__).resolve().parents[1])
    print(f'PASS: {total} distributed source and resource hashes verified; '
          f'{notices} retained attribution comment prefixes verified. '
          'Historical evaluated hashes are audit references only.')


if __name__ == '__main__':
    main()
