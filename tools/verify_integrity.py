"""Check the SHA-256 release manifest; no dataset or weights required."""
import hashlib
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'provenance/package_checksums.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        p=root/row['path']
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==row['sha256'],str(p)
    print('PASS:',len(manifest['files']),'package file checksums')

if __name__=='__main__':main()
