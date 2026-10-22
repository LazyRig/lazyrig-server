"""Build the end-user release and, optionally, a separate GitHub creator kit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
USER_ROOT_FILES = {
    'README.md', 'CHANGELOG.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'requirements-openvino.txt',
    'requirements-converter.txt', 'requirements-qwen-image-21.txt',
    'setup-windows.bat', 'setup-portable.bat',
    'setup-windows.ps1', 'setup.sh', 'start-server.bat', 'start-server.sh', 'update-runtime.bat', 'diagnose-gpu.bat',
}
USER_DOCS = {
    'GETTING-STARTED.md', 'WINDOWS-SETUP.md', 'CONNECTING.md', 'API.md', 'BETA-TEST.md',
    'MODEL-SETTINGS.md', 'IMAGE-SETTINGS.md', 'PROMPT-ENHANCEMENT.md', 'MODEL-LICENSES.md', 'RUNTIME-UPDATES.md',
}
USER_TREE_PATTERNS = {
    'lazyrig': ('*.py', 'prompts/image_generation.txt', 'prompts/image_editing.txt',
                'web/*.html', 'web/*.js', 'web/*.css', 'web/*.svg', 'web/*.png'),
    'docs': ('images/tutorial/*.png',), 'configs': ('*.example.json',),
}
CREATOR_ROOT_FILES = {
    '.gitignore', '.gitattributes', '.editorconfig', 'CONTRIBUTING.md',
    'requirements-dev.txt', 'CREATOR-START-HERE.md',
}
CREATOR_TREE_PATTERNS = {
    'tests': ('*.py',), 'tools': ('*.py',), 'maintainer': ('*.md', 'publication/*.md'),
    '.github': ('*.md',), '.github/ISSUE_TEMPLATE': ('*.md', '*.yml', '*.yaml'),
    '.github/workflows': ('*.yml', '*.yaml'),
}


def _select(root, names, trees):
    root = Path(root)
    files = {root / name for name in names if (root / name).is_file()}
    for directory, patterns in trees.items():
        for pattern in patterns:
            files.update((root / directory).glob(pattern))
    return {p for p in files if p.is_file() and not p.is_symlink()
            and not any((root / parent).is_symlink() for parent in p.relative_to(root).parents)}


def release_files(root=ROOT):
    """Select only runnable application files and end-user documentation."""
    names = USER_ROOT_FILES | {'docs/' + name for name in USER_DOCS} | {'tools/image_job.py'}
    return sorted(_select(root, names, USER_TREE_PATTERNS))


def repository_files(root=ROOT):
    """Select the complete publishable repository, without local runtime data."""
    return sorted(set(release_files(root)) | _select(root, CREATOR_ROOT_FILES, CREATOR_TREE_PATTERNS))


def _build(destination, root, creator):
    root = Path(root)
    files = repository_files(root) if creator else release_files(root)
    required = USER_ROOT_FILES | {'docs/' + name for name in USER_DOCS} | {
        'lazyrig/__init__.py', 'lazyrig/__main__.py', 'lazyrig/server.py', 'lazyrig/profile_check.py',
        'lazyrig/web/index.html', 'lazyrig/web/ui.js', 'lazyrig/web/redesign.css',
        'lazyrig/prompts/image_generation.txt',
        'lazyrig/prompts/image_editing.txt', 'tools/image_job.py',
    }
    if creator:
        required |= CREATOR_ROOT_FILES | {'tools/build_release.py', 'maintainer/PUBLISHING.md',
                                         'maintainer/publication/GITHUB-RELEASE.md'}
    missing = required - {p.relative_to(root).as_posix() for p in files}
    if missing:
        raise ValueError('Required package files missing: ' + ', '.join(sorted(missing)))
    version_match = re.search(r'__version__ = "([0-9A-Za-z.-]+)"',
                              (root / 'lazyrig/__init__.py').read_text())
    if version_match is None:
        raise ValueError('A safe release version is required in lazyrig/__init__.py')
    version = version_match[1]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    label = 'Creator_Kit' if creator else 'Server'
    archive = destination / f'LazyRig_{label}_v{version}.zip'
    payload = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    payload['MANIFEST.sha256.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo('lazyrig-server/' + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            zipped.writestr(info, data, compresslevel=9)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Archive integrity verification failed')
        for name, digest in manifest.items():
            if hashlib.sha256(zipped.read('lazyrig-server/' + name)).hexdigest() != digest:
                raise ValueError('Archive hash mismatch: ' + name)
    print(f'{archive.resolve()}\n{len(payload)} files; {archive.stat().st_size:,} bytes\nSHA256 {hashlib.sha256(archive.read_bytes()).hexdigest()}')
    return archive


def build(destination, root=ROOT):
    return _build(destination, root, creator=False)


def build_creator_kit(destination, root=ROOT):
    return _build(destination, root, creator=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(ROOT / 'dist'))
    parser.add_argument('--include-creator-kit', action='store_true',
                        help='Also build a complete GitHub repository with maintainer files')
    args = parser.parse_args()
    build(args.output_dir)
    if args.include_creator_kit:
        build_creator_kit(args.output_dir)
