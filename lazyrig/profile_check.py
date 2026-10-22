"""Read-only compatibility probe, executed inside an existing model runtime."""
from importlib import import_module, metadata
import json
from pathlib import Path
import sys


def check_requirements(text):
    from packaging.requirements import Requirement

    # These dependencies are installed separately from the profile's -r file.
    lines = ['torch', 'torchvision', 'numpy>=1.24,<2.5', *text.splitlines()]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith(('--index-url ', '--extra-index-url ',
                            '--index-url=', '--extra-index-url=')):
            continue
        # Unsupported requirement syntax fails closed: do not infer that a
        # runtime with unverified constraints or extras is safe to reuse.
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.extras:
            raise RuntimeError('Cannot verify installed extras for ' + requirement.name)
        installed = metadata.distribution(requirement.name)
        if not requirement.specifier.contains(installed.version, prereleases=True):
            raise RuntimeError('Installed version does not satisfy ' + str(requirement))
        if requirement.url:
            origin = json.loads(installed.read_text('direct_url.json') or '{}')
            if origin.get('url') != requirement.url:
                raise RuntimeError('Installed source does not match ' + requirement.name)


def main():
    check_requirements(Path(sys.argv[1]).read_text(encoding='utf-8'))
    for name in ('torch', 'torchvision', 'openvino', 'openvino_genai', 'optimum.intel', 'diffusers'):
        import_module(name)
    print('Installed runtime matches current profile requirements.')


if __name__ == '__main__':
    main()
