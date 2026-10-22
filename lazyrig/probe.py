"""Run before editing the real-device configuration: python -m lazyrig.probe."""
import json

def main():
    try:
        import openvino as ov
        import openvino_genai
    except ImportError:
        raise SystemExit("Install requirements-openvino.txt with 64-bit Python 3.12 or 3.13, then rerun python -m lazyrig.probe.")
    core = ov.Core()
    print(json.dumps({"openvino": ov.__version__, "genai": getattr(openvino_genai, "__version__", "unknown"),
        "devices": [{"id": d, "name": str(core.get_property(d, "FULL_DEVICE_NAME"))} for d in core.available_devices]}, indent=2))

if __name__ == "__main__":
    main()
