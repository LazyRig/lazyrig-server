"""Entry point executed only in the conversion environment."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--precision", choices=["fp16", "int8", "int4"], required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    # Use the supported Optimum CLI, with local weights and no remote-code flag.
    sys.argv = ["optimum-cli", "export", "openvino", "--model", args.source,
                "--task", args.task, "--weight-format", args.precision]
    if args.task == "text-to-image":
        sys.argv += ["--library", "diffusers"]
    if args.profile == "qwen-image-21" and args.precision == "int4":
        sys.argv += ["--group-size-fallback", "ignore"]
    sys.argv.append(args.output)
    from optimum.commands.optimum_cli import main as optimum_main
    optimum_main()


if __name__ == "__main__":
    main()
