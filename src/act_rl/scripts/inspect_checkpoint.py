from __future__ import annotations

import argparse
import json

from act_rl.checkpoint import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
