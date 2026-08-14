from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an explicit episode outcome manifest for ACT RL")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assume-all-success", action="store_true",
                        help="Only for a dataset already known to contain successful episodes")
    args = parser.parse_args()
    if not args.assume_all_success:
        raise SystemExit(
            "Refusing to invent RL outcomes. Supply a labeled manifest, or add --assume-all-success "
            "only for the filtered success dataset. Add failed episodes before formal IQL training."
        )
    info = json.loads((Path(args.dataset_root) / "meta" / "info.json").read_text())
    episodes = {
        str(index): {"success": True, "terminal_reward": 1.0, "source": "filtered_success"}
        for index in range(int(info["total_episodes"]))
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"format_version": 1, "warning": "success-only; not failure-aware RL",
                                  "episodes": episodes}, indent=2, ensure_ascii=False))
    print(f"Wrote {len(episodes)} episode outcomes to {output}")


if __name__ == "__main__":
    main()
