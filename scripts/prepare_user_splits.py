from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cabria.data.personalized_dataset import load_personalized_frame
from cabria.data.user_split import UserSplitConfig, build_episode_specs, serialize_episode_specs, split_users
from cabria.utils.common import dump_json, load_yaml, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CABRIA user splits and support/query episodes.")
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data_flickr_aes_prior_remote.yaml"))
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    set_seed(int(data_config["user_split"]["seed"]))

    personalized_cfg = data_config["personalized_dataset"]
    dataset_name = personalized_cfg["name"]
    dataset_root = personalized_cfg["root"]
    print(f"[CABRIA] Preparing user splits for dataset: {dataset_name}", flush=True)
    print(f"[CABRIA] Dataset root: {dataset_root}", flush=True)

    frame = load_personalized_frame(personalized_cfg["name"], personalized_cfg["root"])
    print(f"[CABRIA] Usable personalized samples: {len(frame)}", flush=True)
    print(f"[CABRIA] Unique users after filtering: {frame['user_id'].nunique()}", flush=True)

    user_split_cfg = data_config["user_split"]
    split_config = UserSplitConfig(
        seed=int(user_split_cfg["seed"]),
        support_sizes=[int(value) for value in data_config["support_sizes"]],
        protocol=str(user_split_cfg.get("protocol", "flickr_aes_prior_173_37")),
        train_user_count=user_split_cfg.get("train_user_count"),
        val_user_count=user_split_cfg.get("val_user_count"),
        test_user_count=user_split_cfg.get("test_user_count"),
        val_from_train_users=bool(user_split_cfg.get("val_from_train_users", False)),
        min_train_user_rows=user_split_cfg.get("min_train_user_rows"),
        min_val_user_rows=user_split_cfg.get("min_val_user_rows"),
        min_test_user_rows=user_split_cfg.get("min_test_user_rows"),
    )
    print(
        "[CABRIA] Split protocol: "
        f"{split_config.protocol}; "
        f"counts train={split_config.train_user_count}, val={split_config.val_user_count}, "
        f"test={split_config.test_user_count}, val_from_train={split_config.val_from_train_users}",
        flush=True,
    )
    print(f"[CABRIA] Support sizes: {split_config.support_sizes}", flush=True)
    splits = split_users(frame, split_config)
    payload = {"splits": splits, "episodes": {}}
    for split_name, users in splits.items():
        default_episodes = data_config["user_split"].get("train_episodes_per_user", 1) if split_name == "train_fit" else 1
        episodes_per_user = int(data_config["user_split"].get(f"{split_name}_episodes_per_user", default_episodes))
        print(f"[CABRIA] Building episodes for split={split_name}, users={len(users)}", flush=True)
        print(f"[CABRIA] Episodes per user for {split_name}: {episodes_per_user}", flush=True)
        split_episodes = build_episode_specs(
            frame,
            users,
            split_config.support_sizes,
            seed=split_config.seed,
            episodes_per_user=episodes_per_user,
        )
        payload["episodes"][split_name] = serialize_episode_specs(split_episodes)
        summary = ", ".join(
            f"{support_size}-shot={len(specs)}"
            for support_size, specs in split_episodes.items()
        )
        print(f"[CABRIA] Episode summary for {split_name}: {summary}", flush=True)

    output_path = data_config["user_split"]["split_file"]
    dump_json(payload, output_path)
    print(f"[CABRIA] Saved split manifest to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
