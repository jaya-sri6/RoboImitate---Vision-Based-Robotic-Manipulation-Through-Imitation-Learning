"""Expert demonstration collection pipeline for RoboImitate."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import h5py

from roboimitate.env.manipulation_env import RoboImitatePickAndPlaceEnv
from roboimitate.evaluation.baseline import ScriptedPickAndPlaceExpert
from roboimitate.utils.config import load_config
from roboimitate.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def collect_demonstrations(
    num_episodes: int = 50,
    save_dir: Union[str, Path] = "data/demos",
    filename: str = "expert_demonstrations.npz",
    env_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    seed: int = 42,
    save_h5: bool = True,
    max_steps_per_episode: int = 60,
) -> Dict[str, Any]:
    """Collect expert demonstrations for tabletop robotic manipulation.

    Args:
        num_episodes: Number of demonstration episodes to collect.
        save_dir: Directory to save the dataset.
        filename: Base NPZ filename for saving.
        env_config: Environment configuration path or dictionary.
        seed: Random seed for deterministic reproducibility.
        save_h5: Whether to also save in HDF5 format.
        max_steps_per_episode: Maximum steps allowed per episode.

    Returns:
        Summary dictionary containing collection statistics.
    """
    set_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load environment configuration
    if isinstance(env_config, (str, Path)):
        cfg = load_config(env_config)
    elif isinstance(env_config, dict):
        cfg = env_config
    else:
        cfg = {}

    if "physics" not in cfg:
        cfg["physics"] = {}
    cfg["physics"]["max_episode_steps"] = max_steps_per_episode

    env = RoboImitatePickAndPlaceEnv(config=cfg)
    expert = ScriptedPickAndPlaceExpert()

    all_images: List[np.ndarray] = []
    all_proprios: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_dones: List[bool] = []
    episode_ends: List[int] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []

    successful_episodes = 0
    failed_episodes = 0
    total_transitions = 0

    print(f"Starting demonstration collection: {num_episodes} target episodes...")

    for ep in range(num_episodes):
        ep_seed = seed + ep * 100
        obs, info = env.reset(seed=ep_seed)
        expert.reset()

        ep_images: List[np.ndarray] = []
        ep_proprios: List[np.ndarray] = []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        ep_dones: List[bool] = []

        ep_success = False
        step = 0

        while True:
            step += 1
            privileged_state = env.get_privileged_state()
            action = expert.get_action(privileged_state)

            # Record current observation and the expert action taken from it
            ep_images.append(obs["image"])
            ep_proprios.append(obs["proprio"])
            ep_actions.append(action)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_rewards.append(float(reward))
            done = bool(terminated or truncated)
            ep_dones.append(done)

            obs = next_obs
            if step_info.get("success", False):
                ep_success = True

            if done:
                break

        if ep_success:
            successful_episodes += 1
        else:
            failed_episodes += 1

        all_images.extend(ep_images)
        all_proprios.extend(ep_proprios)
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_dones.extend(ep_dones)

        total_transitions += len(ep_actions)
        episode_ends.append(total_transitions)
        episode_lengths.append(len(ep_actions))
        episode_successes.append(ep_success)

        if (ep + 1) % max(1, num_episodes // 10) == 0 or (ep + 1) == num_episodes:
            print(f"  Collected {ep + 1}/{num_episodes} episodes | Success rate: {successful_episodes / (ep + 1):.1%}")

    env.close()

    # Convert to NumPy arrays
    images_np = np.stack(all_images, axis=0) # (N, H, W, 3) uint8
    proprio_np = np.stack(all_proprios, axis=0) # (N, 4) float32
    actions_np = np.stack(all_actions, axis=0) # (N, 4) float32
    rewards_np = np.array(all_rewards, dtype=np.float32)
    dones_np = np.array(all_dones, dtype=bool)
    episode_ends_np = np.array(episode_ends, dtype=np.int64)
    episode_lengths_np = np.array(episode_lengths, dtype=np.int64)
    episode_successes_np = np.array(episode_successes, dtype=bool)

    # Save to NPZ
    npz_path = save_path / filename
    np.savez_compressed(
        npz_path,
        images=images_np,
        proprio=proprio_np,
        actions=actions_np,
        rewards=rewards_np,
        dones=dones_np,
        episode_ends=episode_ends_np,
        episode_lengths=episode_lengths_np,
        episode_successes=episode_successes_np,
        seed=seed,
        num_episodes=num_episodes,
    )
    print(f"Saved NPZ dataset to {npz_path} ({images_np.shape[0]} total steps)")

    # Save to HDF5 if requested
    if save_h5:
        h5_path = save_path / (npz_path.stem + ".h5")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("images", data=images_np, compression="gzip", compression_opts=4)
            hf.create_dataset("proprio", data=proprio_np)
            hf.create_dataset("actions", data=actions_np)
            hf.create_dataset("rewards", data=rewards_np)
            hf.create_dataset("dones", data=dones_np)
            hf.create_dataset("episode_ends", data=episode_ends_np)
            hf.create_dataset("episode_lengths", data=episode_lengths_np)
            hf.create_dataset("episode_successes", data=episode_successes_np)
            hf.attrs["num_episodes"] = num_episodes
            hf.attrs["seed"] = seed
            hf.attrs["total_steps"] = int(images_np.shape[0])
            hf.attrs["success_rate"] = float(successful_episodes / num_episodes)
        print(f"Saved HDF5 dataset to {h5_path}")

    # Save metadata JSON
    meta_path = save_path / (npz_path.stem + "_metadata.json")
    metadata = {
        "num_episodes": num_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": failed_episodes,
        "success_rate": float(successful_episodes / num_episodes),
        "total_transitions": total_transitions,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "observation_shapes": {
            "image": list(images_np.shape[1:]),
            "proprio": list(proprio_np.shape[1:]),
        },
        "action_shape": list(actions_np.shape[1:]),
        "seed": seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

