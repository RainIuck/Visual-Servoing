from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vpg_bridge.heightmap import DEFAULT_HEIGHTMAP_RESOLUTION, DEFAULT_WORKSPACE_LIMITS
from src.vpg_training.environment import SapienVPGEnvironment


PROJECT_DIR = Path(__file__).resolve().parents[1]
VPG_DIR = PROJECT_DIR / "visual-pushing-grasping-master"


def main(args: argparse.Namespace) -> None:
    if str(VPG_DIR) not in sys.path:
        sys.path.insert(0, str(VPG_DIR))

    try:
        import torch
        from logger import Logger
        from trainer import Trainer
        import utils
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "VPG training requires the original VPG Python dependencies, including "
            "torch, torchvision, scipy, opencv-python, matplotlib.\n"
            f"Missing dependency: {exc.name}"
        ) from exc

    workspace_limits = DEFAULT_WORKSPACE_LIMITS.copy()
    heightmap_resolution = args.heightmap_resolution
    np.random.seed(args.random_seed)

    env = SapienVPGEnvironment(
        num_obj=args.num_obj,
        workspace_limits=workspace_limits,
        heightmap_resolution=heightmap_resolution,
        random_seed=args.random_seed,
        save_video=args.save_video,
    )
    env.move_to_observation()
    trainer = Trainer(
        args.method,
        args.push_rewards if args.method == "reinforcement" else None,
        args.future_reward_discount,
        args.is_testing,
        args.load_snapshot,
        os.path.abspath(args.snapshot_file) if args.load_snapshot else None,
        args.force_cpu,
    )
    logger = Logger(
        args.continue_logging,
        os.path.abspath(args.logging_directory) if args.continue_logging else os.path.abspath(args.logging_directory),
    )
    logger.save_camera_info(env.cam_intrinsics, env.cam_pose, env.cam_depth_scale)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)

    if args.continue_logging:
        trainer.preload(logger.transitions_directory)

    no_change_count = [2, 2] if not args.is_testing else [0, 0]
    explore_prob = 0.5 if not args.is_testing else 0.0
    prev_state = None
    exit_called = False

    try:
        while True:
            if args.max_iterations and trainer.iteration >= args.max_iterations:
                break

            print("\n%s iteration: %d" % ("Testing" if args.is_testing else "Training", trainer.iteration))
            iteration_time_0 = time.time()
            env.check_sim()

            color_img, depth_img = env.get_camera_data()
            depth_img = depth_img * env.cam_depth_scale
            color_heightmap, depth_heightmap = utils.get_heightmap(
                color_img,
                depth_img,
                env.cam_intrinsics,
                env.cam_pose,
                workspace_limits,
                heightmap_resolution,
            )
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            logger.save_images(trainer.iteration, color_img, depth_img, "0")
            logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, "0")

            stuff_count = np.zeros(valid_depth_heightmap.shape)
            stuff_count[valid_depth_heightmap > 0.02] = 1
            empty_threshold = 10 if args.is_testing else 300
            if np.sum(stuff_count) < empty_threshold or no_change_count[0] + no_change_count[1] > 10:
                no_change_count = [0, 0]
                print("Not enough objects in view (value: %d)! Repositioning objects." % (np.sum(stuff_count)))
                env.restart_sim()
                trainer.clearance_log.append([trainer.iteration])
                logger.write_to_log("clearance", trainer.clearance_log)
                if args.is_testing and len(trainer.clearance_log) >= args.max_test_trials:
                    exit_called = True
                if exit_called:
                    break
                continue

            push_predictions, grasp_predictions, _ = trainer.forward(
                color_heightmap,
                valid_depth_heightmap,
                is_volatile=True,
            )
            selected = select_action(
                trainer=trainer,
                logger=logger,
                push_predictions=push_predictions,
                grasp_predictions=grasp_predictions,
                valid_depth_heightmap=valid_depth_heightmap,
                no_change_count=no_change_count,
                heuristic_bootstrap=args.heuristic_bootstrap,
                grasp_only=args.grasp_only,
                method=args.method,
                is_testing=args.is_testing,
                explore_prob=explore_prob,
                heightmap_resolution=heightmap_resolution,
                workspace_limits=workspace_limits,
                save_visualizations=args.save_visualizations,
                color_heightmap=color_heightmap,
            )

            if prev_state is not None:
                no_change_count = train_previous_transition(
                    trainer=trainer,
                    logger=logger,
                    prev_state=prev_state,
                    color_heightmap=color_heightmap,
                    depth_heightmap=depth_heightmap,
                    valid_depth_heightmap=valid_depth_heightmap,
                    no_change_count=no_change_count,
                    experience_replay=args.experience_replay and not args.is_testing,
                    method=args.method,
                )

                if not args.is_testing:
                    explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if args.explore_rate_decay else 0.5
                    logger.save_backup_model(trainer.model, args.method)
                    if trainer.iteration % 50 == 0:
                        logger.save_model(trainer.iteration, trainer.model, args.method)
                        if trainer.use_cuda:
                            trainer.model = trainer.model.cuda()

            primitive_position = selected["primitive_position"]
            primitive_action = selected["primitive_action"]
            best_rotation_angle = selected["best_rotation_angle"]
            if primitive_action == "push":
                push_success = env.push(primitive_position, best_rotation_angle, workspace_limits)
                grasp_success = False
                print("Push successful: %r" % (push_success))
            else:
                grasp_success = env.grasp(primitive_position, best_rotation_angle, workspace_limits)
                push_success = False
                print("Grasp successful: %r" % (grasp_success))

            prev_state = {
                "color_heightmap": color_heightmap.copy(),
                "depth_heightmap": depth_heightmap.copy(),
                "valid_depth_heightmap": valid_depth_heightmap.copy(),
                "push_predictions": push_predictions.copy(),
                "grasp_predictions": grasp_predictions.copy(),
                "primitive_action": primitive_action,
                "best_pix_ind": selected["best_pix_ind"],
                "push_success": push_success,
                "grasp_success": grasp_success,
            }

            trainer.iteration += 1
            print("Time elapsed: %f" % (time.time() - iteration_time_0))
    finally:
        env.close()


def select_action(
    *,
    trainer,
    logger,
    push_predictions,
    grasp_predictions,
    valid_depth_heightmap,
    no_change_count,
    heuristic_bootstrap,
    grasp_only,
    method,
    is_testing,
    explore_prob,
    heightmap_resolution,
    workspace_limits,
    save_visualizations,
    color_heightmap,
):
    best_push_conf = np.max(push_predictions)
    best_grasp_conf = np.max(grasp_predictions)
    print("Primitive confidence scores: %f (push), %f (grasp)" % (best_push_conf, best_grasp_conf))

    primitive_action = "grasp"
    explore_actions = False
    if not grasp_only:
        if is_testing and method == "reactive":
            if best_push_conf > 2 * best_grasp_conf:
                primitive_action = "push"
        else:
            if best_push_conf > best_grasp_conf:
                primitive_action = "push"
        explore_actions = np.random.uniform() < explore_prob
        if explore_actions:
            print("Strategy: explore (exploration probability: %f)" % explore_prob)
            primitive_action = "push" if np.random.randint(0, 2) == 0 else "grasp"
        else:
            print("Strategy: exploit (exploration probability: %f)" % explore_prob)
    trainer.is_exploit_log.append([0 if explore_actions else 1])
    logger.write_to_log("is-exploit", trainer.is_exploit_log)

    if heuristic_bootstrap and primitive_action == "push" and no_change_count[0] >= 2:
        print("Change not detected for more than two pushes. Running heuristic pushing.")
        best_pix_ind = trainer.push_heuristic(valid_depth_heightmap)
        no_change_count[0] = 0
        predicted_value = push_predictions[best_pix_ind]
        use_heuristic = True
    elif heuristic_bootstrap and primitive_action == "grasp" and no_change_count[1] >= 2:
        print("Change not detected for more than two grasps. Running heuristic grasping.")
        best_pix_ind = trainer.grasp_heuristic(valid_depth_heightmap)
        no_change_count[1] = 0
        predicted_value = grasp_predictions[best_pix_ind]
        use_heuristic = True
    else:
        use_heuristic = False
        if primitive_action == "push":
            best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
            predicted_value = np.max(push_predictions)
        else:
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            predicted_value = np.max(grasp_predictions)
    best_pix_ind = tuple(int(v) for v in best_pix_ind)

    trainer.use_heuristic_log.append([1 if use_heuristic else 0])
    logger.write_to_log("use-heuristic", trainer.use_heuristic_log)
    trainer.predicted_value_log.append([predicted_value])
    logger.write_to_log("predicted-value", trainer.predicted_value_log)

    print("Action: %s at (%d, %d, %d)" % (primitive_action, best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))
    best_rotation_angle = np.deg2rad(best_pix_ind[0] * (360.0 / trainer.model.num_rotations))
    best_pix_x = best_pix_ind[2]
    best_pix_y = best_pix_ind[1]
    primitive_position = [
        best_pix_x * heightmap_resolution + workspace_limits[0][0],
        best_pix_y * heightmap_resolution + workspace_limits[1][0],
        valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0],
    ]

    if primitive_action == "push":
        finger_width = 0.02
        safe_kernel_width = int(np.round((finger_width / 2) / heightmap_resolution))
        local_region = valid_depth_heightmap[
            max(best_pix_y - safe_kernel_width, 0) : min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]),
            max(best_pix_x - safe_kernel_width, 0) : min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1]),
        ]
        primitive_position[2] = workspace_limits[2][0] if local_region.size == 0 else np.max(local_region) + workspace_limits[2][0]

    if primitive_action == "push":
        trainer.executed_action_log.append([0, best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]])
    else:
        trainer.executed_action_log.append([1, best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]])
    logger.write_to_log("executed-action", trainer.executed_action_log)

    if save_visualizations:
        push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, best_pix_ind)
        logger.save_visualizations(trainer.iteration, push_pred_vis, "push")
        cv2.imwrite("visualization.push.png", push_pred_vis)
        grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, best_pix_ind)
        logger.save_visualizations(trainer.iteration, grasp_pred_vis, "grasp")
        cv2.imwrite("visualization.grasp.png", grasp_pred_vis)

    return {
        "primitive_action": primitive_action,
        "best_pix_ind": best_pix_ind,
        "best_rotation_angle": best_rotation_angle,
        "primitive_position": primitive_position,
    }


def train_previous_transition(
    *,
    trainer,
    logger,
    prev_state,
    color_heightmap,
    depth_heightmap,
    valid_depth_heightmap,
    no_change_count,
    experience_replay,
    method,
):
    depth_diff = abs(depth_heightmap - prev_state["depth_heightmap"])
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.01] = 0
    depth_diff[depth_diff > 0] = 1
    change_threshold = 300
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold or prev_state["grasp_success"]
    print("Change detected: %r (value: %d)" % (change_detected, change_value))

    if change_detected:
        if prev_state["primitive_action"] == "push":
            no_change_count[0] = 0
        elif prev_state["primitive_action"] == "grasp":
            no_change_count[1] = 0
    else:
        if prev_state["primitive_action"] == "push":
            no_change_count[0] += 1
        elif prev_state["primitive_action"] == "grasp":
            no_change_count[1] += 1

    label_value, prev_reward_value = trainer.get_label_value(
        prev_state["primitive_action"],
        prev_state["push_success"],
        prev_state["grasp_success"],
        change_detected,
        prev_state["push_predictions"],
        prev_state["grasp_predictions"],
        color_heightmap,
        valid_depth_heightmap,
    )
    trainer.label_value_log.append([label_value])
    logger.write_to_log("label-value", trainer.label_value_log)
    trainer.reward_value_log.append([prev_reward_value])
    logger.write_to_log("reward-value", trainer.reward_value_log)
    trainer.backprop(
        prev_state["color_heightmap"],
        prev_state["valid_depth_heightmap"],
        prev_state["primitive_action"],
        prev_state["best_pix_ind"],
        label_value,
    )

    if experience_replay:
        do_experience_replay(
            trainer=trainer,
            logger=logger,
            method=method,
            prev_primitive_action=prev_state["primitive_action"],
            prev_reward_value=prev_reward_value,
        )
    return no_change_count


def do_experience_replay(*, trainer, logger, method, prev_primitive_action, prev_reward_value):
    sample_primitive_action = prev_primitive_action
    if sample_primitive_action == "push":
        sample_primitive_action_id = 0
        if method == "reactive":
            sample_reward_value = 0 if prev_reward_value == 1 else 1
        else:
            sample_reward_value = 0 if prev_reward_value == 0.5 else 0.5
    else:
        sample_primitive_action_id = 1
        sample_reward_value = 0 if prev_reward_value == 1 else 1

    sample_ind = np.argwhere(
        np.logical_and(
            np.asarray(trainer.reward_value_log)[1 : trainer.iteration, 0] == sample_reward_value,
            np.asarray(trainer.executed_action_log)[1 : trainer.iteration, 0] == sample_primitive_action_id,
        )
    )
    if sample_ind.size == 0:
        print("Not enough prior training samples. Skipping experience replay.")
        return

    if method == "reactive":
        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - (1 - sample_reward_value))
    else:
        sample_surprise_values = np.abs(
            np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]]
            - np.asarray(trainer.label_value_log)[sample_ind[:, 0]]
        )
    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
    rand_sample_ind = int(np.round(np.random.power(2, 1) * (sample_ind.size - 1)))
    sample_iteration = sorted_sample_ind[rand_sample_ind]
    print("Experience replay: iteration %d (surprise value: %f)" % (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

    sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, "%06d.0.color.png" % sample_iteration))
    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
    sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, "%06d.0.depth.png" % sample_iteration), -1)
    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000
    sample_push_predictions, sample_grasp_predictions, _ = trainer.forward(sample_color_heightmap, sample_depth_heightmap, is_volatile=True)
    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
    trainer.backprop(
        sample_color_heightmap,
        sample_depth_heightmap,
        sample_primitive_action,
        sample_best_pix_ind,
        trainer.label_value_log[sample_iteration],
    )
    if sample_primitive_action == "push":
        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
    else:
        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VPG in SAPIEN/Panda while preserving the original VPG learning loop."
    )
    parser.add_argument("--num_obj", "--num-objects", dest="num_obj", type=int, default=8)
    parser.add_argument("--heightmap_resolution", "--heightmap-resolution", dest="heightmap_resolution", type=float, default=0.002)
    parser.add_argument("--random_seed", "--random-seed", dest="random_seed", type=int, default=1234)
    parser.add_argument("--cpu", dest="force_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", dest="force_cpu", action="store_false")
    parser.add_argument("--max_iterations", "--max-iterations", dest="max_iterations", type=int, default=0)

    parser.add_argument("--method", dest="method", default="reinforcement")
    parser.add_argument("--push_rewards", "--push-rewards", dest="push_rewards", action="store_true", default=False)
    parser.add_argument("--future_reward_discount", "--future-reward-discount", dest="future_reward_discount", type=float, default=0.5)
    parser.add_argument("--experience_replay", "--experience-replay", dest="experience_replay", action="store_true", default=False)
    parser.add_argument("--heuristic_bootstrap", "--heuristic-bootstrap", dest="heuristic_bootstrap", action="store_true", default=False)
    parser.add_argument("--explore_rate_decay", "--explore-rate-decay", dest="explore_rate_decay", action="store_true", default=False)
    parser.add_argument("--grasp_only", "--grasp-only", dest="grasp_only", action="store_true", default=False)

    parser.add_argument("--is_testing", "--is-testing", dest="is_testing", action="store_true", default=False)
    parser.add_argument("--max_test_trials", "--max-test-trials", dest="max_test_trials", type=int, default=30)

    parser.add_argument("--load_snapshot", "--load-snapshot", dest="load_snapshot", action="store_true", default=False)
    parser.add_argument("--snapshot_file", "--snapshot-file", dest="snapshot_file")
    parser.add_argument("--continue_logging", "--continue-logging", dest="continue_logging", action="store_true", default=False)
    parser.add_argument("--logging_directory", "--logging-directory", dest="logging_directory", default="logs")
    parser.add_argument("--save_visualizations", "--save-visualizations", dest="save_visualizations", action="store_true", default=False)
    parser.add_argument("--save_video", "--save-video", dest="save_video", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
