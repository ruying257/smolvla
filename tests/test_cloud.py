"""验证 SmolVLA 云端训练命令。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cloud.common import load_yaml_config
from cloud.train import build_parser as build_train_parser, build_train_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CloudTrainingCommandTests(unittest.TestCase):
    """验证训练配置被稳定转换为 LeRobot CLI。"""

    def test_smoke_command_locks_dataset_features_and_single_step(self) -> None:
        """smoke 模式必须推断七维特征并只训练一步。"""
        config = load_yaml_config(PROJECT_ROOT / "configs" / "train" / "mug_b8_s8000.yaml")
        command = build_train_command(
            config,
            Path("/srv/smolvla-data/smolvla_ur10e_mug_v1"),
            Path("/srv/smolvla/outputs/smoke/train"),
            "smoke",
            smoke=True,
        )
        self.assertIn("--policy.input_features=null", command)
        self.assertIn("--policy.output_features=null", command)
        self.assertIn("--policy.empty_cameras=1", command)
        self.assertIn("--steps=1", command)
        self.assertIn("--batch_size=1", command)
        self.assertIn("--policy.use_amp=true", command)
        self.assertIn("--policy.push_to_hub=false", command)

    def test_resume_command_restores_state(self) -> None:
        """resume 模式必须恢复 step/优化器/调度器：--resume=true + --config_path，且不带 --policy.path。"""
        config = load_yaml_config(PROJECT_ROOT / "configs" / "train" / "mug_b8_s12000_resume.yaml")
        checkpoint = Path("/srv/smolvla/outputs/train/smolvla_ur10e_mug_v1_b8_s8000/checkpoints/last")
        command = build_train_command(
            config,
            Path("/srv/smolvla-data/smolvla_ur10e_mug_v1"),
            Path("/srv/smolvla/outputs/train/smolvla_ur10e_mug_v1_b8_s12000"),
            "smolvla_ur10e_mug_v1_b8_s12000",
            resume_from=checkpoint,
        )
        joined = " ".join(command)
        self.assertIn("--resume=true", command)
        self.assertIn(
            f"--config_path={checkpoint / 'pretrained_model' / 'train_config.json'}", command
        )
        self.assertIn("--scheduler.num_warmup_steps=333", command)
        self.assertIn("--scheduler.num_decay_steps=10000", command)
        self.assertIn("--steps=12000", command)
        self.assertNotIn("--policy.path", joined)

    def test_resume_with_smoke_raises(self) -> None:
        """resume 与 smoke 互斥：恢复的 step>0 时 smoke 的 steps=1 无法生效。"""
        config = load_yaml_config(PROJECT_ROOT / "configs" / "train" / "mug_b8_s12000_resume.yaml")
        with self.assertRaises(ValueError):
            build_train_command(
                config,
                Path("/srv/smolvla-data/smolvla_ur10e_mug_v1"),
                Path("/srv/smolvla/outputs/train/smolvla_ur10e_mug_v1_b8_s12000"),
                "job",
                smoke=True,
                resume_from=Path("/srv/smolvla/outputs/train/xxx/checkpoints/last"),
            )

    def test_image_transforms_serialized_as_json(self) -> None:
        """dataset.image_transforms 必须序列化为 --dataset.image_transforms.tfs=<JSON>。

        draccus 不支持嵌套的 ``tfs.<name>.*`` CLI 参数，整个 tfs 字典必须作为
        单个 JSON 字符串传入；enable/max_num_transforms 单独传参。
        """
        config = load_yaml_config(PROJECT_ROOT / "configs" / "train" / "mug_b8_s15000_dr.yaml")
        command = build_train_command(
            config,
            Path("/srv/smolvla-data/smolvla_ur10e_mug_v1"),
            Path("/srv/smolvla/outputs/train/smolvla_ur10e_mug_v1_b8_s11000_dr"),
            "smolvla_ur10e_mug_v1_b8_s11000_dr",
        )
        self.assertIn("--dataset.image_transforms.enable=true", command)
        self.assertIn("--dataset.image_transforms.max_num_transforms=3", command)
        prefix = "--dataset.image_transforms.tfs="
        tfs_arg = next(arg for arg in command if arg.startswith(prefix))
        payload = json.loads(tfs_arg[len(prefix):])
        self.assertIn("gaussian_noise", payload)
        self.assertEqual(payload["brightness"]["kwargs"]["brightness"], [0.6, 1.5])
        self.assertEqual(payload["contrast"]["kwargs"]["contrast"], [0.6, 1.4])
        self.assertEqual(payload["gaussian_noise"]["kwargs"]["sigma"], 0.06)
        self.assertEqual(payload["gaussian_blur"]["kwargs"]["sigma"], [0.5, 3.0])
        self.assertEqual(payload["gaussian_blur"]["kwargs"]["kernel_size"], 5)

    def test_default_dataset_is_inside_project(self) -> None:
        """未显式传参时应读取项目内的数据集目录。"""
        args = build_train_parser().parse_args([])
        self.assertEqual(args.dataset_root, PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_grounding_v2")


if __name__ == "__main__":
    unittest.main()
