"""验证 SmolVLA 云端训练命令。"""

from __future__ import annotations

import unittest
from pathlib import Path

from cloud.common import load_yaml_config
from cloud.train import build_parser as build_train_parser, build_train_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CloudTrainingCommandTests(unittest.TestCase):
    """验证训练配置被稳定转换为 LeRobot CLI。"""

    def test_smoke_command_locks_dataset_features_and_single_step(self) -> None:
        """smoke 模式必须推断七维特征并只训练一步。"""
        config = load_yaml_config(PROJECT_ROOT / "configs" / "cloud_train_Tencent.yaml")
        command = build_train_command(
            config,
            Path("/srv/smolvla-data/smolvla_ur10e_grounding_v2"),
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

    def test_default_dataset_is_inside_project(self) -> None:
        """未显式传参时应读取项目内的数据集目录。"""
        args = build_train_parser().parse_args([])
        self.assertEqual(args.dataset_root, PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_grounding_v2")


if __name__ == "__main__":
    unittest.main()
