import os
import tempfile
import unittest

import core.learn as cflearn

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from unittest.mock import Mock


class TestTrainer(unittest.TestCase):
    def setUp(self) -> None:
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(4, use_validation=True)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            scheduler_name="warmup",
            scheduler_config=dict(warmup_step=2),
            monitor_names="conservative",
            loss_name="mse",
            clip_norm=1.0,
            state_config=dict(max_snapshot_file=5),
            tqdm_settings=cflearn.TqdmSettings(),
        )
        self.data = data
        self.config = config.to_debug()

    def test_functions(self):
        with TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            metrics_path = workspace / cflearn.Trainer.metrics_log_file
            self.assertEqual(cflearn.trainer.get_metrics_path(workspace), metrics_path)
            self.assertFalse(cflearn.trainer.is_started_workspace(workspace))
            self.assertFalse(cflearn.trainer.is_finished_workspace(workspace))
            metrics_path.touch()
            self.assertTrue(cflearn.trainer.is_started_workspace(workspace))
            self.assertFalse(cflearn.trainer.is_finished_workspace(workspace))
            self.assertTrue(cflearn.trainer.is_crashed_workspace(workspace))

    def test_training(self):
        config = self.config.copy()
        config.num_steps = 10
        config.update_scheduler_per_epoch = True
        p = cflearn.TrainingPipeline.init(config).fit(self.data)
        trainer = p.training.build_trainer.trainer
        ckpt_folder = Path(trainer.checkpoint_folder)
        checkpoints = cflearn.trainer.get_sorted_checkpoints(ckpt_folder)
        (ckpt_folder / checkpoints[0]).unlink()
        trainer.restore_checkpoint(state_dict_callback=lambda _: None)
        trainer.state = None
        trainer.save_checkpoint(float("nan"))
        with self.assertRaises(AttributeError):
            trainer.log_with(None)
        trainer.accelerator = Mock()
        trainer.accelerator.is_local_main_process = False
        self.assertFalse(trainer.use_tqdm_in_validation)
        trainer.log_with(None)

    def test_tqdm(self):
        config = self.config.copy()
        config.num_steps = 10
        cflearn.TrainingPipeline.init(config).fit(self.data)
        tqdm_settings = cflearn.TqdmSettings()
        tqdm_settings.use_tqdm = True
        config.tqdm_settings = tqdm_settings.asdict()
        cflearn.TrainingPipeline.init(config).fit(self.data)
        tqdm_settings.use_step_tqdm = True
        config.tqdm_settings = tqdm_settings.asdict()
        cflearn.TrainingPipeline.init(config).fit(self.data)
        tqdm_settings.use_tqdm_in_validation = True
        config.tqdm_settings = tqdm_settings.asdict()
        cflearn.TrainingPipeline.init(config).fit(self.data)

    def test_keyboard_interrupt(self):
        @cflearn.TrainerCallback.register("test", allow_duplicate=True)
        class _(cflearn.TrainerCallback):
            def after_train_step(self, batch, step_outputs, trainer):
                raise KeyboardInterrupt

        config = self.config.copy()
        config.callback_names = "test"
        cflearn.TrainingPipeline.init(config).fit(self.data)

        with patch("core.learn.trainer.is_dist_initialized") as mock:
            mock.return_value = True
            with self.assertRaises(KeyboardInterrupt):
                cflearn.TrainingPipeline.init(config).fit(self.data)

    def test_finetune(self):
        config = self.config.copy()
        m = cflearn.IModel.from_config(config)
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "model.pt")
            m.save(path)
            config.finetune_config = dict(pretrained_ckpt=path)
            cflearn.TrainingPipeline.init(config).fit(self.data)
            pattern = r".*\.weight"
            config.finetune_config["freeze"] = pattern
            cflearn.TrainingPipeline.init(config).fit(self.data)
            config.finetune_config["freeze_except"] = pattern
            with self.assertRaises(ValueError):
                cflearn.TrainingPipeline.init(config).fit(self.data)
            config.finetune_config.pop("freeze")
            cflearn.TrainingPipeline.init(config).fit(self.data)
            config.finetune_config = {}
            with self.assertRaises(ValueError):
                cflearn.TrainingPipeline.init(config).fit(self.data)

    def test_monitor(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(4)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config.to_debug().num_steps = 10
        cflearn.TrainingPipeline.init(config).fit(data)
        config.use_incrementer_for_train_losses_in_eval = False
        cflearn.TrainingPipeline.init(config).fit(data)
        config.recompute_train_losses_in_eval = False
        cflearn.TrainingPipeline.init(config).fit(data)

        @cflearn.TrainerMonitor.register("foo")
        class FooMonitor(cflearn.TrainerMonitor):
            def should_snapshot(self, new_score: float) -> bool:
                return False

            def should_terminate(self, new_score: float) -> bool:
                return True

        config.monitor_names = "foo"
        cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
