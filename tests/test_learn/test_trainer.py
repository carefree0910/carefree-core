import torch
import pytest
import unittest

import core.learn as cflearn

from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock


class TestTrainer(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _prepare_training_case(self, tmp_path: Path) -> None:
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(4, use_validation=True)
        config = cflearn.Config(
            workspace=str(tmp_path / "workspace"),
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
        self.tmp_path = tmp_path
        self.data = data
        self.config = config.to_debug()

    def test_functions(self):
        workspace = self.tmp_path / "functions"
        workspace.mkdir()
        metrics_path = workspace / cflearn.Trainer.metrics_log_file
        self.assertEqual(cflearn.trainer.get_metrics_path(workspace), metrics_path)
        self.assertFalse(cflearn.trainer.is_started_workspace(workspace))
        self.assertFalse(cflearn.trainer.is_finished_workspace(workspace))
        metrics_path.touch()
        self.assertTrue(cflearn.trainer.is_started_workspace(workspace))
        self.assertFalse(cflearn.trainer.is_finished_workspace(workspace))
        self.assertTrue(cflearn.trainer.is_crashed_workspace(workspace))

    def test_metrics_progress_without_callback(self) -> None:
        trainer = cflearn.Trainer(self.config)
        expected = cflearn.MetricsOutputs(0.0, {}, {})
        trainer.state = Mock(is_terminate=False)
        trainer.model = Mock()
        trainer.model.evaluate.return_value = Mock(metric_outputs=expected)
        trainer.metrics = Mock()
        trainer.inference = Mock()
        trainer.callbacks = []
        trainer.accelerator = Mock(is_local_main_process=True)
        trainer.tqdm_settings.use_tqdm_in_validation = True
        with patch("core.learn.trainer.Progress") as mock_progress:
            output = trainer.get_metrics(Mock())
        self.assertIs(output, expected)
        mock_progress.assert_called_once_with()

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

    def test_scheduler_post_prepare_refresh(self) -> None:
        refresh_calls = []

        @cflearn.register_scheduler("post_prepare_refresh")
        class PostPrepareRefreshScheduler(torch.optim.lr_scheduler.LRScheduler):
            def get_lr(self):
                return self.base_lrs

            def load_state_dict(self, state_dict):
                refresh_calls.append(state_dict.copy())
                super().load_state_dict(state_dict)

        config = self.config.copy()
        config.num_steps = 1
        config.scheduler_name = "post_prepare_refresh"
        config.scheduler_config = {}
        pipeline = cflearn.TrainingPipeline.init(config).fit(self.data)
        scheduler = pipeline.training.build_optimizers.schedulers["all"]

        self.assertIsInstance(scheduler, PostPrepareRefreshScheduler)
        self.assertEqual(len(refresh_calls), 1)
        self.assertIn("last_epoch", refresh_calls[0])
        self.assertNotIn("optimizer", refresh_calls[0])

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
        config.callback_names = ["test"]
        cflearn.TrainingPipeline.init(config).fit(self.data)

        with patch("core.learn.trainer.is_dist_initialized") as mock:
            mock.return_value = True
            with self.assertRaises(KeyboardInterrupt):
                cflearn.TrainingPipeline.init(config).fit(self.data)

    def test_finetune(self):
        config = self.config.copy()
        m = cflearn.IModel.from_config(config)
        path = self.tmp_path / "model.pt"
        m.save(str(path))
        config.finetune_config = dict(pretrained_ckpt=str(path))
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
            workspace=str(self.tmp_path / "monitor"),
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

    def test_monitor_evaluates_every_decision(self) -> None:
        events = []

        class RecordingMonitor(cflearn.TrainerMonitor):
            def __init__(
                self,
                name: str,
                snapshot: bool,
                terminate: bool,
            ) -> None:
                self.name = name
                self.snapshot = snapshot
                self.terminate = terminate

            def should_snapshot(self, new_score: float) -> bool:
                events.append((self.name, "snapshot", new_score))
                return self.snapshot

            def should_terminate(self, new_score: float) -> bool:
                events.append((self.name, "terminate", new_score))
                return self.terminate

        metrics = cflearn.MetricsOutputs(1.0, {}, {})
        trainer = cflearn.Trainer(self.config)
        trainer.state = Mock(
            should_extend_epoch=False,
            should_monitor=True,
            should_start_snapshot=True,
            can_snapshot=True,
        )
        trainer.monitors = [
            RecordingMonitor("first", True, True),
            RecordingMonitor("second", False, False),
        ]
        trainer.callbacks = []
        trainer.get_metrics = Mock(return_value=metrics)
        trainer.log_with = Mock()
        step_outputs = cflearn.StepOutputs({}, {})

        with patch("core.learn.trainer.is_ddp", return_value=False):
            monitored = trainer.monitor(Mock(), Mock(), step_outputs)

        self.assertEqual(
            events,
            [
                ("first", "snapshot", 1.0),
                ("second", "snapshot", 1.0),
                ("first", "terminate", 1.0),
                ("second", "terminate", 1.0),
            ],
        )
        self.assertTrue(monitored.save_checkpoint)
        self.assertTrue(monitored.terminate)
        self.assertIs(monitored.metric_outputs, metrics)
        trainer.state.update_snapshot_epoch.assert_called_once_with()

        events.clear()
        trainer.state.update_snapshot_epoch.reset_mock()
        with patch("core.learn.trainer.is_ddp", return_value=True):
            monitored = trainer.monitor(Mock(), Mock(), step_outputs)

        self.assertEqual(
            events,
            [
                ("first", "snapshot", 1.0),
                ("second", "snapshot", 1.0),
            ],
        )
        self.assertTrue(monitored.save_checkpoint)
        self.assertFalse(monitored.terminate)
        trainer.state.update_snapshot_epoch.assert_called_once_with()

    def test_monitor_updates_state_regardless_of_order(self) -> None:
        metrics = cflearn.MetricsOutputs(1.0, {}, {})
        trainer = cflearn.Trainer(self.config)
        trainer.state = Mock(
            should_extend_epoch=False,
            should_monitor=True,
            should_start_snapshot=True,
            can_snapshot=True,
        )
        trainer.callbacks = []
        trainer.get_metrics = Mock(return_value=metrics)
        trainer.log_with = Mock()
        step_outputs = cflearn.StepOutputs({}, {})

        for stateful_first in [False, True]:
            with self.subTest(stateful_first=stateful_first):
                mean_std = cflearn.MeanStdMonitor()
                plateau = cflearn.PlateauMonitor()
                conservative = cflearn.ConservativeMonitor()
                if stateful_first:
                    trainer.monitors = [mean_std, plateau, conservative]
                else:
                    trainer.monitors = [conservative, mean_std, plateau]

                with patch("core.learn.trainer.is_ddp", return_value=False):
                    monitored = trainer.monitor(Mock(), Mock(), step_outputs)

                self.assertListEqual(mean_std.history, [1.0])
                self.assertEqual(mean_std._incrementer.num_record, 1)
                self.assertListEqual(plateau.history, [1.0])
                self.assertEqual(plateau._incrementer.num_record, 1)
                self.assertTrue(monitored.save_checkpoint)


if __name__ == "__main__":
    unittest.main()
