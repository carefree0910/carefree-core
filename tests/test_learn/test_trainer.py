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

    def test_callback_cleanup_before_loop_failure(self) -> None:
        events = []
        failure_stage = ""
        loop_error = RuntimeError("before loop failed")

        @cflearn.TrainerCallback.register("before_loop_failure", allow_duplicate=True)
        class BeforeLoopFailureCallback(cflearn.TrainerCallback):
            def after_workspace_prepared(self, trainer):
                events.append("failure.after_workspace_prepared")
                if failure_stage == "after_workspace_prepared":
                    raise loop_error

            def before_summary(self, trainer):
                events.append("failure.before_summary")
                if failure_stage == "before_summary":
                    raise loop_error

            def before_loop(self, trainer):
                events.append("failure.before_loop")
                if failure_stage == "before_loop":
                    raise loop_error

            def before_loop_with_loaders(self, trainer, train_loader, valid_loader):
                events.append("failure.before_loop_with_loaders")
                raise loop_error

            def after_loop(self, trainer):
                events.append("failure.after_loop")

            def finalize(self, trainer):
                events.append("failure.finalize")

        @cflearn.TrainerCallback.register("before_loop_follower", allow_duplicate=True)
        class BeforeLoopFollowerCallback(cflearn.TrainerCallback):
            def after_workspace_prepared(self, trainer):
                events.append("follower.after_workspace_prepared")

            def before_summary(self, trainer):
                events.append("follower.before_summary")

            def before_loop(self, trainer):
                events.append("follower.before_loop")

            def before_loop_with_loaders(self, trainer, train_loader, valid_loader):
                events.append("follower.before_loop_with_loaders")

            def after_loop(self, trainer):
                events.append("follower.after_loop")

            def finalize(self, trainer):
                events.append("follower.finalize")

        for failure_stage in [
            "after_workspace_prepared",
            "before_summary",
            "before_loop",
            "before_loop_with_loaders",
        ]:
            with self.subTest(failure_stage=failure_stage):
                events.clear()
                loop_error = RuntimeError(f"{failure_stage} failed")
                config = self.config.copy()
                config.workspace = str(self.tmp_path / failure_stage)
                config.auto_callback = False
                config.callback_names = [
                    "before_loop_failure",
                    "before_loop_follower",
                ]
                pipeline = cflearn.TrainingPipeline.init(config)
                with patch.object(
                    cflearn.Trainer,
                    "get_metrics",
                ) as get_metrics, patch.object(
                    cflearn.Trainer,
                    "restore_checkpoint",
                ) as restore_checkpoint, patch.object(
                    cflearn.Trainer,
                    "save_checkpoint",
                ) as save_checkpoint:
                    with self.assertRaises(RuntimeError) as context:
                        pipeline.fit(self.data, do_summary=False)

                self.assertIs(context.exception, loop_error)
                if failure_stage == "after_workspace_prepared":
                    expected_events = [
                        "failure.after_workspace_prepared",
                        "failure.finalize",
                        "follower.finalize",
                    ]
                elif failure_stage == "before_summary":
                    expected_events = [
                        "failure.after_workspace_prepared",
                        "follower.after_workspace_prepared",
                        "failure.before_summary",
                        "failure.finalize",
                        "follower.finalize",
                    ]
                elif failure_stage == "before_loop":
                    expected_events = [
                        "failure.after_workspace_prepared",
                        "follower.after_workspace_prepared",
                        "failure.before_summary",
                        "follower.before_summary",
                        "failure.before_loop",
                        "failure.finalize",
                        "follower.finalize",
                    ]
                else:
                    expected_events = [
                        "failure.after_workspace_prepared",
                        "follower.after_workspace_prepared",
                        "failure.before_summary",
                        "follower.before_summary",
                        "failure.before_loop",
                        "follower.before_loop",
                        "failure.before_loop_with_loaders",
                        "failure.finalize",
                        "follower.finalize",
                    ]
                self.assertListEqual(events, expected_events)
                get_metrics.assert_not_called()
                restore_checkpoint.assert_not_called()
                save_checkpoint.assert_not_called()
                trainer = pipeline.training.build_trainer.trainer
                if failure_stage == "after_workspace_prepared":
                    self.assertIsNone(trainer.state)
                else:
                    self.assertFalse(trainer.state.is_terminate)

    def test_callback_cleanup_preserves_training_error(self) -> None:
        events = []
        training_error = RuntimeError("training failed")
        after_loop_error = RuntimeError("after loop failed")
        finalize_error = ValueError("finalize failed")

        @cflearn.TrainerCallback.register("training_failure", allow_duplicate=True)
        class TrainingFailureCallback(cflearn.TrainerCallback):
            def after_train_step(self, batch, step_outputs, trainer):
                events.append("failure.train")
                raise training_error

            def after_loop(self, trainer):
                events.append("failure.after_loop")
                raise after_loop_error

            def finalize(self, trainer):
                events.append("failure.finalize")
                raise finalize_error

        @cflearn.TrainerCallback.register(
            "training_failure_follower", allow_duplicate=True
        )
        class TrainingFailureFollowerCallback(cflearn.TrainerCallback):
            def after_loop(self, trainer):
                events.append("follower.after_loop")
                raise RuntimeError("follower after loop failed")

            def finalize(self, trainer):
                events.append("follower.finalize")
                raise RuntimeError("follower finalize failed")

        config = self.config.copy()
        config.workspace = str(self.tmp_path / "training_failure")
        config.auto_callback = False
        config.callback_names = [
            "training_failure",
            "training_failure_follower",
        ]
        pipeline = cflearn.TrainingPipeline.init(config)

        def wait_for_everyone() -> None:
            events.append("barrier")

        with patch(
            "core.learn.trainer.Accelerator.wait_for_everyone",
            side_effect=wait_for_everyone,
        ), patch.object(cflearn.Trainer, "get_metrics") as get_metrics, patch.object(
            cflearn.Trainer,
            "restore_checkpoint",
        ) as restore_checkpoint, patch.object(
            cflearn.Trainer,
            "save_checkpoint",
        ) as save_checkpoint:
            with self.assertRaises(RuntimeError) as context:
                pipeline.fit(self.data, do_summary=False)

        self.assertIs(context.exception, training_error)
        failure_index = events.index("failure.train")
        self.assertNotIn("barrier", events[failure_index + 1 :])
        self.assertListEqual(
            events[failure_index:],
            [
                "failure.train",
                "failure.after_loop",
                "follower.after_loop",
                "failure.finalize",
                "follower.finalize",
            ],
        )
        get_metrics.assert_not_called()
        restore_checkpoint.assert_not_called()
        save_checkpoint.assert_not_called()
        trainer = pipeline.training.build_trainer.trainer
        self.assertFalse(trainer.state.is_terminate)

    def test_callback_cleanup_error_priority(self) -> None:
        events = []
        failure_stage = ""
        first_error = RuntimeError()
        second_error = RuntimeError()

        @cflearn.TrainerCallback.register("cleanup_failure", allow_duplicate=True)
        class CleanupFailureCallback(cflearn.TrainerCallback):
            def after_loop(self, trainer):
                events.append("failure.after_loop")
                if failure_stage == "after_loop":
                    raise first_error

            def finalize(self, trainer):
                events.append("failure.finalize")
                raise first_error

        @cflearn.TrainerCallback.register(
            "cleanup_failure_follower", allow_duplicate=True
        )
        class CleanupFailureFollowerCallback(cflearn.TrainerCallback):
            def after_loop(self, trainer):
                events.append("follower.after_loop")
                if failure_stage == "after_loop":
                    raise second_error

            def finalize(self, trainer):
                events.append("follower.finalize")
                raise second_error

        final_results = cflearn.MetricsOutputs(1.0, {}, {})
        for failure_stage in ["after_loop", "finalize"]:
            with self.subTest(failure_stage=failure_stage):
                events.clear()
                first_error = RuntimeError(f"first {failure_stage} failure")
                second_error = RuntimeError(f"second {failure_stage} failure")
                config = self.config.copy()
                config.workspace = str(
                    self.tmp_path / f"cleanup_failure_{failure_stage}"
                )
                config.num_steps = 0
                config.auto_callback = False
                config.callback_names = [
                    "cleanup_failure",
                    "cleanup_failure_follower",
                ]
                pipeline = cflearn.TrainingPipeline.init(config)

                def restore_checkpoint(*args, **kwargs):
                    events.append("restore")
                    return False

                def get_metrics(*args, **kwargs):
                    events.append("evaluate")
                    return final_results

                def log_with(*args, **kwargs):
                    events.append("log")

                def save_checkpoint(*args, **kwargs):
                    events.append("save")

                with patch.object(
                    cflearn.Trainer,
                    "restore_checkpoint",
                    side_effect=restore_checkpoint,
                ), patch.object(
                    cflearn.Trainer,
                    "get_metrics",
                    side_effect=get_metrics,
                ), patch.object(
                    cflearn.Trainer,
                    "log_with",
                    side_effect=log_with,
                ), patch.object(
                    cflearn.Trainer,
                    "save_checkpoint",
                    side_effect=save_checkpoint,
                ):
                    with self.assertRaises(RuntimeError) as context:
                        pipeline.fit(self.data, do_summary=False)

                self.assertIs(context.exception, first_error)
                trainer = pipeline.training.build_trainer.trainer
                if failure_stage == "after_loop":
                    self.assertListEqual(
                        events,
                        [
                            "failure.after_loop",
                            "follower.after_loop",
                            "failure.finalize",
                            "follower.finalize",
                        ],
                    )
                    self.assertFalse(trainer.state.is_terminate)
                else:
                    self.assertListEqual(
                        events,
                        [
                            "failure.after_loop",
                            "follower.after_loop",
                            "restore",
                            "evaluate",
                            "log",
                            "save",
                            "failure.finalize",
                            "follower.finalize",
                        ],
                    )
                    self.assertTrue(trainer.state.is_terminate)

    def test_callback_cleanup_restores_state(self) -> None:
        events = []
        failure_stage = ""
        finalization_error = RuntimeError()

        @cflearn.TrainerCallback.register("final_evaluation", allow_duplicate=True)
        class FinalEvaluationCallback(cflearn.TrainerCallback):
            def after_loop(self, trainer):
                events.append("after_loop")

            def finalize(self, trainer):
                events.append("finalize")

        final_results = cflearn.MetricsOutputs(1.0, {}, {})
        for failure_stage in ["evaluation", "final_barrier"]:
            with self.subTest(failure_stage=failure_stage):
                events.clear()
                finalization_error = RuntimeError(f"{failure_stage} failed")
                config = self.config.copy()
                config.workspace = str(self.tmp_path / failure_stage)
                config.num_steps = 0
                config.auto_callback = False
                config.callback_names = ["final_evaluation"]
                pipeline = cflearn.TrainingPipeline.init(config)

                def wait_for_everyone() -> None:
                    if failure_stage == "final_barrier" and "finalize" in events:
                        events.append("barrier")
                        raise finalization_error

                def restore_checkpoint(*args, **kwargs):
                    events.append("restore")
                    return False

                def get_metrics(*args, **kwargs):
                    events.append("evaluate")
                    if failure_stage == "evaluation":
                        raise finalization_error
                    return final_results

                def log_with(*args, **kwargs):
                    events.append("log")

                def save_checkpoint(*args, **kwargs):
                    events.append("save")

                with patch(
                    "core.learn.trainer.Accelerator.wait_for_everyone",
                    side_effect=wait_for_everyone,
                ), patch.object(
                    cflearn.Trainer,
                    "restore_checkpoint",
                    side_effect=restore_checkpoint,
                ), patch.object(
                    cflearn.Trainer,
                    "get_metrics",
                    side_effect=get_metrics,
                ), patch.object(
                    cflearn.Trainer,
                    "log_with",
                    side_effect=log_with,
                ), patch.object(
                    cflearn.Trainer,
                    "save_checkpoint",
                    side_effect=save_checkpoint,
                ):
                    with self.assertRaises(RuntimeError) as context:
                        pipeline.fit(self.data, do_summary=False)

                self.assertIs(context.exception, finalization_error)
                if failure_stage == "evaluation":
                    expected_events = [
                        "after_loop",
                        "restore",
                        "evaluate",
                        "finalize",
                    ]
                else:
                    expected_events = [
                        "after_loop",
                        "restore",
                        "evaluate",
                        "log",
                        "save",
                        "finalize",
                        "barrier",
                    ]
                self.assertListEqual(events, expected_events)
                trainer = pipeline.training.build_trainer.trainer
                self.assertEqual(trainer.state.step, 0)
                self.assertEqual(trainer.state.epoch, 0)
                self.assertIsNone(trainer.state._last_step)

    def test_keyboard_interrupt(self):
        events = []
        interrupt = KeyboardInterrupt()

        @cflearn.TrainerCallback.register("test_interrupt", allow_duplicate=True)
        class InterruptCallback(cflearn.TrainerCallback):
            def after_train_step(self, batch, step_outputs, trainer):
                events.append("interrupt")
                raise interrupt

            def after_loop(self, trainer):
                events.append("after_loop")

            def finalize(self, trainer):
                events.append("finalize")

        final_results = cflearn.MetricsOutputs(1.0, {}, {})
        for distributed in [False, True]:
            with self.subTest(distributed=distributed):
                events.clear()
                config = self.config.copy()
                config.workspace = str(self.tmp_path / f"interrupt_{distributed}")
                config.auto_callback = False
                config.callback_names = ["test_interrupt"]
                pipeline = cflearn.TrainingPipeline.init(config)

                def wait_for_everyone() -> None:
                    events.append("barrier")

                def restore_checkpoint(*args, **kwargs):
                    events.append("restore")
                    return False

                def get_metrics(*args, **kwargs):
                    events.append("evaluate")
                    return final_results

                def log_with(*args, **kwargs):
                    events.append("log")

                def save_checkpoint(*args, **kwargs):
                    events.append("save")

                with patch(
                    "core.learn.trainer.is_dist_initialized",
                    return_value=distributed,
                ), patch(
                    "core.learn.trainer.Accelerator.wait_for_everyone",
                    side_effect=wait_for_everyone,
                ), patch.object(
                    cflearn.Trainer,
                    "restore_checkpoint",
                    side_effect=restore_checkpoint,
                ) as restore_checkpoint_mock, patch.object(
                    cflearn.Trainer,
                    "get_metrics",
                    side_effect=get_metrics,
                ) as get_metrics_mock, patch.object(
                    cflearn.Trainer,
                    "log_with",
                    side_effect=log_with,
                ) as log_with_mock, patch.object(
                    cflearn.Trainer,
                    "save_checkpoint",
                    side_effect=save_checkpoint,
                ) as save_checkpoint_mock:
                    if distributed:
                        with self.assertRaises(KeyboardInterrupt) as context:
                            pipeline.fit(self.data, do_summary=False)
                        self.assertIs(context.exception, interrupt)
                    else:
                        pipeline.fit(self.data, do_summary=False)

                trainer = pipeline.training.build_trainer.trainer
                lifecycle_events = [
                    event
                    for event in events
                    if event
                    in {
                        "interrupt",
                        "after_loop",
                        "restore",
                        "evaluate",
                        "log",
                        "save",
                        "finalize",
                    }
                ]
                if distributed:
                    self.assertListEqual(
                        lifecycle_events,
                        [
                            "interrupt",
                            "after_loop",
                            "finalize",
                        ],
                    )
                    interrupt_index = events.index("interrupt")
                    self.assertNotIn("barrier", events[interrupt_index + 1 :])
                    restore_checkpoint_mock.assert_not_called()
                    get_metrics_mock.assert_not_called()
                    log_with_mock.assert_not_called()
                    save_checkpoint_mock.assert_not_called()
                    self.assertFalse(trainer.state.is_terminate)
                else:
                    self.assertListEqual(
                        lifecycle_events,
                        [
                            "interrupt",
                            "after_loop",
                            "restore",
                            "evaluate",
                            "log",
                            "save",
                            "finalize",
                        ],
                    )
                    restore_checkpoint_mock.assert_called_once()
                    get_metrics_mock.assert_called_once()
                    log_with_mock.assert_called_once()
                    save_checkpoint_mock.assert_called_once()
                    self.assertTrue(trainer.state.is_terminate)

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
