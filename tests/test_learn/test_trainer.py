import torch
import pytest
import inspect
import unittest

import core.learn as cflearn

from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from unittest.mock import PropertyMock


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

    def make_monitor_trainer(self, name):
        config = self.config.copy()
        config.workspace = str(self.tmp_path / name)
        pipeline = cflearn.TrainingPipeline.init(config).fit(
            self.data,
            do_summary=False,
        )
        trainer = pipeline.training.build_trainer.trainer
        train_loader, valid_loader = pipeline.data.build_loaders(for_inference=True)
        self.assertIsNotNone(valid_loader)
        trainer.state = cflearn.TrainerState(
            num_epoch=2,
            batch_size=train_loader.batch_size,
            loader_length=len(train_loader),
            snapshot_start_step=0,
        )
        trainer.state.step = trainer.state.epoch = 1
        trainer.callbacks = []
        return (
            trainer,
            train_loader,
            valid_loader,
            cflearn.StepOutputs({}, {}),
        )

    def test_public_fit_signature(self) -> None:
        signature = inspect.signature(cflearn.Trainer.fit)
        positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
        keyword_only = inspect.Parameter.KEYWORD_ONLY
        empty = inspect.Parameter.empty
        expected = [
            ("self", positional, empty),
            ("data", positional, empty),
            ("model", positional, empty),
            ("metrics", positional, empty),
            ("inference", positional, empty),
            ("optimizers", positional, empty),
            ("schedulers", positional, empty),
            ("monitors", positional, empty),
            ("callbacks", positional, empty),
            ("schedulers_requires_metric", positional, empty),
            ("do_summary", keyword_only, True),
            ("show_summary", keyword_only, True),
            ("summary_kwargs", keyword_only, None),
            ("loaded_state", keyword_only, None),
            ("skip_final_evaluation", keyword_only, False),
            ("only_touch", keyword_only, False),
            ("device", keyword_only, None),
            ("p", keyword_only, None),
        ]
        actual = [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in signature.parameters.values()
        ]
        self.assertListEqual(actual, expected)
        self.assertEqual(signature.return_annotation, "Trainer")

    def test_finalize_entry_interrupt_preserves_cleanup(self) -> None:
        events = []
        interrupt = KeyboardInterrupt("finalize entry")

        @cflearn.TrainerCallback.register(
            "finalize_entry_interrupt", allow_duplicate=True
        )
        class FinalizeEntryCallback(cflearn.TrainerCallback):
            def after_loop(self, trainer):
                events.append("after_loop")

            def finalize(self, trainer):
                events.append("finalize")

        config = self.config.copy()
        config.workspace = str(self.tmp_path / "finalize_entry_interrupt")
        config.num_steps = 0
        config.auto_callback = False
        config.callback_names = ["finalize_entry_interrupt"]
        pipeline = cflearn.TrainingPipeline.init(config)
        with patch.object(
            cflearn.Trainer,
            "_finalize",
            side_effect=interrupt,
        ) as finalize:
            with self.assertRaises(KeyboardInterrupt) as context:
                pipeline.fit(self.data, do_summary=False)

        self.assertIs(context.exception, interrupt)
        finalize.assert_called_once()
        self.assertListEqual(events, ["after_loop", "finalize"])
        trainer = pipeline.training.build_trainer.trainer
        self.assertEqual(trainer.state.step, 0)
        self.assertEqual(trainer.state.epoch, 0)
        self.assertIsNone(trainer.state._last_step)

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
        config = self.config.copy()
        config.workspace = str(self.tmp_path / "metrics_progress")
        pipeline = cflearn.TrainingPipeline.init(config).fit(
            self.data,
            do_summary=False,
        )
        trainer = pipeline.training.build_trainer.trainer
        _, valid_loader = pipeline.data.build_loaders(for_inference=True)
        self.assertIsNotNone(valid_loader)

        inference_outputs = []
        inference = trainer.inference
        get_outputs = inference.get_outputs
        progresses = []
        progress_type = cflearn.trainer.Progress

        def capture_outputs(*args, **kwargs):
            outputs = get_outputs(*args, **kwargs)
            inference_outputs.append(outputs)
            return outputs

        def make_progress():
            progress = progress_type()
            progresses.append(progress)
            return progress

        trainer.callbacks = []
        trainer.state = cflearn.TrainerState(
            num_epoch=1,
            batch_size=1,
            loader_length=len(valid_loader),
        )
        self.assertFalse(trainer.state.is_terminate)

        trainer.tqdm_settings.use_tqdm_in_validation = True
        with patch.object(
            inference,
            "get_outputs",
            new=capture_outputs,
        ), patch("core.learn.trainer.Progress", new=make_progress):
            output = trainer.get_metrics(valid_loader)

        self.assertIsInstance(output, cflearn.MetricsOutputs)
        self.assertEqual(len(progresses), 1)
        self.assertIs(output, inference_outputs[0].metric_outputs)
        self.assertIn(cflearn.LOSS_KEY, output.metric_values)

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
        with patch.object(
            type(trainer.accelerator),
            "is_local_main_process",
            new_callable=PropertyMock,
            return_value=False,
        ):
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

    def test_callback_success_contract(self) -> None:
        @cflearn.TrainerCallback.register("success_contract", allow_duplicate=True)
        class SuccessContractCallback(cflearn.TrainerCallback):
            pass

        callback = SuccessContractCallback()
        callback_calls = Mock()
        callback_hooks = {
            name
            for name, value in vars(cflearn.TrainerCallback).items()
            if callable(value) and not name.startswith("_")
        }
        for name in callback_hooks:
            hook = Mock(wraps=getattr(callback, name))
            callback_calls.attach_mock(hook, name)
            setattr(callback, name, hook)
        config = self.config.copy()
        config.workspace = str(self.tmp_path / "success_contract")
        config.auto_callback = False
        config.callback_names = ["success_contract"]
        with patch.object(
            SuccessContractCallback,
            "__new__",
            return_value=callback,
        ):
            pipeline = cflearn.TrainingPipeline.init(config).fit(
                self.data,
                do_summary=False,
            )
        trainer = pipeline.training.build_trainer.trainer
        self.assertIn(callback, trainer.callbacks)

        self.assertListEqual(
            [record[0] for record in callback_calls.method_calls],
            [
                "initialize",
                "after_workspace_prepared",
                "before_summary",
                "before_loop",
                "before_loop_with_loaders",
                "at_epoch_start",
                "at_step_start",
                "mutate_forward_kwargs",
                "mutate_loss_kwargs",
                "before_gradient_update",
                "log_lr",
                "after_gradient_update",
                "log_train_step",
                "after_train_step",
                "before_monitor_logging",
                "log_metrics_msg",
                "log_metrics",
                "log_artifacts",
                "after_monitor",
                "after_save_checkpoint",
                "at_step_end",
                "at_terminate",
                "at_epoch_end",
                "after_loop",
                "log_metrics_msg",
                "log_metrics",
                "log_artifacts",
                "finalize",
            ],
        )
        self.assertSetEqual(
            {record[0] for record in callback_calls.method_calls},
            callback_hooks,
        )
        callback.finalize.assert_called_once_with(trainer)
        train_loader = callback.before_loop_with_loaders.call_args.args[1]
        self.assertIs(callback.at_epoch_start.call_args.args[1], train_loader)
        batch = callback.at_step_start.call_args.args[0]
        self.assertIs(callback.before_gradient_update.call_args.args[1], batch)
        self.assertIs(callback.after_gradient_update.call_args.args[1], batch)
        self.assertIs(callback.after_train_step.call_args.args[0], batch)
        self.assertIs(
            callback.before_gradient_update.call_args.args[2],
            callback.after_gradient_update.call_args.args[2],
        )
        step_outputs = callback.after_train_step.call_args.args[1]
        self.assertIs(callback.log_train_step.call_args.args[0], step_outputs)
        monitored = callback.after_monitor.call_args.args[0]
        self.assertIs(
            callback.log_metrics.call_args_list[0].args[0],
            monitored.metric_outputs,
        )

        callback_calls.reset_mock()
        with patch.object(
            type(trainer.accelerator),
            "is_local_main_process",
            new_callable=PropertyMock,
            return_value=False,
        ), patch.object(
            type(trainer.accelerator),
            "is_main_process",
            new_callable=PropertyMock,
            return_value=False,
        ):
            trainer.log_with(trainer.final_results)
        self.assertListEqual(callback_calls.method_calls, [])

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

        def make_decision(name, kind, result):
            def decision(score):
                events.append((name, kind, score))
                return result

            return decision

        first = Mock(spec=cflearn.TrainerMonitor)
        first.should_snapshot.side_effect = make_decision("first", "snapshot", True)
        first.should_terminate.side_effect = make_decision("first", "terminate", True)
        second = Mock(spec=cflearn.TrainerMonitor)
        second.should_snapshot.side_effect = make_decision(
            "second",
            "snapshot",
            False,
        )
        second.should_terminate.side_effect = make_decision(
            "second",
            "terminate",
            False,
        )
        trainer, train_loader, valid_loader, step_outputs = self.make_monitor_trainer(
            "monitor_decisions"
        )
        trainer.monitors = [first, second]
        with patch(
            "core.learn.trainer.is_ddp",
            return_value=False,
        ), patch.object(
            trainer,
            "get_metrics",
            wraps=trainer.get_metrics,
        ) as get_metrics, patch.object(
            trainer,
            "log_with",
            wraps=trainer.log_with,
        ) as log_with, patch.object(
            trainer.state,
            "update_snapshot_epoch",
            wraps=trainer.state.update_snapshot_epoch,
        ) as update_snapshot_epoch:
            monitored = trainer.monitor(train_loader, valid_loader, step_outputs)

        self.assertIsNotNone(monitored.metric_outputs)
        score = monitored.metric_outputs.final_score
        self.assertEqual(
            events,
            [
                ("first", "snapshot", score),
                ("second", "snapshot", score),
                ("first", "terminate", score),
                ("second", "terminate", score),
            ],
        )
        self.assertTrue(monitored.save_checkpoint)
        self.assertTrue(monitored.terminate)
        self.assertIs(monitored.metric_outputs, trainer.intermediate)
        update_snapshot_epoch.assert_called_once_with()
        get_metrics.assert_called_once_with(
            valid_loader,
            trainer.config.valid_portion,
        )
        log_with.assert_called_once_with(monitored.metric_outputs)

        events.clear()
        with patch(
            "core.learn.trainer.is_ddp",
            return_value=True,
        ), patch.object(
            trainer,
            "get_metrics",
            wraps=trainer.get_metrics,
        ) as get_metrics, patch.object(
            trainer,
            "log_with",
            wraps=trainer.log_with,
        ) as log_with, patch.object(
            trainer.state,
            "update_snapshot_epoch",
            wraps=trainer.state.update_snapshot_epoch,
        ) as update_snapshot_epoch:
            monitored = trainer.monitor(train_loader, valid_loader, step_outputs)

        self.assertIsNotNone(monitored.metric_outputs)
        score = monitored.metric_outputs.final_score
        self.assertEqual(
            events,
            [
                ("first", "snapshot", score),
                ("second", "snapshot", score),
            ],
        )
        self.assertTrue(monitored.save_checkpoint)
        self.assertFalse(monitored.terminate)
        update_snapshot_epoch.assert_called_once_with()
        get_metrics.assert_called_once_with(
            valid_loader,
            trainer.config.valid_portion,
        )
        log_with.assert_called_once_with(monitored.metric_outputs)

    def test_monitor_updates_state_regardless_of_order(self) -> None:
        trainer, train_loader, valid_loader, step_outputs = self.make_monitor_trainer(
            "monitor_order"
        )

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
                    monitored = trainer.monitor(
                        train_loader, valid_loader, step_outputs
                    )

                self.assertIsNotNone(monitored.metric_outputs)
                score = monitored.metric_outputs.final_score
                self.assertListEqual(mean_std.history, [score])
                self.assertEqual(mean_std._incrementer.num_record, 1)
                self.assertListEqual(plateau.history, [score])
                self.assertEqual(plateau._incrementer.num_record, 1)
                self.assertTrue(monitored.save_checkpoint)


if __name__ == "__main__":
    unittest.main()
