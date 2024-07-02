import torch
import unittest

import core.learn as cflearn

from typing import List
from typing import Type
from typing import Optional
from unittest.mock import patch
from unittest.mock import Mock


class TestModel(unittest.TestCase):
    def test_model(self):
        with self.assertRaises(ValueError):
            cflearn.IModel.from_config(cflearn.Config())
        in_dim = 13
        out_dim = 7
        num_ensemble = 3
        models = [
            cflearn.IModel.from_config(
                cflearn.Config(
                    module_name="fcnn",
                    module_config=dict(input_dim=in_dim, output_dim=out_dim),
                    loss_name="mse",
                )
            )
            for _ in range(3)
        ]
        ensemble = cflearn.EnsembleModel(models[0], num_ensemble)
        self.assertIs(ensemble.all_modules[0], ensemble.m)
        self.assertEqual(len(ensemble.train_steps), 1)
        self.assertEqual(len(ensemble.m), num_ensemble)
        with self.assertRaises(RuntimeError):
            ensemble.build(ensemble.config)
        for i, model in enumerate(models):
            self.assertEqual(model.config, ensemble.config)
            ensemble.m[i].load_state_dict(model.state_dict())
        batch = {cflearn.INPUT_KEY: torch.randn(17, in_dim)}
        naive_out = 0.0
        for model in models:
            naive_out += model.run(0, batch)[cflearn.PREDICTIONS_KEY]
        naive_out /= 3.0
        ensemble_out = ensemble.run(0, batch)[cflearn.PREDICTIONS_KEY]
        torch.testing.assert_close(naive_out, ensemble_out)
        with self.assertRaises(ValueError):
            models[0].postprocess(0, batch, "foo")
        with patch("core.learn.schema.is_fsdp") as mock:
            mock.return_value = True
            m_mock = Mock()
            models[0].m = m_mock
            models[0].state_dict()
            m_mock.state_dict.assert_called_once()
            models[0].load_state_dict({})
            m_mock.load_state_dict.assert_called_once()

    def test_train_steps(self):
        class FooTrainStep(cflearn.CommonTrainStep):
            def __init__(
                self,
                scope: str = "all",
                *,
                grad_accumulate: Optional[int] = None,
                requires_new_forward: bool = False,
                requires_grad_in_forward: bool = True,
                enable_toggle_optimizer: bool = True,
            ) -> None:
                super().__init__(cflearn.MSELoss())
                self.scope = scope
                self.grad_accumulate = grad_accumulate
                self.requires_new_forward = requires_new_forward
                self.requires_grad_in_forward = requires_grad_in_forward
                self.enable_toggle_optimizer = enable_toggle_optimizer

        class SkipTrainStep(FooTrainStep):
            def should_skip(
                self,
                m: cflearn.IModel,
                state: Optional[cflearn.TrainerState],
            ) -> bool:
                return True

        @cflearn.IModel.register("$test_no_train_steps_model")
        class NoTrainStepsModel(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return []

        @cflearn.IModel.register("$test_skip_train_steps_model")
        class SkipTrainStepsModel(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [SkipTrainStep()]

        @cflearn.IModel.register("$test_grad_ok_model_0")
        class GradOkModel0(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [SkipTrainStep(requires_grad_in_forward=False), FooTrainStep()]

        @cflearn.IModel.register("$test_grad_ok_model_1")
        class GradOkModel1(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [FooTrainStep(), FooTrainStep(requires_new_forward=True)]

        @cflearn.IModel.register("$test_grad_buggy_model_0")
        class GradBuggyModel0(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [FooTrainStep(), FooTrainStep()]

        @cflearn.IModel.register("$test_grad_buggy_model_1")
        class GradBuggyModel1(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [FooTrainStep(requires_grad_in_forward=False), FooTrainStep()]

        @cflearn.IModel.register("$test_grad_buggy_model_2")
        class GradBuggyModel2(cflearn.CommonModel):
            @property
            def train_steps(self) -> List[cflearn.TrainStep]:
                return [
                    FooTrainStep(requires_grad_in_forward=False),
                    FooTrainStep(requires_new_forward=True),
                ]

        in_dim = 13
        out_dim = 7
        batch = {
            cflearn.INPUT_KEY: torch.randn(17, in_dim),
            cflearn.LABEL_KEY: torch.randn(17, out_dim),
        }
        config = cflearn.Config(
            module_name="fcnn",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            loss_name="mse",
        )
        config = config.to_debug()

        model = cflearn.IModel.from_config(config)
        ls0 = model.step(0, batch, get_losses=True).loss_tensors
        ls1 = model.step(0, batch, get_losses=True, detach_losses=False).loss_tensors
        self.assertFalse(ls0[cflearn.LOSS_KEY].requires_grad)
        self.assertTrue(ls1[cflearn.LOSS_KEY].requires_grad)

        config.model = NoTrainStepsModel.__identifier__
        model = cflearn.IModel.from_config(config)
        o0 = model.run(0, batch)[cflearn.PREDICTIONS_KEY]
        o1 = model.step(0, batch).forward_results[cflearn.PREDICTIONS_KEY]
        torch.testing.assert_close(o0, o1)

        config.model = SkipTrainStepsModel.__identifier__
        model = cflearn.IModel.from_config(config)
        o0 = model.run(0, batch)[cflearn.PREDICTIONS_KEY]
        o1 = model.step(0, batch).forward_results[cflearn.PREDICTIONS_KEY]
        torch.testing.assert_close(o0, o1)
        self.assertDictEqual(model.step(0, batch, get_losses=True).loss_items, {})

        def get_train_outputs(cls: Type[cflearn.IModel]) -> cflearn.StepOutputs:
            config.model = cls.__identifier__
            p = cflearn.TrainingPipeline.init(config).fit(data)
            trainer = p.training.build_trainer.trainer
            model = p.training.build_model.model
            batch = data.build_loaders()[0].get_one_batch()
            return model.train(0, batch, trainer, {}, {})

        data, *_ = cflearn.testing.linear_data(4, in_dim, out_dim=out_dim)
        outputs = get_train_outputs(SkipTrainStepsModel)
        self.assertDictEqual(outputs.loss_items, {})
        outputs = get_train_outputs(GradOkModel0)
        outputs = get_train_outputs(GradOkModel1)
        with self.assertRaises(RuntimeError):
            get_train_outputs(GradBuggyModel0)
        with self.assertRaises(RuntimeError):
            get_train_outputs(GradBuggyModel1)
        with self.assertRaises(RuntimeError):
            get_train_outputs(GradBuggyModel2)


if __name__ == "__main__":
    unittest.main()
