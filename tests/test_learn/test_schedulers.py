import unittest

import core.learn as cflearn


class TestSchedulers(unittest.TestCase):
    def test_schedulers(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        scheduler_config = dict(start_epoch=0, end_epoch=1, warmup_step=2)
        for scheduler in [
            "linear",
            "linear_inverse",
            "step",
            "exponential",
            "plateau",
            "warmup",
            "op.cosine_warmup",
            "op.linear_warmup",
        ]:
            if scheduler.startswith("op."):
                scheduler, op_type = scheduler.split(".")
                scheduler_config["op_type"] = op_type
                scheduler_config["op_config"] = dict(
                    warmup_steps=[2],
                    cycle_lengths=[1],
                    f_start=[0.0],
                    f_min=[0.0],
                    f_max=[0.1],
                )
            config = cflearn.Config(
                module_name="linear",
                module_config=dict(input_dim=in_dim, output_dim=out_dim),
                scheduler_name=scheduler,
                scheduler_config=scheduler_config,
                loss_name="mse",
            )
            config.to_debug().num_steps = 10
            cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    unittest.main()
