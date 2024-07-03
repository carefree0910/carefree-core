import unittest

import core.learn as cflearn


class TestMonitors(unittest.TestCase):
    def test_monitors(self):
        data, in_dim, out_dim, _ = cflearn.testing.linear_data(6, batch_size=4)
        config = cflearn.Config(
            module_name="linear",
            module_config=dict(input_dim=in_dim, output_dim=out_dim),
            monitor_names=["basic", "mean_std", "plateau", "conservative", "lazy"],
            monitor_configs=dict(plateau=dict(window_size=2)),
            scheduler_name="warmup",
            loss_name="mse",
        )
        config.to_debug().num_steps = 10
        config.num_epoch = 2
        cflearn.TrainingPipeline.init(config).fit(data)

    def test_basic_monitor(self):
        with self.assertRaises(ValueError):
            monitor = cflearn.BasicMonitor(num_keep=0)
        monitor = cflearn.BasicMonitor(num_keep=1)
        self.assertTrue(monitor.should_snapshot(0.0))
        self.assertFalse(monitor.should_terminate(0.0))
        self.assertFalse(monitor.should_snapshot(-1.0))
        self.assertTrue(monitor.should_terminate(-1.0))

    def test_mean_std_monitor(self):
        monitor = cflearn.MeanStdMonitor(num_keep=1, patience=1)
        self.assertTrue(monitor.should_snapshot(0.0))
        self.assertFalse(monitor.should_terminate(0.0))
        self.assertFalse(monitor.should_snapshot(-1.0))
        self.assertTrue(monitor.should_terminate(-1.0))
        monitor.should_snapshot(1.0)
        self.assertFalse(monitor.should_terminate(1.0))
        monitor.should_snapshot(-20.0)
        monitor.should_snapshot(-10.0)
        self.assertTrue(monitor.should_terminate(-10.0))

    def test_plateau_monitor(self):
        monitor = cflearn.PlateauMonitor(num_keep=1, patience=1)
        self.assertEqual(monitor.plateau_tolerance, 25.0)
        self.assertEqual(monitor.get_extension(None), 5)
        self.assertFalse(monitor.should_terminate(0.0))
        monitor.punish_extension()
        self.assertTrue(monitor.should_terminate(0.0))
        monitor = cflearn.PlateauMonitor(num_keep=1, patience=1)
        self.assertTrue(monitor.should_snapshot(0.0))
        self.assertFalse(monitor.should_terminate(0.0))
        self.assertFalse(monitor.should_snapshot(-1.0))
        self.assertTrue(monitor.should_terminate(-1.0))
        monitor = cflearn.PlateauMonitor(num_keep=1, patience=1, window_size=2)
        self.assertFalse(monitor.should_terminate(0.0))
        self.assertTrue(monitor.should_terminate(0.0))

    def test_lazy_monitor(self):
        monitor = cflearn.LazyMonitor()
        self.assertFalse(monitor.should_snapshot(0.0))
        self.assertFalse(monitor.should_terminate(0.0))
        self.assertFalse(monitor.should_snapshot(-1.0))
        self.assertFalse(monitor.should_terminate(-1.0))


if __name__ == "__main__":
    unittest.main()
