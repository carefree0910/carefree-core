import numpy as np
import core.learn as cflearn


def main() -> None:
    x = np.random.random([10000, 10])
    w = np.random.random([10, 1])
    y = x @ w
    data = cflearn.ArrayData.init().fit(x, y)
    data.config.batch_size = 100
    config = cflearn.Config(
        workspace="_ddp",
        module_name="linear",
        module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1], bias=False),
        loss_name="mse",
        num_steps=10**4,
    )
    config.to_debug()  # comment this line to disable debug mode
    cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    main()
