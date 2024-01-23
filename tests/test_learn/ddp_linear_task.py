import core.learn as cflearn


def main() -> None:
    data, in_dim, out_dim, _ = cflearn.testing.linear_data()
    config = cflearn.Config(
        workspace="_ddp",
        module_name="linear",
        module_config=dict(input_dim=in_dim, output_dim=out_dim, bias=False),
        loss_name="mse",
        num_steps=10**4,
    )
    config.to_debug()  # comment this line to disable debug mode
    cflearn.TrainingPipeline.init(config).fit(data)


if __name__ == "__main__":
    main()
