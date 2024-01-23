import csv

import numpy as np
import core.learn as cflearn

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
from collections import Counter
from core.learn.schema import DataBundle
from core.learn.toolkit import seed_everything


def read(path: Path, delimiter: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r") as f:
        data = list(csv.reader(f, delimiter=delimiter))
        for i in range(len(data) - 1, -1, -1):
            if not data[i]:
                data.pop(i)
    header = [e.strip() for e in data.pop(0)]
    return header, data


@cflearn.IDataBlock.register("titanic")
class TitanicBlock(cflearn.IDataBlock):
    num_features: int
    num_classes: int
    label_column: str
    feature_columns: List[str]
    sex_mapping: Dict[str, int]

    def _transform(self, header: List[str], data: List[List[str]]) -> DataBundle:
        data_T = list(zip(*data))
        try:
            label_index = header.index(self.label_column)
        except ValueError:
            label_index = None
        feature_indices = [header.index(e) for e in self.feature_columns]
        raw_x = [data_T[i] for i in feature_indices]
        # convert 'Sex' string to index
        sex_index = self.feature_columns.index("Sex")
        raw_x[sex_index] = [self.sex_mapping[elem] for elem in raw_x[sex_index]]
        # fill missing values with average
        ## to make it simple, we calculate the average on the fly, but in practice
        ## we should calculate the average on the training set and use it to fill
        ## missing values in both training and testing sets
        for i, line in enumerate(raw_x):
            if all(map(bool, line)):
                continue
            valid_values = [float(value) for value in line if value]
            mean = sum(valid_values) / len(valid_values)
            line = [elem if elem else str(mean) for elem in line]
            raw_x[i] = line
        # construct numpy arrays
        x = np.array(raw_x, dtype=np.float32).T
        if label_index is None:
            y = None
        else:
            y = np.array(data_T[label_index], dtype=np.int64).reshape([-1, 1])
        return DataBundle(x, y)

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        header, data = read(bundle.x_train, ",")
        result = self._transform(header, data)
        if bundle.x_valid is not None:
            valid_result = self._transform(*read(bundle.x_valid, ","))
            result.x_valid = valid_result.x_train
            result.y_valid = valid_result.y_train
        return result

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        header, data = read(bundle.x_train, ",")
        self.label_column = "Survived"
        self.feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
        self.sex_mapping = {}
        sex_column = [line[header.index("Sex")] for line in data]
        for sex, _ in Counter(sex_column).most_common():
            self.sex_mapping[sex] = len(self.sex_mapping)
        bundle = self.transform(bundle, False)
        self.num_features = bundle.x_train.shape[1]
        self.num_classes = len(np.unique(bundle.y_train))
        return bundle

    def to_info(self) -> Dict[str, Any]:
        return dict(
            num_features=self.num_features,
            num_classes=self.num_classes,
            label_column=self.label_column,
            feature_columns=self.feature_columns,
            sex_mapping=self.sex_mapping,
        )


def main() -> None:
    seed_everything(123)
    assets_folder = Path(__file__).parent / "assets" / "titanic"
    train_path = assets_folder / "train.csv"
    test_path = assets_folder / "test.csv"

    data_config = cflearn.DataConfig()
    data_config.batch_size = 16
    data_config.add_blocks(TitanicBlock)
    data = cflearn.ArrayData.init(data_config).fit(train_path, x_valid=train_path)
    titanic_block = data.get_block(TitanicBlock)
    config = cflearn.Config(
        module_name="fcnn",
        module_config=dict(
            input_dim=titanic_block.num_features,
            output_dim=titanic_block.num_classes,
        ),
        loss_name="cross_entropy",
        metric_names="acc",
        num_steps=150,
        log_steps=3,
        lr=1.0e-4,
    )
    pipeline = cflearn.TrainingPipeline.init(config).fit(data)

    test_loader = pipeline.data.build_loader(test_path)
    predictions = pipeline.predict(test_loader)[cflearn.PREDICTIONS_KEY]

    loaded = cflearn.PipelineSerializer.load_inference(pipeline.config.workspace)
    loaded_loader = loaded.data.build_loader(test_path)
    loaded_predictions = loaded.predict(loaded_loader)[cflearn.PREDICTIONS_KEY]
    assert np.allclose(predictions, loaded_predictions)


if __name__ == "__main__":
    main()
