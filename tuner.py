import os
import sys
from typing import Dict, Tuple
import nevergrad as ng
import nltk
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from ray import air, tune
from ray.air import session
from ray.tune.result_grid import ResultGrid
from ray.tune.search.nevergrad import NevergradSearch
from sklearn.metrics import auc, roc_curve, f1_score

from sklearn.model_selection import KFold

_path = os.path.join(os.getcwd(), "..")
sys.path.insert(0, _path)

from processor import CustomVectorizer, get_pipe  # noqa


class RayTuner:
    def __init__(self):
        """init

        Args:
            pipe (Pipeline): untrained sklearn pipeline
        """
        self.all_data = pd.read_csv("data/train.csv")


        self.pipe = get_pipe()
        self.df_metrics = False
        self.preprocessed_data = {}


    def split_preprocessed_data(self, train_index: int, test_index: int, i: int):
        # pre-process each step and save
        if i not in self.preprocessed_data:
            train = self.all_data.iloc[train_index].copy()
            X_train = self.pipe[:-2].fit_transform(train)
            test = self.all_data.iloc[test_index].copy()
            X_test = self.pipe[:-2].transform(test)
            y_train = train["target"]
            y_test = test["target"]
            self.preprocessed_data[i] = (X_train, X_test, y_train, y_test)
        return self.preprocessed_data[i]

    def train_classifier(self, config: dict, n_splits: int = 5, log: bool = False) -> Tuple[Dict]:
        """Runs a time series split and returns the auc and accuracy for each split

        Args:
            n_splits (int, optional): number of splits. Defaults to 5.
            log (bool, optional): whether to log metrics to mlflow. Defaults to False.

        Returns:
            Tuple(Dict): dictionary of metrics and dictionary of graphs
        """
        scores = []
        f1_scores = []
        kf = KFold(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(kf.split(self.all_data)):
            X_train, X_test, y_train, y_test = self.split_preprocessed_data(train_index, test_index, i)
            vectoriser = CustomVectorizer(min_df=config["min_df"])
            model = LGBMClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                num_leaves=config["num_leaves"],
                learning_rate=config["learning_rate"],
                bagging_freq=config["bagging_freq"],
                bagging_fraction=config["bagging_fraction"],
            )
            vectorised_X_train = vectoriser.fit_transform(X_train, y_train)
            vectorised_X_test = vectoriser.transform(X_test, y_test)
            model.fit(vectorised_X_train, y_train)
            y_proba = model.predict_proba(vectorised_X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            score = auc(fpr, tpr)
            scores.append(score)
            f1_score = f1_score(y_test, np.around(y_proba))
            f1_scores.append(f1_score)

        summary_auc = np.mean(scores)
        mean_f1 = np.mean(f1_scores)

        min_auc = np.min(scores)

        se = np.std(scores, ddof=1) / np.sqrt(np.size(scores))

        score_optimise = summary_auc - se


        model_metrics = {"mean_auc": summary_auc, "min_auc": min_auc, "standard_error": se, "mean_f1": mean_f1}
        session.report(metrics=model_metrics)

    def tune_model(self) -> ResultGrid:
        search_space = {
            "min_df": tune.uniform(0, 0.1),
            "max_depth": tune.randint(5, 100),
            "bagging_freq": tune.randint(2, 10),
            "num_leaves": tune.randint(1, 20),
            "bagging_fraction": tune.uniform(0.5, 0.99),
            "learning_rate": tune.uniform(0.01, 0.5),
            "n_estimators": tune.randint(50, 2000),
        }
        algo = NevergradSearch(optimizer=ng.optimizers.TBPSA, metric="mean_f1", mode="max")
        tuner = tune.Tuner(
            tune.with_parameters(self.train_classifier),
            tune_config=tune.TuneConfig(metric="mean_f1", mode="max", num_samples=1000, search_alg=algo),
            run_config=air.RunConfig(log_to_file=False),
            param_space=search_space,
        )
        results = tuner.fit()

        return results


if __name__ == "__main__":
    tuner = RayTuner()
    tuner.tune_model()