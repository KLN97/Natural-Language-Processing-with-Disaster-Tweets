import pandas as pd
import json
import logging
import re
from typing import Dict
from sklearn.metrics import f1_score
import lightgbm as lgb
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("words")
nltk.download("wordnet")




class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopword_list = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()



    def process_text(self, text: str) -> str:
        """Takes in a raw text document and performs the following steps in order:

            - punctuation removal
            - case normalization
            - tokenization
            - remove stopwords
            - lemmatization

        Args:
            text (str): text to process

        Returns:
            str: processed text
        """

        processed_text = text.lower()
        processed_text = re.sub(r"[^a-zA-Z0-9]", " ", processed_text)
        processed_text = word_tokenize(processed_text)
        processed_text = [word for word in processed_text if word not in self.stopword_list]
        processed_text = [self.lemmatizer.lemmatize(word) for word in processed_text]
        return " ".join(processed_text)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.Series:
        """applies text processing to a dataframe

        Args:
            X (pd.DataFrame): feature dataframe
            y (_type_, optional): true values. Defaults to None.

        Returns:
            pd.Series: processed text
        """
        text = X["text"]
        text = text.astype("str")
        
        processed_body = X["text"].apply(self.process_text)

        
        text_series = processed_body
        text_series = text_series.astype("str")

        return text_series


class CustomVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X: pd.Series, y=None):
        """fits a TfidfVectorizer

        Args:
            X (pd.Series): text series
            y (_type_, optional): true values. Defaults to None.

        """
        logging.info("fitting vectorizer")
        self.vectorizer.fit(X)
        return self

    def transform(self, X: pd.Series, y=None) -> pd.DataFrame:
        """applies TfidfVectorizer to text series

        Args:
            X (pd.Series): text series
            y (_type_, optional): true values. Defaults to None.

        Returns:
            pd.DataFrame: dataframe with tfidf counts
        """
        logging.info("transforming preprocessed data to dataframe")
        df = pd.DataFrame(
            self.vectorizer.transform(X).toarray(),
            columns=self.vectorizer.get_feature_names_out(),
        )
        # df.columns
        return df


def get_pipe(
    model: lgb.LGBMClassifier = lgb.LGBMClassifier(),
    preprocess: PreprocessText = PreprocessText(),
    vectorizer: CustomVectorizer = CustomVectorizer(min_df=0.00),
) -> Pipeline:
    """Get the training pipeline

    Args:
        preprocess (bool, optional):
            whether to include text preprocess in the pipeline.
            Defaults to True.
        vectorizer (bool, optional):
            whether to include the vectorisation step in the pipeline.
            Defaults to True.
        model (bool, optional):
            whether to include the lgb.LGBMClassifier in the pipeline.
            Defaults to True.
        feature_selection (bool, optional):
            whether to select the top 500 features


    Returns:
        Pipeline: sklearn pipeline
    """

    pipe_steps = []
    pipe_steps.append(("preprocess", preprocess))
    pipe_steps.append(("vectorizer", vectorizer))
    pipe_steps.append(("clf", model))

    pipe = Pipeline(pipe_steps)
    return pipe


def get_pipe_with_params() -> Pipeline:
    """Get the training pipeline

    Args:
        preprocess (bool, optional):
            whether to include text preprocess in the pipeline.
            Defaults to True.
        vectorizer (bool, optional):
            whether to include the vectorisation step in the pipeline.
            Defaults to True.
        model (bool, optional):
            whether to include the lgb.LGBMClassifier in the pipeline.
            Defaults to True.
        feature_selection (bool, optional):
            whether to select the top 500 features


    Returns:
        Pipeline: sklearn pipeline
    """

    pipe_steps = []

    with open("model_parameters.json", "r") as f:
        paras = json.load(f)

    if "min_df" in paras:
        vectorizer = CustomVectorizer(min_df=paras.pop("min_df"))
    else:
        vectorizer = CustomVectorizer(min_df=0.0329)

    pipe_steps.append(("preprocess", PreprocessText()))
    pipe_steps.append(("vectorizer", vectorizer))
    pipe_steps.append(("clf", lgb.LGBMClassifier(**paras)))

    pipe = Pipeline(pipe_steps)
    return pipe

class GraphGenerator:
    def __init__(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        pipe: Pipeline,
    ):
        """initialise graph generator

        Args:
            X_test (pd.DataFrame): feature dataframe
            y_test (pd.Series): true y values
            pipe (Pipeline): sklearn pipeline
        """
        self.y_test = y_test
        self.y_proba = pd.Series(pipe.predict_proba(X_test)[:, 1])
        self.y_pred = np.around(self.y_proba)
        self.pipe = pipe
        self.X_test = X_test
        self.fpr, self.tpr, self.thresholds = roc_curve(y_test, self.y_proba)

    def shap_plot(self, pipe: Pipeline = None, dataset: pd.DataFrame = None) -> plt.figure:
        """generates a shap plot

        Returns:
            plt.figure: shap plot
        """
        if dataset is None:
            dataset = self.X_test
        if pipe is None:
            pipe = self.pipe
        X = pipe[:-1].transform(dataset)
        shap_values = shap.TreeExplainer(pipe["clf"]).shap_values(X)
        shap.summary_plot(shap_values[1], X, show=False)
        return plt.gcf()

    def probability_frequency_figure(self) -> plt.figure:
        """generates a frequency histogram showing how frequently
        emails of different probabilities appear

        Returns:
            plt.figure: histogram
        """

        hist_ax = self.y_proba.plot.hist(bins=20)
        hist_ax.set_xlabel("Mean predicted probability")
        hist_ax.set_ylabel("Frequency")
        return hist_ax.figure

    def confidence_positive_figure(self) -> plt.figure:
        """plots false postitive rate and true positive rate
            against confidence threshold

        Returns:
            plt.figure: line graph
        """

        d = {
            "false positive rate": self.fpr[1:],
            "confidence": self.thresholds[1:],
            "true positive rate": self.tpr[1:],
        }
        df = pd.DataFrame(d)
        df = df[df.confidence > 0.5]
        fig, ax1 = plt.subplots(figsize=(9, 9))
        ax2 = ax1.twinx()
        df.plot(x="confidence", y="false positive rate", ax=ax1, color="r")
        ax1.set_ylabel("false positive rate", color="r")
        ax1.legend(loc="lower left")
        df.plot(x="confidence", y="true positive rate", ax=ax2, color="g")
        ax2.set_ylabel("true positive rate", color="g")
        ax2.legend(loc="upper right")

        return ax2.figure

    def generate_confusion_matrix(self) -> sns.heatmap:
        """generates a confusion matrix

        Returns:
            sns.heatmap: confusion matrix figure
        """
        cf_matrix = confusion_matrix(self.y_test, self.y_pred)
        cf_matrix_ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues", fmt="g")
        cf_matrix_ax.set_title("Confusion Matrix")
        cf_matrix_ax.set_xlabel("\nPredicted Values")
        cf_matrix_ax.set_ylabel("Actual Values ")
        cf_matrix_ax.xaxis.set_ticklabels(["False", "True"])
        cf_matrix_ax.yaxis.set_ticklabels(["False", "True"])
        return cf_matrix_ax.figure

    def generate_roc_curve(self) -> plt.figure:
        """generates a roc curve
        Returns:
            plt.figure: roc curve
        """
        roc_auc = auc(self.fpr, self.tpr)
        roc_display = RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=roc_auc).plot()
        roc_display.ax_.set_title("ROC curve")
        return roc_display.figure_

    def generate_pr_curve(self) -> plt.figure:
        """generates a pr curve
        Returns:
            plt.figure: pr curve
        """
        pr_display = PrecisionRecallDisplay.from_predictions(self.y_test, self.y_proba)
        pr_display.ax_.set_title("Precision-Recall curve")
        return pr_display.figure_

    """
        Returns:
            plt.figure: calibration curve
        """

    def generate_calibration_curve(self, bins: int = 4) -> plt.figure:
        """generates a calibration curve

        Args:
            bins (int, optional): number of bins for calibration curve. Defaults to 4.

        Returns:
            plt.figure: calibration curve
        """

        cal_display = CalibrationDisplay.from_predictions(self.y_test, self.y_proba, n_bins=bins)
        return cal_display.figure_

    def generate_graphs(self) -> Dict[str, plt.figure]:
        """generates a dictionary of graphs with filenames
            as keys and figures as values
        Returns:
            Dict[str, plt.figure]:
                filenames are keys and values are pyplot figures
        """

        graph_dict = {}
        graph_dict["roc_curve.png"] = self.generate_roc_curve()
        plt.close()

        graph_dict["pos_conf.png"] = self.confidence_positive_figure()
        plt.close()

        graph_dict["probability_freq.png"] = self.probability_frequency_figure()
        plt.close()

        graph_dict["pr_curve.png"] = self.generate_pr_curve()
        plt.close()

        graph_dict["calibration curve.png"] = self.generate_calibration_curve()
        plt.close()

        graph_dict["confusion_matrix.png"] = self.generate_confusion_matrix()
        plt.close()

        graph_dict["shap.png"] = self.shap_plot()
        plt.close()

        return graph_dict
