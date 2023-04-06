from metaflow import FlowSpec, step, Parameter, current, conda, conda_base, card
from metaflow.cards import Markdown, Image


@conda_base(python='3.8.1',
            libraries={'numpy': '1.20.3', 'matplotlib': '3.4.2', 'pandas': '2.0.0', 'scikit-learn': '1.2.2'})
class MBTI_Flow(FlowSpec):
    filename = Parameter('filename',
                         help='filename path for the CSV',
                         type=str,
                         default="data/twitter_MBTI.csv")
    test_size = Parameter('test_size',
                          help='test size in percentage',
                          type=float,
                          default=0.2)

    @card
    @step
    def start(self):
        import pandas as pd
        self.dataset = pd.read_csv(str(self.filename))
        self.next(self.compute_dataset_stats)

    @step
    def compute_dataset_stats(self):
        current.card.append(Markdown("**Dataset EDA**"))
        current.card.append(Image.from_matplotlib(self.dataset.groupby("label").count()["text"].plot()))
        current.card.append(Markdown("**End Example dataset EDA**"))
        self.next(self.split_data)

    @step
    def split_data(self):
        from sklearn.model_selection import train_test_split
        self.train, self.test = train_test_split(self.dataset, stratify=self.dataset["label"], test_size=self.test_size)
        self.next(self.train_model)

    @step
    def train_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression())
        ])
        parameters = {"tfidf__ngram_range": [(1, 2)],
                      "tfidf__min_df": [1],
                      "tfidf__analyzer": ["word"],
                      'clf__C': [10]}
        pipe.fit(self.train.text, self.train.label)
        self.model = pipe
        self.next(self.end)

    @step
    def end(self):
        from sklearn.metrics import classification_report
        report = classification_report(y_true=self.test["label"].to_list(),
                                       y_pred=list(self.model.predict(self.test.text.tolist())))
        print(report)
        current.card.append(Markdown("**Model results**"))
        current.card.append(Markdown(f"{report}"))
        current.card.append(Markdown("**End Model results**"))


if __name__ == '__main__':
    MBTI_Flow()
