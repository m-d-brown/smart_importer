"""Smart importer for Beancount and Fava."""
from smart_importer.entries import update_postings
from smart_importer.hooks import apply_hooks  # noqa
from smart_importer.predictor import EntryPredictor
from smart_importer.pipelines import NumericAttribute, ConcatAttribute


COMMON_MODEL = [
    ConcatAttribute(["payee", "narration"], 0.5),
    NumericAttribute("date.day", 0.1),
]


class PredictPayees(EntryPredictor):
    """Predicts payees."""

    attribute = "payee"
    model_attributes = COMMON_MODEL


class PredictPostings(EntryPredictor):
    """Predicts posting accounts."""

    model_attributes = COMMON_MODEL

    @property
    def targets(self):
        return [
            " ".join(posting.account for posting in txn.postings)
            for txn in self.training_data
        ]

    def apply_prediction(self, entry, prediction):
        return update_postings(entry, prediction.split(" "))
