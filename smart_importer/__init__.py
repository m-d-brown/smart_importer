"""Smart importer for Beancount and Fava."""
from smart_importer.entries import update_postings
from smart_importer.hooks import apply_hooks  # noqa
from smart_importer.predictor import EntryPredictor, ModelAttribute


class PredictPayees(EntryPredictor):
    """Predicts payees."""

    attribute = "payee"
    model_attributes = [
        ModelAttribute("narration", 0.8),
        ModelAttribute("payee", 0.5),
        ModelAttribute("date.day", 0.1),
    ]


class PredictPostings(EntryPredictor):
    """Predicts posting accounts."""

    model_attributes = [
        ModelAttribute("narration", 0.8),
        ModelAttribute("payee", 0.5),
        ModelAttribute("date.day", 0.1),
    ]

    @property
    def targets(self):
        return [
            " ".join(posting.account for posting in txn.postings)
            for txn in self.training_data
        ]

    def apply_prediction(self, entry, prediction):
        return update_postings(entry, prediction.split(" "))
