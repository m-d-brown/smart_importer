"""Machine learning pipelines for data extraction."""

from __future__ import annotations

import operator

import numpy
from beancount.core.data import Transaction
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline


class NoFitMixin:
    """Mixin that implements a transformer's fit method that returns self."""

    def fit(self, *_, **__):
        """A noop."""
        return self


def get_txn_attr(txn: Transaction, attribute_name: str, default=None):
    """Returns the named attribute from a Transaction."""
    if attribute_name.startswith("meta."):
        val = txn.meta.get(attribute_name[5:])
    else:
        val = operator.attrgetter(attribute_name)(txn)
    return val or default


def txn_attr_getter(attribute_name: str, default=None):
    """Return attribute getter for a transaction that also handles metadata."""

    def getter(txn):
        return get_txn_attr(txn, attribute_name, default=default)

    return getter


class NumericEstimator(BaseEstimator, TransformerMixin, NoFitMixin):
    """Get a numeric transaction attribute and vectorize."""

    def __init__(self, txn_to_data):
        self.txn_to_data = txn_to_data

    def transform(self, txns: list[Transaction], _y=None):
        """Return list of entry attributes."""
        return numpy.array([self.txn_to_data(t) for t in txns], ndmin=2).T


class StringEstimator(BaseEstimator, TransformerMixin, NoFitMixin):
    """Get a string transaction attribute."""

    def __init__(self, txn_to_data):
        self.txn_to_data = txn_to_data

    def transform(self, txns: list[Transaction], _y=None):
        """Return list of entry attributes."""
        return [self.txn_to_data(t) for t in txns]


class StringVectorizer(CountVectorizer):
    """Subclass of CountVectorizer that handles empty data."""

    def __init__(self, tokenizer=None):
        super().__init__(ngram_range=(1, 4), tokenizer=tokenizer)

    def fit_transform(self, raw_documents: list[str], y=None):
        try:
            return super().fit_transform(raw_documents, y)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))

    def transform(self, raw_documents: list[str], _y=None):
        try:
            return super().transform(raw_documents)
        except ValueError:
            return numpy.zeros(shape=(len(raw_documents), 0))


class ModelAttribute:
    """ModelAttribute has a weight and creates a training pipeline."""

    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    def create_pipeline(self, tokenizer):
        """Returns an sklearn Pipeline."""
        raise NotImplementedError


class NumericAttribute(ModelAttribute):
    """An attribute for numbers, like day-of-the-month."""

    def create_pipeline(self, tokenizer):
        return NumericEstimator(txn_attr_getter(self.name))


class StringAttribute(ModelAttribute):
    """An attribute for strings, like payee or narration."""

    def create_pipeline(self, tokenizer):
        return make_pipeline(
            StringEstimator(txn_attr_getter(self.name, default="")),
            StringVectorizer(tokenizer),
        )


class ConcatAttribute(ModelAttribute):
    """An attribute that concatenates multiple string fields."""

    def __init__(self, attributes: list[str], weight: float):
        super().__init__("-".join(attributes), weight)
        self.attributes = attributes

    def _get_data(self, txn):
        parts = []
        for attr in self.attributes:
            val = get_txn_attr(txn, attr)
            if val:
                # Provide a header to allow some n-grams that can help
                # improve prediction quality if the source field is
                # important.
                #parts.append(f"{attr}")
                parts.append(val)
        print('|', ' '.join(p.account for p in txn.postings), "|", " ".join(parts))
        return " ".join(parts)

    def create_pipeline(self, tokenizer):
        return make_pipeline(
            StringEstimator(self._get_data),
            StringVectorizer(tokenizer),
        )
