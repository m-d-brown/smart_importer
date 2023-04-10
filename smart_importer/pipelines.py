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


def txn_attr_getter(attribute_name: str, default=None):
    """Return attribute getter for a transaction that also handles metadata."""
    if attribute_name.startswith("meta."):
        meta_attr = attribute_name[5:]

        def getter(txn):
            return txn.meta.get(meta_attr) or default

        return getter

    def base_getter(txn):
        get = operator.attrgetter(attribute_name)
        return get(txn) or default

    return base_getter


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
        super().__init__(ngram_range=(1, 3), tokenizer=tokenizer)

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
