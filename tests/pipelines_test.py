"""Tests for the Machine Learning Helpers."""
# pylint: disable=missing-docstring
import numpy as np
from beancount.parser import parser

from smart_importer.pipelines import (
    StringEstimator,
    NumericEstimator,
    txn_attr_getter,
)

TEST_DATA, _, __ = parser.parse_string(
    """
2016-01-01 open Assets:US:BofA:Checking USD
2016-01-01 open Expenses:Food:Groceries USD
2016-01-01 open Expenses:Food:Coffee USD

2016-01-06 * "Farmer Fresh" "Buying groceries"
  Assets:US:BofA:Checking  -10.00 USD

2016-01-07 * "Starbucks" "Coffee"
  Assets:US:BofA:Checking  -4.00 USD
  Expenses:Food:Coffee

2016-01-07 * "Farmer Fresh" "Groceries"
  Assets:US:BofA:Checking  -11.20 USD
  Expenses:Food:Groceries

2016-01-08 * "Gimme Coffee" "Coffee"
  Assets:US:BofA:Checking  -3.50 USD
  Expenses:Food:Coffee
"""
)
TEST_TRANSACTIONS = TEST_DATA[3:]
TEST_TRANSACTION = TEST_TRANSACTIONS[0]


def attr_getter(attribute, default=None):
    return StringEstimator(txn_attr_getter(attribute, default=default))


def test_get_payee():
    assert attr_getter("payee").transform(TEST_TRANSACTIONS) == [
        "Farmer Fresh",
        "Starbucks",
        "Farmer Fresh",
        "Gimme Coffee",
    ]


def test_get_narration():
    assert attr_getter("narration").transform(TEST_TRANSACTIONS) == [
        "Buying groceries",
        "Coffee",
        "Groceries",
        "Coffee",
    ]


def test_get_metadata():
    txn = TEST_TRANSACTION
    txn.meta["attr"] = "value"
    assert attr_getter("meta.attr").transform([txn]) == ["value"]
    assert attr_getter("meta.attr", "default").transform(
        TEST_TRANSACTIONS
    ) == [
        "value",
        "default",
        "default",
        "default",
    ]


def test_get_day_of_month():
    get_day = txn_attr_getter("date.day")
    assert list(map(get_day, TEST_TRANSACTIONS)) == [6, 7, 7, 8]

    extract_day = NumericEstimator(txn_attr_getter("date.day"))
    transformed = extract_day.transform(TEST_TRANSACTIONS)
    assert (transformed == np.array([[6], [7], [7], [8]])).all()
