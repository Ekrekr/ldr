# -*- coding: utf-8 -*-
import pytest

from ldr import LDR
from examples.classification import Classification


@pytest.fixture()
def examples_output_path():
    return "examples/output"


@pytest.fixture()
def classification_example(examples_output_path):
    return Classification(output_path=examples_output_path)
