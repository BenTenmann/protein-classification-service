import pytest


@pytest.fixture
def mock_read_json(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.srsly.read_json',
        lambda *args, **kwargs: {'A': 1, 'S': 2, 'Q': 3}
    )


@pytest.fixture
def mock_read_jsonl(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.srsly.read_jsonl',
        lambda *args, **kwargs: [{'sequence': 'AASQ', 'family_accession': 'XX'}]
    )


@pytest.fixture
def label_map():
    def _method():
        return {'XX': 0}
    return _method
