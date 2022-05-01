import pytest

from protein_classification import tokenize


class Cases:
    TOKENIZER = [
        ('AASQ', {'A': 1, 'S': 2, 'Q': 3}, 7, [1, 1, 2, 3, 0, 0, 0]),
        ('GTA', {'G': [0, 1], 'A': [1, 0]}, 5, [[0, 1], [0, 0], [1, 0], [0, 0], [0, 0]]),
        ('XDDCTY', {'D': 1}, 5, [0, 1, 1, 0, 0])
    ]


@pytest.mark.parametrize(['sequence', 'token_map', 'max_length', 'target'], Cases.TOKENIZER)
def test_tokenizer(sequence, token_map, max_length, target):
    tokenizer = tokenize.Tokenizer(token_map)

    result = tokenizer(sequence, padding='max_length', truncation=True, max_length=max_length)
    assert result == target

    val, *_ = tokenizer.token_map.values()
    assert tokenizer.feature_dim == (0 if type(val) == int else len(val))
