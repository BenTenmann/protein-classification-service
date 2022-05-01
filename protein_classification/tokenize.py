from typing import Dict, List


class Tokenizer:
    """
    The sequence tokenizer object. Maps a string of letters into a list of (of lists) of numerics, based on a provided
    token map.
    Sequences can be padded and truncated to give uniform length. If an unknown token is encountered, the pad token is
    inserted. The pad token is hard coded and depends on the token map value type: if the values are integers, then the
    pad token is 0, otherwise it is a list of 0s of length `len(value)`. Thus, 0 cannot be used in the token map.

    Attributes
    ----------
    token_map: Dict[str, int or List[float]]
        The mapping from alphabetical to numeric representation.
    feature_dim: int
        The dimensionality of the numeric token representation. Is 0 when index tokens are used.

    Examples
    --------
    A simple example; notice we avoid 0 in the token map:
    >>> token_map = {'A': 1, 'G': 2}
    >>> tokenizer = Tokenizer(token_map)
    >>> tokenizer('AGGAX')
    ... [1, 2, 2, 1, 0]

    The 'X' token is mapped to 0, which is the pad token.

    We can also specify truncation and padding etc.:
    >>> tokenizer('AGGAX', padding='max_length', truncation=True, max_length=7)
    ... [1, 2, 2, 1, 0, 0, 0]
    >>> tokenizer('AGGAX', padding='max_length', truncation=True, max_length=4)
    ... [1, 2, 2, 1]

    Notice that we are exclusively right padding / truncating.
    """
    def __init__(self, token_map: Dict[str, int or List[float]]):
        """
        Initialise a Tokenizer object.

        Parameters
        ----------
        token_map: Dict[str, int ot List[float]]
            The mapping from alphabetical to numeric representation.
        """
        self.token_map = token_map

        elem, *_ = token_map.values()
        self._pad_token = 0 if type(elem) == int else [0] * len(elem)

    def __call__(self, sequence: str, padding: str, truncation: bool, max_length: int) -> list:
        sequence_len = len(sequence)

        toks = []
        toks.extend([self.token_map.get(token, self._pad_token) for token in sequence])
        if padding == 'max_length':
            toks.extend([self._pad_token] * (max_length - sequence_len))

        if truncation:
            toks = toks[:max_length]
        return toks

    @property
    def feature_dim(self) -> int:
        if type(self._pad_token) == int:
            return 0
        out = len(self._pad_token)
        return out
