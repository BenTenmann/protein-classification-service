import enum
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Callable, List, Tuple, Type

import numpy as np
import pandas as pd
import srsly
from hmm_profile import reader, models
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import rv_discrete

# ----- Logging Setup ------------------------------------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s.%(msecs)03d - %(levelname)s - %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ----- Environment Variables ---------------------------------------------------------------------------------------- #
INPUT = os.environ.get('INPUT')
OUTPUT = os.environ.get('OUTPUT')

CLASS_COLUMN = os.environ.get('CLASS_COLUMN', 'family_accession')
N_SAMPLES_LOWER_BOUND = os.environ.get('N_SAMPLES_LOWER_BOUND', 200)
N_SAMPLES_UPPER_BOUND = os.environ.get('N_SAMPLES_UPPER_BOUND', 200)

UNDER_SAMPLING_METHOD = os.environ.get('UNDER_SAMPLING_METHOD', 'NONE')
OVER_SAMPLING_METHOD = os.environ.get('OVER_SAMPLING_METHOD', 'NONE')
CLASS_WEIGHTING_METHOD = os.environ.get('CLASS_WEIGHTING_METHOD', 'NONE')
BETA = os.environ.get('BETA', 0.99)


# ----- Helpers ------------------------------------------------------------------------------------------------------ #
def time_it(fn):
    @wraps(fn)
    def _method(*args, **kwargs):
        logging.info(f'running: {fn.__name__}')
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        logging.info(f'time taken: {t1 - t0:.3f}s')

        return out
    return _method


@time_it
def read_dataframe(path: Path) -> pd.DataFrame:
    lines = srsly.read_jsonl(path)
    df = pd.DataFrame(lines)
    return df


def identifier_check(enumerator: Type[enum.Enum]):
    def _decorator(fn):
        @wraps(fn)
        def _method(identifier: str, *args, **kwargs):
            try:
                cls = enumerator[identifier]
            except KeyError:
                msg = ', '.join(map(str, enumerator))
                raise ValueError(f'{enumerator.__class__.__name__} must be one of: {msg}')
            out = fn(cls, *args, **kwargs)
            return out
        return _method
    return _decorator


@time_it
def get_label_map(class_column: pd.Series) -> dict:
    num_classes = class_column.nunique()
    out = dict(zip(class_column.sort_values().unique(), range(num_classes)))
    return out


def get_transition_matrix(profile_step: models.BaseStep) -> np.ndarray:
    states = ('emission', 'insertion', 'deletion')
    mat = []
    for state_i in states:
        row = []
        for state_j in states:
            try:
                prob = getattr(profile_step, f'p_{state_i}_to_{state_j}')
            except AttributeError:
                prob = 0.
            row.append(prob)
        mat.append(row)
    out = np.array(mat)
    return out


def get_emission_matrix(profile_step: models.BaseStep) -> Tuple[np.ndarray, list]:
    emission_probs = profile_step.p_emission_char
    insertion_probs = profile_step.p_insertion_char

    states = list(emission_probs.keys())
    mat = []
    for hidden_state in [emission_probs, insertion_probs]:
        row = [hidden_state.get(state) for state in states] + [0]
        mat.append(row)
    deletion_state = ([0] * len(states)) + [1]
    states.append('')

    arr = np.array([*mat, deletion_state])
    return arr, states


def step_to_mat(profile_step: models.BaseStep) -> Tuple[np.ndarray, np.ndarray, list]:
    transition_matrix = get_transition_matrix(profile_step)
    emission_matrix, emission_states = get_emission_matrix(profile_step)
    return transition_matrix, emission_matrix, emission_states


# ----- Methods ------------------------------------------------------------------------------------------------------ #
class UnderSamplingMethods(enum.Enum):
    NONE = 0
    RANDOM = 1


class OverSamplingMethods(enum.Enum):
    NONE = 0
    RANDOM = 1
    HMM = 2


class ClassWeightingMethods(enum.Enum):
    NONE = 0
    INVERSE_CLASS_FREQUENCY = 1
    INVERSE_SQUARE_ROOT_CLASS_FREQUENCY = 2
    EFFECTIVE_SAMPLE_NUMBER = 3


# ----- Resampling --------------------------------------------------------------------------------------------------- #
class HMM:
    def __init__(self, profile: models.HMM):
        start_trans, start_emm, start_state = step_to_mat(profile.start_step)

        prior = start_trans.sum(axis=0) / start_trans.sum()
        hidden_states = range(len(prior))

        self.hidden_prior = rv_discrete(values=(list(hidden_states), prior))
        self.start_emm = start_emm
        self.start_states = start_state

        steps = map(step_to_mat, profile.steps)
        self.steps = list(steps)

    def sample(self) -> list:
        init_state = self.hidden_prior.rvs()


def get_sampling_strategy(df: pd.DataFrame, condition: Callable, value: int) -> dict:
    counts = df[CLASS_COLUMN].value_counts()
    clamp_targets = counts.loc[counts.apply(condition)].index
    out = dict(zip(clamp_targets, [value] * len(clamp_targets)))
    return out


@identifier_check(UnderSamplingMethods)
def get_under_sampling_function(cls):
    if cls == UnderSamplingMethods.NONE:
        def _method(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
            return df

    elif cls == UnderSamplingMethods.RANDOM:
        def _method(df: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
            bound = int(N_SAMPLES_UPPER_BOUND)
            sampling_strategy = get_sampling_strategy(df, lambda x: x > bound, bound)
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                         random_state=42)
            out, _ = sampler.fit_resample(df, target)
            return out

    else:
        raise RuntimeError
    return time_it(_method)


@identifier_check(OverSamplingMethods)
def get_over_sampling_function(cls):
    if cls == OverSamplingMethods.NONE:
        def _method(df: pd.DataFrame, target: pd.Series, *args, **kwargs) -> pd.DataFrame:
            return df

    elif cls == OverSamplingMethods.RANDOM:
        def _method(df: pd.DataFrame, target: pd.Series, *args, **kwargs) -> pd.DataFrame:
            bound = int(N_SAMPLES_LOWER_BOUND)
            sampling_strategy = get_sampling_strategy(df, lambda x: x < bound, bound)
            sampler = RandomOverSampler(*args,
                                        sampling_strategy=sampling_strategy,
                                        random_state=42,
                                        **kwargs)
            out, _ = sampler.fit_resample(df, target)
            return out

    elif cls == OverSamplingMethods.HMM:
        def _method(df: pd.DataFrame, target: pd.Series, hmm_profiles, *args, **kwargs) -> pd.DataFrame:
            pass

    else:
        raise RuntimeError
    return time_it(_method)


# ----- Reweighting -------------------------------------------------------------------------------------------------- #
@identifier_check(ClassWeightingMethods)
def get_class_weighting_function(cls):
    if cls == ClassWeightingMethods.NONE:
        def _method(counts: np.ndarray, *args, **kwargs) -> np.ndarray:
            return np.ones(len(counts))

    elif cls == ClassWeightingMethods.INVERSE_CLASS_FREQUENCY:
        def _method(counts: np.ndarray, *args, **kwargs) -> np.ndarray:
            return 1 / counts

    elif cls == ClassWeightingMethods.INVERSE_SQUARE_ROOT_CLASS_FREQUENCY:
        def _method(counts: np.ndarray, *args, **kwargs):
            return 1 / np.sqrt(counts)

    elif cls == ClassWeightingMethods.EFFECTIVE_SAMPLE_NUMBER:
        def _method(class_counts: np.ndarray, beta: float) -> np.ndarray:
            return (1 - beta ** class_counts) / (1 - beta)

    else:
        raise RuntimeError
    return time_it(_method)


@time_it
def compute_weighting(class_column: pd.Series, fn: Callable, *args, **kwargs) -> dict:
    counts = class_column.value_counts()
    weights = fn(counts.to_numpy(), *args, **kwargs)
    out = {'weights': weights.tolist()}
    return out


# ----- Script ------------------------------------------------------------------------------------------------------- #
def main():
    # step 1: under-sample majority classes
    # step 2: over-sample minority classes
    # step 3: create label-weighting
    input_path = Path(INPUT)
    output_path = Path(OUTPUT)

    logging.debug(f'input path: {input_path}')
    logging.debug(f'output path: {output_path}')
    logging.debug(f'under sampling strategy: {UNDER_SAMPLING_METHOD}')
    logging.debug(f'under sampling strategy: {OVER_SAMPLING_METHOD}')
    logging.debug(f're-weighting strategy: {CLASS_WEIGHTING_METHOD}')

    df = read_dataframe(input_path)

    under_sampling_fn = get_under_sampling_function(UNDER_SAMPLING_METHOD)
    over_sampling_fn = get_over_sampling_function(OVER_SAMPLING_METHOD)
    weighting_fn = get_class_weighting_function(CLASS_WEIGHTING_METHOD)

    df = under_sampling_fn(df, df[CLASS_COLUMN])
    df = over_sampling_fn(df, df[CLASS_COLUMN])

    label_weights = compute_weighting(df[CLASS_COLUMN], weighting_fn, float(BETA))
    label_map = get_label_map(df[CLASS_COLUMN])

    srsly.write_jsonl(output_path / f'{input_path.stem}-resampled.jsonl', df.to_dict(orient='records'))
    srsly.write_json(output_path / 'label-map.json', label_map)
    srsly.write_json(output_path / 'label-weights.json', label_weights)


if __name__ == '__main__':
    main()
