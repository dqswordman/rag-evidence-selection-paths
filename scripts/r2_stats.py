"""Shared question-cluster bootstrap, with matched draws across conditions."""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


class QuestionBootstrap:
    def __init__(self, repetitions=20000, seed=20261107):
        self.repetitions = repetitions
        self.seed = seed
        self.draws = {}

    def weights(self, questions):
        key = tuple(questions)
        if key not in self.draws:
            n = len(key)
            rng = np.random.Generator(np.random.PCG64(self.seed))
            weights = np.zeros((self.repetitions, n), dtype=np.float64)
            datasets = sorted({d for d, q in key})
            for dataset in datasets:
                positions = np.asarray([i for i, (d, q) in enumerate(key) if d == dataset])
                draws = positions[rng.integers(0, len(positions), size=(self.repetitions, len(positions))) ]
                row_ids = np.arange(self.repetitions)[:, None]
                np.add.at(weights, (row_ids, draws), 1.0)
            assert np.all(weights.sum(axis=1) == n)
            self.draws[key] = weights / n
        return self.draws[key]

    def summarize(self, questions, columns):
        assert questions == sorted(questions) and len(set(questions)) == len(questions) and questions
        names = list(columns)
        matrix = np.column_stack([np.asarray(columns[name], dtype=float) for name in names])
        assert matrix.shape == (len(questions), len(names)) and np.all(np.isfinite(matrix))
        values = self.weights(questions) @ matrix
        intervals = np.quantile(values, [.025, .975], axis=0)
        return [dict(metric=name, mean=float(matrix[:,i].mean()), ci_low=float(intervals[0,i]),
                     ci_high=float(intervals[1,i]), n_questions=len(questions),
                     strata=dict(Counter(d for d, q in questions)), bootstrap_repetitions=self.repetitions,
                     bootstrap_seed=self.seed, independent_unit='question', interval='pointwise paired percentile')
                for i, name in enumerate(names)]


def paired_models(rows, metrics, extra_keys):
    by_pair = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ('dataset', 'qid', 'track', *extra_keys))
        assert row['model'] not in by_pair[key], key
        by_pair[key][row['model']] = row
    result = []
    order = ['qwen','gemma','mistral']
    for key, models in sorted(by_pair.items()):
        for first, second in combinations((m for m in order if m in models), 2):
            result.append(dict(dataset=key[0], qid=key[1], track=key[2], model=first+'_minus_'+second,
                               **{k: v for k,v in zip(extra_keys,key[3:])},
                               **{metric: float(models[first][metric])-float(models[second][metric]) for metric in metrics}))
    return result


def summarize_tracks(rows, group_fields, metrics, bootstrap):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        for track in ('A','B','AB'):
            by_question = defaultdict(list)
            for row in group:
                if track == 'AB' or row['track'] == track:
                    by_question[row['dataset'],row['qid']].append(row)
            if not by_question:
                continue
            questions = sorted(by_question)
            if track == 'AB':
                assert all({row['track'] for row in by_question[q]} == {'A','B'} and len(by_question[q]) == 2 for q in questions), key
            else:
                assert all(len(by_question[q]) == 1 for q in questions), key
            columns = {metric: [sum(float(row[metric]) for row in by_question[q])/len(by_question[q]) for q in questions]
                       for metric in metrics}
            for row in bootstrap.summarize(questions,columns):
                result.append(dict(**{k:v for k,v in zip(group_fields,key)}, track=track,
                                   repeated_rows=sum(len(values) for values in by_question.values()), **row))
    return result
