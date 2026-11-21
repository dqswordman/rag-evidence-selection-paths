"""Recompute V2 coverage and crossed-order analyses from the supplied frozen data.

No model inference or external data are used. Scores are internally in [0, 1].
The original question-stratified, matched bootstrap uses seed 20261107.
Run: OPENBLAS_NUM_THREADS=1 python scripts/analyze_v2.py
"""
from __future__ import annotations
import os
import csv, gzip, json, math, re, string
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable
from r2_stats import QuestionBootstrap, summarize_tracks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = Path(os.environ.get('REPLAY_OUT', ROOT / 'replay'))
OUT.mkdir(exist_ok=True)


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def norm(s):
    s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def score(pred, answers):
    a = norm(pred); at = a.split()
    f1 = em = 0.0
    for target in answers:
        b = norm(target); bt = b.split()
        em = max(em, float(a == b))
        overlap = sum((Counter(at) & Counter(bt)).values())
        val = (2.0 * overlap / (len(at) + len(bt))) if at and bt else float(at == bt)
        f1 = max(f1, val)
    return f1, em


def identity(r):
    return (r['dataset'], r['qid'], r['model'], r['track'])


def best(u, masks):
    return max(masks, key=lambda m: (float(u[m]), -m))


def greedy(u, stop=False):
    mask = 0
    for _ in range(3):
        choices = [i for i in range(6) if not mask & (1 << i)]
        i = max(choices, key=lambda i: (float(u[mask | (1 << i)]), -i))
        nxt = mask | (1 << i)
        if stop and float(u[nxt]) <= float(u[mask]):
            break
        mask = nxt
    return mask


def summarize(rows, fields, metrics, boot):
    return summarize_tracks(rows, fields, metrics, boot)


def main():
    questions = {(r['dataset'], r['qid']): r for r in jsonl(DATA/'freeze/questions.jsonl.gz')}
    cubes = {identity(r): r for r in jsonl(DATA/'freeze/old_cubes.jsonl.gz')}
    answers = defaultdict(dict)
    roles = defaultdict(dict)
    for r in jsonl(DATA/'freeze/old_answers.jsonl.gz'):
        k = identity(r); m = int(r['mask'])
        if m in answers[k]:
            raise ValueError(('Duplicate canonical answer', k, m))
        f1, em = score(r['prediction'], questions[k[:2]]['answers'])
        answers[k][m] = dict(prediction=r['prediction'], f1=f1, em=em)
        for role in r.get('legacy_roles', []):
            roles[k][role] = m
    complete = {k: v for k,v in answers.items() if len(v)==64}
    assert len(complete)==356
    feasible = [m for m in range(64) if m.bit_count()<=3]
    exact3 = [m for m in feasible if m.bit_count()==3]

    # Verify all three A-path sets and their recorded answer readouts.
    arows = read_csv(DATA/'analysis/A_per_question.csv')
    role_answers = read_csv(DATA/'analysis/A_role_answers.csv')
    ar = {(identity(r),r['role']): r for r in role_answers}
    errors = []
    for r in arows:
        k=identity(r); u=cubes[k]['u']
        selected={'E_le':best(u,feasible),'E_eq':best(u,exact3),'G3':greedy(u)}
        for role,m in selected.items():
            assert int(r[role+'_mask'])==m, (k,role)
            raw=ar[k,role]
            assert int(raw['mask'])==m
            f1,em=score(raw['prediction'],questions[k[:2]]['answers'])
            for met,val in [('f1',f1),('em',em)]:
                errors.append(abs(float(raw[met])-val))
                errors.append(abs(float(r[met+'_'+role])-val))
    assert max(errors)<1e-12, ('A scores',max(errors))

    # The six historical selection rules, with saved masks; add exact-3 and
    # all five cross-encoder rules for a stronger twelve-rule sensitivity.
    six={
        'BM25_top3':'bm25_top_b_b3',
        'BGE_top3':'bge_top_b_b3',
        'U_singleton_top3':'singleton_top_b_b3',
        'U_greedy_stop3':'true_marginal_greedy_stop_b3',
        'U_greedy_fixed3':'true_marginal_greedy_fixed_b3',
        'U_exact_le3':'exact_oracle_b3',
    }
    ce=defaultdict(dict)
    for r in read_csv(DATA/'analysis/C_selected_masks.csv'):
        ce[(r['dataset'],r['qid'],r['track'])][r['policy']]=int(r['mask'])
    coverage=[]; selected_manifest=[]
    for k,table in sorted(complete.items()):
        masks={name:roles[k][role] for name,role in six.items()}
        u=cubes[k]['u']
        assert masks['U_exact_le3']==best(u,feasible)
        assert masks['U_greedy_fixed3']==greedy(u)
        assert masks['U_greedy_stop3']==greedy(u,stop=True)
        masks12=dict(masks,U_exact_eq3=best(u,exact3),**ce[(k[0],k[1],k[3])])
        assert len(masks12)==12 and all(m.bit_count()<=3 for m in masks12.values())
        optimum=max(table[m]['f1'] for m in feasible)
        v6=max(table[m]['f1'] for m in masks.values())
        v12=max(table[m]['f1'] for m in masks12.values())
        coverage.append(dict(zip(('dataset','qid','model','track'),k),
            best_feasible_f1=optimum, best_six_f1=v6,best_twelve_f1=v12,
            likelihood_selected_f1=table[masks['U_exact_le3']]['f1'],
            greedy_selected_f1=table[masks['U_greedy_fixed3']]['f1'],
            likelihood_gap_f1=optimum-table[masks['U_exact_le3']]['f1'],
            extra_six_f1=optimum-v6,extra_twelve_f1=optimum-v12,
            six_unique_sets=len(set(masks.values())),twelve_unique_sets=len(set(masks12.values())),
            six_attains_best=float(abs(optimum-v6)<1e-12),
            twelve_attains_best=float(abs(optimum-v12)<1e-12)))
        for name,m in masks12.items():
            selected_manifest.append(dict(zip(('dataset','qid','model','track'),k),
                policy=name,mask=m,f1=table[m]['f1'],prediction=table[m]['prediction']))
    boot=QuestionBootstrap(repetitions=20000,seed=20261107)
    coverage_summary=summarize(coverage,('model',),
        ('best_feasible_f1','best_six_f1','best_twelve_f1','extra_six_f1','extra_twelve_f1',
         'likelihood_selected_f1','greedy_selected_f1','likelihood_gap_f1',
         'six_unique_sets','twelve_unique_sets','six_attains_best','twelve_attains_best'),boot)
    save_csv(OUT/'coverage_per_pool.csv',coverage)
    save_csv(OUT/'coverage_selected_sets.csv',selected_manifest)
    save_csv(OUT/'coverage_summary.csv',coverage_summary)

    # Cross canonical/reverse-selected membership with both presentation orders.
    banswers={}
    berr=[]
    for r in jsonl(DATA/'analysis/B_answers_42.jsonl.gz'):
        k=identity(r); m=int(r['mask']); key=(k,r['order'],m)
        assert key not in banswers
        f1,em=score(r['prediction'],questions[k[:2]]['answers'])
        berr += [abs(f1-r['f1']),abs(em-r['em'])]
        banswers[key]=dict(prediction=r['prediction'],f1=f1,em=em)
        if r['order']=='canonical':
            assert answers[k][m]['prediction']==r['prediction']
    assert max(berr)<1e-12
    bcubes={(identity(r),r['order']):r for r in jsonl(DATA/'analysis/B_utility_cubes.jsonl.gz')}
    crossed=[]
    for r in read_csv(DATA/'analysis/B_fixed_mask_and_reselection.csv'):
        k=identity(r); mc=int(r['canonical_mask']); mr=int(r['reverse_mask']); role=r['role']
        for order,mask in [('canonical',mc),('reverse',mr)]:
            u=bcubes[k,order]['U']
            expected=(best(u,feasible) if role=='E_le' else best(u,exact3) if role=='E_eq'
                      else greedy(u,stop=(role=='G_stop')) if role in ('G3','G_stop')
                      else max(feasible,key=lambda m:(banswers[k,order,m]['f1'],-m)))
            assert expected==mask, (k, role, order, expected, mask, bcubes[k,order]['masks'])
        # First letter: presentation. Second letter: selection objective order.
        cells={'cc':banswers[k,'canonical',mc], 'rc':banswers[k,'reverse',mc],
               'cr':banswers[k,'canonical',mr], 'rr':banswers[k,'reverse',mr]}
        row=dict(zip(('dataset','qid','model','track'),k),role=role,
                 canonical_mask=mc,reverse_mask=mr)
        for met in ('f1','em'):
            for state,val in cells.items():row[state+'_'+met]=val[met]
            row['canonical_reselection_'+met]=cells['cr'][met]-cells['cc'][met]
            row['reverse_reselection_'+met]=cells['rr'][met]-cells['rc'][met]
            row['crossed_difference_'+met]=row['reverse_reselection_'+met]-row['canonical_reselection_'+met]
            assert abs(cells['cc'][met]-float(r[met+'_canonical']))<1e-12
            assert abs(cells['rc'][met]-float(r[met+'_fixed_reverse']))<1e-12
            assert abs(cells['rr'][met]-float(r[met+'_reselected_reverse']))<1e-12
        crossed.append(row)
    crossed_summary=summarize(crossed,('model','role'),
        ('cc_f1','rc_f1','cr_f1','rr_f1','canonical_reselection_f1','reverse_reselection_f1','crossed_difference_f1',
         'cc_em','rc_em','cr_em','rr_em','crossed_difference_em'),boot)
    save_csv(OUT/'crossed_order_per_pool.csv',crossed)
    save_csv(OUT/'crossed_order_summary.csv',crossed_summary)

    # Verify cross-encoder selection under the raw score table, including MMR.
    logits=defaultdict(dict)
    for r in read_csv(DATA/'analysis/C_joint_scores.csv'):
        logits[(r['dataset'],r['qid'],r['track'])][int(r['mask'])]=float(r['raw_logit'])
    sim=defaultdict(dict)
    for r in jsonl(DATA/'freeze/embedding_cosines.jsonl.gz'):
        key=(r['dataset'],r['qid'],r['track'])
        sim[key][int(r['i']),int(r['j'])]=(1+float(r['cosine']))/2
        sim[key][int(r['j']),int(r['i'])]=(1+float(r['cosine']))/2
    for key,policies in sorted(ce.items()):
        u=logits[key]
        single=sorted(range(6),key=lambda i:(-u[1<<i],i))
        assert policies['CE_top3']==sum(1<<i for i in single[:3])
        # Empty set has no score; greedy first step chooses the best singleton.
        assert policies['CE_joint_greedy3']==greedy(u)
        assert policies['CE_joint_exact3']==best(u,exact3)
        assert policies['CE_joint_best1to3']==best(u,[m for m in feasible if m])
        chosen=[single[0]]
        while len(chosen)<3:
            candidates=[i for i in range(6) if i not in chosen]
            def mmr(i):
                relevance=1/(1+math.exp(-u[1<<i]))
                return .5*relevance-.5*max(sim[key][i,j] for j in chosen)
            chosen.append(max(candidates,key=lambda i:(mmr(i),-i)))
        assert policies['CE_MMR3']==sum(1<<i for i in chosen),key
    cerr=[]
    crows=read_csv(DATA/'analysis/C_policy_per_question.csv')
    for r in crows:
        k=identity(r);m=int(r['mask'])
        if r['policy'].startswith('CE_'):
            assert m==ce[(k[0],k[1],k[3])][r['policy']]
        for met in ('f1','em'):cerr.append(abs(float(r[met])-answers[k][m][met]))
    assert max(cerr)<1e-12

    # Figure fixture reads the true saved question, titles, and all four answers.
    k=('hotpotqa','5ac300905542996773102682','qwen','A')
    case_row=next(r for r in crossed if identity(r)==k and r['role']=='E_le')
    candidate=next(r for r in jsonl(DATA/'freeze/candidates.jsonl.gz') if (r['dataset'],r['qid'],r['track'])==(k[0],k[1],k[3]))
    case=dict(dataset=k[0],qid=k[1],model=k[2],track=k[3],role='E_le',
        question=questions[k[:2]]['question'],reference=questions[k[:2]]['answers'],
        candidates=candidate['candidates'],canonical_mask=case_row['canonical_mask'],reverse_mask=case_row['reverse_mask'],states=[])
    for name,order,m in [('Original','canonical',case_row['canonical_mask']),
                         ('Reversed','reverse',case_row['canonical_mask']),
                         ('Reselected','reverse',case_row['reverse_mask'])]:
        ids=[i for i in range(6) if m&(1<<i)]
        if order=='reverse':ids.reverse()
        answer=banswers[k,order,m]
        case['states'].append(dict(name=name,order=order,mask=m,documents=['D'+str(i+1) for i in ids],
            titles=[candidate['candidates'][i]['title'] for i in ids],**answer))
    (OUT/'figure1_case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n')

    validation=dict(input_scope='Frozen evidence selection and answer records',
        A_questions=len(questions),A_pool_model_conditions=len(arows),A_role_answers_checked=len(ar),
        A_max_readout_error=max(errors), complete_canonical_conditions=len(complete),
        complete_canonical_answers=sum(map(len,complete.values())),
        B_answer_records_checked=len(banswers),B_max_readout_error=max(berr),
        B_crossed_paths=len(crossed),C_pool_rules_recomputed=len(ce)*5,
        C_answer_readouts_checked=len(crows),C_max_readout_error=max(cerr),
        coverage_policy_union6=list(six),coverage_policy_union12=list(masks12),
        bootstrap_repetitions=20000,bootstrap_seed=20261107)
    (OUT/'v2_raw_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))
    print('\nCOVERAGE (F1 points; AB):')
    for r in coverage_summary:
        if r['track']=='AB' and r['metric'] in ('best_feasible_f1','best_six_f1','best_twelve_f1','extra_six_f1','extra_twelve_f1'):
            print(r['model'],r['metric'],*[round(100*r[x],4) for x in ('mean','ci_low','ci_high')])
    print('\nCROSSED ORDER (F1 points; AB; exact three):')
    for r in crossed_summary:
        if r['track']=='AB' and r['role']=='E_eq' and r['metric'].endswith('_f1'):
            print(r['model'],r['metric'],*[round(100*r[x],4) for x in ('mean','ci_low','ci_high')])

if __name__=='__main__':
    main()
