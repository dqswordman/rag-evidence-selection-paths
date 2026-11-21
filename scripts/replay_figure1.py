"""Replay Figure 1's saved example from raw scores and cached model responses.
This recomputes selection and answer scoring, without executing a generator.
"""
from pathlib import Path
import json
from analyze_v2 import jsonl, best, score
ROOT=Path(__file__).resolve().parents[1]

def main():
    dataset,qid,model,track='hotpotqa','5ac300905542996773102682','qwen','A'
    def same(r):return (r['dataset'],r['qid'],r['model'],r['track'])==(dataset,qid,model,track)
    q=next(r for r in jsonl(ROOT/'data/freeze/questions.jsonl.gz') if (r['dataset'],r['qid'])==(dataset,qid))
    cubes={r['order']:r for r in jsonl(ROOT/'data/analysis/B_utility_cubes.jsonl.gz') if same(r)}
    responses={(r['order'],r['mask']):r for r in jsonl(ROOT/'data/analysis/B_answers_42.jsonl.gz') if same(r)}
    feasible=[m for m in range(64) if m.bit_count()<=3]
    mc=best(cubes['canonical']['U'],feasible);mr=best(cubes['reverse']['U'],feasible)
    states=[]
    for name,order,mask in [('Original','canonical',mc),('Reversed','reverse',mc),('Reselected','reverse',mr)]:
        r=responses[order,mask];f1,em=score(r['prediction'],q['answers'])
        docs=[i+1 for i in range(6) if mask&(1<<i)]
        if order=='reverse':docs.reverse()
        states.append(dict(state=name,mask=mask,ordered_document_ids=docs,answer=r['prediction'],F1=100*f1,EM=100*em))
    assert [r['F1'] for r in states]==[100.,0.,100.]
    print(json.dumps(dict(question=q['question'],reference=q['answers'],states=states),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
