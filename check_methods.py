"""Independently reselect the two evidence masks from the full saved utility surfaces."""
import gzip
import json
from pathlib import Path
from methods.eug_core import exact_optimum
from methods.eug_inference import render_user_prompt

ROOT=Path(__file__).resolve().parent
def main():
    selected={}
    with gzip.open(ROOT/'data/analysis/B_utility_cubes.jsonl.gz','rt',encoding='utf-8') as stream:
        for record in map(json.loads,stream):
            if (record['dataset'],record['qid'],record['model'],record['track'])!=('hotpotqa','5ac300905542996773102682','qwen','A'):continue
            values=dict(enumerate(record['U']))
            result=exact_optimum(values,6,3,at_most=True)
            assert result.bitmask==record['masks']['E_le']
            selected[record['order']]=result.bitmask
    with gzip.open(ROOT/'data/freeze/questions.jsonl.gz','rt',encoding='utf-8') as stream:
        case=next(x for x in map(json.loads,stream) if x['qid']=='5ac300905542996773102682')
    assert case['question'] in render_user_prompt(case['question'],'')
    assert selected=={'canonical':5,'reverse':33},selected
    print(json.dumps({'status':'PASS','question':case['question'],'maximum_three_passages':selected},indent=2))
if __name__=='__main__':main()
