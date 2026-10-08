import sys
import unittest
import hashlib
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from retrieval import BM25, calibrate, read_jsonl, retrieve, context_messages

ROOT = Path(__file__).resolve().parents[1]

class ResearchIntegrityTests(unittest.TestCase):
    def test_data_checksums(self):
        expected=json.loads((ROOT/'data/checksums.json').read_text())
        for name,digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest(),digest,name)

    def test_split_disjoint_and_holdout_only(self):
        rows = {s:read_jsonl(ROOT/'data'/f'{s}.jsonl') for s in ('train','validation','test')}
        sets = {s:{r['group_id'] for r in v} for s,v in rows.items()}
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['test'] & sets['validation'])
        for row in read_jsonl(ROOT/'data/benchmark.jsonl'):
            if row['kind']=='answerable':
                self.assertIn(row['target_id'], sets['test'])
        for row in read_jsonl(ROOT/'data/calibration.jsonl'):
            if row['kind']=='answerable':
                self.assertIn(row['target_id'], sets['validation'])

    def test_zero_overlap_refuses(self):
        rows = read_jsonl(ROOT/'data/monash_qa.jsonl')
        hits, accepted = retrieve(BM25(rows), 'zyxwvu qqqqq', 0.1)
        self.assertFalse(accepted)
        self.assertEqual(hits[0]['score'], 0)

    def test_known_fact_ranking_and_reference_not_injected(self):
        index = BM25(read_jsonl(ROOT/'data/monash_qa.jsonl'))
        hits = index.search('Do research scholarships pay visa application charges?')
        self.assertEqual(hits[0]['id'], 'research_offers_02')
        messages = context_messages('Who am I?', hits)
        self.assertEqual([r['role'] for r in messages], ['system','user'])
        self.assertNotIn('reference_answer', str(messages))
        self.assertNotIn('https://', str(messages))

    def test_fixed_calibration_reproducible(self):
        index = BM25(read_jsonl(ROOT/'data/monash_qa.jsonl'))
        cases = read_jsonl(ROOT/'data/calibration.jsonl')
        self.assertEqual(calibrate(index,cases), calibrate(index,cases))

if __name__=='__main__':
    unittest.main()
