"""Auditable BM25 retrieval over a frozen, paraphrased FAQ snapshot."""
import json
import math
import re
from collections import Counter
from pathlib import Path

STOP = set('a an the i my me we you your it is are was be do does did to of for in on at and or can how what where when will have may with as each'.split())
REFUSAL = 'I cannot establish that from the supplied Monash FAQ snapshot. Please check with the relevant official Monash team.'

def tokens(text):
    return [word for word in re.findall(r'[a-z0-9]+', text.lower()) if word not in STOP]

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

class BM25:
    def __init__(self, records, k1=1.5, b=0.75):
        if not records:
            raise ValueError('The knowledge base is empty')
        self.records = records
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(r['question'] + ' ' + r['answer'])) for r in records]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(records)
        df = Counter(t for doc in self.docs for t in doc)
        self.idf = {t: math.log(1 + (len(records) - n + .5)/(n + .5)) for t, n in df.items()}

    def search(self, query, top_k=3):
        terms = set(tokens(query))
        hits = []
        for record, doc, length in zip(self.records, self.docs, self.lengths):
            score = sum(self.idf.get(t, 0) * doc.get(t, 0) * (self.k1+1) /
                        (doc.get(t, 0) + self.k1*(1-self.b+self.b*length/self.avg_length)) for t in terms)
            # Coverage is bounded and includes query terms absent from the corpus.
            coverage = len(terms.intersection(doc)) / max(1, len(terms))
            hits.append({'id': record['id'], 'score': score, 'coverage': coverage, 'record': record})
        return sorted(hits, key=lambda h: (-h['score'], h['id']))[:top_k]

def calibrate(index, cases):
    candidates = [x/20 for x in range(21)] + [1.01]
    details = []
    for threshold in candidates:
        correct = 0
        for case in cases:
            hit = index.search(case['question'], 1)[0]
            accepted = hit['score'] > 0 and hit['coverage'] >= threshold
            correct += ((accepted and hit['id'] == case['target_id']) if case['kind'] == 'answerable' else not accepted)
        details.append({'threshold': threshold, 'correct_decisions': int(correct), 'total': len(cases)})
    best = max(details, key=lambda row: (row['correct_decisions'], row['threshold']))
    return {'threshold': best['threshold'], 'selection': 'Maximize calibration decision count; ties prefer greater abstention.',
            'calibration_only': True, 'candidates': details}

def retrieve(index, question, threshold):
    hits = index.search(question, 3)
    accepted = hits[0]['score'] > 0 and hits[0]['coverage'] >= threshold
    return hits, accepted

def context_messages(question, hits):
    context = '\n\n'.join(f"[{n}] Audience: {h['record']['audience']}\nQuestion: {h['record']['question']}\nAnswer: {h['record']['answer']}" for n,h in enumerate(hits, 1))
    return [
        {'role': 'system', 'content': 'Answer briefly in English using only the supplied FAQ excerpts. Respect their audience. If they do not establish the requested fact, say you cannot establish it from this snapshot. Do not invent numbers, dates, personal outcomes, or URLs. Do not output source URLs; the application attaches retrieval metadata separately. Snapshot date: 2026-10-08. You are not an official university representative.'},
        {'role': 'user', 'content': f'FAQ excerpts:\n{context}\n\nUser question: {question}'},
    ]

def is_refusal(text):
    # A surface heuristic, never used as a factual correctness label.
    phrases = ['cannot establish', 'cannot determine', "can't determine", 'not provided',
               'not available', 'not specified', 'not included', 'does not specify',
               'do not have access', "don't have access", 'cannot access', 'not contain',
               'insufficient information', 'no information', 'cannot confirm', 'cannot guarantee']
    return any(phrase in text.lower() for phrase in phrases)

def source_urls(text):
    return [s.rstrip('.,);]') for s in re.findall(r'https?://[^\s<>]+', text)]
