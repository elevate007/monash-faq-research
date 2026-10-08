"""Local extractive FAQ demo. No LLM generation and no hosted service."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from retrieval import BM25, read_jsonl, calibrate, retrieve, REFUSAL

ROOT = Path(__file__).resolve().parents[1]
INDEX = BM25(read_jsonl(ROOT/'data/monash_qa.jsonl'))
THRESHOLD = calibrate(INDEX, read_jsonl(ROOT/'data/calibration.jsonl'))['threshold']

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == '/api/search':
            question = parse_qs(url.query).get('q', [''])[0][:1000]
            hits, accepted = retrieve(INDEX, question, THRESHOLD)
            result = {'accepted':accepted, 'answer':hits[0]['record']['answer'] if accepted else REFUSAL,
                      'source':hits[0]['record']['source_url'] if accepted else None,
                      'coverage':hits[0]['coverage'], 'threshold':THRESHOLD,
                      'candidates':[{'question':h['record']['question'], 'answer':h['record']['answer'],
                                     'source':h['record']['source_url']} for h in hits]}
            payload, mime = json.dumps(result).encode(), 'application/json'
        elif url.path == '/':
            payload, mime = (ROOT/'docs/demo.html').read_bytes(), 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8000)
    args = parser.parse_args()
    print(f'Extractive FAQ demo: http://127.0.0.1:{args.port} (Ctrl+C to stop)',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
