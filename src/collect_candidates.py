"""Collect public Monash FAQ candidates for manual review; never auto-add to training."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ALLOWED_HOSTS = {'www.monash.edu', 'monash.edu', 'online.monash.edu'}
USER_AGENT = 'MonashFAQResearch/1.0'


def extract_candidates(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    container = soup.select_one('main, #main-content, #main, article') or soup.body
    if container is None:
        return []
    for tag in container.select('script, style, nav, footer, header, form'):
        tag.decompose()
    candidates = []
    seen = set()
    # Squiz accordion pages generally keep each header and its answer in one
    # repeated panel. Online FAQ pages often use h5 plus paragraph siblings.
    for heading in container.find_all(['summary', 'h2', 'h3', 'h4', 'h5', 'button']):
        question = heading.get_text(' ', strip=True)
        if '?' not in question or len(question) > 400:
            continue
        panel = heading.find_parent(class_=lambda value: value and any(
            token in str(value).lower() for token in ('accordion-item', 'accordion__item', 'accordion-panel')))
        if panel:
            answer_nodes = panel.find_all(['p', 'ul', 'ol'])
            parts = [node.get_text(' ', strip=True) for node in answer_nodes
                     if node.find_parent(['ul', 'ol']) is None]
        else:
            parts = []
            for sibling in heading.next_siblings:
                name = getattr(sibling, 'name', None)
                if name in ('h2', 'h3', 'h4', 'h5', 'summary', 'button'):
                    break
                if name:
                    text = sibling.get_text(' ', strip=True)
                    if text:
                        parts.append(text)
        answer = ' '.join(parts).strip()
        key = question.casefold()
        if not answer or key in seen:
            continue
        seen.add(key)
        candidates.append({'question': question, 'answer_candidate': answer,
            'source_url': url, 'review_status': 'unreviewed',
            'note': 'Check scope, complete answer boundaries, linked conditions and currency before use.'})
    return candidates


def collect(sources, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    robots_cache, log, candidates = {}, [], []
    checked_at = datetime.now(timezone.utc).isoformat()
    for source in sources:
        url = source['url']
        parsed = urlparse(url)
        entry = {'source_url': url, 'checked_at_utc': checked_at}
        if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS:
            entry['status'] = 'disallowed_domain'
            log.append(entry)
            continue
        origin = f'{parsed.scheme}://{parsed.netloc}'
        try:
            if origin not in robots_cache:
                response = session.get(origin + '/robots.txt', timeout=30, allow_redirects=False)
                time.sleep(2)
                parser = RobotFileParser(origin + '/robots.txt')
                if response.status_code == 404:
                    parser.parse([])
                elif response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    raise RuntimeError(f'robots.txt unavailable: HTTP {response.status_code}; skip this origin')
                robots_cache[origin] = parser
            if not robots_cache[origin].can_fetch(USER_AGENT, url):
                entry['status'] = 'robots_disallowed'
                log.append(entry)
                continue
            response = session.get(url, timeout=30, allow_redirects=False)
            time.sleep(2)
            # Deliberately do not follow redirects into an uninspected destination.
            if 300 <= response.status_code < 400:
                entry.update(status='redirect_requires_review', location=response.headers.get('Location'))
            elif response.status_code != 200:
                entry.update(status='http_error', http_status=response.status_code)
            elif 'text/html' not in response.headers.get('Content-Type', '').lower():
                entry['status'] = 'unsupported_content_type'
            else:
                raw_path = output_dir / (source['source_id'] + '.html')
                raw_path.write_bytes(response.content)
                items = extract_candidates(response.text, url)
                candidates.extend(items)
                entry.update(status='downloaded', candidates=len(items),
                    sha256=hashlib.sha256(response.content).hexdigest(), raw_file=raw_path.name)
        except (requests.RequestException, RuntimeError) as exc:
            entry.update(status='skipped', error=str(exc))
        log.append(entry)
    (output_dir / 'unreviewed_candidates.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in candidates), encoding='utf-8')
    (output_dir / 'collection_log.json').write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'candidate_count': len(candidates), 'page_count': len(log), 'output_dir': str(output_dir),
            'note': 'Candidates remain unreviewed and are not training data.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path(__file__).parent / 'sources.json')
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'monash_raw_candidates')
    args = parser.parse_args()
    print(json.dumps(collect(json.loads(args.sources.read_text(encoding='utf-8')), args.output), indent=2))
