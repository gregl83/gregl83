#!/usr/bin/env python3
"""Collect public project evidence, cache concise summaries, and build a storyboard."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'prompts/showcase-system.txt'
MODEL = 'gemini-3.5-flash-lite'
FEATURED_REPO = 'agent-vigilo/vigilo'
MAX_PROJECTS = 12
MAX_DOCUMENT_CHARS = 6000
BAD_COPY = re.compile(r'\b(hir(?:e|ing)|recruit\w*|job\w*|career\w*|resume|résumé|candidate|frontier|openai|anthropic|seeking|available for|looking for|employment|opportunities|aspiring|best|fastest|world.class|production.ready)\b', re.I)


def gh(*args):
    result = subprocess.run(['gh', *args], check=True, capture_output=True, text=True, timeout=90)
    return json.loads(result.stdout)


def candidates(owner):
    fields = '''nameWithOwner description url isPrivate isFork isArchived pushedAt
      defaultBranchRef { target { oid } }
      owner { __typename login ... on Organization { name } }'''
    query = '''query($owner:String!) { user(login:$owner) {
      pinnedItems(first:6, types:REPOSITORY) { nodes { ... on Repository { FIELDS } } }
      repositories(first:30, privacy:PUBLIC, ownerAffiliations:OWNER,
        orderBy:{field:PUSHED_AT,direction:DESC}) { nodes { FIELDS } }
    }
    featured: repository(owner:"agent-vigilo", name:"vigilo") { FIELDS }
    }'''.replace('FIELDS', fields)
    result = gh('api', 'graphql', '-f', f'query={query}', '-f', f'owner={owner}')
    if result.get('errors'):
        raise ValueError('GitHub candidate query failed')
    user = result['data']['user']
    found = {}
    featured = result['data'].get('featured')
    for pinned, nodes in ((True, user['pinnedItems']['nodes']),
                          (False, [featured] if featured else []),
                          (False, user['repositories']['nodes'])):
        for repo in nodes:
            name = repo['nameWithOwner']
            if repo['isPrivate'] or repo['isArchived'] or repo['isFork'] or not repo['defaultBranchRef']:
                continue
            if name.lower() == f'{owner}/{owner}'.lower() or name in found:
                continue
            found[name] = dict(repo, pinned=pinned)
    return list(found.values())[:MAX_PROJECTS]


def collect(owner):
    projects = []
    for repo in candidates(owner):
        name = repo['nameWithOwner']
        sha = repo['defaultBranchRef']['target']['oid']
        tree = gh('api', f'repos/{name}/git/trees/{sha}?recursive=1')
        # Never execute repository content. Read bounded documentation at an immutable SHA.
        paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob' and x.get('size', 0) < 200000]
        readmes = sorted(p for p in paths if p.lower() in ('readme.md', 'readme.rst', 'readme'))
        docs = sorted(p for p in paths if p.endswith(('.md', '.mdx')) and
                      re.search(r'(architecture|benchmark|design|reliability)', p, re.I))
        selected = (readmes[:1] + docs[:2])[:3]
        documents = []
        budget = MAX_DOCUMENT_CHARS
        for path in selected:
            data = gh('api', f'repos/{name}/contents/{quote(path, safe="/")}?ref={sha}')
            content = base64.b64decode(data['content']).decode('utf-8', errors='replace')
            text = content[:min(budget, 4000 if not documents else 1000)]
            budget -= len(text)
            documents.append({'path': path, 'text': text,
                              'url': f'https://github.com/{name}/blob/{sha}/{quote(path, safe="/")}'})
        if not documents:
            continue
        projects.append({'repo': name, 'url': repo['url'], 'pinned': repo['pinned'],
                         'pushed_at': repo['pushedAt'], 'commit': sha,
                         'description': repo['description'] or '',
                         'organization': repo.get('owner') if repo.get('owner', {}).get('__typename') == 'Organization' else None,
                         'documents': documents})
    if len(projects) < 3:
        raise ValueError('Need at least three public projects with documentation')
    return projects


def fingerprint(project, model):
    # A new commit without changed supplied evidence should not trigger an AI call.
    material = {'repo': project['repo'], 'description': project['description'],
                'documents': [(d['path'], d['text']) for d in project['documents']],
                'organization': project.get('organization'),
                'policy': POLICY.read_text(), 'model': model, 'copy_format': 'title-newline-v1',
                'layout': json.loads((ROOT / 'assets/profile/storyboard.json').read_text())['layouts']}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def validate_summary(item, project):
    if item['repo'] != project['repo']:
        raise ValueError('Summary repository mismatch')
    for key, limit in (('title', 26), ('summary', 120)):
        value = item[key]
        if not isinstance(value, str) or not 1 <= len(value) <= limit or value != value.strip():
            raise ValueError(f'Invalid {key} length for {project["repo"]}')
        if not value.isascii() or any(ord(c) < 32 for c in value) or BAD_COPY.search(value):
            raise ValueError(f'Unsupported showcase copy for {project["repo"]}')
    if item['title'].endswith('.') or any(c in item['title'] + item['summary'] for c in '<>[]`'):
        raise ValueError('Showcase copy must be plain text without markup')
    # Avoid uncontextualized benchmark claims in the tiny animation.
    if re.search(r'\b\d|%|\b(faster|fastest|best|only|guarantee\w*)\b', item['summary'], re.I):
        raise ValueError('Quantitative/comparative claims belong in linked evidence, not short copy')
    for key in ('relevance', 'substance', 'evidence_strength'):
        if type(item[key]) is not int or not 0 <= item[key] <= 5:
            raise ValueError('Invalid ranking score')
    docs = {d['path']: d for d in project['documents']}
    evidence = item['evidence']
    if not 1 <= len(evidence) <= 2:
        raise ValueError('Need one or two supporting excerpts')
    for ref in evidence:
        if ref['path'] not in docs or not 12 <= len(ref['quote']) <= 300:
            raise ValueError('Invalid evidence reference')
        if ref['quote'] not in docs[ref['path']]['text']:
            raise ValueError('Evidence quote not present in supplied source')
    template = json.loads((ROOT / 'assets/profile/storyboard.json').read_text())
    project_text_length(template, item)
    return item


def generate(changed, model, *, output=None, feedback=None):
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field, create_model

    class Evidence(BaseModel):
        path: str
        quote: str

    class Summary(BaseModel):
        title: str = Field(max_length=26)
        summary: str = Field(max_length=120)
        relevance: int = Field(ge=0, le=5)
        substance: int = Field(ge=0, le=5)
        evidence_strength: int = Field(ge=0, le=5)
        evidence: list[Evidence]

    # Required object keys enforce coverage and preserve exact repository identity.
    # An unconstrained list allowed omissions, duplicates, and renamed repos.
    ProjectSummaries = create_model('ProjectSummaries', **{
        p['repo']: (Summary, ...) for p in changed})

    class Batch(BaseModel):
        projects: ProjectSummaries

    if not os.environ.get('GOOGLE_API_KEY'):
        raise ValueError('GOOGLE_API_KEY is required for changed summaries. Set it in the environment or repository .env; never paste it into logs.')
    # One initial request and at most one targeted repair; no transport retries.
    payload = [{'repo': p['repo'], 'description': p['description'],
                'organization': p.get('organization'),
                'documents': [{'path': d['path'], 'text': d['text']} for d in p['documents']]} for p in changed]
    if feedback:
        payload = {'repositories': payload, 'corrections': feedback,
                   'instruction': 'Correct these rejected summaries. Aim for 55-80 summary '
                                  'characters, shorter if needed to fit four mobile lines '
                                  'including the title. Preserve exact source evidence.'}
    with genai.Client(api_key=os.environ['GOOGLE_API_KEY'],
                      http_options=types.HttpOptions(timeout=120000,
                          retry_options=types.HttpRetryOptions(attempts=1))) as client:
        response = client.models.generate_content(model=model, contents=json.dumps(payload),
            config=types.GenerateContentConfig(system_instruction=POLICY.read_text(),
                response_mime_type='application/json', response_schema=Batch,
                max_output_tokens=8192, temperature=0.2))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        name = 'model-repair.json' if feedback else 'model-response.json'
        (output / name).write_text(response.model_dump_json(indent=2) + '\n')
    candidates = response.candidates or []
    reason = candidates[0].finish_reason if candidates else None
    if reason != types.FinishReason.STOP:
        raise ValueError(f'Model did not complete summaries (finish_reason={reason})')
    if not response.text:
        raise ValueError('Model returned no summary text')
    decoded = json.loads(response.text)
    actual = decoded.get('projects')
    expected = {p['repo'] for p in changed}
    if not isinstance(actual, dict):
        raise ValueError('Model projects must be an object keyed by repository')
    if set(actual) != expected:
        raise ValueError(f'Model repository coverage: missing={sorted(expected - set(actual))}; '
                         f'unexpected={sorted(set(actual) - expected)}')
    valid, invalid, errors = {}, [], {}
    for p in changed:
        name = p['repo']
        try:
            item = Summary.model_validate(actual[name]).model_dump() | {'repo': name}
            validate_summary(item, p)
            valid[name] = item
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append(p)
            errors[name] = {'rejected': actual[name], 'error': str(exc)}
    if invalid:
        if feedback is not None:
            for p in invalid:
                name = p['repo']
                print(f'Excluding invalid candidate after one repair: {name}')
                valid[name] = {'repo': name, 'rejected': errors[name]['error']}
        else:
            print(f'Repairing {len(invalid)} invalid summaries; retaining {len(valid)} valid summaries')
            repaired = generate(invalid, model, output=output, feedback=errors)
            valid.update({item['repo']: item for item in repaired})
    return [valid[p['repo']] for p in changed]


def summarize(projects, cache, model, generate_fn=generate):
    old = cache.get('entries', {})
    changed = [p for p in projects if old.get(p['repo'], {}).get('fingerprint') != fingerprint(p, model)]
    new = generate_fn(changed, model) if changed else []
    if len(new) != len(changed) or {i['repo'] for i in new} != {p['repo'] for p in changed}:
        expected = {p['repo'] for p in changed}
        names = [i['repo'] for i in new]
        actual = set(names)
        duplicates = sorted(name for name in actual if names.count(name) > 1)
        raise ValueError(f'Model repository coverage: expected={len(changed)}, returned={len(new)}; '
                         f'missing={sorted(expected - actual)}; '
                         f'unexpected={sorted(actual - expected)}; duplicates={duplicates}')
    returned = {i['repo']: i for i in new}
    entries = {}
    for p in projects:
        name = p['repo']
        if name not in returned:
            entries[name] = old[name]
            if 'summary' in old[name]:
                validate_summary(old[name]['summary'], p)
            continue
        item = returned[name]
        if 'rejected' in item:
            # Cache the exclusion so unchanged evidence does not spend again.
            entries[name] = {'fingerprint': fingerprint(p, model), 'rejection': item['rejected']}
            continue
        validate_summary(item, p)
        entries[name] = {'fingerprint': fingerprint(p, model), 'summary': item}
    return {'version': 1, 'entries': entries}, len(changed)


def choose(projects, cache, count=4):
    def rank(p):
        s = cache['entries'][p['repo']]['summary']
        return (p['repo'] == FEATURED_REPO,
                s['relevance'] * 2 + s['substance'] * 3 + s['evidence_strength'] * 3,
                p['pinned'], p['pushed_at'], p['repo'])
    selected, families = [], set()
    eligible = [p for p in projects if 'summary' in cache['entries'].get(p['repo'], {})]
    for p in sorted(eligible, key=rank, reverse=True):
        s = cache['entries'][p['repo']]['summary']
        if min(s['relevance'], s['substance'], s['evidence_strength']) < 3:
            continue
        name = p['repo'].split('/')[-1]
        family = 'paq' if name in ('paq', 'paqpy', 'paqjs', 'bazel-paq') else p['repo']
        if family in families:
            continue
        selected.append((p, s))
        families.add(family)
        if len(selected) == count:
            return selected
    raise ValueError(f'Need {count} distinct projects with supported, relevant summaries')


def project_text_length(template, item):
    """Validate actual rendered fit and return the longest wrapped text length."""
    from PIL import ImageFont
    from scripts.render_profile import wrap_message
    text = f'{item["title"]}.\n{item["summary"]}'
    longest = 0
    for name, layout in template['layouts'].items():
        cfg = template | layout
        font = ImageFont.truetype(str(ROOT / 'assets/profile/fonts/CourierPrime-Regular.ttf'), cfg['font_size'])
        wrapped = wrap_message(text, cfg, font)
        rows = len(wrapped.splitlines())
        if rows > 4:
            raise ValueError(f'Project copy exceeds four terminal lines in {name}; shorten title/summary')
        if any(font.getlength(line) > cfg['size'][0] - 2 * cfg['text_origin'][0] - 24
               for line in wrapped.splitlines()):
            raise ValueError(f'Project copy exceeds horizontal space in {name}')
        bottom = cfg['text_origin'][1] + (len(cfg['persistent_lines']) + 2 + rows) * cfg['line_height']
        if bottom > cfg['size'][1] - 20:
            raise ValueError(f'Project copy clips the terminal viewport in {name}')
        longest = max(longest, len(wrapped))
    return longest


def make_storyboard(template, selected):
    scenes = [dict(s) for s in template['scenes'][:2]]
    start = 18
    for project, item in selected:
        text = f'{item["title"]}.\n{item["summary"]}'
        longest = project_text_length(template, item)
        scenes.append({'start': start, 'text': text})
        start += math.ceil(longest / template['characters_per_second'] + 0.35 + 5)
    scenes += [{'start': start, 'text': 'Follow gregl83.'}, {'start': start + 8, 'text': 'Knock, knock.'}]
    return template | {'scenes': scenes, 'shutdown_start': start + 14, 'duration': start + 20}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', default='gregl83')
    parser.add_argument('--model', default=os.environ.get('SHOWCASE_MODEL') or MODEL)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/showcase')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--sources', type=Path, help='Reuse a previously collected public snapshot')
    parser.add_argument('--project-count', type=int, choices=(3, 4), default=4)
    parser.add_argument('--collect-only', action='store_true', help='Read GitHub only; no AI or rendering')
    args = parser.parse_args()
    projects = json.loads(args.sources.read_text()) if args.sources else collect(args.owner)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'sources.json').write_text(json.dumps(projects, indent=2) + '\n')
    if args.collect_only:
        print(f'Collected {len(projects)} public projects; no model calls')
        return
    if len(projects) < args.project_count:
        raise ValueError('Not enough documented candidates for the requested project count')
    cache = json.loads(args.cache.read_text()) if args.cache else {}
    cache, changed = summarize(projects, cache, args.model,
                               lambda changed, model: generate(changed, model, output=args.output))
    # Preserve valid paid work locally even if selection/rendering later fails.
    (args.output / 'showcase-cache.json').write_text(json.dumps(cache, indent=2) + '\n')
    selected = choose(projects, cache, args.project_count)
    template = json.loads((ROOT / 'assets/profile/storyboard.json').read_text())
    storyboard = make_storyboard(template, selected)
    manifest = {'generated_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
                'changed_projects': changed, 'projects': []}
    notes = ['# Project showcase', '', 'Selected public engineering projects.', '']
    for p, s in selected:
        docs = {d['path']: d for d in p['documents']}
        sources = [{'url': docs[e['path']]['url'], 'quote': e['quote']} for e in s['evidence']]
        manifest['projects'].append({'repo': p['repo'], 'url': p['url'], 'commit': p['commit'],
                                    'title': s['title'], 'summary': s['summary'], 'evidence': sources})
        notes += [f'## [{s["title"]}]({p["url"]})', '', s['summary'], '']
        notes += [f'- [Supporting source]({e["url"]}): {e["quote"]}' for e in sources]
        notes += ['']
    for name, data in [('showcase-cache.json', cache), ('showcase.json', manifest), ('storyboard.json', storyboard)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    (args.output / 'release-notes.md').write_text('\n'.join(notes))
    print(f'Selected {[p["repo"] for p, _ in selected]}; {changed} changed projects; AI used: {bool(changed)} (at most two requests)')


if __name__ == '__main__':
    main()
