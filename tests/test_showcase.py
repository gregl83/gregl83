import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import showcase as s
from scripts import publish_showcase as release


def project(name='user/engine', text='Rust workers isolate evaluators in WebAssembly sandboxes.'):
    return {'repo': name, 'description': 'An evaluation engine', 'url': f'https://github.com/{name}',
            'commit': 'a' * 40, 'pinned': True, 'pushed_at': '2026-11-01T00:00:00Z',
            'documents': [{'path': 'README.md', 'text': text, 'url': f'https://github.com/{name}/blob/main/README.md'}]}


def summary(p):
    return {'repo': p['repo'], 'title': 'Engine', 'summary': 'Isolate evaluators with WebAssembly sandboxes.',
            'relevance': 5, 'substance': 5, 'evidence_strength': 5,
            'evidence': [{'path': 'README.md', 'quote': p['documents'][0]['text']}]}


class ShowcaseTests(unittest.TestCase):
    def test_cache_and_source_change(self):
        p = project()
        calls = []
        def model(projects, name):
            calls.append(projects)
            return [summary(x) for x in projects]
        cache, changed = s.summarize([p], {}, 'model', model)
        self.assertEqual(changed, 1)
        p['commit'] = 'b' * 40
        cache, changed = s.summarize([p], cache, 'model', model)
        self.assertEqual(changed, 0)
        self.assertEqual(len(calls), 1)
        p['documents'][0]['text'] += ' Durable runs are persisted.'
        s.summarize([p], cache, 'model', model)
        self.assertEqual(len(calls), 2)

    def test_invalid_copy_and_evidence(self):
        p = project()
        for bad in ('Seeking a job at OpenAI.', 'Hire me for systems engineering.', 'Processes 1000 requests per second.', 'x' * 121, 'Claims\nwith newlines'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                s.validate_summary(summary(p) | {'summary': bad}, p)
        item = summary(p)
        item['evidence'][0]['quote'] = 'This claim is not in the source.'
        with self.assertRaises(ValueError):
            s.validate_summary(item, p)

    def test_actual_fit_accepts_longer_copy_and_rejects_overflow(self):
        text = 'Streamline HTTP cookie validation assertions within SuperTest workflows'
        p = project(text=text)
        item = summary(p) | {'title': 'Expect Cookies', 'summary': text}
        self.assertEqual(len(text), 71)
        self.assertEqual(s.validate_summary(item, p), item)
        long_text = 'instrumentation ' * 7
        item['summary'] = long_text.strip()
        with self.assertRaisesRegex(ValueError, 'four terminal lines'):
            s.validate_summary(item, p)

    def test_technology_names_can_contain_digits(self):
        p = project(text='BLAKE3 fingerprints identify changed build artifacts.')
        item = summary(p) | {'summary': 'BLAKE3 fingerprints identify changed build artifacts.'}
        self.assertEqual(s.validate_summary(item, p), item)

    def test_missing_or_duplicate_model_results_fail(self):
        p = project()
        for values in ([], [summary(p), summary(p)], [summary(p) | {'repo': 'other/repo'}]):
            with self.assertRaises(ValueError):
                s.summarize([p], {}, 'model', lambda *_: values)

    def test_generation_contract_through_real_sdk(self):
        import httpx
        from google import genai
        projects = [project(f'org/project-{i}') for i in range(12)]
        expected = {p['repo']: {k: v for k, v in summary(p).items() if k != 'repo'}
                    for p in projects}
        real_client = genai.Client
        requests = []
        body = {'projects': expected}
        finish_reason = 'STOP'

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{
                'finishReason': finish_reason,
                'content': {'role': 'model', 'parts': [{'text': json.dumps(body)}]}}]})

        def client(**kwargs):
            kwargs['http_options'].client_args = {'transport': httpx.MockTransport(respond)}
            return real_client(**kwargs)

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'offline-test-key'}), \
             patch.object(genai, 'Client', side_effect=client):
            results = s.generate(projects, 'test-model')
            self.assertEqual(results, [summary(p) for p in projects])
            schema = requests[0]['generationConfig']['responseSchema']
            required = schema['properties']['projects']
            self.assertEqual(set(required['required']), set(expected))
            self.assertEqual(set(required['properties']), set(expected))
            self.assertEqual(len(requests), 1)

            body = {'projects': dict(list(expected.items())[1:])}
            with self.assertRaisesRegex(ValueError, 'missing=.*org/project-0'):
                s.generate(projects, 'test-model')
            body = {'projects': expected | {'wrong/repo': next(iter(expected.values()))}}
            with self.assertRaisesRegex(ValueError, 'unexpected=.*wrong/repo'):
                s.generate(projects, 'test-model')
            body = {'projects': expected}
            finish_reason = 'MAX_TOKENS'
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                s.generate(projects, 'test-model')
            self.assertEqual(len(requests), 4)  # No automatic paid retries.

    def test_overlong_summary_is_repaired_once_and_responses_saved(self):
        import httpx
        from google import genai
        projects = [project('org/good'), project('org/long')]
        original = {p['repo']: {k: v for k, v in summary(p).items() if k != 'repo'}
                    for p in projects}
        original['org/long']['summary'] = 'x' * 121
        corrected = {k: v for k, v in summary(projects[1]).items() if k != 'repo'}
        requests = []
        real_client = genai.Client
        repair_succeeds = True

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            body = original if len(requests) % 2 else {
                'org/long': corrected if repair_succeeds else original['org/long']}
            return httpx.Response(200, json={'candidates': [{
                'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': [
                    {'text': json.dumps({'projects': body})}]}}]})

        def client(**kwargs):
            kwargs['http_options'].client_args = {'transport': httpx.MockTransport(respond)}
            return real_client(**kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'GOOGLE_API_KEY': 'offline-test-key'}), \
             patch.object(genai, 'Client', side_effect=client):
            output = Path(directory)
            self.assertEqual(s.generate(projects, 'test-model', output=output),
                             [summary(p) for p in projects])
            self.assertTrue((output / 'model-response.json').exists())
            self.assertTrue((output / 'model-repair.json').exists())
            required = requests[1]['generationConfig']['responseSchema']['properties']['projects']
            self.assertEqual(required['required'], ['org/long'])
            repair_succeeds = False
            results = s.generate(projects, 'test-model', output=output)
            self.assertEqual(results[0], summary(projects[0]))
            self.assertIn('rejected', results[1])
            self.assertEqual(len(requests), 4)

    def test_rejected_candidate_is_cached_and_never_selected(self):
        projects = [project(f'org/{name}') for name in ('a', 'b', 'c', 'bad')]
        results = [summary(p) for p in projects[:3]] + [
            {'repo': 'org/bad', 'rejected': 'Summary too long'}]
        cache, _ = s.summarize(projects, {}, 'model', lambda *_: results)
        self.assertEqual(len(s.choose(projects, cache, count=3)), 3)
        with self.assertRaisesRegex(ValueError, 'Need 4 distinct'):
            s.choose(projects, cache, count=4)
        with patch.object(s, 'generate') as model:
            reused, changed = s.summarize(projects, cache, 'model', model)
            model.assert_not_called()
        self.assertEqual(changed, 0)
        self.assertEqual(cache, reused)

    def test_coverage_error_identifies_missing_unexpected_and_duplicates(self):
        p = project()
        wrong = summary(p) | {'repo': 'other/repo'}
        with self.assertRaisesRegex(ValueError,
                "missing=.*user/engine.*unexpected=.*other/repo.*duplicates=.*other/repo"):
            s.summarize([p], {}, 'model', lambda *_: [wrong, wrong])

    def test_discovery_includes_org_pins_filters_private_forks_and_archives(self):
        def repo(name, **kw):
            return dict(nameWithOwner=name, isPrivate=False, isFork=False, isArchived=False,
                        defaultBranchRef={'target': {'oid': 'abc'}}, **kw)
        private = repo('org/private'); private['isPrivate'] = True
        fork = repo('user/fork'); fork['isFork'] = True
        archived = repo('user/old'); archived['isArchived'] = True
        data = {'data': {'featured': repo(s.FEATURED_REPO), 'user': {'pinnedItems': {'nodes': [repo('org/pin'), private]},
                'repositories': {'nodes': [repo('user/user'), repo('org/pin'), fork, archived, repo('user/new')]}}}}
        with patch.object(s, 'gh', return_value=data):
            found = s.candidates('user')
        self.assertEqual([p['nameWithOwner'] for p in found], ['org/pin', s.FEATURED_REPO, 'user/new'])
        self.assertTrue(found[0]['pinned'])

    def test_family_diversity(self):
        projects = [project(f'user/{name}') for name in ('paq', 'paqpy', 'engine')]
        cache = {'entries': {p['repo']: {'summary': summary(p)} for p in projects}}
        chosen = s.choose(projects, cache, count=2)
        self.assertEqual(len(chosen), 2)
        self.assertIn('user/engine', [p['repo'] for p, _ in chosen])

    def test_ranking_precedes_selection_and_beats_pin_recency(self):
        projects = [project(f'user/{name}') for name in ('weak', 'strong', 'middle', 'third')]
        projects[0]['pushed_at'] = '2099-01-01'
        entries = {}
        for p, score in zip(projects, (3, 5, 4, 4)):
            entries[p['repo']] = {'summary': summary(p) | {
                'relevance': score, 'substance': score, 'evidence_strength': score}}
        projects[1]['pinned'] = False
        selected = s.choose(projects, {'entries': entries}, count=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(s.choose(projects, {'entries': entries}, count=4)), 4)
        self.assertEqual(selected[0][0]['repo'], 'user/strong')
        self.assertNotIn('user/weak', [p['repo'] for p, _ in selected])
        entries['user/third']['summary']['evidence_strength'] = 2
        entries['user/weak']['summary']['substance'] = 2
        with self.assertRaises(ValueError):
            s.choose(projects, {'entries': entries})

    def test_featured_project_first_with_four_by_default_and_no_quality_bypass(self):
        projects = [project(f'user/{name}') for name in ('a', 'b', 'c', 'd')]
        featured = project(s.FEATURED_REPO)
        projects.append(featured)
        entries = {p['repo']: {'summary': summary(p)} for p in projects}
        entries[s.FEATURED_REPO]['summary'].update(
            relevance=3, substance=3, evidence_strength=3)
        selected = s.choose(projects, {'entries': entries})
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0][0]['repo'], s.FEATURED_REPO)
        entries[s.FEATURED_REPO]['summary']['evidence_strength'] = 2
        selected = s.choose(projects, {'entries': entries})
        self.assertNotIn(s.FEATURED_REPO, [p['repo'] for p, _ in selected])

    def test_project_title_occupies_its_own_line_in_both_layouts(self):
        from PIL import ImageFont
        from scripts.render_profile import wrap_message
        template = json.loads((s.ROOT / 'assets/profile/storyboard.json').read_text())
        p = project()
        board = s.make_storyboard(template, [(p, summary(p))])
        text = board['scenes'][2]['text']
        self.assertEqual(text, 'Engine.\nIsolate evaluators with WebAssembly sandboxes.')
        self.assertNotIn('gregl83', template['persistent_lines'])
        for layout in template['layouts'].values():
            cfg = template | layout
            font = ImageFont.truetype(str(s.ROOT / 'assets/profile/fonts/CourierPrime-Regular.ttf'),
                                      cfg['font_size'])
            wrapped = wrap_message(text, cfg, font)
            self.assertEqual(wrapped.splitlines()[0], 'Engine.')
            self.assertTrue(wrapped.splitlines()[1].startswith('Isolate'))
            self.assertNotIn('\n\n', wrapped)

    def test_storyboard_fit_and_reading_time(self):
        template = json.loads((s.ROOT / 'assets/profile/storyboard.json').read_text())
        original = copy.deepcopy(template)
        p = project()
        pairs = [(project(f'user/{name}'), summary(project(f'user/{name}'))) for name in ('first', 'second', 'third')]
        board = s.make_storyboard(template, pairs)
        self.assertEqual(template, original)
        for index in (2, 3, 4):
            scene = board['scenes'][index]
            following = board['scenes'][index + 1]['start']
            self.assertGreaterEqual(following - scene['start'], len(scene['text']) / board['characters_per_second'] + 5)
        self.assertEqual(board['duration'] - board['shutdown_start'], 6)
        self.assertEqual(board['intro_pause'], 3)

    def test_restore_failure_does_not_silently_create_empty_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cache.json'
            with patch.object(release, 'command', side_effect=RuntimeError('network')):
                with self.assertRaises(RuntimeError):
                    release.restore('user/repo', path)
            self.assertFalse(path.exists())

    def test_publish_upload_failure_never_advances_latest(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def fail_upload(*args):
                calls.append(args)
                if 'upload' in args:
                    raise RuntimeError('upload failed')
                return ''
            with patch('sys.argv', ['publish', 'publish', '--directory', d,
                                    '--tag', 'showcase-test', '--commit', 'abc']), \
                 patch.object(release, 'validate_assets', return_value=['intro.gif']), \
                 patch.object(release, 'command', side_effect=fail_upload):
                with self.assertRaises(RuntimeError):
                    release.main()
            self.assertTrue(any('--draft' in args for args in calls))
            self.assertFalse(any('edit' in args for args in calls))

    def test_missing_assets_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            template = json.loads((s.ROOT / 'assets/profile/storyboard.json').read_text())
            (path / 'storyboard.json').write_text(json.dumps(template))
            with patch.object(release, 'command') as cmd, self.assertRaises(ValueError):
                release.validate_assets(path)
            cmd.assert_not_called()


if __name__ == '__main__':
    unittest.main()
