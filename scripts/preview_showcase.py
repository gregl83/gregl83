#!/usr/bin/env python3
"""Build a local preview in ignored out/. This command never publishes or edits Git."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def output_directory(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT / 'out') or path == ROOT / 'out':
        raise argparse.ArgumentTypeError('Preview output must be a subdirectory of this repository\'s ignored out/')
    return path


def write_preview(output, example=False):
    title = 'Checked-in example' if example else 'Generated project showcase'
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{max-width:960px;margin:24px auto;padding:0 16px;background:#101512;color:#d9e8dd;font:16px/1.6 system-ui}}img,video{{display:block;width:100%;max-width:720px;height:auto}}a{{color:#83f5a4}}</style>
<h1>{title}</h1><p>Mobile framing at 600px or narrower. Desktop framing above 600px.</p>
<picture><source media="(max-width:600px)" srcset="intro-mobile.gif"><img src="intro.gif" alt="gregl83 — Software Systems Engineer. Applied AI and Data Systems." width="720"></picture>
<p><a href="showcase.json">Project sources</a> · <a href="release-notes.md">Release notes</a></p>
<h2>Desktop video</h2><video controls playsinline preload="metadata" poster="poster.png"><source src="intro.mp4" type="video/mp4"></video>
<h2>Mobile video</h2><video controls playsinline preload="metadata" poster="poster-mobile.png"><source src="intro-mobile.mp4" type="video/mp4"></video></html>
'''
    if example:
        html = html.replace('<p><a href="showcase.json">Project sources</a> · <a href="release-notes.md">Release notes</a></p>',
                            '<p>Manually reviewed examples; no model call.</p>')
    (output / 'preview.html').write_text(html)
    readme = (ROOT / 'README.md').read_text()
    base = 'https://github.com/gregl83/gregl83/releases/latest'
    readme = readme.replace(base + '/download/', '')
    readme = readme.replace(base, 'showcase-example.md' if example else 'release-notes.md')
    (output / 'README.md').write_text(readme)
    if example:
        shutil.copy2(ROOT / 'assets/profile/showcase-example.md', output / 'showcase-example.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=output_directory, default=ROOT / 'out/local-preview')
    p.add_argument('--example', action='store_true', help='Render the checked-in example; no network or API key')
    p.add_argument('--project-count', choices=(3, 4), type=int, default=3)
    args = p.parse_args()
    for tool in ('ffmpeg', 'ffprobe') + (() if args.example else ('gh',)):
        if not shutil.which(tool):
            p.error(f'{tool} is required')
    args.output.mkdir(parents=True, exist_ok=True)
    def run(*arguments):
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
    if args.example:
        board = ROOT / 'assets/profile/storyboard.json'
    else:
        cache = args.output / 'showcase-cache.json'
        if not cache.exists():
            run('-m', 'scripts.publish_showcase', 'restore', '--cache', str(cache))
        run('-m', 'scripts.showcase', '--cache', str(cache), '--output', str(args.output),
            '--project-count', str(args.project_count))
        board = args.output / 'storyboard.json'
    run('scripts/render_profile.py', '--storyboard', str(board), '--output', str(args.output))
    if not args.example:
        run('-m', 'scripts.publish_showcase', 'validate', '--directory', str(args.output))
    write_preview(args.output, args.example)
    print(f'Local preview: {args.output / "preview.html"}')
    print('No release, Git commit, push, or tracked-file update performed.')


if __name__ == '__main__':
    main()
