#!/usr/bin/env python3
"""Restore a public showcase cache or publish validated media via a draft release."""
import argparse
import json
from pathlib import Path
import subprocess


def command(*args):
    return subprocess.run(list(args), check=True, capture_output=True, text=True, timeout=120).stdout


def restore(repo, destination):
    releases = json.loads(command('gh', 'release', 'list', '--repo', repo, '--limit', '100',
                                  '--json', 'tagName,isDraft,isPrerelease'))
    previous = next((r for r in releases if r['tagName'].startswith('showcase-')
                     and not r['isDraft'] and not r['isPrerelease']), None)
    if previous is None:
        destination.write_text('{"version": 1, "entries": {}}\n')
        return
    # Download errors must fail, not silently trigger expensive re-generation.
    command('gh', 'release', 'download', previous['tagName'], '--repo', repo,
            '--pattern', 'showcase-cache.json', '--clobber', '--output', str(destination))
    cache = json.loads(destination.read_text())
    if cache.get('version') != 1 or not isinstance(cache.get('entries'), dict):
        raise ValueError('Invalid previous showcase cache')


def validate_assets(directory):
    from PIL import Image
    board = json.loads((directory / 'storyboard.json').read_text())
    names = ['showcase-cache.json', 'showcase.json', 'storyboard.json', 'release-notes.md']
    for layout, suffix in (('desktop', ''), ('mobile', '-mobile')):
        expected_size = tuple(board['layouts'][layout]['size'])
        for stem, extension in (('intro', 'gif'), ('intro', 'mp4'), ('poster', 'png')):
            name = f'{stem}{suffix}.{extension}'
            path = directory / name
            if not path.is_file() or not 0 < path.stat().st_size < 10_000_000:
                raise ValueError(f'Missing or oversized asset: {name}')
            names.append(name)
        with Image.open(directory / f'intro{suffix}.gif') as gif:
            if gif.size != expected_size or gif.n_frames < 2 or gif.info.get('loop') != 0:
                raise ValueError('Invalid GIF dimensions, frames, or looping')
            duration = 0
            for index in range(gif.n_frames):
                gif.seek(index)
                gif.load()
                duration += gif.info.get('duration', 0)
            if abs(duration - board['duration'] * 1000) > 100:
                raise ValueError('Incorrect GIF duration')
        with Image.open(directory / f'poster{suffix}.png') as poster:
            if poster.size != expected_size:
                raise ValueError('Incorrect poster dimensions')
            poster.verify()
        video = directory / f'intro{suffix}.mp4'
        info = json.loads(command('ffprobe', '-v', 'error', '-show_entries',
                                 'stream=width,height:format=duration', '-of', 'json', str(video)))
        stream = info['streams'][0]
        if (stream['width'], stream['height']) != expected_size or abs(float(info['format']['duration']) - board['duration']) > .1:
            raise ValueError('Incorrect video dimensions or duration')
        command('ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-')
    for name in names:
        if not (directory / name).is_file():
            raise ValueError(f'Missing release asset: {name}')
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('restore', 'validate', 'publish'))
    p.add_argument('--repo', default='gregl83/gregl83')
    p.add_argument('--directory', type=Path, default=Path('out/showcase'))
    p.add_argument('--cache', type=Path, default=Path('out/previous-showcase.json'))
    p.add_argument('--tag')
    p.add_argument('--commit')
    args = p.parse_args()
    if args.action == 'restore':
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        restore(args.repo, args.cache)
        return
    names = validate_assets(args.directory)
    if args.action == 'validate':
        print('Both media variants and release assets validated')
        return
    if not args.tag or not args.tag.startswith('showcase-') or not args.commit:
        p.error('Publishing requires a showcase tag and target commit')
    # All uploads happen on an unpublished draft. Only a complete draft becomes latest.
    command('gh', 'release', 'create', args.tag, '--repo', args.repo, '--target', args.commit,
            '--draft', '--title', 'Project showcase', '--notes-file', str(args.directory / 'release-notes.md'))
    command('gh', 'release', 'upload', args.tag, '--repo', args.repo,
            *(str(args.directory / name) for name in names))
    uploaded = json.loads(command('gh', 'release', 'view', args.tag, '--repo', args.repo, '--json', 'assets'))
    remote = {a['name']: a for a in uploaded['assets']}
    if set(remote) != set(names) or any(remote[n]['size'] != (args.directory / n).stat().st_size for n in names):
        raise ValueError('Draft uploads do not match local files; leaving previous release current')
    command('gh', 'release', 'edit', args.tag, '--repo', args.repo, '--draft=false', '--latest')


if __name__ == '__main__':
    main()
