#!/usr/bin/env python3
"""Render the profile's scripted terminal introduction without a desktop or API."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def render_frame(config, frame, font):
    width, height = config['size']
    time = frame / config['fps']
    scene = config['scenes'][0]
    for candidate in config['scenes']:
        if time >= candidate['start']:
            scene = candidate
    elapsed = time - scene['start']
    count = max(0, int((elapsed - 0.35) * config['characters_per_second']))
    message = wrap_message(scene['text'], config, font)
    visible = message[:count] if time >= config.get('intro_pause', 0) else ''
    base = Image.new('RGB', (width, height), '#010502')
    ink = Image.new('RGB', base.size)
    draw = ImageDraw.Draw(ink)
    green = '#a4dfc5'
    # Persistent output uses the same terminal grid, not a title bar or panel.
    x, top, step = config['text_origin'][0], config['text_origin'][1], config['line_height']
    for index, line in enumerate(config['persistent_lines']):
        draw.text((x, top + index * step), line, font=font, fill='#82b69d')
    y = top + (len(config['persistent_lines']) + 2) * step
    lines = visible.split('\n')
    for index, line in enumerate(lines):
        draw.text((x, y + index * step), line, font=font, fill=green)
    # A softened core plus two bloom radii approximates luminous CRT phosphor.
    core = ink.filter(ImageFilter.GaussianBlur(0.45))
    red, green_channel, blue = ink.split()
    phosphor = Image.merge('RGB', (red.point(lambda v: int(v * 0.55)),
                                  green_channel, blue.point(lambda v: int(v * 0.65))))
    near = phosphor.filter(ImageFilter.GaussianBlur(1.6)).point(lambda v: int(v * 0.36))
    halo = phosphor.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.20))
    result = ImageChops.add(base, ImageChops.add(core, ImageChops.add(near, halo)))
    # Static, very faint scanlines: texture without animated noise or flickering.
    scan = Image.new('RGB', base.size, 'white')
    raster = ImageDraw.Draw(scan)
    for row in range(0, height, 3):
        raster.line((0, row, width, row), fill=(238, 248, 238))
    result = ImageChops.multiply(result, scan)
    if time >= config['shutdown_start']:
        return shutdown_frame(result, time - config['shutdown_start'])
    return result


def shutdown_frame(screen, elapsed):
    """Collapse the whole screen to a phosphor line, then a dot and black."""
    width, height = screen.size
    black = Image.new('RGB', screen.size)
    if elapsed < 0.25:
        remaining = 1 - elapsed / 0.25
        compressed_height = max(2, round(height * remaining ** 2))
        compressed = screen.resize((width, compressed_height), Image.Resampling.LANCZOS)
        black.paste(compressed, (0, (height - compressed_height) // 2))
        return black
    if elapsed < 0.65:
        remaining = 1 - (elapsed - 0.25) / 0.4
        half = max(1, round(width * 0.46 * remaining ** 2))
        light = Image.new('RGB', screen.size)
        draw = ImageDraw.Draw(light)
        draw.line((width // 2 - half, height // 2, width // 2 + half, height // 2),
                  fill='#c7eed8', width=2)
        return ImageChops.add(light, light.filter(ImageFilter.GaussianBlur(4)))
    if elapsed < 1:
        intensity = int(180 * (1 - (elapsed - 0.65) / 0.35))
        dot = ImageDraw.Draw(black)
        dot.ellipse((width // 2 - 2, height // 2 - 2, width // 2 + 2, height // 2 + 2),
                    fill=(intensity // 2, intensity, intensity * 3 // 4))
        return black.filter(ImageFilter.GaussianBlur(0.8))
    return black



def wrap_message(text, config, font):
    """Lay out the complete message once so words never jump while typing."""
    available = config['size'][0] - 2 * config['text_origin'][0] - 24
    columns = int(available // font.getlength('M'))
    return '\n'.join(
        '\n'.join(textwrap.wrap(line, width=columns, break_long_words=True,
                                break_on_hyphens=False))
        for line in text.split('\n'))


def render_variant(config, output, suffix):
    font = ImageFont.truetype(str(ROOT / 'assets/profile/fonts/CourierPrime-Regular.ttf'), config['font_size'])
    output.mkdir(parents=True, exist_ok=True)
    lines = list(config['persistent_lines'])
    for scene in config['scenes']:
        lines.extend(wrap_message(scene['text'], config, font).split('\n'))
    for line in lines:
        if font.getlength(line) > config['size'][0] - 2 * config['text_origin'][0] - 24:
            raise ValueError(f'Line is too wide: {line}')
    with tempfile.TemporaryDirectory(prefix='profile-frames-') as directory:
        frames = Path(directory)
        count = config['fps'] * config['duration']
        for index in range(count):
            frame = render_frame(config, index, font)
            frame.save(frames / f'{index:05d}.png')
        render_frame(config, int((config['shutdown_start'] - 1) * config['fps']), font).save(output / f'poster{suffix}.png')
        run = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(config['fps']), '-i', str(frames / '%05d.png')]
        subprocess.run(run + ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / f'intro{suffix}.mp4')], check=True)
        subprocess.run(run + ['-filter_complex', '[0:v]split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer', '-loop', '0', str(output / f'intro{suffix}.gif')], check=True)
        # A small contact sheet makes narrow-width and scene review repeatable.
        sheet = Image.new('RGB', (960, len(config['scenes']) * 270), '#121212')
        scene_ends = [scene['start'] for scene in config['scenes'][1:]] + [config['shutdown_start']]
        for row, end in enumerate(scene_ends):
            time = end - 1
            preview = render_frame(config, int(time * config['fps']), font)
            for offset, width in ((0, 280), (280, 320), (600, 360)):
                thumb = preview.resize((width, round(width * config['size'][1] / config['size'][0])), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (offset, row * 270))
        sheet.save(output / f'review{suffix}.png')
    for name in (f'intro{suffix}.gif', f'intro{suffix}.mp4', f'poster{suffix}.png'):
        path = output / name
        print(f'{path}: {path.stat().st_size:,} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/profile')
    parser.add_argument('--storyboard', type=Path, default=ROOT / 'assets/profile/storyboard.json')
    args = parser.parse_args()
    config = json.loads(args.storyboard.read_text())
    for variant, layout in config['layouts'].items():
        render_variant(config | layout, args.output, '' if variant == 'desktop' else '-mobile')


if __name__ == '__main__':
    main()
