# Profile animation

Render the checked-in example with Python/Pillow and FFmpeg:

```sh
uv run --no-project --with pillow==12.0.0 python scripts/render_profile.py
```

Open `preview.html` in a browser. At viewport widths up to 600px it selects the
mobile GIF; above that it selects the desktop GIF. Videos have playback controls.
The checked-in storyboard contains manually reviewed, source-backed showcase
examples; provenance is in `showcase-example.json`. These are not presented as
live model output. The static introduction, CRT styling, pauses, and ending are
preserved. Current example duration: 73 seconds.

For the scheduled content-generation and release pipeline, see
[docs/showcase.md](../../docs/showcase.md). It supplies a generated storyboard:

```sh
python scripts/render_profile.py --storyboard out/showcase/storyboard.json --output out/showcase
```

The renderer uses fixed frame times and the bundled OFL-licensed Courier Prime
font. It makes no network/model calls. Desktop: 960x480, 26px type; mobile:
480x360, 22px type. Shared timeline, 12 fps, continuous GIF loop. Exact output
bytes require the same Python, Pillow, and FFmpeg versions.

Persistent role and specialty text occupy two ordinary terminal lines. Animated messages start
after two blank lines. Each project title ends with a period on its own line;
the description starts on the next line. Messages wrap at word boundaries using available width, with
a small right-side text margin. Layout is determined before typing to avoid word
jumps. Text uses pale green cores, green phosphor bloom, and faint scanlines.

Each loop opens with a three-second pause showing only the static introduction. After the project scenes,
“Follow gregl83.” and “Knock, knock.” lead into a one-second CRT collapse and a
five-second dark hold. Posters are captured before shutdown. The script exports
GIFs, MP4s, posters, and scene review sheets for both layouts.

The root README always uses latest-release assets. This directory’s `README.md`
mirrors it with local asset references for Markdown previews. Freshly generated
previews also include README.md alongside their media. No activation step or
pipeline edits to the root README are needed. GitHub/IDE settings can suppress GIF animation; the
first frame and descriptive alt text retain the static introduction. Browser preview is tested; the IDE rendering issue is unresolved and
has not been worked around.
