# Weekly project showcase

The profile animation highlights four public projects with concrete engineering
capabilities. Static identity, terminal styling, responsive framing, pauses, and
CRT ending remain controlled by `assets/profile/storyboard.json` and the renderer.
AI produces text only. No image-generation API is called.

## Pipeline

`.github/workflows/weekly-showcase.yml` runs Sunday at **00:00 UTC**, or manually.

1. Restore `showcase-cache.json` from the most recent successful `showcase-*`
   release. With no previous showcase, start empty. Download failures stop the job.
2. Read six public pinned repositories (including organization repositories) and
   recently pushed owned public repositories. Exclude forks, archived/empty repos,
   and the profile repo; deduplicate and cap the combined pool at twelve, pins first.
3. Collect README plus up to two architecture/design/benchmark documents at the
   default branch commit. Cap supplied text at 6,000 characters per repository.
   This is a bounded documentation review, not a comprehensive code audit. Changes
   outside those excerpts do not trigger regeneration.
4. Hash supplied content, description, model, and prompt policy. Unchanged content
   reuses its cached summary even if a new commit only changes unrelated files.
5. Send all changed candidates in **one initial request**, with at most one
   correction request containing only invalid summaries. No transport retries.
   Save raw responses as review artifacts before validation. At most
   twelve candidates / 72,000 document characters and 8,192 output tokens. No
   search, tools, image APIs, or separate model-ranking call. Unchanged runs use
   zero model requests. API charges still depend on the configured model.
6. Validate ASCII text, title <=26 characters, summary targeted at 55-80 characters
   (120-character safety bound), combined title/summary fitting four lines in both layouts, no hiring
   or employer-targeting copy, no numeric/comparative claims in short summaries,
   ranking scores, and exact evidence quotes from supplied documents. Invalid
   candidates get one targeted correction attempt, then are excluded if still
   invalid. Text is never silently truncated. Batch/transport failures stop the job.
7. Feature Agent Vigilo first when it passes validation and the scoring thresholds.
   Rank the remaining candidates to fill four slots by default (or three with
   `--project-count 3`). Weight relevance ×2, engineering substance ×3, and
   evidence strength ×3; require at least 3/5 on every axis.
   Pins and recent activity break ties. Never fill a slot with a weak candidate;
   insufficient qualifying projects stops the run and preserves the previous release. Only one member of the paq family is selected.
   No forced weekly rotation: strong unchanged work can remain featured.
8. Check wrapping and available height in both layouts. Each project gets enough
   time for typing plus at least five seconds of reading. Render GIF/MP4/posters.
9. For scheduled runs or explicitly requested publication, validate media, upload
   all files into a **draft** release, verify names and byte
   sizes, then publish and mark latest. Failed uploads leave the previous public
   release current. A failed draft can be inspected/deleted separately.

Public summaries describe project capabilities, not personal authorship or
employment ambitions. Evidence quotes and immutable source URLs appear in release
notes and `showcase.json`. Exact-quote validation proves the source contains the
quote; it cannot prove that a model's paraphrase is fully entailed. Inspect the
initial output and adjust the prompt if it overstates a claim. Performance metrics
are intentionally left to linked documentation where methodology can be explained.

## Local development

Use Python **3.13**, `gh`, and FFmpeg (`ffmpeg` + `ffprobe`). Run commands from
this repository root. The showcase is a repository tool, not an installed library.
`pyproject.toml` preserves the old image experiment; the lighter showcase runtime
uses its own hash-pinned dependency file, shared with CI.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements/showcase.txt
```

Or, with uv:

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements/showcase.txt
```

Both `.venv/` and `out/` are Git-ignored. To update dependency pins deliberately,
edit `requirements/showcase.in`, regenerate the lock, and rerun checks:

```sh
uv pip compile requirements/showcase.in --generate-hashes --python-version 3.13 -o requirements/showcase.txt
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` locally, or export it in
your shell. `.env` is ignored and loaded without overriding existing environment
variables. Never paste the key into chat, logs, or tracked files. Optionally set
`SHOWCASE_MODEL`; the default is `gemini-3.5-flash-lite`. GitHub Actions uses the
existing repository secret `GOOGLE_API_KEY` and optional variable `SHOWCASE_MODEL`.

Authenticate `gh` for public repository reads, then run the complete local pipeline:

```sh
.venv/bin/python -m scripts.preview_showcase
```

Open `out/local-preview/README.md` in a Markdown preview, or `preview.html`
in a browser. The generated Markdown mirrors the profile README using local assets. This command reads GitHub, summarizes changed
sources, renders both layouts, and validates media. It writes **only inside out/**;
it never publishes a release, changes tracked files, commits, or pushes. It reuses
its local summary cache on subsequent runs. A fresh run without local cache restores
the previous successful release cache. A changed-content run needs the API key.

For a completely offline render of the checked-in, manually reviewed examples:

```sh
.venv/bin/python -m scripts.preview_showcase --example --output out/example-preview
```

For source collection alone, with no model calls:

```sh
.venv/bin/python -m scripts.showcase --collect-only
```

Lower-level commands remain available for troubleshooting:

```sh
.venv/bin/python -m scripts.publish_showcase restore
.venv/bin/python -m scripts.showcase --cache out/previous-showcase.json
.venv/bin/python scripts/render_profile.py --storyboard out/showcase/storyboard.json --output out/showcase
.venv/bin/python -m scripts.publish_showcase validate
.venv/bin/python -m unittest discover -s tests -v
```

`--sources PATH` reuses collected documentation for offline tests. Valid paid
summaries are saved before ranking/rendering so a later local failure need not
repeat the model call. Exact excerpt validation is not semantic fact-checking;
review the first live output for accurate claims and useful project selection.

## First release

The main README permanently references `releases/latest/download/...`. The
pipeline publishes assets; it **never edits README or creates Git commits**.
Local Markdown previews are separate:

- `assets/profile/README.md`: checked-in example.
- `out/local-preview/README.md`: newly generated output.

After committing and pushing the reviewed changes to `master`, immediately run:

```sh
gh workflow run weekly-showcase.yml --ref master -f publish=true -f project_count=4
gh run list --workflow weekly-showcase.yml --limit 1
# Substitute the run ID printed above:
gh run watch RUN_ID --exit-status
```

Or use Actions → **Weekly Project Showcase** → **Run workflow**. Publishing is
checked by default. The existing `GOOGLE_API_KEY` repository secret supplies the
first live summarization call. No local key is needed for Actions.

The profile images will be unavailable between the push and the first successful
release. On failure, inspect the logs/artifacts, fix the issue, and rerun. Once a
complete release exists, every later successful weekly release refreshes the same
URLs. No activation command or follow-up README change is needed.

Verify after the first run:

```sh
gh release view --json tagName,url,assets
```

Check the live profile on desktop and mobile and inspect the release's selected
projects and supporting sources. The live model's copy still needs this first
review; offline tests cannot establish its quality.

### Optional preview-only Actions run

Set `publish=false` to generate a downloadable `showcase-preview-*` artifact
without creating a release. Extract it locally and open `README.md` or
`preview.html`. To publish those exact reviewed files without another model call:

```sh
.venv/bin/python -m scripts.publish_showcase publish --directory out/reviewed-release --tag showcase-UNIQUE-RUN-ID --commit WORKFLOW-COMMIT-SHA
```

Substitute a unique tag and the actual pushed SHA from the run. This is an explicit
remote publish command; local `preview_showcase` never publishes. A second fresh
Actions generation run before any published cache exists will need another model
request. Failed-run artifacts retain any validated summaries already produced.

Sunday **00:00 UTC** runs publish automatically. Review artifacts are retained for
14 days. Failed publication leaves the previous successful release current.

`/releases/latest` is repository-wide, not workflow-specific: keep Weekly Imagine
disabled and do not mark an unrelated release latest unless it includes the same
showcase asset names. GitHub scheduling and image caching can delay the visible
refresh beyond exactly 00:00 UTC.

## Published assets

- `intro.gif`, `intro-mobile.gif`: README sources, responsive at 600px.
- `intro.mp4`, `intro-mobile.mp4`: videos with the same timeline.
- `poster.png`, `poster-mobile.png`: final message before shutdown.
- `storyboard.json`: exact generated render inputs.
- `showcase.json`: selected projects, source commits, and supporting excerpts.
- `showcase-cache.json`: reusable summaries and content/policy fingerprints.
- `release-notes.md`: readable project links and evidence.

References: [GitHub release asset URLs](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases),
[Google structured output guidance](https://ai.google.dev/gemini-api/docs/structured-output).

Ranking uses an anchored technical rubric: failure handling, correctness,
evaluation, isolation, data integrity, observability, resource constraints, and
reproducibility. Generic AI branding and popularity confer no advantage. The
public copy never mentions ranking, audience targeting, or employment goals.
Changing this rubric invalidates cached summaries once; subsequent unchanged
runs still make zero model calls.

Project titles prefer documented product names. Organization display metadata
is supplied to the model, so Agent Vigilo can keep its public branding. Distinct
products within one organization retain their own identities.

If an individual summary remains invalid after the one correction request, cache
its exclusion and rank only validated candidates. Unchanged excluded candidates
do not trigger another paid request. Changed evidence or policy reevaluates them.
Publication still requires the requested number of distinct, qualified projects.
Raw initial and repair responses are retained in review artifacts, not published
as profile copy.

Agent Vigilo is explicitly collected even if it is no longer pinned. Its featured
position does not bypass evidence, relevance, or rendered-fit checks.
