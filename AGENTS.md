# Project instructions

## Context

This is gregl83's GitHub profile repository and weekly AI image experiment.
Read `README.md` and the relevant implementation before editing.

- `imagine.py`: Python CLI, Grok/Gemini adapters, prompt rendering, JSON parsing,
  image generation, and cropping/resizing.
- `prompts/*.j2`: Jinja templates for four news-inspired image prompts.
- `pyproject.toml`: Python >=3.10 and runtime dependencies; CI uses Python 3.13.
- `.github/workflows/weekly-imagine.yml`: paused image generation; manual release workflow retained.
- `README.md`: public profile with a scripted terminal introduction and project links.

- `scripts/render_profile.py` and `assets/profile/`: reproducible terminal animation;
  see `assets/profile/BUILD.md` for rendering and draft limitations.

- `scripts/showcase.py`, `scripts/publish_showcase.py`, and
  `.github/workflows/weekly-showcase.yml`: cached project summaries and weekly
  release assets. See `docs/showcase.md`. Public copy must remain project-focused
  and must not include job-seeking or employer-targeting language.

## Working approach

- Define the requested outcome and how to verify it before changing code.
- State consequential assumptions. Resolve routine choices independently; ask
  when missing information would materially change the result.
- Make the smallest complete change. Follow nearby style and avoid unrelated
  cleanup, speculative abstractions, or unnecessary dependencies.
- Inspect Git status first and preserve existing user changes. Do not revert
  unrelated work or commit/push unless requested.
- Diagnose failures from evidence and verify fixes. Report what changed, checks
  performed, and any remaining limitations accurately.

## Local plans

For substantial or multi-step work, keep a short Markdown plan in
`.agents/plans/<task-name>.md`. Include the objective, constraints, steps,
validation, progress, and consequential decisions. Keep it current as work evolves.
Small edits do not need a plan.

`.agents/` is ignored local working space. Create `.agents/plans/` as needed on
fresh clones; never depend on ignored plans for durable project documentation.
Keep shared instructions in this file and public project documentation in tracked files.

## Project contracts

- Local preview: `.venv/bin/python -m scripts.preview_showcase` writes only to
  ignored `out/`; do not publish as part of preview. `--example` needs no API key.
- Root README always points to latest-release assets and is never rewritten by
  the pipeline. Local Markdown preview: `assets/profile/README.md` (example) or
  `out/local-preview/README.md` (generated).

- Preserve `alpha.png`, `bravo.png`, `charlie.png`, and `delta.png` as the existing
  image experiment release contract. Default output is `out/`, resized to 130x130.
- Keep prompt templates and consumers aligned: JSON contains an `images` array
  of four entries with `date_relevance`, `category`, `style`, `colors`, and `prompt`.
- Preserve the templates' date context and visual constraints unless the task
  explicitly changes them. Prompt instructions alone do not verify news accuracy.
- Run the CLI from the repository root; template paths are relative to it.
- Maintain independently selectable language/image providers and their existing
  protocols. Verify provider documentation before changing model/API assumptions.
- Keep credentials in environment variables or GitHub secrets (`XAI_API_KEY`,
  `GOOGLE_API_KEY`). Never include their values in code, plans, or logs.
- Live generation spends API credits; workflow dispatch also publishes a release.
  Use offline checks for ordinary validation unless live execution is requested.
- Preserve the workflow's release asset contract and keep permissions limited to
  what the job requires. Generated images belong in ignored output or releases.

## Validation

Install `requirements/showcase.txt` into `.venv` and run
`.venv/bin/python -m unittest discover -s tests -v` for showcase changes.
The showcase tests are offline. There is no configured lint tool.
Install dependencies in an isolated environment with `python -m pip install .`
when runtime checks are needed. With dependencies installed, these checks do not
call generation APIs or require credentials:

```sh
python imagine.py --help
python imagine.py --dry-run -l gemini -i gemini
git diff --check
```

For behavior changes, add focused offline regression coverage where useful:
JSON parsing, prompt rendering, image dimensions using temporary files, and
provider selection using fakes. Avoid live provider calls in automated tests.
For documentation-only edits, inspect the diff and run `git diff --check`.

## Design references

Adapted to this repository from:

- [Karpathy's autoresearch instructions](https://github.com/karpathy/autoresearch/blob/master/program.md): explicit scope, measurable outcomes, simplicity.
- [Karpathy-inspired community guidelines](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md): assumptions, minimal changes, verification. This file is community-authored, not Karpathy's own.
- [Anthropic's repository instructions](https://github.com/anthropics/claude-code/blob/main/CLAUDE.md): concrete workflow constraints and credential handling.
