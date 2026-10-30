#!/usr/bin/env python
"""§45: render the maps as plain HTML (no server, no script): one page per prompt and an index per stage.

    .venv/bin/python scripts/exp45_render.py --dir runs/exp45_maps/j56-jrefit-logit/stage1 [--split dev] [--limit N]

Per prompt: the prompt, the generated text and outcome; per lens a heat map (layers x positions) of each pinned
entity's best rank over its forms (log10 rank, rank 1 darkest, clipped at 10^4; the dashed line is the last prompt
token, everything right of it is generated); and the top-10 strip at the last prompt token and each generated token
(the cell shows the top-1, its tooltip the top-10 with log-probs; a cell whose top-1 is a pinned entity is outlined).
For F2 and stage 3 a dotted green line marks where the counted answer span starts (rev 1).
Presentation only: nothing here changes a criterion. `--split dev` renders the dev half only (§3: read first).
Writes <dir>/html/<key>.html and <dir>/html/index.html.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import maps as M  # noqa: E402

ROLE_ORDER = ("gold", "intermediate", "competitor", "substitute", "entity", "unknown")
STYLE = """
:root { --ink:#1d2330; --muted:#5b6475; --line:#d9dde5; --bg:#ffffff; --panel:#f5f7fa; --mark:#1f5fbf; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ink:#e6e9ef; --muted:#a3abba; --line:#3a4150; --bg:#14171d; --panel:#1c2028; --mark:#7fb0ff; } }
:root[data-theme="dark"] { --ink:#e6e9ef; --muted:#a3abba; --line:#3a4150; --bg:#14171d; --panel:#1c2028; --mark:#7fb0ff; }
body { background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; margin:0 16px 40px; max-width:1400px; }
pre { background:var(--panel); padding:8px; white-space:pre-wrap; border:1px solid var(--line); }
table.strip { border-collapse:collapse; font:11px ui-monospace, monospace; }
table.strip td, table.strip th { border:1px solid var(--line); padding:1px 4px; white-space:pre; }
table.strip td.pin { outline:2px solid var(--mark); outline-offset:-2px; font-weight:600; }
.scroll { overflow-x:auto; } img { max-width:100%; height:auto; background:#fff; }
.muted { color:var(--muted); } a { color:var(--mark); }
"""


def load_pair(json_path: Path) -> tuple[dict, dict]:
    payload = json.loads(json_path.read_text())
    arrays = dict(np.load(json_path.with_suffix(".npz")))
    return payload, arrays


def entity_rows(pinned: list[dict]) -> list[tuple[str, str, list[int]]]:
    """(label, surface, pinned columns) per entity, ordered by role."""
    seen, rows = set(), []
    for role in ROLE_ORDER:
        for p in pinned:
            if role in p["roles"] and p["surface"] not in seen:
                seen.add(p["surface"])
                rows.append((f"{role}: {p['surface']}", p["surface"], M.entity_columns(pinned, p["surface"])))
    return rows


def heatmap_png(ranks: np.ndarray, layers, tokens: list[str], last_prompt: int, rows, title: str,
                answer_from: int | None = None) -> str:
    """Base64 PNG: one panel per entity, y = layer (final at top), x = position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pos = ranks.shape[1]
    width = min(22, max(6, 0.09 * n_pos + 2))
    fig, axes = plt.subplots(len(rows), 1, figsize=(width, 2.3 * len(rows) + 0.6), squeeze=False,
                             gridspec_kw={"hspace": 0.9})
    for ax, (label, _, cols) in zip(axes[:, 0], rows):
        grid = np.log10(np.clip(ranks[:, :, cols].min(axis=-1), 1, 10 ** 4))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Blues_r", vmin=0, vmax=4, interpolation="nearest")
        ax.axvline(last_prompt + 0.5, color="#c0392b", lw=1, ls="--")
        if answer_from is not None and answer_from > last_prompt:
            ax.axvline(answer_from - 0.5, color="#1e8449", lw=1, ls=":")
        ax.set_ylabel("layer", fontsize=7)
        ax.set_title(label, fontsize=8, loc="left")
        step = max(1, n_pos // 120)
        ax.set_xticks(range(0, n_pos, step))
        ax.set_xticklabels([tokens[i].replace("\n", "\\n")[:10] for i in range(0, n_pos, step)], rotation=90, fontsize=5)
        ax.set_yticks(range(0, len(layers), max(1, len(layers) // 6)))
        ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 6))], fontsize=6)
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("log10 rank (0 = top-1)", fontsize=7)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def strip_table(top_ids: np.ndarray, top_logp: np.ndarray, layers, tokens, last_prompt: int, tokenizer_decode,
                pinned_ids: set[int]) -> str:
    cols = list(range(last_prompt, top_ids.shape[1]))
    head = "".join(f"<th>{html.escape(repr(tokens[c])[1:-1][:12])}</th>" for c in cols)
    body = []
    for li in range(len(layers) - 1, -1, -1):
        cells = []
        for c in cols:
            ids = top_ids[li, c]
            tip = " | ".join(f"{tokenizer_decode(int(t))!r} {float(lp):.2f}" for t, lp in zip(ids, top_logp[li, c]))
            top1 = tokenizer_decode(int(ids[0]))
            cls = ' class="pin"' if int(ids[0]) in pinned_ids else ""
            cells.append(f'<td{cls} title="{html.escape(tip)}">{html.escape(repr(top1)[1:-1][:12])}</td>')
        body.append(f"<tr><th>{layers[li]}</th>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table class="strip"><tr><th>layer</th>{head}</tr>{"".join(body)}</table></div>'


def make_decoder(payload: dict, arrays: dict):
    """Token text from a tokenizer when one is cached, else the ids themselves."""
    try:
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        return lambda t: tok.decode([t])
    except Exception:  # noqa: BLE001
        return lambda t: f"#{t}"


def render_one(payload: dict, arrays: dict, decode) -> str:
    rows = entity_rows(payload["pinned"])
    layers = [int(l) for l in arrays["layers"]]
    tokens = payload["position_tokens"]
    lp = payload["last_prompt_index"]
    pinned_ids = {int(p["id"]) for p in payload["pinned"]}
    parts = [f"<h1>{html.escape(payload['key'])}</h1>",
             f"<p>outcome <b>{html.escape(payload['outcome'])}</b>{' (ambiguous)' if payload.get('ambiguous') else ''}; "
             f"gold {html.escape(str(payload['gold']))}; answer marker {payload.get('answer_marker')}; "
             f"{payload['n_prompt_tokens']} prompt tokens, {payload['n_generated']} generated; lenses "
             f"{', '.join(payload['lens_names'])}</p>",
             f"<p class='muted'>{html.escape(json.dumps({k: payload.get(k) for k in ('task', 'variant', 'format', 'hardness', 'split', 'label23', 'kind') if payload.get(k) is not None}))}</p>",
             "<h2>Generated</h2>", f"<pre>{html.escape(payload['generated_text'])}</pre>",
             "<details><summary>Prompt</summary>", f"<pre>{html.escape(payload.get('prompt', '') or '(see rows44.jsonl)')}</pre></details>"]
    for li, name in enumerate(payload["lens_names"]):
        counted = payload.get("counted_positions") or [lp]
        span = [c for c in counted if c != lp]
        png = heatmap_png(arrays["ranks"][li], layers, tokens, lp, rows, f"{name}: best rank per layer and position",
                          answer_from=span[0] if span else None)
        parts += [f"<h2>{html.escape(name)}</h2>", f'<img alt="{html.escape(name)} rank heat map" src="data:image/png;base64,{png}">',
                  "<h3>Top-10 strip (last prompt token, generated tokens)</h3>",
                  strip_table(arrays["top_ids"][li], arrays["top_logp"][li].astype(np.float32), layers, tokens, lp, decode,
                              pinned_ids)]
    return page(payload["key"], "".join(parts))


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>{body}</body></html>")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--split", choices=("dev", "held", "all"), default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keys", default=None, help="comma-separated keys to render")
    args = p.parse_args(argv)
    d = Path(args.dir)
    out = d / "html"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(x for x in d.glob("*.json") if not x.name.startswith("_"))
    keys = set(args.keys.split(",")) if args.keys else None
    decode, index = None, []
    for path in paths:
        payload = json.loads(path.read_text())
        if keys and payload["key"] not in keys:
            continue
        if args.split != "all" and payload.get("split") not in (None, args.split):
            continue
        if args.limit is not None and len(index) >= args.limit:
            break
        arrays = dict(np.load(path.with_suffix(".npz")))
        decode = decode or make_decoder(payload, arrays)
        (out / f"{payload['key']}.html").write_text(render_one(payload, arrays, decode))
        index.append(payload)
    rows = "".join(f"<tr><td><a href='{html.escape(x['key'])}.html'>{html.escape(x['key'])}</a></td>"
                   f"<td>{html.escape(x['outcome'])}</td><td>{html.escape(x['generated_text'][:80])}</td></tr>" for x in index)
    (out / "index.html").write_text(page(f"§45 maps {d.name}", f"<h1>{html.escape(str(d))}</h1><table>"
                                                                  f"<tr><th>key</th><th>outcome</th><th>generated</th></tr>{rows}</table>"))
    print(f"render: {len(index)} pages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
