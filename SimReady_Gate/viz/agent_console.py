"""Build a self-contained HTML "agent run" page from a run directory: the request, the program
the model wrote, every tool call with its JSON verdict and exit code, the rendered videos (as
data URIs) and the certificate. Nothing is typed by hand: every number is read from the files
the tools produced.

Usage: python viz/agent_console.py <run_dir> <request_text_file> <out.html> [--videos gate.mp4 settle.mp4]
"""
import base64
import html
import json
import sys
from pathlib import Path


def esc(s):
    return html.escape(str(s))


def video_tag(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<video controls muted playsinline preload="metadata" src="data:video/mp4;base64,{b}"></video>'


def block(title, body, kind=""):
    return f'<section class="step {kind}"><h3>{esc(title)}</h3>{body}</section>'


def pre(text, cls=""):
    return f'<pre class="{cls}">{esc(text)}</pre>'


def read(p):
    return open(p).read() if Path(p).exists() else "(missing)"


def main():
    run = Path(sys.argv[1]); request = read(sys.argv[2]).strip(); out = sys.argv[3]
    videos = []
    if "--videos" in sys.argv:
        videos = sys.argv[sys.argv.index("--videos") + 1:]
    summarize = read(run / "agent_summarize.txt")
    # rounds: agent_round{k}_*.txt + program_round{k}.json are earlier attempts; the plain files are the final round
    rounds = []
    k = 1
    while (run / f"agent_round{k}_repair.txt").exists():
        rounds.append({"program": read(run / f"program_round{k}.json"), "check": read(run / f"agent_round{k}_check.txt"),
                       "repair": read(run / f"agent_round{k}_repair.txt"), "settle": read(run / f"agent_round{k}_settle.txt"), "final": False})
        k += 1
    rounds.append({"program": read(run / "program.json") if (run / "program.json").exists() else read(run / "program.dsl"),
                   "check": read(run / "agent_check.txt"), "repair": read(run / "agent_repair.txt"), "settle": read(run / "agent_settle.txt"), "final": True})
    cert_path = next((c for c in run.glob("*.certificate.json") if "round" not in c.name), None)
    cert = json.load(open(cert_path)) if cert_path else {}
    verdict = {"pen_before": cert.get("pen_before"), "pen_after": cert.get("pen_after"), "rmsd_xy": cert.get("rmsd_xy"),
               "time_s": cert.get("time_s"), "failed_predicates": cert.get("failed_predicates"), "g5_pass": cert.get("g5", {}).get("pass")}
    steps = [block("1 · summarize — the agent reads the scene as text", pre(summarize.strip()[:4000])),
             block("request", f'<p class="req">“{esc(request)}”</p>')]
    for i, r in enumerate(rounds, 1):
        tag = f"round {i}" + (" (accepted)" if r["final"] else "")
        steps.append(block(f"{tag} · text2function — the program the model wrote", pre(r["program"].strip()[:9000], "dsl")))
        steps.append(block(f"{tag} · check", pre(r["check"].strip()[-1500:]), "ok" if "exit=0" in r["check"] else "fail"))
        steps.append(block(f"{tag} · repair — S4R under the program, verified on full meshes, written back", pre(r["repair"].strip()[-3500:]), "ok" if "exit=0" in r["repair"] else "fail"))
        steps.append(block(f"{tag} · settle — G5 in MuJoCo, thresholds from the program's gate", pre(r["settle"].strip()[-2500:]), "ok" if "exit=0" in r["settle"] else "fail"))
    vids = "".join(f'<figure>{video_tag(v)}<figcaption>{esc(Path(v).stem)}</figcaption></figure>' for v in videos if Path(v).exists())
    cert_html = pre(json.dumps({k: v for k, v in cert.items() if k not in ("provenance", "predicates")}, indent=1)[:6000]) + \
        pre(json.dumps(cert.get("provenance", {}), indent=1)[:4000], "prov") if cert else "<p>(no certificate)</p>"
    page = f"""<title>SimReady Gate · agent run</title>
<style>
:root{{--bg:#F4F6F9;--surface:#fff;--ink:#16202B;--muted:#5B6977;--line:#D6DDE5;--accent:#2A5D9F;--pass:#1F8A5B;--fail:#B93A2B;--code:#EEF2F6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0F151C;--surface:#161E27;--ink:#E6EBF0;--muted:#98A5B2;--line:#2A3641;--accent:#7DB0E6;--pass:#52C48F;--fail:#E76B58;--code:#1C2631}}}}
:root[data-theme="dark"]{{--bg:#0F151C;--surface:#161E27;--ink:#E6EBF0;--muted:#98A5B2;--line:#2A3641;--accent:#7DB0E6;--pass:#52C48F;--fail:#E76B58;--code:#1C2631}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;margin:0;line-height:1.55}}
.page{{max-width:1180px;margin:0 auto;padding:36px 22px 80px}}
h1{{font-size:2rem;margin:0 0 .3rem}} .lede{{color:var(--muted);margin:0 0 1.6rem}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
.step{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 14px}}
.step h3{{margin:0 0 .5rem;font-size:1rem}} .step.ok h3::after{{content:"  exit 0";color:var(--pass);font-family:monospace}} .step.fail h3::after{{content:"  exit ≠ 0";color:var(--fail);font-family:monospace}}
pre{{background:var(--code);padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:.8rem;line-height:1.45;margin:.4rem 0}} pre.dsl{{border-left:3px solid var(--accent)}} pre.prov{{color:var(--muted)}}
.req{{font-size:1.05rem;margin:.2rem 0 .6rem}}
figure{{margin:0 0 16px}} video{{width:100%;border-radius:8px;background:#000}} figcaption{{color:var(--muted);font-size:.85rem;margin-top:4px}}
.verdict{{display:flex;flex-wrap:wrap;gap:10px 22px;font-family:monospace;font-size:.9rem;margin:0 0 18px}} .verdict b{{color:var(--accent)}}
</style>
<div class="page">
<h1>SimReady Gate · one agent run</h1>
<p class="lede">A language request, a typed program written by the model, and the deterministic gates that decide. Every number on this page was read from the tools' own outputs in <code>{esc(run)}</code>.</p>
<div class="verdict"><span>penetrating pairs <b>{verdict['pen_before']} → {verdict['pen_after']}</b></span><span>planar RMSD <b>{verdict['rmsd_xy']:.3f} m</b></span><span>repair <b>{verdict['time_s']} s</b></span><span>failed predicates <b>{verdict['failed_predicates']}</b></span><span>G5 <b>{'pass' if verdict['g5_pass'] else verdict['g5_pass']}</b></span></div>
<div class="grid"><div>{''.join(steps)}</div><div>{vids}<section class="step"><h3>certificate</h3>{cert_html}</section></div></div>
</div>
"""
    Path(out).write_text(page); print("wrote", out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
