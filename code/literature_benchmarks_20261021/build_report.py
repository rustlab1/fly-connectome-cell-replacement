"""Build a self-contained HTML report from completed, audited results."""
import base64
import html
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/literature_benchmarks_20261021'
DEST=ROOT/'report/literature_benchmarks_20261021'


def main():
    qa=json.loads((OUT/'analysis_qa.json').read_text());assert qa['passed'] and qa['states']==70
    stats=json.loads((OUT/'statistics.json').read_text())['results']
    lookup={(r['condition'],r['metric']):r for r in stats}
    verdict=json.loads((OUT/'interpretation.json').read_text())
    refs=json.loads((OUT/'literature_sources.json').read_text())
    completion_date=datetime.now(timezone.utc).strftime('%d %B %Y')
    esc=html.escape
    def st(c,m,k='perturbed'):return lookup[c,m][k]
    def value(c,m,k='perturbed'):return f"{st(c,m,k)['mean']:.1f}"
    def retained(c,m):
        x=st(c,m,'percent_of_paired_intact')
        return f"{x['mean']:.1f}% ({x['ci95_low']:.1f}–{x['ci95_high']:.1f})"
    def reduction(c,m):
        x=st(c,m,'percent_of_paired_intact')
        return f"{100-x['mean']:.1f}% ({100-x['ci95_high']:.1f}–{100-x['ci95_low']:.1f})"
    def img(filename,alt):
        b64=base64.b64encode((DEST/filename).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}">'
    study_rows=''.join(f'<tr><td><strong>{esc(v["label"])}</strong><br><span class="muted">{esc(v["study"])}</span></td><td>{esc(v["result"])}</td><td>{esc(v["judgment"])}</td></tr>' for v in verdict['studies'])
    hs_rows=[]
    for c,label in [('HS_left_loss','Left HS off'),('HS_right_loss','Right HS off'),('HS_bilateral_loss','Both HS off')]:
        hs_rows.append(f'<tr><td>{label}</td><td>{retained(c,"y_R")}</td><td>{retained(c,"y_L")}</td><td>{value(c,"y_F","intact")} → {value(c,"y_F")}</td></tr>')
    gain_rows=[]
    for c,label in [('HS_left_loss','Left HS off'),('HS_right_loss','Right HS off'),('HS_bilateral_loss','Both HS off')]:
        gain_rows.append(f'<tr><td>{label}</td><td>{retained(c,"gain_R")}</td><td>{retained(c,"gain_L")}</td><td>{retained(c,"gain")}</td></tr>')
    loom_rows=[]
    for c,label in [('LPLC2_block','LPLC2 off'),('LC4_block','LC4 off'),('LC4_LPLC2_block','Both off')]:
        loom_rows.append(f'<tr><td>{label}</td>'+''.join(f'<td>{value(c,m)} Hz<br><span class="muted">{retained(c,m)} retained</span></td>' for m in ['GF_loom','TTMn_loom','DLMn_loom'])+'</tr>')
    references=''.join(f'<li><strong>{esc(r["authors_year"])}</strong>. {esc(r["title"])}. <em>{esc(r["journal"])}</em>. <a href="{esc(r["url"])}">doi:{esc(r["doi"])}</a></li>' for r in refs['studies'])
    context=refs['interpretive_context']
    references+=f'<li><strong>{esc(context["authors_year"])}</strong>. {esc(context["title"])}. <a href="{esc(context["url"])}">doi:{esc(context["doi"])}</a>. Context for omitted electrical interactions; not a fourth benchmark.</li>'
    context=refs['hs_specificity_context']
    references+=f'<li><strong>{esc(context["authors_year"])}</strong>. {esc(context["title"])}. <a href="{esc(context["url"])}">doi:{esc(context["doi"])}</a>. Context for cell specificity and head/wing endpoint differences; not an additional benchmark.</li>'
    recommendations=''.join(f'<li>{esc(x)}</li>' for x in verdict['recommendations'])
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLY literature benchmark pilot</title>
<style>
:root{{--ink:#172b3a;--muted:#586b7b;--accent:#216884;--line:#dbe3e8;--pale:#f0f5f8}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1180px;margin:36px auto;background:#fff;padding:45px 52px;border:1px solid var(--line);border-radius:14px}}h1{{font-size:34px;line-height:1.15;letter-spacing:-.7px;max-width:850px;margin:12px 0 16px}}h2{{font-size:23px;margin:42px 0 12px}}h3{{font-size:18px;margin:24px 0 8px}}p{{margin:12px 0}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:var(--accent);font-weight:750}}.lead{{font-size:20px;max-width:900px}}.muted,.caption{{color:var(--muted);font-size:13px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.stat{{padding:15px;background:var(--pale);border-radius:8px}}.stat strong{{display:block;font-size:25px;line-height:1.2}}.stat span{{font-size:13px;color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:17px 0 10px}}th{{text-align:left;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.4px}}td,th{{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top}}td:first-child{{min-width:145px}}img{{display:block;width:100%;height:auto;margin:20px 0;border:1px solid #edf0f3}}.note{{border-left:4px solid var(--accent);padding:12px 18px;background:var(--pale);margin:22px 0}}.formula{{font:15px ui-monospace,monospace;background:#f6f8fa;padding:14px;overflow:auto}}a{{color:#1a637e}}li{{margin:10px 0}}code{{font-size:13px}}footer{{margin-top:35px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}@media(max-width:700px){{main{{margin:0;padding:26px 19px;border:0;border-radius:0}}h1{{font-size:29px}}.stats{{grid-template-columns:repeat(2,1fr)}}table{{display:block;overflow:auto}}}}@media print{{body{{background:#fff}}main{{border:0;margin:0;padding:0;max-width:none}}h2,h3{{break-after:avoid}}table,img,.note{{break-inside:avoid}}a{{color:inherit}}}}
</style></head><body><main>
<div class="eyebrow">Frozen-model exploratory pilot · 21 October 2026</div>
<h1>Three published perturbations tested in the FLY v8 model</h1>
<p class="lead">{esc(verdict['lead'])}</p>
<div class="stats"><div class="stat"><strong>3</strong><span>primary studies</span></div><div class="stat"><strong>70</strong><span>intervention states</span></div><div class="stat"><strong>190</strong><span>perturbation trials</span></div><div class="stat"><strong>10</strong><span>paired seeds per condition</span></div></div>
<p>The model, connectome and steering decoder were held fixed. Expected effects and all seven intervention conditions were recorded before perturbation outcomes were inspected. Four replay trials verified the available simulator against archived intact v8 rates for seed 1000. The other nine intact controls were also replayed (36 trials) because v8 retained their summaries but not their full neuron arrays.</p>
<table><thead><tr><th>Experiment</th><th>Observed model result</th><th>Assessment</th></tr></thead><tbody>{study_rows}</tbody></table>
<div class="note"><strong>These are qualitative circuit-consistency tests.</strong> No published response curves were digitized, no empirical effect sizes were fitted, and the model endpoints differ from the animal assays. The ten seeds are random input realizations of one connectome, not ten flies. A common numerical “reproduction score” would be misleading.</div>
{img('benchmark_overview.png','Observed responses retained after HS, T4/T5 and looming-pathway silencing')}

<h2>1. HS-cell loss: the most relevant test for MP’s steering question</h2>
<p>Hausen and Wehrhahn (1983) linked microsurgical HS lesions to altered optomotor yaw torque in blowflies. We silenced the canonical HSN, HSE and HSS cells on either side of the fruit-fly connectome (three cells per side), plus a bilateral control. HST cells were excluded.</p>
<p>Left HS loss reduced the magnitude of the leftward raw command by <strong>{reduction('HS_left_loss','y_L')}</strong>; right HS loss reduced the rightward command by <strong>{reduction('HS_right_loss','y_R')}</strong>. Values in parentheses are 95% intervals across paired simulation seeds. These are model effect sizes, not estimates from the paper.</p>
<div class="note">{esc(verdict['hs_qualification'])}</div>
<table><thead><tr><th>Condition</th><th>Rightward command retained</th><th>Leftward command retained</th><th>Forward command<br>intact → perturbed</th></tr></thead><tbody>{''.join(hs_rows)}</tbody></table>
<p class="caption">Retained = mean of 100 × perturbed/intact for each paired seed. Raw command ratios preserve the sign of any reversal. Forward commands are decoder projections in weighted Hz.</p>
<h3>Why the manuscript’s normalized gain needs care here</h3>
<p>Each directional gain subtracts the <em>perturbed state’s own forward response</em>. HS loss changes that forward response, so a normalized gain can increase while the corresponding raw rotation command weakens. Both sets of readouts are retained below.</p>
<div class="formula">gain_R = (y_R − y_F) / intact_reference_R<br>gain_L = (y_F − y_L) / intact_reference_L</div>
<table><thead><tr><th>Condition</th><th>Rightward gain retained</th><th>Leftward gain retained</th><th>Overall R–L gain retained</th></tr></thead><tbody>{''.join(gain_rows)}</tbody></table>
<p><strong>Boundary:</strong> this supports a qualitative causal role for HS cells in the model’s steering pathway. Species, lesion method and readout differ from the original study. The full experimental waveform and magnitude have not been reproduced. The model omits graded HS physiology and electrical coupling; later Drosophila work highlights the behavioral importance of these interactions.</p>
<p>More selective HS/VS hyperpolarization in Drosophila produced strong head-response deficits and more modest wing effects (Kim et al., 2017). Older microsurgical lesions could also affect neighboring pathways. The present modest wing-decoder effect should therefore be interpreted in that context; the model has not reproduced the head-versus-wing dissociation.</p>

<h2>2. T4/T5 blockade: a clear necessity check</h2>
<p>Bahl et al. (2013) found that blocking T4/T5 output abolished the walking optomotor response, while some object fixation remained. Silencing all <strong>13,580 T4/T5 cells</strong> here left <strong>{retained('T4T5_block','gain_R')}</strong> of rightward gain and <strong>{retained('T4T5_block','gain_L')}</strong> of leftward gain. The raw R/L/F steering commands were also zero.</p>
<p>The separately driven giant-fiber looming response retained <strong>{retained('T4T5_block','GF_loom')}</strong>. This control stimulates LC4/LPLC2 directly, bypassing upstream motion processing; it does not demonstrate that real T4/T5-silenced flies retain normal looming responses.</p>
<p><strong>Assessment:</strong> the expected optomotor loss is reproduced at the level of the model readout. Its evidential value is limited because T4/T5 supply the model’s direct motion inputs. The black-bar fixation result was not tested, and the model reads wing motor neurons rather than walking behavior.</p>

<h2>3. LPLC2 silencing: partial pathway agreement</h2>
<p>Ache et al. (2019) showed that LPLC2 contributes the angular-size component of the giant-fiber looming response, with LC4 contributing a velocity component. LPLC2 was necessary for GF-mediated escape in the animal assay. We silenced all <strong>185 LPLC2 neurons</strong> under the existing combined LC4/LPLC2 input protocol, with LC4-only and combined blockade as controls.</p>
<p>LPLC2 silencing reduced mean giant-fiber firing from <strong>{value('LPLC2_block','GF_loom','intact')} to {value('LPLC2_block','GF_loom')} Hz</strong>: a <strong>{reduction('LPLC2_block','GF_loom')}</strong> decrease. Substantial GF and motor activity remained.</p>
<table><thead><tr><th>Condition</th><th>Giant fiber</th><th>Jump motor neurons<br>TTMn</th><th>Flight-power motor neurons<br>DLMn</th></tr></thead><tbody>{''.join(loom_rows)}</tbody></table>
<p><strong>Assessment:</strong> the reduction is consistent with LPLC2 contributing excitatory drive. The experiment’s size/velocity decomposition and loss of GF-mediated escape are not reproduced by this assay. Constant 100-Hz Poisson drive has no angular-size or velocity trajectory, and GF mean spike rate is not the measured graded voltage response or takeoff probability. The residual motor output is a limitation worth retaining in the report.</p>
<p>As an additional check, LPLC2 blockade retained <strong>{retained('LPLC2_block','gain')}</strong> of the overall motion-steering gain. This is reported to show cross-pathway effects; it was not an endpoint of the cited animal experiment.</p>
{img('benchmark_diagnostics.png','Raw HS steering readouts and giant-fiber, jump-motor and flight-power-motor responses under looming pathway blockade')}

<h2>What to do with these results</h2><ol>{recommendations}</ol>

<h2>Methods and audit</h2>
<p>All trials used the v8 MaleCNS matrix: 165,122 neurons, original signed chemical synapses, original V5 LIF reference equations and frozen decoder. Each trial lasted 300 ms, with firing measured over 50–300 ms and existing input drive set to 100 Hz. Seeds 1000–1009 were paired to their saved intact v8 responses. Original stimulus slots remained present for silenced inputs so the surviving neurons received the same random event streams.</p>
<p>The archived compiled environment was unavailable. A portable loader executed the exact original NumPy reference class and readout functions; only file loading was adapted. All 165,122 firing rates matched the saved intact seed-1000 arrays exactly for R, L, F and looming. Scalar readouts permit rtol=10<sup>−7</sup>, atol=10<sup>−5</sup> for floating-point reduction differences, documented before perturbation outcomes. The nine other intact controls were replayed with the same original event seeds; their full arrays and validation are included. All ten intact readouts were computed in the current environment and checked against the archived summaries.</p>
<p>Every specified condition and seed is present. All targeted cells were silent in every saved trial, and raw rate files reproduce the recorded readouts. Source hashes confirm that the original model inputs and source files are unchanged. The existing v8 manuscript and results were not edited.</p>
<p>Statistics are means and two-sided t intervals (df=9). Paired differences and per-seed fractions are supplied in machine-readable tables. These intervals describe sensory-event variability only; they exclude uncertainty across flies, synaptic parameters, connectomes and assay mappings. There are no formal multiple-comparison significance claims.</p>
<p>The bundle contains the complete new raw rates, cell lists, protocol and numerical amendment, validation logs, CSV/JSON summaries, plotting code and this report. The existing large connectome inputs are referenced by path and hash. See <code>code/literature_benchmarks_20261021/README.txt</code> for reproduction commands.</p>

<h2>Primary sources</h2><ol>{references}</ol>
<footer>Exploratory analysis completed {completion_date} (UTC). All three selected experiments and their predefined controls are reported. No graft repair or transplantation outcome is validated by these tests.</footer>
</main></body></html>'''
    DEST.mkdir(exist_ok=True,parents=True)
    path=DEST/'literature_benchmark_report.html';path.write_text(report)
    print(path)


if __name__=='__main__':main()
