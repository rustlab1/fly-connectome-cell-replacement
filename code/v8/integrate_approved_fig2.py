"""Apply the approved three-panel Figure 2 amendment without changing model data."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'report/v8'
REVIEW = ROOT / 'figures/v8/review_20261012'
EXTENSIONS = ['figure2_full_replacement_20261012', 'figure2_random_dose_20261012']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_paragraph(text, start, replacement):
    pattern = re.compile(r'^' + re.escape(start) + r'[^\n]*(?:\n(?!\n)[^\n]*)*', re.M)
    text, count = pattern.subn(lambda _: replacement, text)
    assert count == 1, (start, count)
    return text


FIGURE2_LEGEND = '''**Figure 2 | Cell identity and synaptic wiring determine replacement benefit.** **a**, Five repair conditions. Circles denote matched identities, triangles non-target identities, grey dots surviving cells and the cross no graft. Original repair restores the lost cell's available partners; random wiring samples nearby cells. “By cell type” means partners from a surviving neuron of the assigned type. All conditions share the 2,000-cell lesion (pink); the four grafts share 100 intended replacement positions. Anatomy does not depict wiring differences. **b**, Ten-second paths with 100 or 2,000 replacements, using neural seed 1000, intervention seed 2000 and gust seed 3. Grey dashed paths show matched intact. Fly symbols have equal size and illustrative posture; dots mark one-second intervals. All paths start at the origin and share axis ranges. **c**, Distance, repeated virtual crossings, wing-motor activity, looming-evoked motor activity and spared visual activity, each divided by matched intact. Zero replacement is the shared damaged baseline (orange dashed); intact is one (grey dotted). Points and ribbons show means and pointwise 95% confidence intervals across 100 realizations; movement averages 20 matched gusts first. Lines join tested doses without smoothing. At full dose, 42 untyped positions receive sampled lost-cell identities in the correct-type graft.'''


def update_sources():
    # A later author-approved supplement revision has its own compact callouts.
    if 'Three supplementary figures support the main results.' in (REPORT / 'supplementary_information_v8.md').read_text():
        return
    p = REPORT / 'manuscript_v8.md'
    main = p.read_text()
    citation_sequence = re.findall(r'\^(\d+(?:[–,]\d+)*)\^', main)
    main = main.replace('when choosing which lost cells to restore (Fig. 2b)',
                        'when choosing which lost cells to restore (Supplementary Fig. 2g)')
    main = main.replace('18.4–22.1%; Fig. 2b and Supplementary Movies 2 and 5)',
                        '18.4–22.1%; Fig. 2b, Supplementary Fig. 2g and Supplementary Movies 2 and 5)')
    main = main.replace('(solid curves in Fig. 2b)', '(Supplementary Fig. 2g)')
    main = replace_paragraph(main, '**Figure 2 |', FIGURE2_LEGEND)
    main = main.replace('(Fig. 2a,c–e and Supplementary Movie 3)',
                        '(Fig. 2a–c and Supplementary Movie 3)')
    main = main.replace('Original partners restored 62.5%', 'Original repair restored 62.5%')
    dose_start = 'Matched and non-target identities were also tested' if 'Matched and non-target identities were also tested' in main else 'The four repair rules were tested'
    main = replace_paragraph(main, dose_start,
        'The four repair rules were tested from 50 to 2,000 replacements, with the same intended positions at each dose (Fig. 2b,c). At 2,000 cells, displacement reached 87.6% with matched identities and 9.3% with non-target identities. Random wiring reached 14.3% (95% confidence interval 12.7–16.0%), whereas complete original repair reproduced intact output. Thus, replacing as many cells as were lost was insufficient when their identities or partners differed from those of the original circuit. The correct-type graft used sampled lost-cell identities at the 42 untyped positions. An earlier control that shuffled lost identities between positions is retained in Supplementary Figures 2 and 5. These grafts can still supply useful missing types at larger doses.')
    main = main.replace('(Fig. 2e and Supplementary Fig. 5b)', '(Fig. 2c and Supplementary Fig. 5b)')
    if 'The dose curves start at the measured function remaining after damage' not in main:
        main = main.replace('Movement recovery did not imply complete restoration of neural responses.',
            'The dose curves start at the measured function remaining after damage, with matched intact defined as one (Fig. 2c). Distance, virtual crossings and wing-motor activity were markedly reduced, whereas the looming response retained 37.4% of intact activity. Movement recovery did not imply complete restoration of neural responses.')
    method_addition = ('A separately recorded random-wiring extension added eight dose conditions: 50, 200, 500, 1,000, 1,250, 1,500, 1,750 and 2,000 cells, reusing the original 100-cell condition. Intended identities and motor-control-priority positions were retained while synaptic partners were sampled locally. The full-dose and intermediate-dose protocols were recorded before their respective simulations, after earlier results were available. Together, the original 46 conditions and the 16 typed-dose, nine non-target and eight random-wiring additions comprise 79 conditions, 7,900 neural/intervention realizations and 158,000 nested movement trajectories. These additions are documented separately from the original protocol.')
    if not main.startswith(method_addition) and method_addition not in main:
        anchor = '\n\nOlder three-seed controls separately varied'
        assert anchor in main
        main = main.replace(anchor, '\n\n' + method_addition + anchor)
    assert re.findall(r'\^(\d+(?:[–,]\d+)*)\^', main) == citation_sequence
    assert not re.search(r'Fig\. 2[de]|Fig\. 2a,c–e|solid curves in Fig\. 2b|dashed curves in Fig\. 2b', main)
    p.write_text(main)

    p = REPORT / 'supplementary_information_v8.md'
    supp = p.read_text()
    supp = replace_paragraph(supp, 'All results derive from one reconstructed nervous system.',
        'All results derive from one reconstructed nervous system. The current analyses contain 100 independent neural/intervention realizations for each of 79 conditions, giving 7,900 neural states and 158,000 nested controller trajectories. Movement summaries average 20 matched gust sequences before estimating uncertainty across realizations. The original 46-condition protocol was extended by 16 required-versus-shuffled typed-dose conditions, nine non-target dose conditions and eight random-wiring dose conditions. Each extension was recorded before its simulations, after earlier results were available. The eight random-wiring additions contribute 800 states in two separately archived batches. The preceding 7,100 states and their raw files are unchanged. Crossing rate was added as a descriptive endpoint during the primary batch. The original 56-location screen and earlier robustness tests remain separate exploratory analyses with their original sample counts.')
    supp = supp.replace('Figure 2 shows dose-dependent neural responses and virtual flight-gate crossings.',
        'Figure 2 shows dose-dependent neural responses, virtual flight-gate crossings and the retained cell-selection distance comparison.')
    supp = supp.replace('Positions follow the visual-and-descending selection rule and match Figure 2c–e.',
        'Positions follow motor-control priority and match the 100-cell row of Figure 2b.')
    supp = supp.replace('projection-first, or visual-and-descending-first.', 'projection-first, or motor-control priority.')
    supp = supp.replace('100 visual-and-descending-selected cells', '100 cells selected by motor-control priority')
    supp = supp.replace('This excess-dose experiment is separate from complete reconstruction in Figure 2b.',
        'This excess-dose experiment is separate from complete reconstruction in Figure 2b,c.')
    s2 = next(b for b in supp.split('\n\n') if b.startswith('**Supplementary Figure 2 |'))
    s2 = s2.replace('Neural responses and flight-gate crossings after replacement.', 'Cell selection, neural responses and flight-gate crossings after replacement.')
    if '**g**,' not in s2:
        s2 = s2.replace('Solid curves use original identities', '**g**, Absolute distance from the starting position under the three cell-selection rules and the correct-type and non-target graft rules. Random selection chooses missing cells with original available partners; it is distinct from random wiring in the main figure. Solid curves use original identities')
    supp = replace_paragraph(supp, '**Supplementary Figure 2 |', s2)
    p.write_text(supp)
    refresh_captions(main, supp)


def refresh_captions(main, supp):
    legends = {}
    for text in [main, supp]:
        for block in text.split('\n\n'):
            m = re.match(r'\*\*(Supplementary )?Figure (\d+) \|', block)
            if m:
                legends[('S' if m[1] else '') + m[2]] = block
    assert set(legends) == {'1', '2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'}
    captions = dict(re.findall(r'\*\*Supplementary Movie (\d+)\.\*\* ([^\n]+)', supp))
    assert set(captions) == set(map(str, range(1, 9)))
    (REPORT / 'figure_legends.json').write_text(json.dumps(legends, indent=2) + '\n')
    (REPORT / 'movie_captions.json').write_text(json.dumps(captions, indent=2) + '\n')


def promote_figure():
    metadata = json.loads((REVIEW / 'Fig2_v8_damage_baseline_review_metadata.json').read_text())
    for name, digest in metadata['outputs'].items():
        source = REVIEW / name
        assert sha(source) == digest
        shutil.copy2(source, ROOT / ('figures/v8/Fig2_v8_cell_replacement.' + source.suffix[1:]))
    for suffix in ['_source_values.csv', '_QA.json', '_metadata.json']:
        shutil.copy2(REVIEW / ('Fig2_v8_damage_baseline_review' + suffix),
                     ROOT / ('figures/v8/Fig2_v8_cell_replacement' + suffix))
    figqa = ROOT / 'figures/v8/figure_qa.json'
    q = json.loads(figqa.read_text())
    metadata.update(status='APPROVED_INTEGRATED', manuscript_and_package_updated_for_this_review=True,
                    panel_a_conditions=metadata['path_conditions']['100_cells'],
                    panel_a_anatomy=metadata['anatomy'], old_panel_b_moved_to='Supplementary Figure 5e',
                    earlier_cell_selection_distance_plot='Supplementary Figure 2g')
    if 'Three supplementary figures support the main results.' in (REPORT / 'supplementary_information_v8.md').read_text():
        metadata['old_panel_b_moved_to'] = 'Supplementary Figure 3d'
        metadata['earlier_cell_selection_distance_plot'] = 'Supplementary Figure 3a'
    q['figures']['Fig2_v8_cell_replacement'] = metadata
    for ext in ['png', 'pdf', 'svg']:
        rel = 'figures/v8/Fig2_v8_cell_replacement.' + ext
        q['output_sha256'][rel] = sha(ROOT / rel)
    q['input_sha256'].update(metadata['sources'])
    # The supplementary builder changes only to retain the former dose comparison.
    q['input_sha256']['code/v8/build_figures.py'] = sha(ROOT / 'code/v8/build_figures.py')
    q['figure2_approved_20261012'] = {'status': 'INTEGRATED_PENDING_DOCUMENT_QA', 'source': str(REVIEW.relative_to(ROOT)), 'primary_states': 7100, 'separate_additional_states': 800}
    figqa.write_text(json.dumps(q, indent=2) + '\n')


def integrate_tables():
    import pandas as pd
    primary = ROOT / 'results/v8/tables'
    out = ROOT / 'results/v8/integrated_20261012'
    out.mkdir(exist_ok=True)
    sources = [primary] + [ROOT / 'results/v8' / n for n in EXTENSIONS]
    records = {}
    for filename in ['per_realization.csv', 'gust_trials.csv']:
        frames = []
        for folder in sources:
            frames.append(pd.read_csv(folder / filename))
            records[str((folder / filename).relative_to(ROOT))] = sha(folder / filename)
        merged = pd.concat(frames, ignore_index=True, sort=False)
        keys = ['name', 'metric'] if filename == 'summary.csv' else ['name', 'rep'] + (['gust_seed'] if filename == 'gust_trials.csv' else [])
        assert not merged.duplicated(keys).any()
        merged.to_csv(out / filename, index=False)
    rows = pd.read_csv(out / 'per_realization.csv')
    trials = pd.read_csv(out / 'gust_trials.csv')
    assert len(rows) == 7900 and rows.name.nunique() == 79 and len(trials) == 158000
    from scipy.stats import t
    import numpy as np
    metrics = pd.read_csv(primary / 'summary.csv').metric.unique()
    summary = []
    for name, group in rows.groupby('name', sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            n = len(values); mean = values.mean(); sd = values.std(ddof=1); sem = sd / np.sqrt(n)
            half = t.ppf(.975, n-1) * sem
            low, high = np.quantile(values, [.025, .975])
            summary.append(dict(name=name, metric=metric, n=n, mean=mean, sd=sd, sem=sem,
                                ci_low=mean-half, ci_high=mean+half, p025=low, p975=high))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    manifest = {'status': 'COMPLETE_PENDING_INDEPENDENT_QA', 'primary_states': 7100, 'additional_review_states': 800,
                'conditions': 79, 'neural_states': 7900, 'nested_gust_trials': 158000,
                'separate_post_primary_extensions': EXTENSIONS, 'source_sha256': records,
                'output_sha256': {p.name: sha(p) for p in out.glob('*.csv')}}
    (out / 'integration_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status = json.loads((REPORT / 'text_build_status.json').read_text())
    status['primary_analysis'] = status.get('primary_analysis', status['analysis'])
    status['analysis'] = {**status['analysis'], 'states': 7900, 'expected_states': 7900,
                          'conditions': 79, 'nested_gust_trials': 158000,
                          'primary_states': 7100, 'separate_review_extension_states': 800}
    status['figure2_integration'] = 'Approved three-panel figure, damage-baseline normalization and complete random-wiring dose series'
    status['main_text_words'] = len((REPORT / 'manuscript_v8.md').read_text().split('## Methods')[0].split())
    (REPORT / 'text_build_status.json').write_text(json.dumps(status, indent=2) + '\n')


def main():
    update_sources()
    promote_figure()
    integrate_tables()
    print('Approved Figure 2 and source text integrated; 79 conditions, 7,900 states, 158,000 nested trajectories.')


if __name__ == '__main__':
    main()
