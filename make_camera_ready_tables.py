#!/usr/bin/env python3
"""Generate camera-ready LaTeX tables from textfix_numbers.json (2026-11).

Red (\\rev) = value changed vs. the submitted paper, purple (\\chk) = co-authors
please verify. Outputs go to sec_camera_ready/tab_*.tex.
"""
import json, os, statistics as st

RES = '/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/textfix_numbers.json'
OUT = '/home/av543f/deep_models/CLIPSAM/Openvocab-SD/LVR-WACV2027-submission/sec_camera_ready'
J = json.load(open(RES))
D6 = ['DUTS-TE', 'DUT-O', 'ECSSD', 'HKU-IS', 'PASCAL-S-full', 'SIP-full']
D5 = D6[:5]
HDR6 = ['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SIP']


def f3(x):
    return f'{x:.3f}'.lstrip('0') if x < 1 else f'{x:.3f}'


def ms(base, d, m, seeds=(42, 1, 2)):
    a = [J[f'{base}_s{s}'][d][m] for s in seeds]
    return st.mean(a), st.stdev(a)


def pm(mean, sd):
    return f'{f3(mean)}{{\\scriptsize$\\pm${sd:.3f}'.replace('0.', '.', 1) + '}'


# ── Table 1: RGB SOTA ────────────────────────────────────────────────────────
# Competitor values: SOURCING_REPORT.md sec. 6b ("WACV table", all cells sourced).
# (Sm, F, MAE) per dataset in D5 order. F: max-F (DU = Dulam CVPRW'25 Table 2,
# MDSAM = own paper) or variant not stated (dagger; STNet paper Table 1).
COMP = [
    ('PiCANet', 'picanet2018', '', [(.869, .860, .051), (.832, .803, .065), (.917, .935, .046), (.904, .918, .043), (.852, .868, .078)]),
    ('BASNet', 'basnet2019', '', [(.866, .860, .048), (.836, .805, .056), (.916, .942, .037), (.909, .928, .032), (.834, .860, .079)]),
    ('F3Net', 'f3net2020', '', [(.888, .891, .035), (.838, .813, .053), (.924, .945, .033), (.917, .937, .028), (.857, .882, .064)]),
    ('LDF', 'ldf2020', '', [(.892, .898, .034), (.838, .820, .051), (.924, .950, .034), (.919, .939, .027), (.859, .887, .062)]),
    ('CSF-R2', 'csnet2020', '', [(.890, .890, .037), (.838, .815, .055), (.930, .950, .033), (.921, .935, .030), (.862, .886, .069)]),
    ('VST', 'vst2021', '', [(.896, .890, .037), (.850, .825, .058), (.932, .951, .033), (.928, .942, .029), (.871, .890, .062)]),
    ('PSGLoss', 'psgloss2021', '', [(.883, .886, .036), (.831, .811, .052), (.925, .949, .031), (.919, .938, .027), (.858, .886, .063)]),
    ('RCSB', 'rcsb2022', '', [(.878, .889, .035), (.820, .810, .045), (.921, .944, .033), (.918, .938, .027), (.857, .886, .061)]),
    ('EDN', 'edn2022', '', [(.892, .895, .035), (.846, .828, .048), (.927, .951, .032), (.924, .941, .026), (.860, .891, .065)]),
    ('MENet', 'menet2023', '', [(.905, .913, .028), (.850, .834, .045), (.928, .955, .031), (.927, .948, .023), (.868, .901, .057)]),
    ('DC-Net', 'sod_dcnet', '$^\\dagger$', [(.896, .829, .035), (.849, .774, .053), (.924, .882, .034), (.924, .910, .027), (.857, .769, .066)]),
    ('ADMNet', 'sod_admnet', '$^\\dagger$', [(.835, .753, .057), (.815, .722, .063), (.889, .870, .051), (.889, .871, .040), (.798, .731, .094)]),
    ('BDFGNet', 'sod_bdfgnet', '$^\\dagger$', [(.884, .794, .049), (.826, .723, .058), (.908, .891, .042), (.909, .882, .042), (.844, .791, .069)]),
    ('STNet', 'sod_stnet', '$^\\dagger$', [(.860, .813, .044), (.823, .752, .056), (.909, .914, .040), (.902, .898, .033), (.840, .821, .072)]),
    ('MDSAM', 'mdsam2024', '', [(.920, .937, .024), (.878, .887, .039), (.948, .974, .021), (.941, .963, .019), (.882, .907, .052)]),
]
# Submitted paper table (4 datasets, no PASCAL-S); "CSNet" row compared to CSF-R2.
SUB = {
    'PiCANet': [(.869, .860, .051), (.832, .803, .065), (.917, .935, .046), (.904, .918, .043)],
    'BASNet': [(.876, .860, .048), (.836, .805, .056), (.916, .942, .037), (.909, .928, .032)],
    'F3Net': [(.888, .891, .035), (.838, .813, .053), (.924, .945, .033), (.917, .937, .028)],
    'LDF': [(.892, .898, .034), (.839, .820, .052), (.925, .950, .034), (.919, .939, .027)],
    'CSF-R2': [(.880, .877, .040), (.833, .805, .055), (.919, .938, .040), (.921, .935, .030)],
    'VST': [(.896, .890, .037), (.850, .825, .058), (.932, .951, .033), (.928, .942, .029)],
    'PSGLoss': [(.887, .890, .036), (.841, .820, .052), (.924, .948, .033), (.919, .938, .027)],
    'EDN': [(.892, .895, .035), (.849, .828, .049), (.927, .951, .032), (.924, .941, .026)],
    'MENet': [(.905, .911, .028), (.850, .834, .045), (.928, .952, .031), (.927, .948, .023)],
    'DC-Net': [(.906, .910, .028), (.851, .833, .046), (.930, .953, .030), (.924, .910, .027)],
    'ADMNet': [(.888, .885, .036), (.840, .818, .052), (.923, .946, .034), (.889, .871, .040)],
    'BDFGNet': [(.893, .896, .033), (.845, .824, .048), (.927, .950, .031), (.909, .882, .042)],
    'STNet': [(.900, .905, .030), (.848, .830, .047), (.929, .952, .030), (.902, .898, .033)],
}
OURS = 'plus_af_s42'


def table1():
    ours = [(J[OURS][d]['Sm'], J[OURS][d]['F05'], J[OURS][d]['MAE']) for d in D5]
    allrows = [r[3] for r in COMP] + [ours]
    best_sm = [max(round(r[i][0], 3) for r in allrows) for i in range(5)]
    best_mae = [min(round(r[i][2], 3) for r in allrows) for i in range(5)]

    def cell(v, i, k, changed):
        s = f3(v)
        if (k == 0 and round(v, 3) == best_sm[i]) or (k == 2 and round(v, 3) == best_mae[i]):
            s = f'\\textbf{{{s}}}'
        return f'\\rev{{{s}}}' if changed else s

    L = []
    for name, key, mark, vals in COMP:
        sub = SUB.get(name)
        cells = []
        for i, trip in enumerate(vals):
            for k, v in enumerate(trip):
                changed = sub is None or i >= 4 or abs(sub[i][k] - v) > 1e-9
                cells.append(cell(v, i, k, changed))
        label = f'{name}{mark}~\\cite{{{key}}}'
        if name == 'CSF-R2':
            label = f'\\chk{{CSF-R2}}~\\cite{{{key}}}'
        if sub is None:
            label = f'\\rev{{{label}}}'
        if name == 'MDSAM':
            label = f'\\rev{{MDSAM~\\cite{{{key}}} (SAM, 512$^2$)}}'
        L.append(f'    {label} & ' + ' & '.join(cells) + ' \\\\')
    oc = [cell(v, i, k, True) for i, t in enumerate(ours) for k, v in enumerate(t)]
    body = '\n'.join(L)
    return f"""\\begin{{table*}}[t]
  \\centering
  \\caption{{Comparison with RGB SOD methods. Best $S_m$ and $\\mathcal{{M}}$ per column in
    \\textbf{{bold}}. \\rev{{Competitor values are taken from published tables (see text);
    their $F_\\beta$ is the maximum F-measure, or a variant not stated in the source
    ($^\\dagger$), while ours is $F_\\beta$ at a fixed threshold of $0.5$, so the
    $F_\\beta$ column is not directly comparable and is not bolded. \\methodname{{}}:
    Baseline+MP+AF, CLIP ViT-L/14 at $336^2$, seed 42; three-seed means are in
    \\cref{{tab:ablation}}.}}}}
  \\label{{tab:sota}}
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\scriptsize
  \\begin{{tabular}}{{l ccc ccc ccc ccc ccc}}
    \\toprule
    & \\multicolumn{{3}}{{c}}{{DUTS-TE}} & \\multicolumn{{3}}{{c}}{{DUT-OMRON}}
    & \\multicolumn{{3}}{{c}}{{ECSSD}} & \\multicolumn{{3}}{{c}}{{HKU-IS}} & \\multicolumn{{3}}{{c}}{{\\rev{{PASCAL-S}}}} \\\\
    \\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\\cmidrule(lr){{11-13}}\\cmidrule(lr){{14-16}}
    Method & $S_m$ & $F_\\beta$ & $\\mathcal{{M}}$ & $S_m$ & $F_\\beta$ & $\\mathcal{{M}}$
           & $S_m$ & $F_\\beta$ & $\\mathcal{{M}}$ & $S_m$ & $F_\\beta$ & $\\mathcal{{M}}$
           & $S_m$ & $F_\\beta$ & $\\mathcal{{M}}$ \\\\
    \\midrule
{body}
    \\midrule
    \\methodname{{}} (ours) & {' & '.join(oc)} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""


# ── Table 2: SIP vs RGB-D ────────────────────────────────────────────────────
# VST++ Table 4 (V+): Sm / max F / max E / MAE on all 929 SIP images.
SIPC = [
    ('S2MA', 'sod_s2ma', (.872, .877, .919, .058), None),
    ('BBS-Net', 'sod_bbsnet', (.879, .884, .922, .055), None),
    ('JL-DCF', 'sod_jldcf', (.885, .894, .931, .049), None),
    ('SPNet', 'sod_spnet', (.894, .904, .933, .043), None),
    ('SPSN', 'sod_spsn', (.892, .900, .936, .043), None),
    ('VST', 'vst2021', (.904, .915, .944, .040), (.904, .895, .944, .040)),
    ('EBMGSOD', 'sod_ebm', (.902, .918, .944, .037), (.902, .890, .943, .037)),
]


def table2():
    L = []
    for name, key, v, sub in SIPC:
        if sub is None:
            cells = [f'\\rev{{{f3(x)}}}' for x in v]
            label = f'\\rev{{{name}~\\cite{{{key}}}}}'
        else:
            cells = [f3(x) if abs(x - y) < 1e-9 else f'\\chk{{{f3(x)}}}' for x, y in zip(v, sub)]
            label = f'{name}~\\cite{{{key}}}'
        L.append(f'    {label} & RGB-D & ' + ' & '.join(cells) + ' \\\\')
    o = J[OURS]['SIP-full']
    m_sm, s_sm = ms('plus_af', 'SIP-full', 'Sm')
    m_mae, s_mae = ms('plus_af', 'SIP-full', 'MAE')
    ours = ' & '.join(f'\\rev{{\\textbf{{{f3(o[k])}}}}}' for k in ('Sm', 'F05', 'meanE', 'MAE'))
    return f"""\\begin{{table}}[t]
  \\centering
  \\caption{{SIP (all 929 images): \\methodname{{}} (RGB only, trained on DUTS-TR) vs.\\
    RGB-D methods (RGB{{+}}depth, trained on RGB-D data). \\rev{{RGB-D values are from
    VST++~\\cite{{vstpp2024}} (max $F_\\beta$, max $E_m$); ours are $F_\\beta$ at threshold
    $0.5$ and mean $E_m$ (our max $F_\\beta$/max $E_m$: {f3(o['maxF'])}/{f3(o['maxE'])}).
    Three seeds: $S_m$ {f3(m_sm)}$\\pm${s_sm:.3f}, $\\mathcal{{M}}$ {f3(m_mae)}$\\pm${s_mae:.3f}.
    Purple: changed from the submitted version; please verify.}}}}
  \\label{{tab:sip_rgbd}}
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\footnotesize
  \\begin{{tabular}}{{llcccc}}
    \\toprule
    Method & Input & $S_m\\!\\uparrow$ & $F_\\beta\\!\\uparrow$ & $E_m\\!\\uparrow$ & $\\mathcal{{M}}\\!\\downarrow$ \\\\
    \\midrule
{chr(10).join(L)}
    \\midrule
    \\methodname{{}} (ours) & RGB & {ours} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


# ── Table 3: cumulative ablation (S_m and MAE, 6 datasets) ───────────────────
def row(label, v, red=True):
    c = [f3(J[v][d]['Sm']) for d in D6] + [f3(J[v][d]['MAE']) for d in D6]
    return f'    {label} & ' + ' & '.join(c) + ' \\\\'


def row_seeds(label, base):
    c = [pm(*ms(base, d, 'Sm')) for d in D6] + [f3(ms(base, d, 'MAE')[0]) for d in D6]
    return f'    {label} & ' + ' & '.join(c) + ' \\\\'


def table3():
    rows = [row('Baseline', 'baseline_s42'), row('$+$ MP', 'plus_mp_s42'),
            row('$+$ AF', 'plus_af_s42'), row('$+$ BiSAM', 'plus_bisam_s42'),
            row('$+$ Dec', 'plus_dec_s42'), row('$+$ Lpt', 'full_max_s42')]
    seeds = [row_seeds('Baseline', 'baseline'), row_seeds('$+$ AF', 'plus_af'),
             row_seeds('$+$ AF, frozen enc.', 'frozen_plus_af')]
    hdr = ' & '.join(f'\\multicolumn{{1}}{{c}}{{{h}}}' for h in ['DT', 'DO', 'EC', 'HK', 'PS', 'SIP'])
    return f"""\\begin{{table*}}[t]
  \\centering
  \\begin{{revpar}}
  \\caption{{Cumulative ablation (each row adds one component to the row above).
    Baseline: CLIP ViT-L/14 + SAM blocks + ESC blocks with the single caption, threshold-only
    PPG, image-to-token SAM attention, U-Net decoder, BCE loss. MP: 10-prompt bank (max);
    AF: adaptive fallback; BiSAM: bidirectional SAM attention; Dec: ASPP/FPN decoder with
    compound loss and deep supervision; Lpt: point loss ($\\gamma_{{\\textit{{pt}}}}{{=}}0.3$).
    Top: seed 42. Bottom: mean$\\pm$std over seeds 42, 1, 2 ($\\mathcal{{M}}$: mean).
    ``Frozen enc.'': CLIP visual encoder and SAM blocks frozen.
    DT/DO/EC/HK/PS: DUTS-TE, DUT-OMRON, ECSSD, HKU-IS, PASCAL-S.}}
  \\label{{tab:ablation}}
  \\setlength{{\\tabcolsep}}{{2.4pt}}
  \\scriptsize
  \\begin{{tabular}}{{l cccccc cccccc}}
    \\toprule
    & \\multicolumn{{6}}{{c}}{{$S_m\\uparrow$}} & \\multicolumn{{6}}{{c}}{{$\\mathcal{{M}}\\downarrow$}} \\\\
    \\cmidrule(lr){{2-7}}\\cmidrule(lr){{8-13}}
    Variant & {hdr} & {hdr} \\\\
    \\midrule
{chr(10).join(rows)}
    \\midrule
{chr(10).join(seeds)}
    \\bottomrule
  \\end{{tabular}}
  \\end{{revpar}}
\\end{{table*}}
"""


# ── Table 4: prompt-side analysis on the full model (S_m) ────────────────────
def table4():
    a = []
    for agg, lab in (('max', 'Max'), ('attention', 'Attention')):
        for suf, r in (('', 'on'), ('_noresid', 'off')):
            base = f'full_{agg}{suf}'
            a.append(f'    {lab} & {r} & ' + ' & '.join(pm(*ms(base, d, 'Sm')) for d in D6) + ' \\\\')
    b = []
    for lab, v in (('$K{=}1$', 'K1_s42'), ('$K{=}5$', 'K5_s42'), ('$K{=}10$, curated', 'full_max_s42'),
                   ('$K{=}10$, random', 'random_bank_s42'), ('$K{=}10$, diverse', 'diverse_bank_s42')):
        b.append(f'    \\multicolumn{{2}}{{l}}{{{lab}}} & ' + ' & '.join(f3(J[v][d]['Sm']) for d in D6) + ' \\\\')
    return f"""\\begin{{table*}}[t]
  \\centering
  \\begin{{revpar}}
  \\caption{{Prompt-side analysis on the correlation-aware (full) model, $S_m$.
    Top: aggregation mode and prompt residual (\\cref{{eq:residual}}), mean$\\pm$std over
    seeds 42, 1, 2. Bottom: bank size and wording (max aggregation, residual on, seed 42).}}
  \\label{{tab:prompt}}
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\footnotesize
  \\begin{{tabular}}{{ll cccccc}}
    \\toprule
    Aggregation & Residual & {' & '.join(HDR6)} \\\\
    \\midrule
{chr(10).join(a)}
    \\midrule
{chr(10).join(b)}
    \\bottomrule
  \\end{{tabular}}
  \\end{{revpar}}
\\end{{table*}}
"""


for name, fn in (('tab_sota', table1), ('tab_sip', table2), ('tab_ablation', table3), ('tab_prompt', table4)):
    with open(os.path.join(OUT, f'{name}.tex'), 'w') as f:
        f.write('% AUTO-GENERATED by make_camera_ready_tables.py from textfix_numbers.json\n' + fn())
    print('wrote', name)
