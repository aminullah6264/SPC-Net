#!/bin/bash
# Every 15 min: score finished no-point-loss runs and write NOLPT_PROGRESS.md.
# Usage: screen -dmS nl_watch bash watch_nolpt.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/av543f/anaconda3/etc/profile.d/conda.sh; conda activate open_clip
R=/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results; O=$R/outputs_textfix
while true; do
  done_runs=$(for d in $O/base_af_* $O/af_* $O/dec_*; do [ -f $d/results.yaml ] && [ ! -f $d/unified_metrics.csv ] && basename $d; done | paste -sd,)
  [ -n "$done_runs" ] && nice -n 5 python score_unified.py --root $O --variants "$done_runs" > /dev/null 2>&1
  nice -n 5 python - << 'EOF'
import csv, os, statistics as st
O='/mnt/data/projects/CAD_Adaptation/CLIPSAM_new_results/outputs_textfix'
D=['DUTS-TE','DUT-O','ECSSD','HKU-IS','PASCAL-S-full','SIP-full']
def get(v):
    p=f'{O}/{v}/unified_metrics.csv'
    if not os.path.exists(p): return None
    r={x['dataset']:float(x['Sm']) for x in csv.DictReader(open(p))}
    if 'SIP-full' not in r:  # older runs: SIP-full scored separately
        q=f'{O}/../outputs_sipfull/{v}/unified_metrics.csv'
        if os.path.exists(q): r['SIP-full']=float(next(csv.DictReader(open(q)))['Sm'])
    return r
def row(name, vs):
    rs=[get(v) for v in vs]; rs=[r for r in rs if r]
    if not rs: return f'| {name} | ' + ' | '.join(['…']*6) + f' | 0/{len(vs)} |'
    c=[]
    for d in D:
        a=[r[d] for r in rs if d in r]
        c.append('–' if not a else (f'{st.mean(a):.4f}' + (f'±{st.stdev(a):.4f}' if len(a)>1 else '')))
    return f'| {name} | ' + ' | '.join(c) + f' | {len(rs)}/{len(vs)} |'
s3=lambda b:[f'{b}_s{s}' for s in (42,1,2)]
L=['# No-point-loss runs: S_m progress (auto-updated)','',
   '| Config | DUTS-TE | DUT-O | ECSSD | HKU-IS | PASCAL-S | SIP (929) | runs done |','|'+'---|'*8,
   row('Baseline (single prompt, simple dec)', s3('baseline')),
   row('+ AF (single prompt)', ['base_af_s42']),
   row('+ MP (bank, no AF)', ['plus_mp_s42']),
   row('+ MP + AF (ours)', s3('plus_af')),
   row('ours, K=1', ['af_K1_s42']), row('ours, K=5', ['af_K5_s42']),
   row('ours, random wording', ['af_random_s42']), row('ours, diverse wording', ['af_diverse_s42']),
   row('Dec, max, residual on', ['plus_dec_s42','dec_max_s1','dec_max_s2']),
   row('Dec, max, residual off', s3('dec_max_noresid')),
   row('Dec, attention, residual on', s3('dec_attention')),
   row('Dec, attention, residual off', s3('dec_attention_noresid'))]
open('/home/av543f/deep_models/CLIPSAM/NOLPT_PROGRESS.md','w').write('\n'.join(L)+'\n')
EOF
  [ $(ls $O/{base_af,af_,dec_}*/unified_metrics.csv 2>/dev/null | wc -l) -ge 16 ] && break
  sleep 900
done
