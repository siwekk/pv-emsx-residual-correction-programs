"""Correct recoverable legacy WIS values and withdraw unverifiable summaries.

Only the small, checked-in legacy JSON/CSV summaries are migrated. Full predictions
are never changed, and all old values survive under explicit legacy-invalid names.
The program is idempotent and records every correction and withdrawal.
"""
import csv
import json
from pathlib import Path
from emsx_paths import CODE_ROOT
from emsx_probabilistic import SCORING_VERSION

WIS_KEYS={'wis_kwh','weighted_interval_score_kwh','mean_wis_kwh'}

def migrate_dict(node,trail,changes):
    if isinstance(node,list):
        for i,item in enumerate(node):migrate_dict(item,trail+[str(i)],changes)
    elif isinstance(node,dict):
        for key in list(node):
            if key.startswith('legacy_invalid_'):continue
            value=node[key]
            if key in WIS_KEYS and value is not None:
                if 'legacy_invalid_'+key in node:
                    continue
                node['legacy_invalid_'+key]=value
                pb=node.get('mean_pinball_kwh')
                outer=node.get('interval_score_90_kwh',node.get('central_90_interval_score_kwh'))
                if isinstance(pb,(int,float)) and isinstance(outer,(int,float)):
                    node[key]=(11*pb-.05*outer)/4.5
                    status='corrected_from_unrounded_pinball_and_outer_interval_score'
                else:
                    node[key]=None
                    status='withdrawn_full_predictions_required'
                changes.append({'field':'/'.join(trail+[key]),'status':status})
            else:migrate_dict(value,trail+[key],changes)

def main():
    log=[]
    roots=[CODE_ROOT/'results'/x for x in ['article_summaries','frozen','moirai_96']]
    for root in roots:
        for path in sorted(root.glob('*.json')):
            data=json.loads(path.read_text(encoding='utf-8'))
            if data.get('wis_migration_version')==SCORING_VERSION:continue
            changes=[];migrate_dict(data,[],changes)
            if changes:
                data['wis_migration_version']=SCORING_VERSION
                data['wis_migration']=changes
                data['evaluation_note']='Original partitions have staggered dates across systems; use results/round1 for the common calendar validation.'
                path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
                log.append({'file':path.relative_to(CODE_ROOT).as_posix(),'changes':changes})
        for path in sorted(root.glob('*.csv')):
            with path.open(encoding='utf-8',newline='') as stream:
                reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
            keys=[key for key in fields if key in WIS_KEYS]
            if not keys or 'wis_status' in fields:continue
            for key in keys:
                fields.append('legacy_invalid_'+key)
                for row in rows:row['legacy_invalid_'+key]=row[key];row[key]=''
            fields.append('wis_status')
            for row in rows:row['wis_status']='withdrawn_full_predictions_required'
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            log.append({'file':path.relative_to(CODE_ROOT).as_posix(),'changes':[{'field':key,'status':'withdrawn_full_predictions_required'} for key in keys]})
    # The complete-test analysis uses the same prediction files as the full
    # foundation audit, so those four WIS values can be linked exactly.
    audit=CODE_ROOT/'results/article_summaries/emsx_daily_foundation_probabilistic.json'
    summary=CODE_ROOT/'results/article_summaries/emsx_daily_manuscript_analyses.json'
    values=json.loads(audit.read_text());data=json.loads(summary.read_text())
    names={'Chronos-2 target-only':'chronos2_target_only','Chronos-2 + vendor':'chronos2_vendor_covariate','Moirai 2.0 target-only':'moirai2_target_only','Moirai 2.0 + vendor':'moirai2_vendor_covariate'}
    for name,key in names.items():
        data['probabilistic_metrics'][name]['wis_kwh']=values['methods'][key]['weighted_interval_score_kwh']
    for change in data.get('wis_migration',[]):
        if change['field'].startswith('probabilistic_metrics/'):
            change['status']='corrected_from_matching_full_prediction_audit'
    summary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report=CODE_ROOT/'results/SCORING_CORRECTIONS.json'
    prior=json.loads(report.read_text()) if report.exists() else {'scoring_version':SCORING_VERSION,'note':'Legacy-invalid values are retained only for provenance, never as valid WIS. Corrected round1 results are separate.','files':[]}
    entries={entry['file']:entry for entry in prior['files']}
    entries.update({entry['file']:entry for entry in log})
    for entry in entries.values():
        if entry['file']==summary.relative_to(CODE_ROOT).as_posix():
            for change in entry['changes']:
                if change['field'].startswith('probabilistic_metrics/'):
                    change['status']='corrected_from_matching_full_prediction_audit'
    prior['files']=[entries[key] for key in sorted(entries)]
    report.write_text(json.dumps(prior,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'migrated_files':len(log),'scoring_version':SCORING_VERSION}))

if __name__=='__main__':main()
