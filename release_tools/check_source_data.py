"""Recalculate risk indices and check manuscript source data without raw AIS.

This archival verification tool was written for the release. It reproduces
M0-M3 from archived temporal features; it does not replace the missing original
raw-input risk pipeline or the final forecasting-training driver.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def check(data_root):
 p=Path(data_root)
 checks=[]
 def record(name,passed,detail):checks.append({'check':name,'passed':bool(passed),'detail':detail})
 annual=pd.read_csv(p/'city_panels/emission_city_annual_2019_2025.csv')
 record('city_year_coverage',set(annual.year)==set(range(2019,2026)) and annual.groupby('year').node_id.nunique().eq(1425).all(),annual.groupby('year').node_id.nunique().to_dict())
 record('city_year_keys_unique',not annual.duplicated(['node_id','year']).any(),len(annual))
 calc=2*annual.SO2_t+5*annual.PM_t+annual.NOx_t
 error=float((calc-annual.composite_t).abs().max())
 record('health_weighted_emissions',np.allclose(calc,annual.composite_t,rtol=1e-10,atol=1e-8),{'max_absolute_error_t':error})
 d=pd.read_csv(p/'city_panels/SPRI_city_annual_2019_2025.csv')
 ablation=pd.read_csv(p/'city_panels/SPRI_model_ablation_city_annual_2019_2025.csv')
 manifest=json.loads((p/'provenance/time_information_ablation_manifest.json').read_text(encoding='utf-8-sig'))
 risk_manifest=json.loads((p/'provenance/risk_method_manifest.json').read_text(encoding='utf-8-sig'))
 mu=float(risk_manifest['mu']); lam=float(risk_manifest['lambda'])
 for mode,hist,future in [('m0',0,0),('m1',1,0),('m2',0,1),('m3',1,1)]:
  dim={g:np.zeros(len(d)) for g in ['E','S','AC']}
  for g in dim:
   for var,weight in manifest['critic_weights'][g].items():
    x=d[var+'_now_z']+hist*mu*d[var+'_hist_z']+future*lam*d[var+'_future_z']
    low,high=manifest['evaluation_endpoints'][var+'_eval']
    dim[g]+=weight*np.clip((x-low)/(high-low),0,1)
  result=d[['node_id','year']].assign(reproduced=100*np.cbrt(dim['E']*dim['S']*(1-dim['AC'])))
  result=result.merge(ablation[['node_id','year','spri_'+mode]],on=['node_id','year'],validate='1:1')
  error=float((result.reproduced-result['spri_'+mode]).abs().max())
  record('reproduce_'+mode,error<1e-9,{'rows':len(result),'max_absolute_error':error})
 err=float((100*np.cbrt(d.E*d.S*d.LAC)-d.SPRI).abs().max())
 record('full_risk_component_identity',err<1e-9,{'max_absolute_error':err})
 last=d[d.year.eq(2025)]
 n=int(last.quadrant.eq('emerging').sum())
 record('emerging_city_count_2025',n==397,{'cities':n,'share':n/1425})
 forecast=pd.read_csv(p/'city_panels/forecast_city_annual_2026_2027.csv')
 record('forecast_panel',set(forecast.year)=={2026,2027} and len(forecast)==2850,{'rows':len(forecast),'years':sorted(forecast.year.unique().tolist())})
 policy=pd.read_csv(p/'policy_tables/scenario_effect_metrics_all.csv')
 s7=policy[policy.scenario.eq('S7')].iloc[0]
 mass=float(s7.global_weighted_reduction_share*100)
 exposure=float(s7.delta_pwe/s7.global_baseline_pwe*100)
 record('S7_reported_percentages',round(mass,2)==2.57 and round(exposure,2)==16.64,{'weighted_emissions_percent':mass,'population_weighted_proxy_percent':exposure})
 return {'passed':all(x['passed'] for x in checks),'scope':'Source-data integrity and numerical reproduction from archived temporal feature columns; not full raw-AIS or training replication','checks':checks}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 report=check(args.data_root)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),'output':str(args.output)}))
 raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
