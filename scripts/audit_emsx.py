"""Audit EMSx inputs, actual PV, and 96-step forecast trajectories."""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
import pandas as pd
from emsx_paths import WORK_ROOT
def main():
 p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=WORK_ROOT);a=p.parse_args();root=a.project_root.resolve();raw=root/"data"/"raw"/"emsx";rows=[]
 for path in sorted(raw.glob("[0-9]*.csv.gz"),key=lambda x:int(x.stem.split(".")[0])):
  sample=pd.read_csv(path,sep=";",nrows=20000);cols=list(sample.columns);pv=[c for c in cols if c.startswith("pv_")];actual=[c for c in cols if "actual" in c.lower() and "pv" in c.lower()];time=[c for c in cols if c.lower() in {"time","timestamp","date","datetime"}]
  rows.append({"site":path.name,"bytes":path.stat().st_size,"columns":len(cols),"pv_forecast_columns":len(pv),"actual_pv_columns":actual,"timestamp_columns":time,"sample_missing_max":float(sample.isna().mean().max()),"sample_rows":int(len(sample)),"first_row":{k:str(sample.iloc[0][k]) for k in cols[:5]}})
 meta=pd.read_csv(raw/"metadata.csv") if (raw/"metadata.csv").exists() else None
 out={"source":"EMSx Zenodo record 5510400","sites":len(rows),"site_rows":rows,"metadata_columns":list(meta.columns) if meta is not None else [],"notes":{"forecast_semantics":"pv_00 through pv_95 are vendor forecasts for successive 15-minute PV-energy intervals.","scope":"External operational forecast-correction benchmark, not a location-aware NWP validation dataset."}}
 destination=root/"data"/"processed"/"emsx_audit.json";destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"sites":len(rows),"first_site":rows[0] if rows else None},indent=2))
if __name__=="__main__":main()
