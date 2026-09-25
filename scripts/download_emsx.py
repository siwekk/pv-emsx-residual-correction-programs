"""Download and checksum the public EMSx Zenodo release on the compute server."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from urllib.request import Request, urlopen
from emsx_paths import WORK_ROOT

RECORD = "https://zenodo.org/api/records/5510400"
def fetch(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "pv-qgbm-research/0.1"}), timeout=120) as response: return response.read()
def md5_file(path: Path) -> str:
    digest=hashlib.md5()
    with path.open("rb") as source:
        while block:=source.read(1024*1024): digest.update(block)
    return digest.hexdigest()
def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=WORK_ROOT);a=p.parse_args();root=a.project_root.resolve();target=root/"data"/"raw"/"emsx";target.mkdir(parents=True,exist_ok=True)
    record=json.loads(fetch(RECORD));(target/"zenodo_record.json").write_text(json.dumps(record,indent=2)+"\n");manifest=[]
    for entry in record["files"]:
        name=entry["key"];path=target/name;expected=entry.get("checksum","").replace("md5:","")
        if path.exists() and md5_file(path)==expected:
            manifest.append({"name":name,"status":"existing_verified","bytes":path.stat().st_size,"md5":expected});continue
        temporary=path.with_suffix(path.suffix+".part")
        for attempt in range(3):
            try:
                with urlopen(Request(entry["links"]["self"],headers={"User-Agent":"pv-qgbm-research/0.1"}),timeout=120) as source, temporary.open("wb") as output:
                    while block:=source.read(1024*1024): output.write(block)
                digest=md5_file(temporary)
                if digest!=expected: raise RuntimeError(f"checksum mismatch for {name}")
                temporary.replace(path);manifest.append({"name":name,"status":"downloaded_verified","bytes":path.stat().st_size,"md5":digest});break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt==2: raise
                time.sleep(5*(attempt+1))
    (target/"download_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"files":len(manifest),"bytes":sum(x["bytes"] for x in manifest)},indent=2))
if __name__=="__main__":main()
