# -*- coding: utf-8 -*-
import os, time, requests

OUT = os.path.join("支撑材料", "02_文献资料")
os.makedirs(OUT, exist_ok=True)

HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

JOBS = [
    ("zugno2015robust", "Zugno2015_robust_energy_reserve.pdf",
     "https://orbit.dtu.dk/files/54304976/tr13_05_Zugno_M_1.pdf",
     "https://orbit.dtu.dk/"),
]

for key, fname, url, ref in JOBS:
    h = dict(HDR)
    if ref:
        h["Referer"] = ref
    try:
        r = requests.get(url, headers=h, timeout=90, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        print(key, r.status_code, ct, len(r.content))
        if r.status_code == 200 and (r.content[:5] == b"%PDF-" or "pdf" in ct.lower()):
            p = os.path.join(OUT, fname)
            with open(p, "wb") as f:
                f.write(r.content)
            print("   SAVED", p, len(r.content))
        else:
            print("   NOT PDF")
    except Exception as e:
        print(key, "FAIL", str(e)[:120])
    time.sleep(1)

