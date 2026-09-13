# -*- coding: utf-8 -*-
import os, requests

OUT = os.path.join("支撑材料", "02_文献资料")
os.makedirs(OUT, exist_ok=True)
HDR = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
       "Accept": "application/pdf,*/*"}

JOBS = [
 ("Li2017_industrial_microgrid", "https://link.springer.com/content/pdf/10.1186/s41601-017-0040-6.pdf"),
 ("Parhizi2015_state_of_art_microgrids", "https://ieeexplore.ieee.org/iel7/6287639/7125200/07114972.pdf"),
]
for name, url in JOBS:
    try:
        r = requests.get(url, headers=HDR, timeout=90, allow_redirects=True)
        ok = r.status_code == 200 and (r.content[:5] == b"%PDF-" or "pdf" in r.headers.get("Content-Type","").lower())
        print(name, r.status_code, r.headers.get("Content-Type"), len(r.content), "PDF" if ok else "no")
        if ok:
            p = os.path.join(OUT, name + ".pdf")
            open(p, "wb").write(r.content)
            print("   SAVED", len(r.content))
    except Exception as e:
        print(name, "FAIL", str(e)[:120])

