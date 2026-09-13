import json, urllib.request, urllib.parse, time

MAIL = "mathmodeling.team26@outlook.com"
REF = [
 ("antonanzas2016review", "10.1016/j.solener.2016.06.069"),
 ("silvente2015rolling", "10.1016/j.apenergy.2015.05.090"),
 ("elkazaz2020ems", "10.1016/j.ijepes.2019.105483"),
 ("zugno2015robust", "10.1016/j.ejor.2015.05.081"),
 ("farzin2017stochastic", "10.1109/TSG.2016.2598678"),
 ("khodabakhsh2016cvar", "10.1109/TSTE.2016.2543024"),
 ("rockafellar2000cvar", "10.21314/JOR.2000.038"),
 ("khouja1999singleperiod", "10.1016/S0305-0483(99)00017-1"),
 ("petruzzi1999pricing", "10.1287/opre.47.2.183"),
 ("hu2021mpc", "10.1016/j.apenergy.2020.116009"),
 ("cominesi2018two", "10.1109/TCST.2017.2754981"),
 ("schwenzer2021mpc", "10.1016/j.est.2021.103111"),
]

out = []
for key, doi in REF:
    url = "https://api.openalex.org/works/https://doi.org/" + doi + "?mailto=" + MAIL
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mailto:"+MAIL})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        out.append((key, doi, "ERROR", str(e)[:80], []))
        time.sleep(1); continue
    title = (d.get("title") or "")[:70]
    oa = d.get("open_access") or {}
    oa_url = oa.get("oa_url")
    cands = []
    if oa_url: cands.append(oa_url)
    for loc in (d.get("locations") or []):
        pu = loc.get("pdf_url")
        if pu: cands.append(pu)
    # dedupe
    seen=set(); cc=[]
    for c in cands:
        if c and c not in seen: seen.add(c); cc.append(c)
    out.append((key, doi, oa.get("is_oa"), title, cc))
    time.sleep(1)

for key, doi, is_oa, title, cc in out:
    print("### " + key + " | oa=" + str(is_oa))
    print("    " + str(title))
    if not cc: print("    (no pdf url)")
    for c in cc[:6]:
        print("    - " + c)

