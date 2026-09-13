import json, urllib.request, time

DIS = ["10.1016/j.solener.2016.06.069","10.1016/j.apenergy.2015.05.090",
       "10.1016/j.ijepes.2019.105483","10.1016/j.ejor.2015.05.081",
       "10.1109/TSG.2016.2598678","10.1109/TSTE.2016.2543024",
       "10.21314/JOR.2000.038","10.1016/S0305-0483(99)00017-1",
       "10.1287/opre.47.2.183","10.1109/TCST.2017.2754981"]

for doi in DIS:
    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:" + doi +
           "?fields=title,openAccessPdf,externalIds")
    got = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"mathmodeling-team26/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
            oa = d.get("openAccessPdf") or {}
            print(doi, "->", (d.get("title") or "")[:55], "|", oa.get("url"))
            got = True
            break
        except Exception as e:
            if attempt == 2:
                print(doi, "FAIL", str(e)[:90])
            time.sleep(4)
    time.sleep(3)

