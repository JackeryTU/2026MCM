import urllib.request, urllib.parse, re, sys, os, time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "application/pdf,text/html,*/*"}

def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type",""), r.read()

cands = [
 ("rockafellar2000cvar", "http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf"),
 ("rockafellar2000cvar", "https://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf"),
 ("elkazaz2020ems", "https://nottingham-repository.worktribe.com/output/2655588"),
 ("silvente2015rolling", "https://www.sciencedirect.com/science/article/pii/S0306261915007230/pdf"),
]

for key, url in cands:
    try:
        st, ct, body = fetch(url)
    except Exception as e:
        print(key, "FAIL", url, str(e)[:100]); continue
    print(key, st, ct, len(body), url)
    if "pdf" in ct.lower() or body[:5] == b"%PDF-":
        fn = "支撑材料/02_文献资料/_probe_" + key + ".pdf"
        open(fn, "wb").write(body)
        print("   saved", fn)
    elif "html" in ct.lower():
        txt = body.decode("utf-8","ignore")
        links = set(re.findall(r'href="([^"]+\.pdf[^"]*)"', txt, re.I))
        for l in list(links)[:10]:
            print("   pdflink:", l)
    time.sleep(1)

