"""Citation-list audit from institutional information only (to examine the demographic
balance of the references). Gender, race and ethnicity are NOT inferred from names. For each cited work with
a DOI or a findable title, OpenAlex gives the institutions (and their countries) that the first and last authors
reported at publication; we tabulate those countries, venue types and years. Affiliation country is a coarse proxy
for geographic diversity, not for any individual's identity.

    python3 scoring/audit_citations.py     # needs network; writes results/citation_audit/citation_audit.json
"""
import json, re, time, urllib.parse, urllib.request, difflib
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import os
# OpenAlex asks polite clients for a contact address; set CONTACT_EMAIL in the environment to supply one.
_contact = os.environ.get("CONTACT_EMAIL")
UA = {"User-Agent": "citation-audit" + (f" (mailto:{_contact})" if _contact else "")}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def parse_bib(text):
    out = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}\n', text, flags=re.S):
        body = m.group(3)
        f = lambda k: (re.search(r'\b' + k + r'\s*=\s*[{"](.*?)[}"],?\s*\n', body + "\n", flags=re.S) or [None, ""])[1]
        out[m.group(2)] = {"type": m.group(1), "title": re.sub(r'[{}\\]', '', re.sub(r'\s+', ' ', f("title"))),
                           "doi": f("doi").strip(), "year": f("year"), "url": f("url")}
    return out


def lookup(e):
    if e["doi"]:
        try:
            return get("https://api.openalex.org/works/doi:" + urllib.parse.quote(e["doi"], safe="/"))
        except Exception:
            pass
    q = urllib.parse.urlencode({"search": e["title"], "per-page": 3})
    try:
        for w in get("https://api.openalex.org/works?" + q)["results"]:
            if difflib.SequenceMatcher(None, e["title"].lower(), (w.get("title") or "").lower()).ratio() > 0.9:
                return w
    except Exception:
        pass
    return None


def main():
    bib = parse_bib((REPO / "references.bib").read_text())
    rows, first, last = [], Counter(), Counter()
    for key, e in bib.items():
        if e["type"] == "book" and not e["doi"]:
            pass
        w = lookup(e); time.sleep(0.15)
        row = {"key": key, "type": e["type"], "year": e["year"], "found": bool(w)}
        if w and w.get("authorships"):
            a = w["authorships"]
            def countries(au): return sorted({c for i in au.get("institutions", []) for c in [i.get("country_code")] if c})
            row["first_author_countries"], row["last_author_countries"] = countries(a[0]), countries(a[-1])
            row["n_authors"] = len(a)
            for c in row["first_author_countries"]: first[c] += 1
            for c in row["last_author_countries"]: last[c] += 1
        rows.append(row)
    found = [r for r in rows if r.get("first_author_countries") is not None]
    def share_non_us(key):
        have = [r for r in found if r.get(key)]
        return (sum(1 for r in have if "US" not in r[key]), len(have))
    res = {"n_references": len(bib), "n_with_institution_data": len(found),
           "first_author_country_counts": first.most_common(), "last_author_country_counts": last.most_common(),
           "first_author_non_US": share_non_us("first_author_countries"), "last_author_non_US": share_non_us("last_author_countries"),
           "type_counts": Counter(e["type"] for e in bib.values()).most_common(),
           "rows": rows,
           "note": "Institutional affiliation only (OpenAlex); no gender, race or ethnicity inferred from names."}
    out = REPO / "results/citation_audit"; out.mkdir(parents=True, exist_ok=True)
    (out / "citation_audit.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
