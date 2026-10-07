"""Fetch OpenAlex metadata+abstract for a DOI, or search by title (audit/literature helper)."""
import json
import sys
import urllib.parse
import urllib.request

if len(sys.argv) > 1 and sys.argv[1] == "search":
    query = urllib.parse.quote(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    with urllib.request.urlopen(
            f"https://api.openalex.org/works?search={query}&per-page={n}", timeout=60) as r:
        d = json.load(r)
    for w in d.get("results", []):
        src = (w.get("primary_location") or {}).get("source") or {}
        print("-", w.get("title"), "|", w.get("publication_year"), "|",
              src.get("display_name"), "|", w.get("doi"))
    sys.exit(0)

doi = sys.argv[1]
url = f"https://api.openalex.org/works/https://doi.org/{doi}"
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)
print("title:", d.get("title"))
print("year:", d.get("publication_year"))
src = (d.get("primary_location") or {}).get("source") or {}
print("venue:", src.get("display_name"))
ai = d.get("abstract_inverted_index") or {}
pos = {}
for k, v in ai.items():
    for i in v:
        pos[i] = k
abstract = " ".join(pos[p] for p in sorted(pos))
print("abstract:", abstract[:2500])
print("oa:", (d.get("open_access") or {}).get("oa_url"))
