from aslite.db import get_papers_db

def search_rank(q: str = ''):
    if not q:
        return [], [] # no query? no results
    qs = q.lower().strip().split() # split query by spaces and lowercase

    pdb = get_papers_db()
    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        score += 10.0 * matchu(' '.join([a['name'] for a in p['authors']]))
        score += 20.0 * matchu(p['title'])
        score += 1.0 * match(p['summary'])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores

def get_summaries(q, top_k=3):
    pdb = get_papers_db()
    pids, _ = search_rank(q)
    summaries = []
    for pid in pids[:top_k]:
        summaries.append(f'Paper {pid}: ' + pdb[pid]["summary"])
    return summaries
