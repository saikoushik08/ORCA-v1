# Minimal in-memory fake of supabase-py used by the test suite (no network, no credentials)
DB = {"jobs": [], "images": [], "job_events": []}
# Set RESTRICT["jobs"] = {"job_id", "status", ...} to simulate an old schema that rejects
# unknown columns (like PostgREST does).
RESTRICT = {"jobs": None, "images": None, "job_events": None}


class R:  # response
    def __init__(s, data=None, count=None):
        s.data = data
        s.count = count


class Q:
    def __init__(s, t):
        s.t = t; s.f = []; s.op = "select"; s.payload = None
        s.single_ = False; s.cnt = False; s.lim = None; s.ord = None

    def select(s, *a, count=None, **k): s.op = "select"; s.cnt = bool(count); return s
    def insert(s, p): s.op = "insert"; s.payload = p; return s
    def update(s, p): s.op = "update"; s.payload = p; return s
    def eq(s, k, v): s.f.append((k, v)); return s
    def single(s): s.single_ = True; return s
    def order(s, col, desc=False): s.ord = (col, desc); return s
    def limit(s, n): s.lim = n; return s

    def _check_columns(s):
        allowed = RESTRICT.get(s.t)
        if allowed is not None and s.payload:
            bad = set(s.payload) - allowed
            if bad:
                raise Exception(f"Could not find the {sorted(bad)} column of '{s.t}'")

    def execute(s):
        rows = DB[s.t]
        m = [r for r in rows if all(r.get(k) == v for k, v in s.f)]
        if s.op == "insert":
            s._check_columns()
            r = dict(s.payload); r.setdefault("id", len(rows) + 1); rows.append(r)
            return R([dict(r)])
        if s.op == "update":
            s._check_columns()
            for r in m:
                r.update(s.payload)
            return R([dict(r) for r in m])
        if s.single_:
            if not m:
                raise Exception("PGRST116: The result contains 0 rows")
            return R(dict(m[0]))
        if s.ord:
            m = sorted(m, key=lambda r: str(r.get(s.ord[0])), reverse=s.ord[1])
        if s.lim:
            m = m[:s.lim]
        return R([dict(r) for r in m], len(m) if s.cnt else None)


class Bucket:
    uploads = []

    def upload(s, path, data, file_options=None):
        Bucket.uploads.append(path)
        return {"path": path}


class Storage:
    def from_(s, b): return Bucket()


class Client:
    storage = Storage()
    def table(s, t): return Q(t)


def create_client(u, k): return Client()


def reset():
    for k in DB:
        DB[k].clear()
    for k in RESTRICT:
        RESTRICT[k] = None
    Bucket.uploads.clear()
