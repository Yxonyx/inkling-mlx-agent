"""Versioned, dependency-free coding tasks and separate executable oracles."""

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    family: str
    instruction: str
    starter: dict[str, str]
    reference: dict[str, str]
    oracle: str


def task(id, split, family, instruction, before, after, oracle):
    def files(value):
        mapping = {"solution.py": value} if isinstance(value, str) else value
        return {name: dedent(text).strip() + "\n" for name, text in mapping.items()}

    return Task(id, split, family, instruction, files(before), files(after), dedent(oracle).strip() + "\n")


TASKS = (
    task("slug-normalization", "train", "text", "Fix slug(value). Lowercase and collapse every run of whitespace or hyphens to one hyphen. Strip edge hyphens and preserve the other characters.",
         r"""def slug(value): return value.lower().replace(" ", "-")""",
         r"""import re
def slug(value): return re.sub(r"[\s-]+", "-", value.lower()).strip("-")""",
         r"""from solution import slug
assert slug(" -- DéJà___ Vu\t -- ") == "déjà___-vu"
assert slug(" \t - ") == ""
assert slug("a-b-c") == "a-b-c" """),
    task("shorten-label", "train", "text", "Fix shorten(text, limit). Keep short text intact. Use one Unicode ellipsis when truncating, count it in limit, and return empty text if limit is at most zero.",
         r"""def shorten(text, limit): return text[:limit] + "…" """,
         r"""def shorten(text, limit):
    if limit <= 0: return ""
    return text if len(text) <= limit else text[:limit-1] + "…" """,
         r"""from solution import shorten
assert shorten("apple", 4) == "app…"
assert shorten("apple", 1) == "…"
assert shorten("ok", 2) == "ok"
assert shorten("ok", -1) == "" """),
    task("escaped-fields", "train", "text", "Fix fields(text). Split on commas except commas escaped with a backslash. Backslash escapes the next character. Preserve empty fields and a trailing unpaired backslash.",
         r"""def fields(text): return text.split(",")""",
         r"""def fields(text):
    result, chars = [], []
    escaped = False
    for ch in text:
        if escaped: chars.append(ch); escaped = False
        elif ch == chr(92): escaped = True
        elif ch == ",": result.append("".join(chars)); chars = []
        else: chars.append(ch)
    if escaped: chars.append(chr(92))
    return result + ["".join(chars)]""",
         r"""from solution import fields
b = chr(92)
assert fields("a" + b + ",b,,c" + b) == ["a,b", "", "c" + b]
assert fields("") == [""]
assert fields(b+b+",") == [b, ""]"""),
    task("strict-hex", "train", "text", "Fix rgb(text). Require exactly six ASCII hexadecimal digits with one optional leading #. Return a tuple of channel integers or raise ValueError.",
         r"""def rgb(text): return tuple(int(text.lstrip("#")[i:i+2],16) for i in (0,2,4))""",
         r"""import re
def rgb(text):
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", text): raise ValueError(text)
    text = text.removeprefix("#")
    return tuple(int(text[i:i+2],16) for i in (0,2,4))""",
         r"""from solution import rgb
assert rgb("#aB001f") == (171,0,31)
for value in ("##ffffff", "1234567", " 1ff00", "ＡＡ0000"):
    try: rgb(value)
    except ValueError: continue
    raise AssertionError(value)"""),
    task("half-open-ranges", "train", "intervals", "Fix overlap(a, b) for half-open intervals [start,end). Touching and empty ranges do not overlap.",
         r"""def overlap(a,b): return max(a[0],b[0]) <= min(a[1],b[1])""",
         r"""def overlap(a,b): return max(a[0],b[0]) < min(a[1],b[1])""",
         r"""from solution import overlap
assert not overlap((0,1),(1,2))
assert not overlap((1,1),(0,3))
assert overlap((-2,3),(0,4))"""),
    task("schedule-gap", "train", "intervals", "Fix first_gap(bookings, start, end, duration). Return the earliest start for a free interval within [start,end), or None. Bookings can be unsorted or overlap. Assume positive duration.",
         r"""def first_gap(bookings,start,end,duration):
    for left,right in bookings:
        if left-start >= duration: return start
        start = right
    return start""",
         r"""def first_gap(bookings,start,end,duration):
    for left,right in sorted(bookings):
        if right <= start: continue
        if min(left,end)-start >= duration: return start
        if left >= end: break
        start = max(start,right)
    return start if end-start >= duration else None""",
         r"""from solution import first_gap
assert first_gap([(7,9),(0,4),(2,6)],0,10,1) == 6
assert first_gap([(0,5)],0,6,2) is None
assert first_gap([(-3,-1),(10,11)],0,3,3) == 0"""),
    task("coalesce", "train", "intervals", "Fix coalesce(ranges). Ignore empty or inverted ranges. Sort and merge overlapping or touching half-open ranges. Return tuples and leave the input intact.",
         r"""def coalesce(ranges): return sorted(ranges)""",
         r"""def coalesce(ranges):
    out=[]
    for lo,hi in sorted(ranges):
        if hi <= lo: continue
        if out and lo <= out[-1][1]: out[-1]=(out[-1][0],max(hi,out[-1][1]))
        else: out.append((lo,hi))
    return out""",
         r"""from solution import coalesce
x=[(4,9),(1,4),(2,3),(8,8),(5,2)]
assert coalesce(x)==[(1,9)] and x[0]==(4,9)
assert coalesce([(2,3),(-1,0)])==[(-1,0),(2,3)]"""),
    task("clip-window", "train", "intervals", "Fix clip(ranges, start, stop). Return nonempty intersections with [start,stop), in the original order. If the window is empty return [].",
         r"""def clip(ranges,start,stop): return [(max(a,start),min(b,stop)) for a,b in ranges]""",
         r"""def clip(ranges,start,stop):
    pairs=[(max(a,start),min(b,stop)) for a,b in ranges]
    return [(a,b) for a,b in pairs if a < b]""",
         r"""from solution import clip
assert clip([(9,12),(-1,3),(4,4),(10,15)],2,10)==[(9,10),(2,3)]
assert clip([(0,5)],2,2)==[]"""),
    task("keyed-order", "train", "collections", "Fix first_by_key(items, key). Keep the first item for each key in input order, including falsey keys. Accept any iterable.",
         r"""def first_by_key(items,key): return list({key(item):item for item in items}.values())""",
         r"""def first_by_key(items,key):
    out={}
    for item in items: out.setdefault(key(item),item)
    return list(out.values())""",
         r"""from solution import first_by_key
assert first_by_key(iter([(0,"first"),(1,"second"),(0,"last")]),lambda i:i[0])==[(0,"first"),(1,"second")]
assert first_by_key([], str)==[]"""),
    task("bucket-values", "train", "collections", "Fix bucket(pairs). Group values per key in insertion order. Repeated values are retained and the returned lists are independent.",
         r"""def bucket(pairs): return {k:[v] for k,v in pairs}""",
         r"""def bucket(pairs):
    out={}
    for key,value in pairs: out.setdefault(key,[]).append(value)
    return out""",
         r"""from solution import bucket
out=bucket(iter([("x",1),("y",2),("x",1),("x",3)]))
assert list(out)==["x","y"] and out=={"x":[1,1,3],"y":[2]}
out["x"].append(8)
assert out["y"]==[2]"""),
    task("chunked", "train", "collections", "Fix chunks(items, size). Accept any iterable, yield independent lists of at most size items, keep the final short chunk, and reject size <= 0 with ValueError.",
         r"""def chunks(items,size):
    for i in range(0,len(items),size): yield items[i:i+size]""",
         r"""def chunks(items,size):
    if size<=0: raise ValueError(size)
    chunk=[]
    for item in items:
        chunk.append(item)
        if len(chunk)==size: yield chunk; chunk=[]
    if chunk: yield chunk""",
         r"""from solution import chunks
assert list(chunks(iter(range(5)),2))==[[0,1],[2,3],[4]]
assert list(chunks([],1))==[]
try: list(chunks([1],-1))
except ValueError: pass
else: raise AssertionError("size")"""),
    task("rank-stable", "train", "collections", "Fix rank(items). Accept (name, score) pairs. Sort descending by score while preserving input order for equal scores. Keep duplicates.",
         r"""def rank(items): return sorted(items,key=lambda item:(item[1],item[0]),reverse=True)""",
         r"""def rank(items): return sorted(items,key=lambda item:-item[1])""",
         r"""from solution import rank
assert rank(iter([("a",2),("z",2),("x",3),("a",2)]))==[("x",3),("a",2),("z",2),("a",2)]
assert rank([("q",-10),("r",0)])==[("r",0),("q",-10)]"""),
    task("money-rounding", "train", "numeric", "Fix cents(value). Convert a decimal string into integer cents, rounding halves away from zero and avoiding binary float.",
         r"""def cents(value): return round(float(value)*100)""",
         r"""from decimal import Decimal, ROUND_HALF_UP
def cents(value): return int((Decimal(value)*100).to_integral_value(rounding=ROUND_HALF_UP))""",
         r"""from solution import cents
assert [cents(x) for x in ("1.005","-0.005","2.674","2")] == [101,-1,267,200]"""),
    task("median-single-pass", "train", "numeric", "Fix median(values). Accept any iterable, return None for empty input, the middle value for odd length and the average of both middle values for even length.",
         r"""def median(values):
    values=sorted(values)
    return values[len(values)//2]""",
         r"""def median(values):
    values=sorted(values)
    n=len(values)
    if not n: return None
    mid=n//2
    return values[mid] if n%2 else (values[mid-1]+values[mid])/2""",
         r"""from solution import median
assert median(iter([7,1,3,5]))==4
assert median([0,-4,6])==0
assert median(iter(())) is None"""),
    task("integer-ceiling", "train", "numeric", "Fix ceil_div(total, width). Return the mathematical ceiling of integer division for arbitrary integers and strictly positive width. Reject nonpositive width with ValueError.",
         r"""def ceil_div(total,width): return int(total/width) + bool(total%width)""",
         r"""def ceil_div(total,width):
    if width<=0: raise ValueError(width)
    return -(-total//width)""",
         r"""from solution import ceil_div
assert ceil_div(-7,3)==-2 and ceil_div(7,3)==3
assert ceil_div(10**40+1,1)==10**40+1
try: ceil_div(4,-1)
except ValueError: pass
else: raise AssertionError("width")"""),
    task("mean-with-missing", "train", "numeric", "Fix mean(values). Ignore None, preserve zero and negative observations, accept generators, and return None when every observation is missing.",
         r"""def mean(values):
    values=[x for x in values if x]
    return sum(values)/len(values)""",
         r"""def mean(values):
    values=[x for x in values if x is not None]
    return sum(values)/len(values) if values else None""",
         r"""from solution import mean
assert mean(iter([0,None,2,-1]))==1/3
assert mean([None]) is None
assert mean([0])==0"""),
    task("price-api", "train", "packages", "Make catalog.quote(quantity, unit) use billing.subtotal so the price calculation has one source of truth. Negative quantities must raise ValueError in billing.subtotal and catalog.quote. Keep both public names.",
         {"billing.py":"def subtotal(quantity, unit): return quantity * unit", "catalog.py":"def quote(quantity, unit): return quantity * unit"},
         {"billing.py":r"""def subtotal(quantity, unit):
    if quantity < 0: raise ValueError(quantity)
    return quantity * unit""", "catalog.py":r"""import billing
def quote(quantity, unit): return billing.subtotal(quantity, unit)"""},
         r"""import billing, catalog
assert catalog.quote(3,5)==15
for function in (billing.subtotal,catalog.quote):
    try: function(-1,5)
    except ValueError: continue
    raise AssertionError(function)
billing.subtotal=lambda *a: "delegated"
assert catalog.quote(1,3)=="delegated" """),
    task("shared-default-port", "train", "packages", "Move the default port to settings.DEFAULT_PORT and update client.endpoint(host, port=None) to look it up at call time. An explicit port, including zero, must take precedence.",
         {"settings.py":"DEFAULT_PORT = 8080", "client.py":r"""def endpoint(host,port=None): return f"{host}:{port or 8000}" """},
         {"settings.py":"DEFAULT_PORT = 8080", "client.py":r"""import settings
def endpoint(host,port=None): return f"{host}:{settings.DEFAULT_PORT if port is None else port}" """},
         r"""import settings, client
assert client.endpoint("local")=="local:8080"
settings.DEFAULT_PORT=1234
assert client.endpoint("local")=="local:1234"
assert client.endpoint("local",0)=="local:0" """),
    task("invoice-reference", "train", "packages", "Update formats.reference to trim external spaces and uppercase invoice identifiers. Update invoice.serialize to delegate to formats.reference at call time. Keep the existing dictionary shape.",
         {"formats.py":"def reference(value): return value", "invoice.py":r"""def serialize(value): return {"reference": value}"""},
         {"formats.py":"def reference(value): return value.strip().upper()", "invoice.py":r"""import formats
def serialize(value): return {"reference": formats.reference(value)}"""},
         r"""import invoice, formats
assert invoice.serialize(" x-3 ")=={"reference":"X-3"}
assert formats.reference(" q ")=="Q"
formats.reference=lambda value:"changed"
assert invoice.serialize("z")=={"reference":"changed"}"""),
    task("csv-records", "train", "packages", "Fix parser.records(text) to use standard CSV parsing for headers and quoted fields. Update report.names to look up name for each record, preserving order.",
         {"parser.py":"def records(text): return text.splitlines()[1:]", "report.py":r"""from parser import records
def names(text): return [x.split(",")[0] for x in records(text)]"""},
         {"parser.py":r"""import csv,io
def records(text): return list(csv.DictReader(io.StringIO(text)))""", "report.py":r"""import parser
def names(text): return [x["name"] for x in parser.records(text)]"""},
         r"""import parser, report
text='id,name\n1,"Doe, Jane"\n2,Ada\n'
assert parser.records(text)[0]=={"id":"1","name":"Doe, Jane"}
assert report.names(text)==["Doe, Jane","Ada"]"""),
    task("time-window-tests", "train", "unit-tests", "Keep solution.contains unchanged. Implement tests.test_contains with assertions that catch inclusivity at both boundaries of the half-open interval and an empty interval.",
         {"solution.py":"def contains(start,stop,point): return start <= point < stop", "tests.py":"def test_contains(): pass"},
         {"solution.py":"def contains(start,stop,point): return start <= point < stop", "tests.py":r"""from solution import contains
def test_contains():
    assert contains(2,5,2)
    assert not contains(2,5,5)
    assert not contains(3,3,3)"""},
         r"""import solution, tests, importlib
original=solution.contains
assert original(0,1,0) and not original(0,1,1)
tests.test_contains()
for mutant in (lambda a,b,c: a<c<b,lambda a,b,c: a<=c<=b,lambda a,b,c: (a==b and c==a) or a<=c<b):
    solution.contains=mutant
    importlib.reload(tests)
    try: tests.test_contains()
    except AssertionError: continue
    raise AssertionError("surviving mutant")"""),
    task("placeholders-tests", "train", "unit-tests", "Keep solution.render unchanged. Implement tests.test_render with assertions that catch missing placeholder fallback and replacement of multiple occurrences.",
         {"solution.py":r"""def render(text, name=None): return text.replace("{name}", name if name is not None else "guest")""", "tests.py":"def test_render(): pass"},
         {"solution.py":r"""def render(text, name=None): return text.replace("{name}", name if name is not None else "guest")""", "tests.py":r"""from solution import render
def test_render():
    assert render("{name}") == "guest"
    assert render("{name}/{name}","A") == "A/A" """},
         r"""import solution, tests, importlib
assert solution.render("{name}")=="guest"
tests.test_render()
for mutant in (lambda s,n=None: s.replace("{name}",n or ""),lambda s,n=None: s.replace("{name}",n or "guest",1)):
    solution.render=mutant
    importlib.reload(tests)
    try: tests.test_render()
    except AssertionError: continue
    raise AssertionError("surviving mutant")"""),
    task("initials-tests", "train", "unit-tests", "Keep solution.initials unchanged. Implement tests.test_initials with assertions covering empty input, multiple words with extra whitespace, and uppercase initials.",
         {"solution.py":r"""def initials(text): return "".join(x[0].upper() for x in text.split())""", "tests.py":"def test_initials(): pass"},
         {"solution.py":r"""def initials(text): return "".join(x[0].upper() for x in text.split())""", "tests.py":r"""from solution import initials
def test_initials():
    assert initials("")==""
    assert initials(" ada   lovelace ")=="AL" """},
         r"""import solution,tests,importlib
assert solution.initials("a b")=="AB"
tests.test_initials()
for mutant in (lambda s: "X" if not s else "".join(x[0].upper() for x in s.split()), lambda s: "".join(x[0] for x in s.split()), lambda s:"".join(x[0].upper() for x in s.split()[:1])):
    solution.initials=mutant
    importlib.reload(tests)
    try: tests.test_initials()
    except AssertionError: continue
    raise AssertionError("surviving mutant")"""),
    task("weights-tests", "train", "unit-tests", "Keep solution.total unchanged. Implement tests.test_total with assertions that detect an incorrectly nonzero empty total, loss of negative contributions and ignoring quantities.",
         {"solution.py":"def total(items): return sum(quantity*weight for quantity,weight in items)", "tests.py":"def test_total(): pass"},
         {"solution.py":"def total(items): return sum(quantity*weight for quantity,weight in items)", "tests.py":r"""from solution import total
def test_total():
    assert total([])==0
    assert total([(2,4),(1,-2)])==6"""},
         r"""import solution,tests,importlib
assert solution.total([(2,3)])==6
tests.test_total()
for mutant in (lambda x:sum(a*b for a,b in x) if x else 1,lambda x:sum(max(0,a*b) for a,b in x),lambda x:sum(b for a,b in x)):
    solution.total=mutant
    importlib.reload(tests)
    try: tests.test_total()
    except AssertionError: continue
    raise AssertionError("surviving mutant")"""),
    task("walk-levels", "holdout", "graphs", "Fix levels(graph, start). Return node distances in an unweighted directed graph. Include start at zero; follow adjacency lists and handle cycles and missing adjacency lists.",
         r"""def levels(graph,start): return {node:i for i,node in enumerate(graph)}""",
         r"""from collections import deque
def levels(graph,start):
    dist={start:0}
    q=deque([start])
    while q:
        node=q.popleft()
        for nxt in graph.get(node,()):
            if nxt not in dist:
                dist[nxt]=dist[node]+1
                q.append(nxt)
    return dist""",
         r"""from solution import levels
assert levels({"b":["a","c"],"a":["b","x"],"c":["x"],"z":[]}, "a")=={"a":0,"b":1,"x":1,"c":2}
assert levels({},42)=={42:0}"""),
    task("valid-build-order", "holdout", "graphs", "Fix valid(order, deps). Require every listed job and prerequisite exactly once in order, with every prerequisite before its dependent. Reject unknown jobs and duplicates.",
         r"""def valid(order,deps): return set(deps) <= set(order)""",
         r"""def valid(order,deps):
    jobs=set(deps)|{p for values in deps.values() for p in values}
    pos={item:i for i,item in enumerate(order)}
    return set(pos)==jobs and len(pos)==len(order) and all(pos[p]<pos[k] for k,req in deps.items() for p in req)""",
         r"""from solution import valid
d={"compile":["fetch"],"ship":["compile"]}
assert valid(["fetch","compile","ship"],d)
for order in (["compile","fetch","ship"],["fetch","compile","ship","ship"],["fetch","compile","ship","extra"]):
    assert not valid(order,d)"""),
    task("tree-weight", "holdout", "graphs", "Fix weight(node). A node is an integer or a nested list of nodes. Return the sum of all integer leaves, and support empty sublists.",
         r"""def weight(node): return sum(node) if isinstance(node,list) else node""",
         r"""def weight(node): return sum(weight(x) for x in node) if isinstance(node,list) else node""",
         r"""from solution import weight
assert weight([[],4,[-1,[3]],0])==6
assert weight(-9)==-9
assert weight([])==0"""),
    task("first-path", "holdout", "graphs", "Fix path(graph, start, goal). Find a shortest directed path including start and goal. Return None if unreachable and break ties using adjacency-list order.",
         r"""def path(graph,start,goal): return [start,goal] if goal in graph.get(start,[]) else None""",
         r"""from collections import deque
def path(graph,start,goal):
    queue=deque([[start]])
    seen={start}
    while queue:
        current=queue.popleft()
        if current[-1]==goal: return current
        for node in graph.get(current[-1],()):
            if node not in seen:
                seen.add(node)
                queue.append(current+[node])
    return None""",
         r"""from solution import path
g={"a":["c","b"],"b":["d"],"c":["a","d"]}
assert path(g,"a","d")==["a","c","d"]
assert path(g,"a","a")==["a"]
assert path(g,"d","a") is None"""),
    task("utc-instant", "holdout", "time", "Fix seconds(a,b). Parse ISO 8601 datetimes with explicit offsets, return elapsed seconds b minus a, and reject naive datetimes with ValueError.",
         r"""from datetime import datetime
def seconds(a,b): return (datetime.fromisoformat(b).replace(tzinfo=None)-datetime.fromisoformat(a).replace(tzinfo=None)).total_seconds()""",
         r"""from datetime import datetime
def seconds(a,b):
    a,b=map(datetime.fromisoformat,(a,b))
    if a.utcoffset() is None or b.utcoffset() is None: raise ValueError("offset")
    return (b-a).total_seconds()""",
         r"""from solution import seconds
assert seconds("2024-01-01T02:00:00+02:00","2024-01-01T00:00:01+00:00")==1
assert seconds("2024-01-01T00:01:00Z","2024-01-01T00:00:00Z")==-60
try: seconds("2024-01-01T00:00:00","2024-01-01T00:00:00")
except ValueError: pass
else: raise AssertionError("missing offset")"""),
    task("calendar-next-month", "holdout", "time", "Fix next_month(iso_date). Advance one calendar month. Preserve the day where possible, otherwise use the last day of the new month. Return an ISO date string.",
         r"""from datetime import date,timedelta
def next_month(iso_date): return (date.fromisoformat(iso_date)+timedelta(days=30)).isoformat()""",
         r"""from datetime import date
from calendar import monthrange
def next_month(iso_date):
    d=date.fromisoformat(iso_date)
    y,m=(d.year+1,1) if d.month==12 else (d.year,d.month+1)
    return date(y,m,min(d.day,monthrange(y,m)[1])).isoformat()""",
         r"""from solution import next_month
assert next_month("2024-01-31")=="2024-02-29"
assert next_month("2023-01-31")=="2023-02-28"
assert next_month("2023-12-03")=="2024-01-03" """),
    task("business-days", "holdout", "time", "Fix add_business_days(iso_date, n). Move by n weekdays, skipping Saturdays and Sundays. Negative n counts backwards. Zero returns the original date.",
         r"""from datetime import date,timedelta
def add_business_days(iso_date,n): return (date.fromisoformat(iso_date)+timedelta(days=n)).isoformat()""",
         r"""from datetime import date,timedelta
def add_business_days(iso_date,n):
    d=date.fromisoformat(iso_date)
    step=1 if n>0 else -1
    for _ in range(abs(n)):
        d+=timedelta(days=step)
        while d.weekday()>=5: d+=timedelta(days=step)
    return d.isoformat()""",
         r"""from solution import add_business_days as move
assert move("2024-05-03",1)=="2024-05-06"
assert move("2024-05-06",-1)=="2024-05-03"
assert move("2024-05-04",0)=="2024-05-04" """),
    task("iso-week", "holdout", "time", "Fix week_key(iso_date). Return the ISO week-year and week as YYYY-WNN, including when the ISO week-year differs from the calendar year.",
         r"""from datetime import date
def week_key(iso_date):
    d=date.fromisoformat(iso_date)
    return f"{d.year}-W{d.isocalendar().week:02}" """,
         r"""from datetime import date
def week_key(iso_date):
    d=date.fromisoformat(iso_date).isocalendar()
    return f"{d.year}-W{d.week:02}" """,
         r"""from solution import week_key
assert week_key("2021-01-01")=="2020-W53"
assert week_key("2018-12-31")=="2019-W01"
assert week_key("2024-01-04")=="2024-W01" """),
    task("cache-recency", "holdout", "state", "Fix Cache to implement LRU eviction. Successful get and every put refresh recency. Missing get returns None. Capacity zero retains nothing.",
         r"""class Cache:
    def __init__(self,limit): self.limit,self.data=limit,{}
    def get(self,key): return self.data.get(key)
    def put(self,key,value):
        self.data[key]=value
        if len(self.data)>self.limit: self.data.pop(next(iter(self.data)))""",
         r"""class Cache:
    def __init__(self,limit): self.limit,self.data=limit,{}
    def get(self,key):
        if key not in self.data: return None
        value=self.data.pop(key)
        self.data[key]=value
        return value
    def put(self,key,value):
        self.data.pop(key,None)
        self.data[key]=value
        if len(self.data)>self.limit: self.data.pop(next(iter(self.data)))""",
         r"""from solution import Cache
c=Cache(2)
c.put("a",1);c.put("b",2);assert c.get("a")==1
c.put("c",3);assert c.get("b") is None
c.put("a",9);c.put("d",4);assert c.get("c") is None and c.get("a")==9
z=Cache(0); z.put("k",0);assert z.get("k") is None"""),
    task("undo-journal", "holdout", "state", "Fix Journal. set(key,value) stores the value. undo() reverts one set, including deletion if the key did not exist. An empty undo is a no-op. Keep the public data dictionary.",
         r"""class Journal:
    def __init__(self): self.data={}
    def set(self,k,v): self.data[k]=v
    def undo(self): self.data.clear()""",
         r"""class Journal:
    def __init__(self): self.data={};self.history=[]
    def set(self,k,v):
        self.history.append((k,k in self.data,self.data.get(k)))
        self.data[k]=v
    def undo(self):
        if not self.history: return
        k,existed,value=self.history.pop()
        if existed: self.data[k]=value
        else: self.data.pop(k)""",
         r"""from solution import Journal
j=Journal();j.undo();j.set("a",None);j.set("b",2);j.set("a",4);j.undo()
assert j.data=={"a":None,"b":2}
j.undo();assert j.data=={"a":None}
j.undo();j.undo();assert j.data=={}"""),
    task("deduplicating-queue", "holdout", "state", "Fix Queue. Ignore push of a value already waiting. pop returns the oldest pending value or None when empty. A popped value can be pushed again.",
         r"""class Queue:
    def __init__(self): self.items=[]
    def push(self,value): self.items.append(value)
    def pop(self): return self.items.pop() if self.items else None""",
         r"""from collections import deque
class Queue:
    def __init__(self): self.items=deque()
    def push(self,value):
        if value not in self.items: self.items.append(value)
    def pop(self): return self.items.popleft() if self.items else None""",
         r"""from solution import Queue
q=Queue();assert q.pop() is None
q.push(0);q.push(1);q.push(0)
assert q.pop()==0
q.push(0)
assert [q.pop(),q.pop(),q.pop()]==[1,0,None]"""),
    task("transaction-staging", "holdout", "state", "Fix Store. Inside begin/commit/rollback a transaction stages writes. Public committed data must not change until commit. Rollback discards writes and nesting raises ValueError. Outside a transaction writes are immediate.",
         r"""class Store:
    def __init__(self): self.data={}
    def begin(self): pass
    def put(self,k,v): self.data[k]=v
    def commit(self): pass
    def rollback(self): self.data.clear()""",
         r"""class Store:
    def __init__(self): self.data={}; self.staged=None
    def begin(self):
        if self.staged is not None: raise ValueError("nested")
        self.staged={}
    def put(self,k,v): (self.data if self.staged is None else self.staged)[k]=v
    def commit(self):
        if self.staged is not None: self.data.update(self.staged)
        self.staged=None
    def rollback(self): self.staged=None""",
         r"""from solution import Store
s=Store();s.put("a",1);s.begin();s.put("a",2);s.put("b",3)
assert s.data=={"a":1}
s.rollback();assert s.data=={"a":1}
s.begin()
try: s.begin()
except ValueError: pass
else: raise AssertionError("nested")
s.put("b",3);s.commit();assert s.data=={"a":1,"b":3}"""),
)

BY_ID = {item.id: item for item in TASKS}
