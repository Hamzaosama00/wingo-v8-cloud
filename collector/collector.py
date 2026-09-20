import os,time,requests
API=os.getenv("SOURCE_API","https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json")
BACKEND=os.environ["BACKEND_URL"].rstrip("/")
SECRET=os.environ["INGEST_SECRET"]
POLL=float(os.getenv("COLLECTOR_POLL_SECONDS","3"))
N=int(os.getenv("COLLECTOR_BACKFILL_SIZE","20"))
H={"User-Agent":"Mozilla/5.0","Accept":"application/json","Origin":"https://www.92pak8.com","Referer":"https://www.92pak8.com/"}
s=requests.Session(); s.headers.update(H); seen=set()
def key(x):
    try:return int(x.get("issueNumber","0"))
    except:return 0
while True:
    try:
        r=s.get(API,params={"pageNo":1,"pageSize":N,"ts":int(time.time()*1000)},timeout=15); r.raise_for_status()
        for x in sorted(r.json().get("data",{}).get("list",[]),key=key):
            issue=str(x.get("issueNumber",""))
            if not issue or issue in seen: continue
            body={"issue":issue,"number":int(x["number"]),"color":str(x.get("color","")),"secret":SECRET}
            ok=False
            for a in range(5):
                try:
                    q=requests.post(BACKEND+"/api/ingest",json=body,timeout=30); q.raise_for_status()
                    print("[OK]",issue,q.json().get("stored_rounds")); seen.add(issue); ok=True; break
                except Exception as e:
                    print("[RETRY]",issue,a+1,type(e).__name__); time.sleep(min(4*(a+1),15))
            if not ok: break
        if len(seen)>500: seen=set(sorted(seen)[-250:])
    except Exception as e: print("[SOURCE ERROR]",type(e).__name__,e)
    time.sleep(POLL)
