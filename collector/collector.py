import os,time,requests
SOURCE_API=os.getenv("SOURCE_API","https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json")
BACKEND_URL=os.environ["BACKEND_URL"].rstrip("/")
SECRET=os.environ["INGEST_SECRET"]
POLL=float(os.getenv("COLLECTOR_POLL_SECONDS","3"))
HEADERS={"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*","Origin":os.getenv("WINGO_ORIGIN","https://www.92pak8.com"),"Referer":os.getenv("WINGO_REFERER","https://www.92pak8.com/")}
last=None
print("V8 collector:",BACKEND_URL)
while True:
 try:
  r=requests.get(SOURCE_API,params={"pageNo":1,"pageSize":20,"ts":int(time.time()*1000)},headers=HEADERS,timeout=12);r.raise_for_status()
  items=r.json().get("data",{}).get("list",[])
  if items:
   x=items[0]; issue=str(x["issueNumber"])
   if issue!=last:
    payload={"issue":issue,"number":int(x["number"]),"color":str(x.get("color","")),"secret":SECRET}
    q=requests.post(BACKEND_URL+"/api/ingest",json=payload,timeout=15);q.raise_for_status();body=q.json()
    if body.get("ok"): last=issue; print("[OK]",issue,"stored=",body.get("stored_rounds"))
    else: print("[INGEST]",body)
 except Exception as e: print("[ERROR]",type(e).__name__,e)
 time.sleep(POLL)
