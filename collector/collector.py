import os, time
import requests

SOURCE_API=os.getenv('SOURCE_API','https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json')
BACKEND_URL=os.getenv('BACKEND_URL','').rstrip('/')
INGEST_SECRET=os.getenv('INGEST_SECRET','')
POLL_SECONDS=max(2.0,float(os.getenv('COLLECTOR_POLL_SECONDS','3')))
BACKFILL_WINDOW=max(5,min(100,int(os.getenv('COLLECTOR_BACKFILL_WINDOW','20'))))
HEADERS={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36','Accept':'application/json','Origin':'https://www.92pak8.com','Referer':'https://www.92pak8.com/','Cache-Control':'no-cache'}
s=requests.Session(); delivered=set()

def color(v):
    c=(v or '').lower()
    if 'violet' in c:return 'violet'
    if 'red' in c:return 'red'
    if 'green' in c:return 'green'
    return ''

def fetch_recent():
    r=s.get(SOURCE_API,params={'ts':int(time.time()*1000)},headers=HEADERS,timeout=15);r.raise_for_status();items=r.json().get('data',{}).get('list',[])
    out=[]
    for x in items[:BACKFILL_WINDOW]:
        issue=str(x.get('issueNumber') or '')
        try:n=int(x.get('number'))
        except Exception:continue
        c=color(str(x.get('color') or ''))
        if issue and 0<=n<=9 and c:out.append({'issue':issue,'number':n,'color':c})
    out.sort(key=lambda x:int(x['issue']));return out

def send(row):
    payload={**row,'secret':INGEST_SECRET}
    for attempt in range(1,6):
        try:
            r=s.post(f'{BACKEND_URL}/api/ingest',json=payload,timeout=45)
            try:b=r.json()
            except Exception:b={}
            if r.ok and b.get('ok') is True:
                print(f"[OK] {row['issue']} {row['number']} {row['color']} stored={b.get('stored_rounds')}");return True
            print(f"[POST] attempt={attempt} http={r.status_code} body={b or r.text[:180]}")
            if b.get('error')=='Unauthorized':
                print('[FATAL] INGEST_SECRET does not match backend.');return False
        except Exception as e:print(f'[POST] attempt={attempt} error={e}')
        time.sleep(min(10,attempt*2))
    return False

def main():
    if not BACKEND_URL:raise SystemExit('BACKEND_URL is required')
    if not INGEST_SECRET:raise SystemExit('INGEST_SECRET is required')
    print('WinGo V9 collector online');print(f'Backend: {BACKEND_URL}');print(f'Poll: {POLL_SECONDS}s | Backfill: {BACKFILL_WINDOW}')
    while True:
        try:
            rows=fetch_recent()
            if rows:print(f"[GET] newest={rows[-1]['issue']} rows={len(rows)}")
            for row in rows:
                if row['issue'] in delivered:continue
                if send(row):
                    delivered.add(row['issue'])
                    if len(delivered)>500:
                        keep=sorted(delivered,key=int)[-300:];delivered.clear();delivered.update(keep)
                else:break
        except Exception as e:print(f'[GET] error={e}')
        time.sleep(POLL_SECONDS)

if __name__=='__main__':main()
