import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import datetime, random, json
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="AdvisorIQ", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ── Module imports ────────────────────────────────────────────────────────────
try:
    import database as db; DB_OK = True
except: DB_OK = False

try:
    from outcomes import (log_outcome, get_client_history, get_outcome_adjusted_score,
                          get_outcome_stats, get_best_calling_patterns,
                          init_outcome_tables, OUTCOME_TYPES)
    OUT_OK = True
except: OUT_OK = False

try:
    from sheets_sync import validate_sheets_url, get_sync_status; SHEETS_OK = True
except: SHEETS_OK = False

try:
    from whatsapp import get_whatsapp_link; WA_OK = True
except: WA_OK = False

try:
    from ml_model import get_model_meta, train_models, load_models, get_top_feature, extract_features
    ML_OK = True
except: ML_OK = False

# ── DB init ───────────────────────────────────────────────────────────────────
if DB_OK:
    db.init_db()
if OUT_OK:
    init_outcome_tables()

# ── Theme ─────────────────────────────────────────────────────────────────────
def inject_theme():
    dark = st.session_state.get("dark_mode", True)
    if dark:
        root = """:root{
  --bg:#0d1117;--s1:#161b22;--s2:#1c2128;--s3:#21262d;
  --bd:#30363d;--bd2:#444c56;--tx:#e6edf3;--t2:#8b949e;--t3:#6e7681;
  --gr:#3fb950;--grbg:rgba(63,185,80,.1);--grbd:rgba(63,185,80,.3);
  --am:#d29922;--ambg:rgba(210,153,34,.1);--ambd:rgba(210,153,34,.3);
  --rd:#f85149;--rdbg:rgba(248,81,73,.1);--rdbd:rgba(248,81,73,.3);
  --bl:#58a6ff;--blbg:rgba(88,166,255,.1);--blbd:rgba(88,166,255,.3);
  --pu:#a371f7;--pubg:rgba(163,113,247,.1);--pubd:rgba(163,113,247,.3)}"""
    else:
        root = """:root{
  --bg:#f5f6f8;--s1:#ffffff;--s2:#f0f2f5;--s3:#e8eaed;
  --bd:#e2e5ea;--bd2:#cdd1d8;--tx:#111318;--t2:#6b7280;--t3:#9ca3af;
  --gr:#16a34a;--grbg:rgba(22,163,74,.08);--grbd:rgba(22,163,74,.22);
  --am:#d97706;--ambg:rgba(217,119,6,.08);--ambd:rgba(217,119,6,.22);
  --rd:#dc2626;--rdbg:rgba(220,38,38,.07);--rdbd:rgba(220,38,38,.20);
  --bl:#2563eb;--blbg:rgba(37,99,235,.07);--blbd:rgba(37,99,235,.20);
  --pu:#7c3aed;--pubg:rgba(124,58,237,.07);--pubd:rgba(124,58,237,.22)}"""
    st.markdown(f"<style>{root}</style>", unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box}
html,body,[data-testid=stAppViewContainer]{background:var(--bg)!important;color:var(--tx)!important;font-family:"Plus Jakarta Sans",sans-serif!important}
[data-testid=stHeader],[data-testid=stDecoration],footer{display:none!important}
[data-testid=stSidebar]{background:var(--s1)!important;border-right:1px solid var(--bd)!important}
.block-container{padding:0!important;max-width:100%!important}
.nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;height:56px;background:var(--s1);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:200}
.nav-logo{display:flex;align-items:center;gap:10px}
.nav-icon{width:30px;height:30px;background:var(--gr);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#000}
.nav-brand{font-size:15px;font-weight:600;color:var(--tx)}.nav-brand em{color:var(--gr);font-style:normal}
.nav-right{display:flex;align-items:center;gap:10px}
.nav-user{font-size:12px;color:var(--t2);font-family:IBM Plex Mono,monospace}
.nav-role{font-size:11px;padding:2px 8px;border-radius:12px;background:var(--grbg);color:var(--gr);border:1px solid var(--grbd);font-weight:600}
.wrap{padding:1.5rem;max-width:1440px;margin:0 auto}
.greet{display:flex;align-items:center;justify-content:space-between;background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem}
.gt{font-size:11px;font-family:IBM Plex Mono,monospace;color:var(--gr);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.gn{font-size:1.4rem;font-weight:600;letter-spacing:-.4px;margin-bottom:4px}
.gsub{font-size:13px;color:var(--t2)}
.gstats{display:flex;gap:2rem;text-align:right}
.gnum{font-size:1.4rem;font-weight:700;display:block;font-family:IBM Plex Mono,monospace}
.glbl{font-size:11px;color:var(--t2);margin-top:2px;display:block}
.kgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:.5rem}
.kc{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;transition:border-color .15s,transform .15s}
.kc:hover{border-color:var(--bd2);transform:translateY(-2px)}
.kc::before{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.kc.gr::before{background:var(--gr)}.kc.bl::before{background:var(--bl)}.kc.rd::before{background:var(--rd)}.kc.am::before{background:var(--am)}.kc.pu::before{background:var(--pu)}
.kl{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;font-family:IBM Plex Mono,monospace;margin-bottom:10px}
.kc.gr .kl{color:var(--gr)}.kc.bl .kl{color:var(--bl)}.kc.rd .kl{color:var(--rd)}.kc.am .kl{color:var(--am)}.kc.pu .kl{color:var(--pu)}
.knum{font-size:2rem;font-weight:700;letter-spacing:-.04em;line-height:1;margin-bottom:5px}
.kdesc{font-size:12px;color:var(--t2);line-height:1.4;margin-bottom:8px}
.ksig{font-size:11px;font-family:IBM Plex Mono,monospace;padding-top:8px;border-top:1px solid var(--bd)}
.kc.gr .ksig{color:var(--gr)}.kc.bl .ksig{color:var(--bl)}.kc.rd .ksig{color:var(--rd)}.kc.am .ksig{color:var(--am)}.kc.pu .ksig{color:var(--pu)}
.kdet{background:var(--s2);border:1px solid var(--bd2);border-radius:10px;padding:1.25rem;margin-bottom:1.5rem}
.kdet-h{display:flex;align-items:center;margin-bottom:.875rem;padding-bottom:.75rem;border-bottom:1px solid var(--bd)}
.kdet-t{font-size:14px;font-weight:600}
.ptable{width:100%;border-collapse:collapse;font-size:13px}
.ptable thead th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-family:IBM Plex Mono,monospace;color:var(--t3);font-weight:500;padding:10px 14px;border-bottom:1px solid var(--bd);text-align:left}
.ptable tbody tr{border-bottom:1px solid var(--bd);transition:background .1s}
.ptable tbody tr:hover{background:var(--s2)}
.prank{font-family:IBM Plex Mono,monospace;font-size:12px;color:var(--t3);width:44px}
.pname{font-weight:600;font-size:14px}.psub{font-size:12px;color:var(--t2);margin-top:3px}
.sbar{display:inline-flex;align-items:center;gap:8px}
.strack{width:52px;height:3px;border-radius:2px;background:var(--bd2);overflow:hidden;display:inline-block;vertical-align:middle}
.sfill{height:100%;border-radius:2px}
.snum{font-family:IBM Plex Mono,monospace;font-size:12px;font-weight:600;min-width:22px}
.chip{display:inline-block;font-size:11px;font-weight:600;font-family:IBM Plex Mono,monospace;padding:2px 9px;border-radius:12px}
.chi{background:var(--grbg);color:var(--gr);border:1px solid var(--grbd)}
.chm{background:var(--ambg);color:var(--am);border:1px solid var(--ambd)}
.chl{background:var(--rdbg);color:var(--rd);border:1px solid var(--rdbd)}
.tag{font-size:10px;padding:2px 7px;border-radius:8px;display:inline-block;margin-right:3px;background:var(--s3);color:var(--t2);border:1px solid var(--bd);font-family:IBM Plex Mono,monospace}
.xin{padding:.875rem 1.1rem;border-left:3px solid var(--bl);margin:0 0 6px 44px;background:var(--s2);border-radius:0 6px 6px 0}
.xlbl{font-size:10px;font-family:IBM Plex Mono,monospace;color:var(--bl);text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;font-weight:600}
.xtxt{font-size:13px;color:var(--t2);line-height:1.6}
.outcome-row{display:flex;gap:6px;margin-top:.75rem;flex-wrap:wrap}
.evgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.evcard{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:1.25rem}
.evcard:hover{border-color:var(--bd2)}
.evtitle{font-size:14px;font-weight:600;margin-bottom:6px}
.evbody{font-size:13px;color:var(--t2);line-height:1.65;margin-bottom:.75rem}
.evroi{font-size:12px;font-family:IBM Plex Mono,monospace;color:var(--gr);font-weight:600;margin-bottom:.5rem}
.evmeta{display:flex;gap:14px;font-size:11px;color:var(--t3);font-family:IBM Plex Mono,monospace}
.btn-wa{font-size:12px;padding:5px 14px;border-radius:6px;font-weight:600;background:rgba(37,211,102,.1);color:#25d366;border:1px solid rgba(37,211,102,.3);text-decoration:none;font-family:IBM Plex Mono,monospace;display:inline-block}
.wprof{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:1.1rem;margin-bottom:1rem}
.wpname{font-size:15px;font-weight:700;margin-bottom:10px}
.wprow{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--bd);font-size:13px;color:var(--t2)}
.wprow:last-child{border:none}.wpval{font-family:IBM Plex Mono,monospace;color:var(--tx)}
.uph{text-align:center;padding:5rem 2rem 2rem}
.upey{font-size:11px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.15em;color:var(--gr);margin-bottom:1rem}
.upt{font-size:2.5rem;font-weight:700;letter-spacing:-.05em;line-height:1.15;margin-bottom:.75rem}
.upt em{color:var(--gr);font-style:normal}
.ups{font-size:14px;color:var(--t2);max-width:480px;margin:0 auto 2rem;line-height:1.7}
.hist-pill{font-size:11px;padding:3px 10px;border-radius:20px;display:inline-block;font-family:IBM Plex Mono,monospace;font-weight:600;margin-right:4px}
.stButton>button{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-size:13px!important;font-weight:500!important;border-radius:6px!important;padding:6px 16px!important}
.stButton>button:hover{background:var(--s3)!important}
.stTextInput>div>div>input{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-size:13px!important}
.stSelectbox>div>div{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important}
.stTabs [data-baseweb=tab-list]{background:var(--s2)!important;border-bottom:1px solid var(--bd)!important;padding:0 .5rem!important;gap:0!important}
.stTabs [data-baseweb=tab]{color:var(--t2)!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-size:13px!important;font-weight:500!important;padding:10px 16px!important;border-radius:0!important;border-bottom:2px solid transparent!important}
.stTabs [aria-selected=true]{color:var(--tx)!important;border-bottom-color:var(--bl)!important;background:transparent!important}
textarea{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important;font-family:"Plus Jakarta Sans",sans-serif!important}
.stRadio label{color:var(--t2)!important;font-size:13px!important}
.stAlert{background:var(--s2)!important;border-radius:6px!important;color:var(--t2)!important}
div[data-testid=stFileUploader]{background:var(--s1)!important;border:1px dashed var(--bd2)!important;border-radius:8px!important;padding:1rem!important}
[data-testid=stMarkdownContainer] p{color:var(--t2)!important;font-size:13px!important}
hr{border-color:var(--bd)!important}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fi(v):
    try: n=float(str(v).replace(",","").replace("\u20b9","") or 0)
    except: n=0
    if n>=1e7: return f"\u20b9{n/1e7:.1f}Cr"
    if n>=1e5: return f"\u20b9{n/1e5:.1f}L"
    if n>=1e3: return f"\u20b9{n/1e3:.0f}K"
    return f"\u20b9{int(n)}"

def num(v):
    try: return float(str(v).replace(",","").replace("\u20b9","").strip())
    except: return 0.0

def mago(d):
    if not d or str(d).strip() in ("","nan","None"): return 99
    try:
        dt=pd.to_datetime(str(d),dayfirst=True,errors="coerce")
        if pd.isna(dt): return 99
        return max(0,(datetime.datetime.now()-dt.to_pydatetime()).days/30)
    except: return 99

def now_ist():
    try:
        import pytz; return datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    except: return datetime.datetime.now()+datetime.timedelta(hours=5,minutes=30)

def score_c(r):
    p=num(r.get("portfolio",0)); sip=num(r.get("sip",0))
    try: age=int(float(r.get("age") or 35))
    except: age=35
    try:
        yr=int(float(str(r.get("tenure","2020")).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=3
    ma=mago(r.get("lastContact",""))
    nom=str(r.get("nominee","")).lower().strip()
    goal=str(r.get("goal","")).lower()
    s=40
    if p>8e6:s+=28
    elif p>4e6:s+=20
    elif p>1.5e6:s+=13
    elif p>5e5:s+=7
    if sip>20000:s+=18
    elif sip>10000:s+=13
    elif sip>3000:s+=8
    elif sip>0:s+=4
    if ma<1:s+=15
    elif ma<3:s+=10
    elif ma<6:s+=5
    elif ma>12:s-=18
    elif ma>6:s-=10
    if ty>15:s+=15
    elif ty>8:s+=10
    elif ty>3:s+=5
    if nom=="no":s-=8
    if "bond" in goal:s+=5
    if age>55 and "lic" in goal:s+=5
    if sip==0 and p>5e5:s-=5
    return max(0,min(100,round(s)))

def churn_c(r):
    r2=0; ma=mago(r.get("lastContact",""))
    sip=num(r.get("sip",0)); nom=str(r.get("nominee","")).lower().strip()
    try:
        yr=int(float(str(r.get("tenure","2020")).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=5
    if ma>12:r2+=40
    elif ma>6:r2+=25
    elif ma>3:r2+=10
    if sip==0:r2+=20
    if nom=="no":r2+=15
    if ty<2:r2+=15
    return min(100,round(r2))

def flags_c(r):
    f=[]; p=num(r.get("portfolio",0)); sip=num(r.get("sip",0))
    ma=mago(r.get("lastContact","")); nom=str(r.get("nominee","")).lower().strip()
    if p>5e6:f.append("High Value")
    if ma>6:f.append("Inactive 6m+")
    if sip==0 and p>5e5:f.append("No SIP")
    if nom=="no":f.append("No Nominee")
    if churn_c(r)>55:f.append("Leaving Risk")
    return f

def cn(v):
    try: return str(float(str(v).replace(",","").replace("\u20b9","").strip()))
    except: return "0"

def cph(v):
    if not v: return ""
    d="".join(filter(str.isdigit,str(v)))
    return ("91"+d) if len(d)==10 else d

HINTS={"name":["name","client","naam"],"age":["age","umur"],
       "portfolio":["portfolio","aum","value","investment","amount","total"],
       "sip":["sip","monthly"],"lastContact":["last","date","meeting","contact","interaction"],
       "goal":["product","goal","scheme","type"],"tenure":["since","tenure","year","clientsince"],
       "nominee":["nominee","nomination"],"phone":["phone","mobile","number"]}

def det(cols):
    m={}
    for f,hints in HINTS.items():
        for c in cols:
            cl=c.lower().replace(" ","").replace("_","")
            for h in hints:
                if h in cl: m[f]=c; break
            if f in m: break
    return m

def process(df,mapping):
    dfl={"name":"","age":"","portfolio":"0","sip":"0","lastContact":"",
         "goal":"","tenure":"2020","nominee":"","phone":""}
    clients=[]
    for _,row in df.iterrows():
        c=dict(dfl)
        for key in dfl:
            col=mapping.get(key)
            if col and col in df.columns:
                val=row[col]
                if pd.notna(val) and str(val).strip() not in ("","nan","None"):
                    c[key]=cn(val) if key in ("portfolio","sip") else (cph(val) if key=="phone" else str(val).strip())
        c["score"]=score_c(c); c["churn"]=churn_c(c)
        c["priority"]="High" if c["score"]>=70 else ("Medium" if c["score"]>=45 else "Low")
        c["flags"]=flags_c(c); clients.append(c)
    seen_p={}; seen_n={}; out=[]; merged=0
    for c in clients:
        ph=c.get("phone","").strip(); nm=c.get("name","").strip().lower()
        p=num(c.get("portfolio",0))
        if ph and len(ph)>=10 and ph in seen_p:
            if p>num(seen_p[ph].get("portfolio",0)): out[out.index(seen_p[ph])]=c; seen_p[ph]=c
            merged+=1
        elif nm and nm in seen_n:
            if p>num(seen_n[nm].get("portfolio",0)): out[out.index(seen_n[nm])]=c; seen_n[nm]=c
            merged+=1
        else:
            out.append(c)
            if ph and len(ph)>=10: seen_p[ph]=c
            if nm: seen_n[nm]=c
    out.sort(key=lambda x:x.get("score",0),reverse=True)
    return out, merged

def export_excel(clients):
    rows=[{"Client":c.get("name",""),"Portfolio":num(c.get("portfolio",0)),
           "SIP/mo":num(c.get("sip",0)),"Priority":c.get("priority",""),
           "Health Score":c.get("score",0),"Leaving Risk":c.get("churn",0),
           "Product":c.get("goal",""),"Last Contact":c.get("lastContact",""),
           "Phone":c.get("phone",""),"Flags":" | ".join(c.get("flags",[]))} for c in clients]
    df=pd.DataFrame(rows); buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="Clients")
        ws=w.sheets["Clients"]
        for col in ws.columns:
            mx=max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width=min(mx+4,40)
    buf.seek(0); return buf.getvalue()

DEMO=[
    {"name":"Ramesh Patel","age":"62","portfolio":"4800000","sip":"15000","lastContact":"2024-01-10","goal":"MF+LIC","tenure":"2010","nominee":"Yes","phone":"9876543210"},
    {"name":"Kavita Joshi","age":"55","portfolio":"7200000","sip":"25000","lastContact":"2024-03-01","goal":"MF+Bonds+LIC","tenure":"2008","nominee":"Yes","phone":"9876543211"},
    {"name":"Hemant Rao","age":"67","portfolio":"9500000","sip":"0","lastContact":"2023-11-20","goal":"Bonds+LIC","tenure":"2005","nominee":"Yes","phone":"9876543212"},
    {"name":"Geeta Sharma","age":"61","portfolio":"6100000","sip":"20000","lastContact":"2023-10-15","goal":"MF+LIC+Bonds","tenure":"2007","nominee":"Yes","phone":"9876543213"},
    {"name":"Suresh Agrawal","age":"70","portfolio":"12000000","sip":"0","lastContact":"2023-05-10","goal":"Bonds+LIC","tenure":"2002","nominee":"Yes","phone":"9876543214"},
    {"name":"Pushpa Rao","age":"64","portfolio":"5500000","sip":"15000","lastContact":"2024-02-01","goal":"MF+LIC+Bonds","tenure":"2006","nominee":"Yes","phone":"9876543215"},
    {"name":"Nisha Gupta","age":"41","portfolio":"2100000","sip":"12000","lastContact":"2024-01-25","goal":"MF+LIC","tenure":"2016","nominee":"Yes","phone":"9876543216"},
    {"name":"Manisha Patel","age":"53","portfolio":"2900000","sip":"10000","lastContact":"2023-07-22","goal":"LIC+MF","tenure":"2013","nominee":"Yes","phone":"9876543217"},
    {"name":"Rekha Jain","age":"58","portfolio":"3400000","sip":"0","lastContact":"2023-12-05","goal":"LIC+Bonds","tenure":"2011","nominee":"Yes","phone":"9876543218"},
    {"name":"Sunita Shah","age":"45","portfolio":"1200000","sip":"8000","lastContact":"2023-09-20","goal":"MF","tenure":"2018","nominee":"No","phone":"9876543219"},
    {"name":"Arun Trivedi","age":"48","portfolio":"900000","sip":"0","lastContact":"2023-06-15","goal":"LIC","tenure":"2015","nominee":"No","phone":"9876543220"},
    {"name":"Vijay Solanki","age":"50","portfolio":"650000","sip":"6000","lastContact":"2023-08-10","goal":"MF","tenure":"2019","nominee":"No","phone":"9876543221"},
    {"name":"Bhavesh Modi","age":"44","portfolio":"520000","sip":"7500","lastContact":"2024-03-10","goal":"MF","tenure":"2020","nominee":"No","phone":"9876543222"},
    {"name":"Jigar Shah","age":"47","portfolio":"1750000","sip":"9000","lastContact":"2023-12-18","goal":"MF+LIC","tenure":"2017","nominee":"No","phone":"9876543223"},
    {"name":"Hetal Trivedi","age":"39","portfolio":"430000","sip":"6000","lastContact":"2024-02-20","goal":"MF","tenure":"2021","nominee":"No","phone":"9876543224"},
    {"name":"Dinesh Mehta","age":"38","portfolio":"350000","sip":"5000","lastContact":"2024-02-28","goal":"SIP","tenure":"2022","nominee":"No","phone":"9876543225"},
    {"name":"Kalpesh Vora","age":"36","portfolio":"210000","sip":"3000","lastContact":"2024-01-30","goal":"SIP","tenure":"2023","nominee":"No","phone":"9876543226"},
    {"name":"Priya Desai","age":"32","portfolio":"180000","sip":"4000","lastContact":"2024-02-10","goal":"SIP","tenure":"2023","nominee":"No","phone":"9876543227"},
    {"name":"Nilesh Mehta","age":"33","portfolio":"95000","sip":"2000","lastContact":"2024-03-05","goal":"SIP","tenure":"2024","nominee":"No","phone":"9876543228"},
    {"name":"Archana Desai","age":"56","portfolio":"4200000","sip":"0","lastContact":"2023-09-05","goal":"LIC+Bonds","tenure":"2009","nominee":"Yes","phone":"9876543229"},
]

AGENDAS=[
    "Your clients are waiting. Let's get the right ones on a call today.",
    "A few things need your attention. Your priority list is ready.",
    "Fresh data loaded. Your top clients are ranked and waiting.",
    "Good to have you back. A few clients need a call today.",
    "Your intelligence engine is active. Best opportunities surfaced.",
]

def prep_demo():
    out=[]
    for c in DEMO:
        c2=dict(c); c2["score"]=score_c(c2); c2["churn"]=churn_c(c2)
        c2["priority"]="High" if c2["score"]>=70 else ("Medium" if c2["score"]>=45 else "Low")
        c2["flags"]=flags_c(c2); out.append(c2)
    out.sort(key=lambda x:x.get("score",0),reverse=True)
    return out

# ── NAV ───────────────────────────────────────────────────────────────────────
def show_nav():
    inject_theme()
    user=st.session_state.get("user_name","")
    company=st.session_state.get("user_company","")
    role=st.session_state.get("user_role","advisor")
    rl="Owner" if role=="owner" else "Advisor"
    dark=st.session_state.get("dark_mode",True)
    st.markdown(f"""<div class="nav">
      <div class="nav-logo">
        <div class="nav-icon">\u26a1</div>
        <span class="nav-brand">Advisor<em>IQ</em></span>
      </div>
      <div class="nav-right">
        <span class="nav-user">{user} \u00b7 {company}</span>
        <span class="nav-role">{rl}</span>
      </div>
    </div>""",unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns([5,1,1,1,1])
    with c2:
        if st.button("\u2600" if dark else "\U0001f319",key="th",help="Toggle theme"):
            st.session_state.dark_mode=not dark; st.rerun()
    with c3:
        if st.button("\u2b06 Upload",key="nav_up"):
            st.session_state.pop("kpi_open",None); st.session_state.screen="upload"; st.rerun()
    with c4:
        if st.button("\u2699 Settings",key="nav_set"):
            st.session_state.screen="settings"; st.rerun()
    with c5:
        if st.button("Sign out",key="nav_so"):
            try: st.query_params.clear()
            except: pass
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def show_login():
    inject_theme()
    _,col,_=st.columns([1,1,1])
    with col:
        st.markdown("""<div style="text-align:center;margin-top:3rem;margin-bottom:2rem">
          <div style="width:48px;height:48px;background:#3fb950;border-radius:10px;
            display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#000;margin-bottom:.875rem">\u26a1</div>
          <div style="font-size:1.3rem;font-weight:700;letter-spacing:-.3px;color:var(--tx)">AdvisorIQ</div>
          <div style="font-size:13px;color:var(--t2);margin-top:4px">Portfolio intelligence for financial advisors</div>
        </div>""",unsafe_allow_html=True)
        t1,t2=st.tabs(["Sign in","Create account"])
        with t1:
            st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
            u=st.text_input("Username",placeholder="your.username",key="li_u")
            p=st.text_input("Password",type="password",placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022",key="li_p")
            st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
            if st.button("Sign in \u2192",use_container_width=True,key="li_b"):
                if u and p:
                    if DB_OK:
                        row=db.login_user(u,p)
                        if row:
                            st.session_state.user_id=row["id"]
                            st.session_state.user_name=row["full_name"]
                            st.session_state.user_company=row["company"]
                            st.session_state.user_role=row["role"]
                            st.session_state.user_plan=row.get("plan","free")
                            saved=db.load_clients(row["id"])
                            if saved: st.session_state.clients=saved
                            st.session_state.screen="upload" if not saved else "dashboard"
                            try: st.query_params["uid"]=str(row["id"])
                            except: pass
                            st.rerun()
                        else: st.error("Incorrect username or password.")
                    else: st.error("Database not available.")
                else: st.warning("Please enter both fields.")
        with t2:
            st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
            rn=st.text_input("Full name",placeholder="Ramesh Patel",key="r_n")
            rc=st.text_input("Company",placeholder="Patel Wealth Advisory",key="r_c")
            ru=st.text_input("Username",placeholder="ramesh.patel",key="r_u")
            rp=st.text_input("Password",type="password",placeholder="Min 6 characters",key="r_p")
            rr=st.selectbox("Role",["Owner / Director","Senior Advisor","Advisor","Team Member"],key="r_r")
            rm={"Owner / Director":"owner","Senior Advisor":"advisor","Advisor":"advisor","Team Member":"staff"}
            st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
            if st.button("Create account \u2192",use_container_width=True,key="r_b"):
                if all([rn,rc,ru,rp]):
                    if len(rp)<6: st.warning("Password must be at least 6 characters.")
                    elif DB_OK:
                        ok,msg=db.create_user(ru,rp,rn,rc,rm[rr])
                        if ok: st.success(f"Account created! Sign in with: {ru}")
                        else: st.error(msg)
                else: st.warning("Please fill in all fields.")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
def show_upload():
    show_nav()
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    clients=st.session_state.get("clients",[])
    if clients:
        st.success(f"\u2713 {len(clients)} clients loaded from your last session.")
        cc,_=st.columns([1,4])
        with cc:
            if st.button("View dashboard \u2192",use_container_width=True):
                st.session_state.screen="dashboard"; st.rerun()
        st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("""<div class="uph">
      <div class="upey">\u26a1 Intelligence Engine</div>
      <div class="upt">Your clients,<br><em>clearly ranked.</em></div>
      <div class="ups">Upload any Excel or CSV. The engine scores every client and shows you exactly who to call today — and why.</div>
    </div>""",unsafe_allow_html=True)
    _,cc,_=st.columns([1,2,1])
    with cc:
        uploaded=st.file_uploader("",type=["xlsx","xls","csv"],label_visibility="collapsed")
        st.markdown("<div style='text-align:center;font-size:11px;color:var(--t3);margin-top:.5rem'>Any column format \u00b7 Excel or CSV \u00b7 Auto-detected</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("\U0001f4ca Load demo data \u2192",use_container_width=True):
            st.session_state.use_demo=True; st.session_state.screen="map"; st.rerun()
    st.markdown("</div>",unsafe_allow_html=True)
    return uploaded

# ── MAPPING ───────────────────────────────────────────────────────────────────
def show_mapping(df):
    show_nav()
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    st.markdown("### Map your columns")
    st.caption("Auto-detected where possible. Adjust if needed.")
    cols=df.columns.tolist(); mapping=det(cols)
    field_labels={"name":"Client name","age":"Age","portfolio":"Portfolio / AUM (\u20b9)",
                  "sip":"Monthly SIP (\u20b9)","lastContact":"Last contact date",
                  "goal":"Product / goal","tenure":"Client since (year)",
                  "nominee":"Nominee updated?","phone":"Phone number"}
    user_map={}; g=st.columns(2)
    for i,key in enumerate(field_labels):
        best=mapping.get(key)
        with g[i%2]:
            opts=["\u2014 skip \u2014"]+cols
            idx=(cols.index(best)+1) if best and best in cols else 0
            sel=st.selectbox(field_labels[key],opts,index=idx,key=f"m_{key}")
            user_map[key]=sel if sel!="\u2014 skip \u2014" else None
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,_=st.columns([1,1,4])
    with c1:
        if st.button("Run engine \u2192",use_container_width=True):
            with st.spinner("Scoring all clients..."):
                clients,merged=process(df,user_map)
            st.session_state.clients=clients; st.session_state.merged_count=merged
            if DB_OK: db.save_clients(st.session_state.user_id,clients)
            st.session_state.screen="dashboard"; st.rerun()
    with c2:
        if st.button("\u2190 Back"):
            st.session_state.screen="upload"; st.rerun()
    st.markdown("</div>",unsafe_allow_html=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
def show_settings():
    show_nav()
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    st.markdown("### Settings")
    user_id=st.session_state.get("user_id")
    t1,t2=st.tabs(["Google Sheets","Account"])
    with t1:
        st.markdown("**Connect Google Sheets for automatic sync**")
        st.caption("Kartik updates the sheet → app detects change → insights refresh every 5 minutes.")
        if SHEETS_OK and user_id:
            status=get_sync_status(user_id)
            if status.get("has_sheets"):
                st.success(f"\u2713 Connected: {status['sheets_url'][:60]}...")
                st.caption(f"Last synced: {status.get('last_synced','Never')}")
                if st.button("Disconnect"):
                    db.update_sheets_url(user_id,""); st.rerun()
            else:
                url=st.text_input("Google Sheets URL",placeholder="https://docs.google.com/spreadsheets/d/...")
                if st.button("Connect \u2192"):
                    if url.strip():
                        v,msg=validate_sheets_url(url)
                        if v: db.update_sheets_url(user_id,url); st.success(msg); st.rerun()
                        else: st.error(msg)
                st.markdown("""<div style="font-size:12px;color:var(--t2);margin-top:1rem;line-height:1.8">
                  <strong style="color:var(--tx)">Setup steps:</strong><br>
                  1. Go to sheets.google.com \u2192 move your Excel data there<br>
                  2. Share the sheet \u2192 copy the link \u2192 paste above<br>
                  3. App auto-syncs every 5 minutes when you update the sheet
                </div>""",unsafe_allow_html=True)
        else:
            st.info("Google Sheets requires: pip install gspread google-auth")
    with t2:
        st.markdown(f"**Name:** {st.session_state.get('user_name','')}")
        st.markdown(f"**Company:** {st.session_state.get('user_company','')}")
        st.markdown(f"**Role:** {st.session_state.get('user_role','')}")
        st.markdown(f"**Plan:** {st.session_state.get('user_plan','free').upper()}")
    st.markdown("</div>",unsafe_allow_html=True)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def show_dashboard(clients):
    show_nav()
    st.markdown('<div class="wrap">',unsafe_allow_html=True)

    # Metrics
    aum=sum(num(c.get("portfolio",0)) for c in clients)
    high=[c for c in clients if c.get("priority")=="High"]
    at_risk=[c for c in clients if c.get("churn",0)>50]
    no_sip=[c for c in clients if "No SIP" in c.get("flags",[])]
    no_nom=[c for c in clients if "No Nominee" in c.get("flags",[])]
    hni=[c for c in clients if "High Value" in c.get("flags",[])]
    risk_aum=sum(num(c.get("portfolio",0)) for c in at_risk)
    user_id=st.session_state.get("user_id",0)

    # Outcome stats
    out_stats={"total_calls":0,"total_converted":0,"conversion_rate":0}
    pattern_insight=""
    if OUT_OK and user_id:
        out_stats=get_outcome_stats(user_id)
        pattern_insight=get_best_calling_patterns(user_id)

    # Greeting
    now=now_ist()
    h=now.hour
    greeting="Good morning" if h<12 else ("Good afternoon" if h<17 else "Good evening")
    user=st.session_state.get("user_name","")
    if "agenda" not in st.session_state: st.session_state.agenda=random.choice(AGENDAS)
    pct=round(len(high)/len(clients)*100) if clients else 0

    st.markdown(f"""<div class="greet">
      <div>
        <div class="gt">\u26a1 {now.strftime("%A, %d %B %Y")} \u00b7 {now.strftime("%I:%M %p")} IST</div>
        <div class="gn">{greeting}, {user}.</div>
        <div class="gsub">{st.session_state.agenda}</div>
      </div>
      <div class="gstats">
        <div><span class="gnum" style="color:#3fb950">{len(high)}</span><span class="glbl">Call today</span></div>
        <div><span class="gnum" style="color:#f85149">{len(at_risk)}</span><span class="glbl">At risk</span></div>
        <div><span class="gnum" style="color:var(--tx)">{fi(aum)}</span><span class="glbl">Total AUM</span></div>
        {f'<div><span class="gnum" style="color:#d29922">{out_stats["total_calls"]}</span><span class="glbl">Calls logged</span></div>' if out_stats["total_calls"]>0 else ""}
      </div>
    </div>""",unsafe_allow_html=True)

    # KPI Cards
    st.markdown(f"""<div class="kgrid">
      <div class="kc gr"><div class="kl">Total AUM</div><div class="knum">{fi(aum)}</div>
        <div class="kdesc">{len(clients)} clients \u00b7 {len(hni)} worth \u20b950L+</div>
        <div class="ksig">\u2191 Full portfolio</div></div>
      <div class="kc bl"><div class="kl">Call today</div><div class="knum">{len(high)}</div>
        <div class="kdesc">Health score 70+ \u2014 highest priority</div>
        <div class="ksig">{pct}% of your clients</div></div>
      <div class="kc rd"><div class="kl">Leaving risk</div><div class="knum">{len(at_risk)}</div>
        <div class="kdesc">May move to another advisor</div>
        <div class="ksig">\u20b9{fi(risk_aum)} at risk</div></div>
      <div class="kc am"><div class="kl">Revenue gap</div><div class="knum">{len(no_sip)}</div>
        <div class="kdesc">Has portfolio, no monthly SIP</div>
        <div class="ksig">Upsell opportunity</div></div>
      <div class="kc pu"><div class="kl">Paperwork</div><div class="knum">{len(no_nom)}</div>
        <div class="kdesc">Nominee form missing</div>
        <div class="ksig">Compliance risk</div></div>
    </div>""",unsafe_allow_html=True)

    # KPI expand buttons
    k1,k2,k3,k4,k5=st.columns(5)
    kdata=[(k1,"kaum",clients),(k2,"khigh",high),(k3,"krisk",at_risk),(k4,"ksip",no_sip),(k5,"knom",no_nom)]
    active=st.session_state.get("kpi_open",None)
    dmap={"kaum":"Total AUM","khigh":"Call today","krisk":"Leaving risk","ksip":"Revenue gap","knom":"Paperwork"}
    for col,key,lst in kdata:
        with col:
            lbl="\u25b2 Close" if active==key else f"\u25bc {len(lst)}"
            if st.button(lbl,key=f"kb_{key}",use_container_width=True):
                st.session_state.kpi_open=None if active==key else key; st.rerun()

    if active and active in {k for _,k,_ in kdata}:
        lst=next(l for _,k,l in kdata if k==active)
        rd=""
        for i,c in enumerate(lst[:10]):
            sc=c.get("score",0); pr=c.get("priority","Low")
            fill="#3fb950" if sc>=70 else ("#d29922" if sc>=45 else "#f85149")
            cc2="chi" if pr=="High" else ("chm" if pr=="Medium" else "chl")
            rd+=f"""<tr><td class="prank">#{i+1}</td>
              <td><div class="pname">{c.get("name","\u2014")}</div>
              <div class="psub">{c.get("goal","\u2014")} \u00b7 {fi(c.get("portfolio",0))}</div></td>
              <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
              <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
              <td><span class="chip {cc2}">{pr}</span></td>
              <td style="font-size:11px;color:var(--t2)">{"\u00b7".join(c.get("flags",[])[:2])}</td></tr>"""
        st.markdown(f"""<div class="kdet" style="margin-top:.75rem">
          <div class="kdet-h"><span class="kdet-t">{dmap.get(active,"")} — {len(lst)} clients</span></div>
          <table class="ptable"><thead><tr><th></th><th>Client</th><th>Score</th><th>Priority</th><th>Alerts</th></tr></thead>
          <tbody>{rd}</tbody></table></div>""",unsafe_allow_html=True)

    # Tabs — 3 only
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["Today's priorities","Analytics","WhatsApp"])

    # ── TAB 1: PRIORITY + OUTCOME TRACKING ──────────────────────────────────
    with tab1:
        # Pattern insight (only when data exists)
        if pattern_insight:
            st.info(f"\U0001f4ca {pattern_insight}")

        # Search + filter
        sf1,sf2,sf3=st.columns([3,2,1])
        with sf1: sq=st.text_input("",placeholder="\U0001f50d Search client...",label_visibility="collapsed",key="sq")
        with sf2:
            fsel=st.selectbox("",["All","Call today (High)","Medium","Needs attention","At risk","No SIP","No Nominee"],label_visibility="collapsed")
        with sf3:
            excel_data=export_excel(clients)
            st.download_button("\u2193 Export",data=excel_data,
                file_name=f"advisoriq_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

        filtered=clients
        if "High" in fsel: filtered=[c for c in clients if c.get("priority")=="High"]
        elif "Medium" in fsel: filtered=[c for c in clients if c.get("priority")=="Medium"]
        elif "Needs" in fsel: filtered=[c for c in clients if c.get("priority")=="Low"]
        elif "risk" in fsel: filtered=[c for c in clients if c.get("churn",0)>50]
        elif "SIP" in fsel: filtered=[c for c in clients if "No SIP" in c.get("flags",[])]
        elif "Nominee" in fsel: filtered=[c for c in clients if "No Nominee" in c.get("flags",[])]
        if sq: filtered=[c for c in filtered if sq.lower() in c.get("name","").lower() or sq.lower() in c.get("goal","").lower()]

        st.markdown(f"<div style='font-size:11px;color:var(--t3);font-family:IBM Plex Mono,monospace;margin:.5rem 0'>{len(filtered)} of {len(clients)} clients</div>",unsafe_allow_html=True)

        if "exp_row" not in st.session_state: st.session_state.exp_row=None

        for i,c in enumerate(filtered[:25]):
            sc=c.get("score",0); ch=c.get("churn",0); pr=c.get("priority","Low")
            fill="#3fb950" if sc>=70 else ("#d29922" if sc>=45 else "#f85149")
            chcol="#f85149" if ch>60 else ("#d29922" if ch>30 else "#3fb950")
            cc2="chi" if pr=="High" else ("chm" if pr=="Medium" else "chl")
            rank="\U0001f947" if i==0 else ("\U0001f948" if i==1 else ("\U0001f949" if i==2 else f"#{i+1}"))
            tags_h="".join(f'<span class="tag">{f}</span>' for f in c.get("flags",[])[:2])
            is_exp=st.session_state.exp_row==i

            # Outcome badge if logged
            hist={}
            if OUT_OK and user_id:
                hist=get_client_history(user_id,c.get("name",""))
            outcome_badge=""
            if hist.get("last_outcome"):
                ot=hist["last_outcome"]
                oc=OUTCOME_TYPES.get(ot,{}).get("color","#8b949e")
                ol=OUTCOME_TYPES.get(ot,{}).get("label",ot)
                outcome_badge=f'<span class="hist-pill" style="background:{oc}18;color:{oc};border:1px solid {oc}44">{ol}</span>'

            st.markdown(f"""<table class="ptable" style="margin-bottom:0"><tbody>
            <tr {"style=\'background:var(--s2)\'" if is_exp else ""}>
              <td class="prank">{rank}</td>
              <td><div class="pname">{c.get("name","\u2014")}</div>
              <div class="psub">{c.get("goal","\u2014")} \u00b7 Age {c.get("age","\u2014")} \u00b7 {fi(c.get("portfolio",0))}</div></td>
              <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
                <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
              <td><span class="chip {cc2}">{pr}</span></td>
              <td style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{chcol}">{ch}% risk</td>
              <td>{tags_h}</td>
              <td>{outcome_badge}</td>
            </tr></tbody></table>""",unsafe_allow_html=True)

            # Expand button
            _,xc=st.columns([11,1])
            with xc:
                if st.button("\u25b2" if is_exp else "\u25bc",key=f"er_{i}"):
                    st.session_state.exp_row=None if is_exp else i; st.rerun()

            # Expanded panel
            if is_exp:
                # Why this client
                if pr=="High": reason=f"Strong portfolio of {fi(c.get('portfolio',0))} with active engagement signals. Best time to pitch a top-up or new product."
                elif pr=="Medium": reason=f"Mid-range client at {fi(c.get('portfolio',0))}. A personalised call could move them to high priority — they are close."
                else: reason=f"Needs re-engagement. Start with a simple check-in — no sales pitch. Build trust first."

                # Call history
                hist_html=""
                if hist.get("total_calls",0)>0:
                    hist_html=f"""<div style="margin-top:.75rem;font-size:12px;color:var(--t2)">
                      \U0001f4de {hist['total_calls']} calls logged \u00b7 {hist['conversions']} converted ({round(hist['conversion_rate']*100)}%)
                    </div>"""

                st.markdown(f"""<div class="xin">
                  <div class="xlbl">Why this client</div>
                  <div class="xtxt">{reason}</div>
                  {hist_html}
                  <div style="margin-top:8px;font-size:11px;color:var(--t3);font-family:IBM Plex Mono,monospace">
                    Since {c.get("tenure","\u2014")} \u00b7 Nominee: {c.get("nominee","\u2014")} \u00b7 SIP: {fi(c.get("sip",0)) if num(c.get("sip",0))>0 else "None"}
                  </div>
                </div>""",unsafe_allow_html=True)

                # Action buttons
                ph=c.get("phone","")
                wa_link=get_whatsapp_link(ph,f"Hi {c.get('name','').split()[0]}! I wanted to connect regarding your portfolio. Can we schedule a quick call?") if WA_OK and ph else f"https://wa.me/?text=Hi!"

                acol1,acol2,acol3=st.columns([1,1,2])
                with acol1:
                    if ph:
                        st.markdown(f'<a class="btn-wa" href="tel:{ph}" style="display:block;text-align:center;color:var(--bl);background:var(--blbg);border:1px solid var(--blbd)">\U0001f4de Call</a>',unsafe_allow_html=True)
                with acol2:
                    st.markdown(f'<a class="btn-wa" href="{wa_link}" target="_blank" style="display:block;text-align:center">\U0001f4f2 WhatsApp</a>',unsafe_allow_html=True)

                # Outcome logging
                if OUT_OK:
                    with acol3:
                        outcome_opts=["Mark outcome..."]+[v["label"] for v in OUTCOME_TYPES.values()]
                        outcome_keys=list(OUTCOME_TYPES.keys())
                        sel_out=st.selectbox("",outcome_opts,key=f"out_{i}",label_visibility="collapsed")
                        if sel_out!="Mark outcome...":
                            out_key=outcome_keys[outcome_opts.index(sel_out)-1]
                            log_outcome(user_id,c.get("name",""),ph,out_key,score=sc,portfolio=c.get("portfolio","0"))
                            st.session_state.exp_row=None; st.rerun()

    # ── TAB 2: ANALYTICS ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div style='height:.5rem'></div>",unsafe_allow_html=True)

        # Outcome performance (if data exists)
        if OUT_OK and user_id and out_stats["total_calls"]>0:
            st.markdown(f"""<div style="background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:1.1rem;margin-bottom:1.5rem;display:flex;gap:2rem">
              <div><span style="font-size:1.5rem;font-weight:700;font-family:IBM Plex Mono,monospace;color:var(--tx)">{out_stats["total_calls"]}</span><div style="font-size:11px;color:var(--t2)">Total calls</div></div>
              <div><span style="font-size:1.5rem;font-weight:700;font-family:IBM Plex Mono,monospace;color:#3fb950">{out_stats["total_converted"]}</span><div style="font-size:11px;color:var(--t2)">Converted</div></div>
              <div><span style="font-size:1.5rem;font-weight:700;font-family:IBM Plex Mono,monospace;color:#d29922">{out_stats["conversion_rate"]}%</span><div style="font-size:11px;color:var(--t2)">Conversion rate</div></div>
            </div>""",unsafe_allow_html=True)

        # 2 charts only
        gc1,gc2=st.columns(2)
        PC={"paper_bgcolor":"transparent","plot_bgcolor":"transparent",
            "font":dict(family="Plus Jakarta Sans",color="#8b949e",size=11),
            "margin":dict(l=8,r=8,t=32,b=8),"showlegend":False,
            "xaxis":dict(showgrid=False,zeroline=False,color="#8b949e",tickfont=dict(size=10)),
            "yaxis":dict(showgrid=True,gridcolor="rgba(255,255,255,.06)",zeroline=False,color="#8b949e",tickfont=dict(size=10))}
        with gc1:
            sv=[sum(num(c.get("portfolio",0)) for c in clients if c.get("priority")==p)/1e5 for p in ["High","Medium","Low"]]
            fig=go.Figure(go.Bar(x=["Call today","Medium","Low"],y=[round(v,1) for v in sv],
                marker_color=["#3fb950","#d29922","#f85149"],marker_line_width=0,
                text=[f"\u20b9{v:.1f}L" for v in sv],textposition="outside",textfont=dict(color="#e6edf3",size=10)))
            fig.update_layout(**{**PC,"title":dict(text="Portfolio by priority",font=dict(size=13,color="#e6edf3"),x=0)})
            fig.update_traces(width=0.5)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with gc2:
            scores=[c.get("score",0) for c in clients]
            bins=[0,20,40,60,80,101]; lbs=["0-20","21-40","41-60","61-80","81-100"]
            cts=[sum(1 for s in scores if bins[i]<=s<bins[i+1]) for i in range(5)]
            fig2=go.Figure(go.Bar(x=lbs,y=cts,
                marker_color=["#f85149","#f85149","#d29922","#3fb950","#3fb950"],
                marker_line_width=0,text=cts,textposition="outside",textfont=dict(color="#e6edf3",size=10)))
            fig2.update_layout(**{**PC,"title":dict(text="Client health distribution",font=dict(size=13,color="#e6edf3"),x=0)})
            fig2.update_traces(width=0.6)
            st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

        # Event intelligence
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px;font-weight:600;margin-bottom:1rem'>Event suggestions</div>",unsafe_allow_html=True)
        mid=[c for c in clients if c.get("priority")=="Medium"]
        senior=[c for c in clients if int(float(c.get("age") or 0))>=55]
        evs=[
            ("#d29922","high impact","Conversion workshop",
             f"{len(mid)} mid-priority clients are close to converting. A focused group session targeting {', '.join(c.get('name','') for c in mid[:3])} could shift 3-4 of them to high priority this quarter.",
             f"Potential: ~{fi(sum(num(c.get('portfolio',0)) for c in mid)*0.08)} uplift",
             f"Workshop \u00b7 {len(mid)} clients \u00b7 This month"),
            ("#58a6ff","medium impact","HNI portfolio review",
             f"Your {len(hni)} high-value clients contribute {round(sum(num(c.get('portfolio',0)) for c in hni)/max(aum,1)*100,1)}% of total AUM. A private 1:1 review is the strongest retention move for this group.",
             f"Retention: {fi(sum(num(c.get('portfolio',0)) for c in hni))} at stake",
             f"Private meeting \u00b7 {len(hni)} clients \u00b7 This quarter"),
            ("#a371f7","medium impact","Senior planning session",
             f"{len(senior)} clients aged 55+ need LIC maturity planning and estate structuring. This builds loyalty no competitor can easily replace.",
             "Long-term retention value",
             f"Workshop \u00b7 {len(senior)} clients \u00b7 Quarterly"),
        ]
        rows_e=""
        for tc,tag,title,body,roi,meta_str in evs:
            rows_e+=f"""<div class="evcard">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.75rem">
                <div class="evtitle">{title}</div>
                <span class="chip" style="background:{tc}18;color:{tc};border:1px solid {tc}44;font-size:10px">{tag}</span>
              </div>
              <div class="evbody">{body}</div>
              <div class="evroi">{roi}</div>
              <div class="evmeta">{"".join(f"<span>{p.strip()}</span>" for p in meta_str.split("\u00b7"))}</div>
            </div>"""
        st.markdown(f'<div class="evgrid">{rows_e}</div>',unsafe_allow_html=True)

    # ── TAB 3: WHATSAPP ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div style='height:.5rem'></div>",unsafe_allow_html=True)
        names=[c.get("name","") for c in clients if c.get("name")]
        seln=st.selectbox("Select client",names,label_visibility="collapsed")
        sel=next((c for c in clients if c.get("name")==seln),None)
        if sel:
            sc2=sel.get("score",0); ch2=sel.get("churn",0)
            scc="#3fb950" if sc2>=70 else ("#d29922" if sc2>=45 else "#f85149")
            chc="#f85149" if ch2>50 else "#3fb950"
            un=st.session_state.get("user_name","Your Advisor")
            uc=st.session_state.get("user_company","")
            ca,cb=st.columns([1,1])
            with ca:
                st.markdown(f"""<div class="wprof">
                  <div class="wpname">{sel.get("name","")}</div>
                  <div class="wprow"><span>Portfolio</span><span class="wpval">{fi(sel.get("portfolio",0))}</span></div>
                  <div class="wprow"><span>Monthly SIP</span><span class="wpval">{fi(sel.get("sip",0)) if num(sel.get("sip",0))>0 else "Not started"}</span></div>
                  <div class="wprow"><span>Health</span><span class="wpval" style="color:{scc}">{sc2}/100</span></div>
                  <div class="wprow"><span>Leaving risk</span><span class="wpval" style="color:{chc}">{ch2}%</span></div>
                  <div class="wprow"><span>Product</span><span class="wpval">{sel.get("goal","\u2014")}</span></div>
                </div>""",unsafe_allow_html=True)
                mt=st.radio("Message",["Check-in","SIP proposal","Portfolio review","Nominee update"],label_visibility="visible")
            with cb:
                tmpls={"Check-in":f"Dear {sel.get('name','')},\n\nI have been reviewing your portfolio and wanted to personally connect. There are a few things worth discussing.\n\nCould we do a quick 20-minute call this week?\n\nWarm regards,\n{un}\n{uc}",
                       "SIP proposal":f"Dear {sel.get('name','')},\n\nBased on your portfolio of {fi(sel.get('portfolio',0))}, I have a personalised SIP plan that could make a real difference over the next 10 years.\n\nCan we find 15 minutes?\n\nWarm regards,\n{un}\n{uc}",
                       "Portfolio review":f"Dear {sel.get('name','')},\n\nYour portfolio review is due. I want to make sure your investments are positioned right for the year ahead.\n\nWhen works best for a quick call?\n\nWarm regards,\n{un}\n{uc}",
                       "Nominee update":f"Dear {sel.get('name','')},\n\nI noticed your nominee details may need updating. This protects your family and takes under 10 minutes.\n\nCan I help with this?\n\nWarm regards,\n{un}\n{uc}"}
                edited=st.text_area("",tmpls[mt],height=220,label_visibility="collapsed")
                ph=sel.get("phone","")
                wt=edited.replace("\n","%0A").replace(" ","%20")
                wl=f"https://wa.me/{ph}?text={wt}" if ph else f"https://wa.me/?text={wt}"
                st.markdown(f'<br><a class="btn-wa" href="{wl}" target="_blank">\U0001f4f2 Open in WhatsApp \u2197</a>',unsafe_allow_html=True)

    # Footer
    st.markdown("<br><hr>",unsafe_allow_html=True)
    mc=st.session_state.get("merged_count",0)
    ms=f" \u00b7 {mc} duplicates merged" if mc else ""
    st.markdown(f"<div style='text-align:center;font-size:11px;color:var(--t3);font-family:IBM Plex Mono,monospace'>AdvisorIQ \u00b7 {len(clients)} clients \u00b7 {fi(aum)} AUM{ms}</div>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    with st.sidebar:
        if st.button("Upload new"): st.session_state.screen="upload"; st.rerun()
        if st.button("Settings"): st.session_state.screen="settings"; st.rerun()
        if st.button("Sign out"):
            try: st.query_params.clear()
            except: pass
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # Persistent login
    if "user_id" not in st.session_state:
        try:
            uid=st.query_params.get("uid",None)
            if uid and DB_OK:
                user=db.get_user(int(uid))
                if user:
                    st.session_state.user_id=user["id"]
                    st.session_state.user_name=user["full_name"]
                    st.session_state.user_company=user["company"]
                    st.session_state.user_role=user["role"]
                    st.session_state.user_plan=user.get("plan","free")
                    saved=db.load_clients(user["id"])
                    if saved: st.session_state.clients=saved
                    st.session_state.screen="dashboard" if saved else "upload"
        except: pass

    if "screen" not in st.session_state: st.session_state.screen="login"
    if "user_id" not in st.session_state and st.session_state.screen!="login":
        st.session_state.screen="login"
    screen=st.session_state.screen

    if screen=="login":   show_login();   return
    if screen=="settings":show_settings();return

    if screen=="upload":
        up=show_upload()
        if up:
            try:
                df=pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                st.session_state.upload_df=df; st.session_state.screen="map"; st.rerun()
            except Exception as e: st.error(f"Could not read file: {e}")
        return

    if screen=="map":
        if st.session_state.get("use_demo"):
            clients=prep_demo(); st.session_state.clients=clients
            if DB_OK: db.save_clients(st.session_state.user_id,clients)
            st.session_state.use_demo=False; st.session_state.screen="dashboard"; st.rerun()
        elif "upload_df" in st.session_state: show_mapping(st.session_state.upload_df)
        else: st.session_state.screen="upload"; st.rerun()
        return

    if screen=="dashboard":
        clients=st.session_state.get("clients",[])
        if not clients: st.session_state.screen="upload"; st.rerun(); return
        show_dashboard(clients); return

if __name__=="__main__": main()
