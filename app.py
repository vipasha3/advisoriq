import streamlit as st
import pandas as pd
import datetime, random, json
import plotly.graph_objects as go
import urllib.parse
from io import BytesIO
from ml_model import predict_batch


# ── Module imports (graceful fallback) ───────────────────────────────────────
try:
    import database as db
    DB_OK = True
except Exception as e:
    DB_OK = False
    st.error(f"database.py import failed: {e}")

try:
    from scoring import auto_map_columns, process_dataframe, fmt_inr, months_ago
    SCORING_OK = True
except Exception as e:
    SCORING_OK = False

# ── Self-contained fallbacks (always work even if scoring.py fails) ───────────
import datetime as _dt

def fmt_inr(v):
    try: n=float(str(v).replace(",","").replace("\u20b9","") or 0)
    except: n=0
    if n>=1e7: return f"\u20b9{n/1e7:.1f}Cr"
    if n>=1e5: return f"\u20b9{n/1e5:.1f}L"
    if n>=1e3: return f"\u20b9{n/1e3:.0f}K"
    return f"\u20b9{int(n)}"

def _mago(d):
    if not d or str(d).strip() in ("","nan","None","NaT"): return 12
    try:
        dt=pd.to_datetime(str(d),dayfirst=True,errors="coerce")
        if pd.isna(dt): return 12
        return max(0,(datetime.datetime.now()-dt.to_pydatetime()).days/30)
    except: return 12

def _score_c(r):
    p=float(str(r.get("portfolio","0")).replace(",","").replace("\u20b9","") or 0)
    sip=float(str(r.get("sip","0")).replace(",","").replace("\u20b9","") or 0)
    try: age=int(float(r.get("age") or 35))
    except: age=35
    try:
        yr=int(float(str(r.get("tenure","2020")).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=3
    ma=_mago(r.get("lastContact",""))
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

def _churn_c(r):
    r2=0; ma=_mago(r.get("lastContact",""))
    sip=float(str(r.get("sip","0")).replace(",","") or 0)
    nom=str(r.get("nominee","")).lower().strip()
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

def _flags_c(r):
    f=[]; p=float(str(r.get("portfolio","0")).replace(",","") or 0)
    sip=float(str(r.get("sip","0")).replace(",","") or 0)
    ma=_mago(r.get("lastContact",""))
    nom=str(r.get("nominee","")).lower().strip()
    if p>5e6:f.append("High Value")
    if ma>6:f.append("Inactive 6m+")
    if sip==0 and p>5e5:f.append("No SIP")
    if nom=="no":f.append("No Nominee")
    if _churn_c(r)>55:f.append("Leaving Risk")
    return f

def _clean_num(v):
    try: return str(float(str(v).replace(",","").replace("\u20b9","").strip()))
    except: return "0"

def _clean_phone(v):
    if not v: return ""
    d="".join(filter(str.isdigit,str(v)))
    return ("91"+d) if len(d)==10 else d

_COL_HINTS = {
    "name":["name","client","naam"],"age":["age","umur"],
    "portfolio":["portfolio","aum","value","investment","amount","total"],
    "sip":["sip","monthly"],"lastContact":["last","date","meeting","contact","interaction"],
    "goal":["product","goal","scheme","type"],"tenure":["since","tenure","year","clientsince"],
    "nominee":["nominee","nomination"],"phone":["phone","mobile","number"],
}

def auto_map_columns(cols):
    mapping={}
    for field,hints in _COL_HINTS.items():
        for c in cols:
            cl=c.lower().replace(" ","").replace("_","")
            for h in hints:
                if h in cl: mapping[field]=c; break
            if field in mapping: break
    return mapping

def process_dataframe(df, mapping):
    defaults={"name":"","age":"","portfolio":"0","sip":"0","lastContact":"",
              "goal":"","tenure":"2020","nominee":"","phone":""}
    clients=[]
    for _,row in df.iterrows():
        c=dict(defaults)
        for key in defaults:
            col=mapping.get(key)
            if col and col in df.columns:
                val=row[col]
                if pd.notna(val) and str(val).strip() not in ("","nan","None"):
                    if key in ("portfolio","sip"): c[key]=_clean_num(val)
                    elif key=="phone": c[key]=_clean_phone(val)
                    else: c[key]=str(val).strip()
        if ML_OK:
            try:
                result=predict_batch([c]); c.update(result[0])
            except:
                c["score"]=_score_c(c); c["churn"]=_churn_c(c)
                c["conv"]=min(95,max(5,round(c["score"]*0.65+(100-c["churn"])*0.35)))
                c["priority"]="High" if c["score"]>=70 else ("Medium" if c["score"]>=45 else "Low")
        else:
            c["score"]=_score_c(c); c["churn"]=_churn_c(c)
            c["conv"]=min(95,max(5,round(c["score"]*0.65+(100-c["churn"])*0.35)))
            c["priority"]="High" if c["score"]>=70 else ("Medium" if c["score"]>=45 else "Low")
        c["flags"]=_flags_c(c)
        clients.append(c)
    seen_p={}; seen_n={}; out=[]; merged=0
    for c in clients:
        ph=c.get("phone","").strip(); nm=c.get("name","").strip().lower()
        p=float(str(c.get("portfolio","0")).replace(",","") or 0)
        if ph and len(ph)>=10 and ph in seen_p:
            ex=seen_p[ph]
            if p>float(str(ex.get("portfolio","0")).replace(",","") or 0):
                out[out.index(ex)]=c; seen_p[ph]=c
            merged+=1
        elif nm and nm in seen_n:
            ex=seen_n[nm]
            if p>float(str(ex.get("portfolio","0")).replace(",","") or 0):
                out[out.index(ex)]=c; seen_n[nm]=c
            merged+=1
        else:
            out.append(c)
            if ph and len(ph)>=10: seen_p[ph]=c
            if nm: seen_n[nm]=c
    out.sort(key=lambda x:x.get("score",0),reverse=True)
    return out

try:
    from subscription import get_plan_info, check_client_limit, can_use_whatsapp, get_upgrade_prompt, get_plan_badge_html, PLANS
    SUB_OK = True
except Exception as e:
    SUB_OK = False
    PLANS = {"free":{"clients":25},"starter":{"clients":100},"growth":{"clients":500},"firm":{"clients":99999}}

try:
    from sheets_sync import validate_sheets_url, get_sync_status
    SHEETS_OK = True
except Exception as e:
    SHEETS_OK = False

try:
    from whatsapp import get_whatsapp_link, build_message, TEMPLATES
    WA_OK = True
except Exception as e:
    WA_OK = False

try:
    from ml_model import get_model_meta, predict_batch, get_top_feature, load_models, train_models
    ML_OK = True
except Exception as e:
    ML_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdvisorIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Init DB ───────────────────────────────────────────────────────────────────
if DB_OK:
    db.init_db()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  --bg:#0d1117;--s1:#161b22;--s2:#1c2128;--s3:#21262d;
  --bd:#30363d;--bd2:#444c56;
  --tx:#e6edf3;--t2:#8b949e;--t3:#6e7681;
  --gr:#3fb950;--grbg:rgba(63,185,80,.1);--grbd:rgba(63,185,80,.3);
  --am:#d29922;--ambg:rgba(210,153,34,.1);--ambd:rgba(210,153,34,.3);
  --rd:#f85149;--rdbg:rgba(248,81,73,.1);--rdbd:rgba(248,81,73,.3);
  --bl:#58a6ff;--blbg:rgba(88,166,255,.1);--blbd:rgba(88,166,255,.3);
  --pu:#a371f7;--pubg:rgba(163,113,247,.1);--pubd:rgba(163,113,247,.3)}
*{box-sizing:border-box}
html,body,[data-testid=stAppViewContainer]{background:var(--bg)!important;color:var(--tx)!important;font-family:Inter,sans-serif!important}
[data-testid=stHeader],[data-testid=stDecoration],footer{display:none!important}
[data-testid=stSidebar]{background:var(--s1)!important;border-right:1px solid var(--bd)!important}
.block-container{padding:0!important;max-width:100%!important}
.nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;height:56px;background:var(--s1);border-bottom:1px solid var(--bd);position:sticky;top:0;z-index:200}
.nav-logo{display:flex;align-items:center;gap:10px}
.nav-icon{width:30px;height:30px;background:var(--gr);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#000}
.nav-brand{font-size:15px;font-weight:600;color:var(--tx)}.nav-brand em{color:var(--gr);font-style:normal}
.nav-right{display:flex;align-items:center;gap:10px}
.nav-user{font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace}
.nav-role{font-size:11px;padding:2px 8px;border-radius:12px;background:var(--grbg);color:var(--gr);border:1px solid var(--grbd);font-weight:600}
.bc{padding:8px 1.5rem;background:var(--s1);border-bottom:1px solid var(--bd);font-size:12px;color:var(--t3);font-family:'JetBrains Mono',monospace}
.bc em{color:var(--bl);font-style:normal}
.wrap{padding:1.5rem;max-width:1440px;margin:0 auto}
.greet{display:flex;align-items:center;justify-content:space-between;background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem}
.gt{font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--gr);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.gn{font-size:1.4rem;font-weight:600;letter-spacing:-.4px;margin-bottom:4px}
.gsub{font-size:13px;color:var(--t2)}
.gstats{display:flex;gap:2rem;text-align:right}
.gst{font-family:'JetBrains Mono',monospace}
.gnum{font-size:1.4rem;font-weight:700;display:block}
.glbl{font-size:11px;color:var(--t2);margin-top:2px;display:block}
.kgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:.5rem}
.kc{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;transition:border-color .15s,transform .15s}
.kc:hover{border-color:var(--bd2);transform:translateY(-2px)}
.kc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.kc.gr::before{background:var(--gr)}.kc.bl::before{background:var(--bl)}.kc.rd::before{background:var(--rd)}.kc.am::before{background:var(--am)}.kc.pu::before{background:var(--pu)}
.kl{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;margin-bottom:10px}
.kc.gr .kl{color:var(--gr)}.kc.bl .kl{color:var(--bl)}.kc.rd .kl{color:var(--rd)}.kc.am .kl{color:var(--am)}.kc.pu .kl{color:var(--pu)}
.knum{font-size:2rem;font-weight:700;letter-spacing:-.04em;line-height:1;margin-bottom:5px}
.kdesc{font-size:12px;color:var(--t2);line-height:1.4;margin-bottom:8px}
.ksig{font-size:11px;font-family:'JetBrains Mono',monospace;padding-top:8px;border-top:1px solid var(--bd)}
.kc.gr .ksig{color:var(--gr)}.kc.bl .ksig{color:var(--bl)}.kc.rd .ksig{color:var(--rd)}.kc.am .ksig{color:var(--am)}.kc.pu .ksig{color:var(--pu)}
.khint{font-size:10px;color:var(--t3);margin-top:3px;font-family:'JetBrains Mono',monospace}
.kdet{background:var(--s2);border:1px solid var(--bd2);border-radius:10px;padding:1.25rem;margin-bottom:1.5rem}
.kdet-h{display:flex;align-items:center;margin-bottom:.875rem;padding-bottom:.75rem;border-bottom:1px solid var(--bd)}
.kdet-t{font-size:14px;font-weight:600}
.mhd{display:flex;align-items:center;gap:12px;padding-bottom:1rem;border-bottom:1px solid var(--bd);margin-bottom:1.25rem}
.mic{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
.mgr{background:var(--grbg);border:1px solid var(--grbd)}.mbl{background:var(--blbg);border:1px solid var(--blbd)}.mam{background:var(--ambg);border:1px solid var(--ambd)}.mpu{background:var(--pubg);border:1px solid var(--pubd)}.mrd{background:var(--rdbg);border:1px solid var(--rdbd)}
.mtitle{font-size:14px;font-weight:600;margin-bottom:2px}.msub{font-size:12px;color:var(--t2)}
.ptable{width:100%;border-collapse:collapse;font-size:13px}
.ptable thead th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;color:var(--t3);font-weight:500;padding:8px 12px;border-bottom:1px solid var(--bd);text-align:left}
.ptable tbody tr{border-bottom:1px solid var(--bd);cursor:pointer;transition:background .1s}
.ptable tbody tr:hover,.ptable tbody tr.xp{background:var(--s2)}
.prank{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--t3);width:44px}
.pname{font-weight:600;font-size:13px}.psub{font-size:11px;color:var(--t2);margin-top:2px}
.xin{padding:.875rem 1.1rem;border-left:3px solid var(--bl);margin:0 0 4px 44px;background:var(--s2)}
.xlbl{font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--bl);text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;font-weight:600}
.xtxt{font-size:13px;color:var(--t2);line-height:1.6}
.sbar{display:inline-flex;align-items:center;gap:8px}
.strack{width:52px;height:3px;border-radius:2px;background:var(--bd2);overflow:hidden;display:inline-block;vertical-align:middle}
.sfill{height:100%;border-radius:2px}
.snum{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;min-width:22px}
.chip{display:inline-block;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;padding:2px 8px;border-radius:12px}
.chi{background:var(--grbg);color:var(--gr);border:1px solid var(--grbd)}
.chm{background:var(--ambg);color:var(--am);border:1px solid var(--ambd)}
.chl{background:var(--rdbg);color:var(--rd);border:1px solid var(--rdbd)}
.tag{font-size:10px;padding:2px 7px;border-radius:8px;display:inline-block;margin-right:3px;background:var(--s3);color:var(--t2);border:1px solid var(--bd);font-family:'JetBrains Mono',monospace}
.acard{border:1px solid var(--bd);border-radius:8px;padding:1.25rem;margin-bottom:16px;background:var(--s2)}
.acard:hover{border-color:var(--bd2)}
.atop{display:flex;align-items:flex-start;gap:10px;margin-bottom:.875rem}
.abadge{font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;flex-shrink:0;margin-top:2px;font-family:'JetBrains Mono',monospace;text-transform:uppercase}
.bhi{background:var(--ambg);color:var(--am);border:1px solid var(--ambd)}
.bur{background:var(--rdbg);color:var(--rd);border:1px solid var(--rdbd)}
.bgr{background:var(--grbg);color:var(--gr);border:1px solid var(--grbd)}
.bbl{background:var(--blbg);color:var(--bl);border:1px solid var(--blbd)}
.achan{font-size:11px;color:var(--t2);margin-bottom:4px}
.atitle{font-size:14px;font-weight:600;margin-bottom:.625rem}
.areason{font-size:13px;color:var(--t2);line-height:1.65;margin-bottom:.625rem}
.aimpact{font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--gr);font-weight:600;margin-bottom:.75rem}
.waq{background:var(--s3);border:1px solid var(--bd2);border-radius:6px;padding:.875rem}
.waql{font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--bl);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;display:block;font-weight:600}
.waqm{font-size:13px;color:var(--t2);font-style:italic;line-height:1.55}
.abtns{display:flex;gap:8px;margin-top:.875rem}
.btn-wa{font-size:12px;padding:5px 14px;border-radius:6px;font-weight:600;background:rgba(37,211,102,.1);color:#25d366;border:1px solid rgba(37,211,102,.3);text-decoration:none;font-family:'JetBrains Mono',monospace;display:inline-block}
.evgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.evcard{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:1.25rem}
.evcard:hover{border-color:var(--bd2)}
.evtop{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.875rem}
.evtitle{font-size:15px;font-weight:600;margin-bottom:0}
.evbody{font-size:13px;color:var(--t2);line-height:1.65;margin-bottom:.875rem}
.evroi{font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--gr);font-weight:600;margin-bottom:.5rem}
.evmeta{display:flex;gap:14px;font-size:11px;color:var(--t3);font-family:'JetBrains Mono',monospace}
.mlhdr{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1.5fr 1fr;padding:8px 14px;border-bottom:1px solid var(--bd)}
.mlhdr span{font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;color:var(--t3);font-weight:500}
.mlrow{display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1.5fr 1fr;padding:12px 14px;border-bottom:1px solid var(--bd);cursor:pointer;transition:background .1s;align-items:center}
.mlrow:hover,.mlex{background:var(--s2)}
.mlxpand{background:var(--s3);border-left:3px solid var(--bl);padding:.875rem 1rem;margin:0 14px 8px;border-radius:0 6px 6px 0}
.mlfl{font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--bl);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;font-weight:600}
.mlft{font-size:13px;color:var(--t2);line-height:1.5}
.tup{color:var(--gr);font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600}
.tdn{color:var(--rd);font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600}
.tsb{color:var(--am);font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600}
.cbr{display:inline-flex;align-items:center;gap:5px}
.cbar{height:3px;border-radius:2px;background:var(--bd2);width:36px;overflow:hidden;display:inline-block;vertical-align:middle}
.cfill{height:100%;background:var(--bl);border-radius:2px}
.wprof{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:1.1rem;margin-bottom:1rem}
.wpname{font-size:15px;font-weight:700;margin-bottom:10px}
.wprow{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--bd);font-size:13px;color:var(--t2)}
.wprow:last-child{border:none}.wpval{font-family:'JetBrains Mono',monospace;color:var(--tx)}
.uph{text-align:center;padding:5rem 2rem 2rem}
.upey{font-size:11px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.15em;color:var(--gr);margin-bottom:1rem}
.upt{font-size:2.5rem;font-weight:700;letter-spacing:-.05em;line-height:1.15;margin-bottom:.75rem}
.upt em{color:var(--gr);font-style:normal}
.ups{font-size:14px;color:var(--t2);max-width:480px;margin:0 auto 2rem;line-height:1.7}
.plan-card{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.25rem;text-align:center}
.plan-card.active{border-color:var(--gr);background:var(--s2)}
.plan-name{font-size:13px;font-weight:600;margin-bottom:4px}
.plan-price{font-size:1.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin-bottom:4px}
.plan-clients{font-size:11px;color:var(--t2);margin-bottom:.875rem}
.sheets-panel{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:1.25rem;margin-bottom:1.5rem}
.sync-badge{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-family:'JetBrains Mono',monospace;padding:3px 10px;border-radius:12px;background:var(--grbg);color:var(--gr);border:1px solid var(--grbd)}
.sync-dot{width:6px;height:6px;border-radius:50%;background:var(--gr);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.stButton>button{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;font-family:Inter,sans-serif!important;font-size:13px!important;font-weight:500!important;border-radius:6px!important;padding:6px 16px!important}
.stButton>button:hover{background:var(--s3)!important}
.stTextInput>div>div>input{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important;font-family:Inter,sans-serif!important;font-size:13px!important}
.stSelectbox>div>div{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important}
.stTabs [data-baseweb=tab-list]{background:var(--s2)!important;border-bottom:1px solid var(--bd)!important;padding:0 .5rem!important;gap:0!important}
.stTabs [data-baseweb=tab]{color:var(--t2)!important;font-family:Inter,sans-serif!important;font-size:13px!important;font-weight:500!important;padding:10px 16px!important;border-radius:0!important;border-bottom:2px solid transparent!important}
.stTabs [aria-selected=true]{color:var(--tx)!important;border-bottom-color:var(--bl)!important;background:transparent!important}
textarea{background:var(--s2)!important;border:1px solid var(--bd2)!important;color:var(--tx)!important;border-radius:6px!important;font-family:Inter,sans-serif!important}
.stRadio label{color:var(--t2)!important;font-size:13px!important}
.stAlert{background:var(--s2)!important;border-radius:6px!important;color:var(--t2)!important}
div[data-testid=stFileUploader]{background:var(--s1)!important;border:1px dashed var(--bd2)!important;border-radius:8px!important;padding:1rem!important}
[data-testid=stMarkdownContainer] p{color:var(--t2)!important;font-size:13px!important}
hr{border-color:var(--bd)!important}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _fi(v):
    try: n=float(str(v).replace(",","").replace("\u20b9","") or 0)
    except: n=0
    if n>=1e7: return f"\u20b9{n/1e7:.1f}Cr"
    if n>=1e5: return f"\u20b9{n/1e5:.1f}L"
    if n>=1e3: return f"\u20b9{n/1e3:.0f}K"
    return f"\u20b9{int(n)}"

_fi = fmt_inr if SCORING_OK else _fi

def _num(v):
    try: return float(str(v).replace(",","").replace("\u20b9","").strip())
    except: return 0.0

def _mago(d):
    if not d or str(d).strip() in ("","nan","None"): return 99
    try:
        dt=pd.to_datetime(str(d),dayfirst=True,errors="coerce")
        if pd.isna(dt): return 99
        return max(0,(datetime.datetime.now()-dt.to_pydatetime()).days/30)
    except: return 99

def now_ist():
    try:
        import pytz
        return datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    except: return datetime.datetime.now()+datetime.timedelta(hours=5,minutes=30)

AGENDAS=[
    "Your portfolio signals are ready. A few clients need your attention today.",
    "Fresh intelligence loaded. The engine has ranked your priorities.",
    "Three things need your eye before you close today.",
    "Your clients health scores are updated. Let make today count.",
    "Intelligence engine active. Your best opportunities are surfaced.",
]

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
    {"name":"Archana Desai","age":"56","portfolio":"4200000","sip":"0","lastContact":"2023-09-05","goal":"LIC+Bonds","tenure":"2009","nominee":"Yes","phone":"9876543219"},
    {"name":"Sunita Shah","age":"45","portfolio":"1200000","sip":"8000","lastContact":"2023-09-20","goal":"MF","tenure":"2018","nominee":"No","phone":"9876543220"},
    {"name":"Arun Trivedi","age":"48","portfolio":"900000","sip":"0","lastContact":"2023-06-15","goal":"LIC","tenure":"2015","nominee":"No","phone":"9876543221"},
    {"name":"Vijay Solanki","age":"50","portfolio":"650000","sip":"6000","lastContact":"2023-08-10","goal":"MF","tenure":"2019","nominee":"No","phone":"9876543222"},
    {"name":"Bhavesh Modi","age":"44","portfolio":"520000","sip":"7500","lastContact":"2024-03-10","goal":"MF","tenure":"2020","nominee":"No","phone":"9876543223"},
    {"name":"Jigar Shah","age":"47","portfolio":"1750000","sip":"9000","lastContact":"2023-12-18","goal":"MF+LIC","tenure":"2017","nominee":"No","phone":"9876543224"},
    {"name":"Hetal Trivedi","age":"39","portfolio":"430000","sip":"6000","lastContact":"2024-02-20","goal":"MF","tenure":"2021","nominee":"No","phone":"9876543225"},
    {"name":"Dinesh Mehta","age":"38","portfolio":"350000","sip":"5000","lastContact":"2024-02-28","goal":"SIP","tenure":"2022","nominee":"No","phone":"9876543226"},
    {"name":"Kalpesh Vora","age":"36","portfolio":"210000","sip":"3000","lastContact":"2024-01-30","goal":"SIP","tenure":"2023","nominee":"No","phone":"9876543227"},
    {"name":"Priya Desai","age":"32","portfolio":"180000","sip":"4000","lastContact":"2024-02-10","goal":"SIP","tenure":"2023","nominee":"No","phone":"9876543228"},
    {"name":"Nilesh Mehta","age":"33","portfolio":"95000","sip":"2000","lastContact":"2024-03-05","goal":"SIP","tenure":"2024","nominee":"No","phone":"9876543229"},
]

def prep_demo():
    if SCORING_OK:
        from scoring import auto_map_columns
        mapping = {k: k for k in ["name","age","portfolio","sip","lastContact","goal","tenure","nominee","phone"]}
        clients = process_dataframe(pd.DataFrame(DEMO), mapping)
    else:
        clients = DEMO
    clients.sort(key=lambda x: x.get("score",0), reverse=True)
    return clients

def export_excel(clients):
    rows = [{
        "Client Name": c.get("name",""),
        "Age": c.get("age",""),
        "Portfolio (INR)": _num(c.get("portfolio",0)),
        "Monthly SIP (INR)": _num(c.get("sip",0)),
        "Health Score": c.get("score",0),
        "Churn Risk (%)": c.get("churn",0),
        "Conv. Prob (%)": c.get("conv",0),
        "Priority": c.get("priority",""),
        "Product / Goal": c.get("goal",""),
        "Last Contact": c.get("lastContact",""),
        "Tenure (Since)": c.get("tenure",""),
        "Nominee": c.get("nominee",""),
        "Phone": c.get("phone",""),
        "Flags": " | ".join(c.get("flags",[])),
    } for c in clients]
    df = pd.DataFrame(rows)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Client Intelligence")
        ws = writer.sheets["Client Intelligence"]
        for col in ws.columns:
            mx = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(mx+4, 40)
    buf.seek(0)
    return buf.getvalue()

PC={"paper_bgcolor":"#0d1117","plot_bgcolor":"#161b22",
    "font":dict(family="Inter,sans-serif",color="#8b949e",size=11),
    "margin":dict(l=8,r=8,t=32,b=8),"showlegend":False,
    "xaxis":dict(showgrid=False,zeroline=False,color="#8b949e",tickfont=dict(size=10),linecolor="#30363d"),
    "yaxis":dict(showgrid=True,gridcolor="#21262d",zeroline=False,color="#8b949e",tickfont=dict(size=10))}

# ── NAV ───────────────────────────────────────────────────────────────────────
def show_nav():
    user = st.session_state.get("user_name","")
    company = st.session_state.get("user_company","")
    role = st.session_state.get("user_role","advisor")
    plan = st.session_state.get("user_plan","free")
    rl = "Owner" if role=="owner" else "Advisor"
    plan_colors = {"free":"#6e7681","starter":"#58a6ff","growth":"#3fb950","firm":"#d29922"}
    pc = plan_colors.get(plan,"#6e7681")
    st.markdown(f"""<div class="nav">
      <div class="nav-logo">
        <div class="nav-icon">\u26a1</div>
        <span class="nav-brand">Advisor<em>IQ</em></span>
      </div>
      <div class="nav-right">
        <span class="nav-user">{user} \u00b7 {company}</span>
        <span class="nav-role">{rl}</span>
        <span style="font-size:11px;padding:2px 8px;border-radius:12px;background:{pc}18;color:{pc};border:1px solid {pc}44;font-family:JetBrains Mono,monospace;font-weight:600">{plan.upper()}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns([6,1,1,1])
    with c2:
        if st.button("\u2b06 Upload", key="nav_up"):
            st.session_state.pop("kpi_open", None)
            st.session_state.screen = "upload"; st.rerun()
    with c3:
        if st.button("\u2699 Settings", key="nav_set"):
            st.session_state.screen = "settings"; st.rerun()
    with c4:
        if st.button("Sign out", key="nav_so"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def show_login():
    _,col,_ = st.columns([1,1,1])
    with col:
        st.markdown("""<div style="text-align:center;margin-top:3rem;margin-bottom:2rem">
          <div style="width:48px;height:48px;background:#3fb950;border-radius:10px;
            display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#000;margin-bottom:.875rem">\u26a1</div>
          <div style="font-size:1.3rem;font-weight:700;letter-spacing:-.3px;color:#e6edf3">AdvisorIQ</div>
          <div style="font-size:13px;color:#8b949e;margin-top:4px">Portfolio intelligence platform</div>
        </div>""", unsafe_allow_html=True)
        t1,t2 = st.tabs(["Sign in","Create account"])
        with t1:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="your.username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022", key="li_p")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Sign in \u2192", use_container_width=True, key="li_b"):
                if u and p:
                    if DB_OK:
                        row = db.login_user(u, p)
                        if row:
                            st.session_state.user_id = row["id"]
                            st.session_state.user_name = row["full_name"]
                            st.session_state.user_company = row["company"]
                            st.session_state.user_role = row["role"]
                            st.session_state.user_plan = row.get("plan","free")
                            saved = db.load_clients(row["id"])
                            if saved: st.session_state.clients = saved
                            st.session_state.screen = "upload" if not saved else "dashboard"
                            st.rerun()
                        else: st.error("Incorrect username or password.")
                    else: st.error("Database not available.")
                else: st.warning("Please enter both fields.")
        with t2:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            rn = st.text_input("Full name", placeholder="Ramesh Patel", key="r_n")
            rc = st.text_input("Company", placeholder="Patel Wealth Advisory", key="r_c")
            ru = st.text_input("Username", placeholder="ramesh.patel", key="r_u")
            rp = st.text_input("Password", type="password", placeholder="Min 6 characters", key="r_p")
            rr = st.selectbox("Role", ["Owner / Director","Senior Advisor","Advisor","Team Member"], key="r_r")
            rm = {"Owner / Director":"owner","Senior Advisor":"advisor","Advisor":"advisor","Team Member":"staff"}
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Create account \u2192", use_container_width=True, key="r_b"):
                if all([rn,rc,ru,rp]):
                    if len(rp) < 6: st.warning("Password must be at least 6 characters.")
                    elif DB_OK:
                        ok,msg = db.create_user(ru, rp, rn, rc, rm[rr])
                        if ok: st.success(f"Account created! Sign in with: {ru}")
                        else: st.error(msg)
                else: st.warning("Please fill in all fields.")

# ── UPLOAD ────────────────────────────────────────────────────────────────────
def show_upload():
    show_nav()
    st.markdown('<div class="bc">Upload \u2192 Mapping \u2192 <em>Dashboard</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    # Sheets sync status banner
    user_id = st.session_state.get("user_id")
    if SHEETS_OK and user_id:
        status = get_sync_status(user_id)
        if status.get("has_sheets"):
            last = status.get("last_synced","Never")
            st.markdown(f"""<div class="sheets-panel">
              <div style="display:flex;align-items:center;justify-content:space-between">
                <div>
                  <span class="sync-badge"><span class="sync-dot"></span> Google Sheets Connected</span>
                  <span style="font-size:12px;color:#8b949e;margin-left:10px;font-family:JetBrains Mono,monospace">Last synced: {last}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            c_sync, c_view, _ = st.columns([1,1,5])
            with c_sync:
                if st.button("\u21bb Sync now", use_container_width=True):
                    with st.spinner("Syncing Google Sheets..."):
                        from sheets_sync import sync_user_sheets
                        import ml_model as ml_mod
                        result = sync_user_sheets(user_id, status["sheets_url"], db, ml_mod)
                        if result.get("changed"):
                            saved = db.load_clients(user_id)
                            if saved:
                                st.session_state.clients = saved
                                st.success(f"\u2713 Synced {result['rows']} clients")
                                st.session_state.screen = "dashboard"; st.rerun()
                        else:
                            st.info("No changes detected in your sheet.")
            with c_view:
                if st.button("View dashboard \u2192", use_container_width=True):
                    st.session_state.screen = "dashboard"; st.rerun()
            st.markdown("<hr>", unsafe_allow_html=True)

    clients = st.session_state.get("clients", [])
    if clients and not (SHEETS_OK and user_id and get_sync_status(user_id).get("has_sheets")):
        st.success(f"\u2713 {len(clients)} clients loaded from your last session.")
        cc,_ = st.columns([1,4])
        with cc:
            if st.button("View dashboard \u2192", use_container_width=True):
                st.session_state.screen = "dashboard"; st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""<div class="uph">
      <div class="upey">\u26a1 Intelligence Engine</div>
      <div class="upt">Your clients,<br><em>clearly ranked.</em></div>
      <div class="ups">Upload any Excel or CSV \u2014 or connect Google Sheets for automatic daily sync. The engine scores every client and tells you exactly who to call.</div>
    </div>""", unsafe_allow_html=True)

    _,cc,_ = st.columns([1,2,1])
    with cc:
        uploaded = st.file_uploader("", type=["xlsx","xls","csv"], label_visibility="collapsed")
        st.markdown("<div style='text-align:center;font-size:11px;color:#6e7681;margin-top:.5rem'>Any column format \u00b7 Excel or CSV</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("\U0001f4ca Load demo data \u2192", use_container_width=True):
            st.session_state.use_demo = True; st.session_state.screen = "map"; st.rerun()

        if SHEETS_OK:
            st.markdown("<div style='text-align:center;font-size:12px;color:#8b949e;margin:.75rem 0 .5rem'>\u2014 or connect Google Sheets \u2014</div>", unsafe_allow_html=True)
            sheets_url = st.text_input("", placeholder="https://docs.google.com/spreadsheets/d/...", label_visibility="collapsed", key="sheets_url_input")
            if st.button("Connect Google Sheets \u2192", use_container_width=True, key="connect_sheets"):
                if sheets_url.strip():
                    with st.spinner("Validating sheet access..."):
                        valid, msg = validate_sheets_url(sheets_url)
                    if valid:
                        db.update_sheets_url(user_id, sheets_url)
                        st.success(f"\u2713 {msg}")
                        st.session_state.screen = "map"
                        st.session_state.use_sheets = True
                        st.session_state.sheets_url = sheets_url
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please paste your Google Sheets URL.")

    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded

# ── MAPPING ───────────────────────────────────────────────────────────────────
def show_mapping(df):
    show_nav()
    st.markdown('<div class="bc">Upload \u2192 <em>Column mapping</em> \u2192 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("### Map your columns")
    st.caption("Auto-detected where possible \u2014 adjust if needed.")
    cols = df.columns.tolist()
    mapping = auto_map_columns(cols) if SCORING_OK else {}
    field_labels = {
        "name":"Client name","age":"Age","portfolio":"Portfolio / AUM (\u20b9)",
        "sip":"Monthly SIP (\u20b9)","lastContact":"Last contact date",
        "goal":"Product / goal","tenure":"Client since (year)",
        "nominee":"Nominee updated?","phone":"Phone number"
    }
    all_fields = list(field_labels.keys())
    user_mapping = {}
    g = st.columns(2)
    for i, key in enumerate(all_fields):
        best = mapping.get(key)
        with g[i % 2]:
            opts = ["\u2014 skip \u2014"] + cols
            idx = (cols.index(best)+1) if best and best in cols else 0
            sel = st.selectbox(field_labels[key], opts, index=idx, key=f"m_{key}")
            user_mapping[key] = sel if sel != "\u2014 skip \u2014" else None
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,_ = st.columns([1,1,4])
    with c1:
        if st.button("Run engine \u2192", use_container_width=True):
            with st.spinner("Processing and scoring all clients..."):
                 

                if SCORING_OK:
                    clients = process_dataframe(df, user_mapping)
                    clients = predict_batch(clients)   # 👈 MOST IMPORTANT LINE
                    merged = 0
                else:
                    clients = df.to_dict("records")
                    clients = predict_batch(clients)   # 👈 aa pan add karo safety mate
                    merged = 0
            st.session_state.clients = clients
            st.session_state.merged_count = merged
            if DB_OK:
                db.save_clients(st.session_state.user_id, clients)
            st.session_state.screen = "dashboard"; st.rerun()
    with c2:
        if st.button("\u2190 Back"):
            st.session_state.screen = "upload"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
def show_settings():
    show_nav()
    st.markdown('<div class="bc">\u2192 <em>Settings</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    user_id = st.session_state.get("user_id")
    plan = st.session_state.get("user_plan","free")

    t1, t2, t3 = st.tabs(["Google Sheets", "Subscription", "ML Model Info"])

    with t1:
        st.markdown('\n        <div class="mhd" style="margin-top:.5rem">\n          <div class="mic mgr">\U0001f4ca</div>\n          <div><div class="mtitle">Google Sheets Auto-Sync</div>\n          <div class="msub">Connect your client Excel \u2192 Google Sheets \u2192 App syncs automatically every 5 minutes</div></div>\n        </div>', unsafe_allow_html=True)
        if SHEETS_OK and user_id:
            status = get_sync_status(user_id)
            if status.get("has_sheets"):
                st.success(f"\u2713 Connected: {status['sheets_url'][:60]}...")
                st.info(f"Last synced: {status.get('last_synced','Never')} \u00b7 Syncs every {status.get('sync_interval_min',5)} minutes")
                if st.button("Disconnect sheet"):
                    db.update_sheets_url(user_id, "")
                    st.success("Disconnected."); st.rerun()
            else:
                new_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...", key="settings_sheets")
                if st.button("Connect \u2192", key="settings_connect"):
                    if new_url.strip():
                        valid, msg = validate_sheets_url(new_url)
                        if valid:
                            db.update_sheets_url(user_id, new_url)
                            st.success(f"\u2713 {msg}"); st.rerun()
                        else: st.error(msg)
                st.markdown("""<div style='font-size:12px;color:#8b949e;margin-top:1rem;line-height:1.7'>
                  <strong style='color:#e6edf3'>How to set up:</strong><br>
                  1. Go to <a href='https://sheets.google.com' target='_blank' style='color:#58a6ff'>sheets.google.com</a> \u2192 Create new sheet<br>
                  2. Copy your Excel data into the sheet<br>
                  3. Click Share \u2192 copy the link \u2192 paste above<br>
                  4. App will auto-sync every 5 minutes when you update the sheet
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("Google Sheets requires gspread package. Run: pip install gspread google-auth")

    with t2:
        st.markdown('\n        <div class="mhd" style="margin-top:.5rem">\n          <div class="mic mam">\U0001f4b0</div>\n          <div><div class="mtitle">Subscription Plans</div>\n          <div class="msub">Upgrade to unlock more clients, WhatsApp automation, and API access</div></div>\n        </div>', unsafe_allow_html=True)
        plan_data = [
            ("free","Free","\u20b90","25 clients","—","—"),
            ("starter","Starter","\u20b91,999/mo","100 clients","—","—"),
            ("growth","Growth","\u20b94,999/mo","500 clients","\u2713","—"),
            ("firm","Firm","\u20b912,999/mo","Unlimited","\u2713","\u2713"),
        ]
        p1,p2,p3,p4 = st.columns(4)
        for col,(pid,pname,pprice,pclients,pwa,papi) in zip([p1,p2,p3,p4],plan_data):
            is_active = plan == pid
            with col:
                border = "border:1px solid #3fb950" if is_active else "border:1px solid #30363d"
                st.markdown(f"""<div style="background:#161b22;{border};border-radius:10px;padding:1.25rem;text-align:center;margin-bottom:1rem">
                  <div style="font-size:13px;font-weight:600;margin-bottom:6px">{pname}</div>
                  <div style="font-size:1.4rem;font-weight:700;font-family:JetBrains Mono,monospace;color:#e6edf3;margin-bottom:4px">{pprice}</div>
                  <div style="font-size:11px;color:#8b949e;margin-bottom:.875rem">{pclients}</div>
                  <div style="font-size:11px;color:#8b949e">WhatsApp: {pwa}</div>
                  <div style="font-size:11px;color:#8b949e">API: {papi}</div>
                  {"<div style=\'margin-top:.875rem;font-size:11px;color:#3fb950;font-family:JetBrains Mono,monospace;font-weight:600\'>CURRENT PLAN</div>" if is_active else ""}
                </div>""", unsafe_allow_html=True)
                if not is_active and pid != "free":
                    if st.button(f"Upgrade \u2192 {pname}", key=f"up_{pid}", use_container_width=True):
                        st.info(f"Razorpay integration: set RAZORPAY_KEY_ID in .env to enable payments for {pname} plan.")

    with t3:
        st.markdown('\n        <div class="mhd" style="margin-top:.5rem">\n          <div class="mic mpu">\u25c8</div>\n          <div><div class="mtitle">ML Model Information</div>\n          <div class="msub">GradientBoosting models for priority scoring and churn prediction</div></div>\n        </div>', unsafe_allow_html=True)
        if ML_OK:
            meta = get_model_meta()
            if meta:
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Priority model AUC", f"{meta.get('priority_auc',0):.4f}")
                with c2:
                    st.metric("Churn model AUC", f"{meta.get('churn_auc',0):.4f}")
                with c3:
                    st.metric("Training samples", f"{meta.get('n_training_samples',0):,}")
                st.caption(f"Trained at: {meta.get('trained_at','Unknown')} \u00b7 Features: {meta.get('n_features',0)}")
                if st.button("\U0001f504 Retrain models"):
                    with st.spinner("Retraining... this takes ~30 seconds"):
                        new_meta = train_models(force=True)
                    st.success(f"\u2713 Retrained \u2014 Priority AUC: {new_meta['priority_auc']}, Churn AUC: {new_meta['churn_auc']}")
            else:
                if st.button("Train models now \u2192"):
                    with st.spinner("Training ML models..."):
                        meta = train_models(force=True)
                    st.success(f"\u2713 Done \u2014 Priority AUC: {meta['priority_auc']}")
        else:
            st.warning("ML module not available. Check ml_model.py.")

    st.markdown("</div>", unsafe_allow_html=True)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def show_dashboard(clients):
    show_nav()
    st.markdown('<div class="bc">Upload \u2192 Mapping \u2192 <em>Intelligence Dashboard</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)

    aum = sum(_num(c.get("portfolio",0)) for c in clients)
    high = [c for c in clients if c.get("priority")=="High"]
    at_risk = [c for c in clients if c.get("churn",0)>50]
    no_sip = [c for c in clients if "No SIP" in c.get("flags",[])]
    no_nom = [c for c in clients if "No Nominee" in c.get("flags",[])]
    hni = [c for c in clients if "High Value" in c.get("flags",[])]
    risk_aum = sum(_num(c.get("portfolio",0)) for c in at_risk)

    user = st.session_state.get("user_name","")
    now = now_ist()
    greeting = "Good morning" if now.hour<12 else ("Good afternoon" if now.hour<17 else "Good evening")
    if "agenda" not in st.session_state: st.session_state.agenda = random.choice(AGENDAS)
    agenda = st.session_state.agenda
    pct = round(len(high)/len(clients)*100) if clients else 0

    # Sheets sync indicator
    user_id = st.session_state.get("user_id")
    if SHEETS_OK and user_id:
        status = get_sync_status(user_id)
        if status.get("has_sheets"):
            last = status.get("last_synced","Never")
            st.markdown(f'<div style="font-size:11px;color:#8b949e;font-family:JetBrains Mono,monospace;margin-bottom:.75rem">\U0001f4ca Google Sheets connected \u00b7 Last synced: {last}</div>', unsafe_allow_html=True)

    # Greeting
    st.markdown(f"""<div class="greet">
      <div>
        <div class="gt">\u26a1 {now.strftime("%A, %d %B %Y")} \u00b7 {now.strftime("%I:%M %p")} IST</div>
        <div class="gn">{greeting}, {user}.</div>
        <div class="gsub">{agenda}</div>
      </div>
      <div class="gstats">
        <div class="gst"><span class="gnum" style="color:#3fb950">{len(high)}</span><span class="glbl">Call today</span></div>
        <div class="gst"><span class="gnum" style="color:#f85149">{len(at_risk)}</span><span class="glbl">Leaving risk</span></div>
        <div class="gst"><span class="gnum" style="color:#e6edf3">{_fi(aum)}</span><span class="glbl">Total AUM</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPI cards
    st.markdown(f"""<div class="kgrid">
      <div class="kc gr"><div class="kl">Total AUM</div><div class="knum">{_fi(aum)}</div>
        <div class="kdesc">{len(clients)} clients \u00b7 {len(hni)} high-value (50L+)</div>
        <div class="ksig">\u2191 Full portfolio pipeline</div><div class="khint">\u25bc Tap below to expand</div></div>
      <div class="kc bl"><div class="kl">Ready to act</div><div class="knum">{len(high)}</div>
        <div class="kdesc">Health score 70+ \u2014 call these first</div>
        <div class="ksig">{pct}% of your client base</div><div class="khint">\u25bc Tap to see who</div></div>
      <div class="kc rd"><div class="kl">Leaving risk</div><div class="knum">{len(at_risk)}</div>
        <div class="kdesc">May move to another advisor soon</div>
        <div class="ksig">\u26a0 {_fi(risk_aum)} at risk</div><div class="khint">\u25bc Tap to see who</div></div>
      <div class="kc am"><div class="kl">Revenue gap</div><div class="knum">{len(no_sip)}</div>
        <div class="kdesc">Portfolio but no monthly SIP</div>
        <div class="ksig">Easy SIP upsell opportunity</div><div class="khint">\u25bc Tap to see who</div></div>
      <div class="kc pu"><div class="kl">Paperwork due</div><div class="knum">{len(no_nom)}</div>
        <div class="kdesc">Nominee form not filed</div>
        <div class="ksig">Compliance risk for family</div><div class="khint">\u25bc Tap to see who</div></div>
    </div>""", unsafe_allow_html=True)

    # KPI expand buttons
    k1,k2,k3,k4,k5 = st.columns(5)
    kdata = [(k1,"kaum","Total AUM",clients),(k2,"khigh","Ready to act",high),
             (k3,"krisk","Leaving risk",at_risk),(k4,"ksip","Revenue gap",no_sip),(k5,"knom","Paperwork due",no_nom)]
    active = st.session_state.get("kpi_open", None)
    for col,key,label,lst in kdata:
        with col:
            lbl = "\u25b2 Close" if active==key else f"\u25bc {len(lst)} clients"
            if st.button(lbl, key=f"kb_{key}", use_container_width=True):
                st.session_state.kpi_open = None if active==key else key; st.rerun()

    if active:
        dmap = {
            "kaum":("Total AUM breakdown",clients),
            "khigh":("Ready to act \u2014 call these first",high),
            "krisk":("Leaving risk \u2014 contact urgently",at_risk),
            "ksip":("Revenue gap \u2014 no SIP despite portfolio",no_sip),
            "knom":("Paperwork due \u2014 nominee missing",no_nom),
        }
        if active in dmap:
            dlbl, dlst = dmap[active]
            rd = ""
            for i,c in enumerate(dlst[:10]):
                sc=c.get("score",0); pr=c.get("priority","Low")
                fill="#3fb950" if sc>=70 else ("#d29922" if sc>=45 else "#f85149")
                cc2="chi" if pr=="High" else ("chm" if pr=="Medium" else "chl")
                rd += f"""<tr><td class="prank">#{i+1}</td>
                  <td><div class="pname">{c.get("name","\u2014")}</div>
                  <div class="psub">{c.get("goal","\u2014")} \u00b7 Age {c.get("age","\u2014")}</div></td>
                  <td style="font-family:\'JetBrains Mono\',monospace;font-size:12px">{_fi(c.get("portfolio",0))}</td>
                  <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
                  <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
                  <td><span class="chip {cc2}">{pr}</span></td>
                  <td style="font-size:11px;font-family:\'JetBrains Mono\',monospace;color:#8b949e">{" \u00b7 ".join(c.get("flags",[])[:2]) or "\u2014"}</td></tr>"""
            st.markdown(f"""<div class="kdet">
              <div class="kdet-h"><span class="kdet-t">{dlbl} <span style="font-size:12px;color:#8b949e;font-weight:400">({len(dlst)} clients)</span></span></div>
              <div style="overflow-x:auto"><table class="ptable" style="margin:0">
              <thead><tr><th></th><th>Client</th><th>Portfolio</th><th>Health score</th><th>Status</th><th>Alerts</th></tr></thead>
              <tbody>{rd}</tbody></table></div></div>""", unsafe_allow_html=True)

    # Charts
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    gc1,gc2,gc3 = st.columns(3)
    with gc1:
        sv = [sum(_num(c.get("portfolio",0)) for c in clients if c.get("priority")==p)/1e5 for p in ["High","Medium","Low"]]
        fig = go.Figure(go.Bar(x=["Ready","Medium","Needs work"],y=[round(v,1) for v in sv],
            marker_color=["#3fb950","#d29922","#f85149"],marker_line_width=0,
            text=[f"\u20b9{v:.1f}L" for v in sv],textposition="outside",textfont=dict(color="#e6edf3",size=10)))
        fig.update_layout(**{**PC,"title":dict(text="Portfolio by status",font=dict(size=12,color="#e6edf3"),x=0)})
        fig.update_traces(width=0.5)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with gc2:
        scores = [c.get("score",0) for c in clients]
        bins=[0,20,40,60,80,101]; lbs=["0-20","21-40","41-60","61-80","81-100"]
        cts=[sum(1 for s in scores if bins[i]<=s<bins[i+1]) for i in range(5)]
        fig2 = go.Figure(go.Bar(x=lbs,y=cts,marker_color=["#f85149","#f85149","#d29922","#3fb950","#3fb950"],
            marker_line_width=0,text=cts,textposition="outside",textfont=dict(color="#e6edf3",size=10)))
        fig2.update_layout(**{**PC,"title":dict(text="Health score spread",font=dict(size=12,color="#e6edf3"),x=0)})
        fig2.update_traces(width=0.6)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
    with gc3:
        sx=[c.get("score",0) for c in clients]; sy=[c.get("churn",0) for c in clients]
        sn=[c.get("name","") for c in clients]
        scc=["#f85149" if c.get("churn",0)>50 else ("#d29922" if c.get("churn",0)>25 else "#3fb950") for c in clients]
        fig3 = go.Figure(go.Scatter(x=sx,y=sy,mode="markers",
            marker=dict(color=scc,size=8,line=dict(width=0)),text=sn,
            hovertemplate="<b>%{text}</b><br>Score: %{x}<br>Risk: %{y}%<extra></extra>"))
        fig3.update_layout(**{**PC,"title":dict(text="Health vs leaving risk",font=dict(size=12,color="#e6edf3"),x=0),
            "xaxis":dict(**PC["xaxis"],title="Health score",range=[0,105]),
            "yaxis":dict(**PC["yaxis"],title="Leaving risk %",range=[0,105])})
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

    # Tabs
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Priority Rankings","Smart Next Best Action","Event Intelligence","ML Prediction Engine","WhatsApp Drafts"])

    # TAB 1
    with tab1:
        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mhd"><div class="mic mbl">\u2261</div><div><div class="mtitle">Priority Rankings</div><div class="msub">Sorted by ML health score \u00b7 Click any row for strategic recommendation</div></div></div>', unsafe_allow_html=True)
        sf1,sf2,sf3,sf4 = st.columns([3,2,1.2,1.2])
        with sf1: sq = st.text_input("", placeholder="\U0001f50d  Search client name or product...", label_visibility="collapsed", key="sq")
        with sf2: fsel = st.selectbox("",["All clients","Ready to act","Medium","Needs attention","Leaving risk","No SIP","No Nominee"],label_visibility="collapsed")
        with sf3: amin = st.number_input("Min AUM (L)", min_value=0, max_value=10000, value=0, step=10, label_visibility="collapsed", key="amin")
        with sf4: amax = st.number_input("Max AUM (L)", min_value=0, max_value=10000, value=10000, step=10, label_visibility="collapsed", key="amax")
        filtered = clients
        if "Ready" in fsel: filtered=[c for c in clients if c.get("priority")=="High"]
        elif "Medium" in fsel: filtered=[c for c in clients if c.get("priority")=="Medium"]
        elif "Needs" in fsel: filtered=[c for c in clients if c.get("priority")=="Low"]
        elif "Leaving" in fsel: filtered=[c for c in clients if c.get("churn",0)>50]
        elif "No SIP" in fsel: filtered=[c for c in clients if "No SIP" in c.get("flags",[])]
        elif "Nominee" in fsel: filtered=[c for c in clients if "No Nominee" in c.get("flags",[])]
        if sq: filtered=[c for c in filtered if sq.lower() in c.get("name","").lower() or sq.lower() in c.get("goal","").lower()]
        if amin>0 or amax<10000: filtered=[c for c in filtered if amin*1e5<=_num(c.get("portfolio",0))<=amax*1e5]
        rc1,rc2 = st.columns([5,1])
        with rc1: st.markdown(f"<div style='font-size:11px;color:#6e7681;font-family:JetBrains Mono,monospace;margin-bottom:.75rem'>{len(filtered)} of {len(clients)} records \u00b7 sorted by ML score</div>", unsafe_allow_html=True)
        with rc2:
            excel_data = export_excel(filtered)
            st.download_button(label="\u2193 Export", data=excel_data,
                file_name=f"advisoriq_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_btn", use_container_width=True)
        if "exp_row" not in st.session_state: st.session_state.exp_row = None
        for i,c in enumerate(filtered[:20]):
            sc=c.get("score",0); ch=c.get("churn",0); pr=c.get("priority","Low")
            fill="#3fb950" if sc>=70 else ("#d29922" if sc>=45 else "#f85149")
            chcol="#f85149" if ch>60 else ("#d29922" if ch>30 else "#3fb950")
            cc2="chi" if pr=="High" else ("chm" if pr=="Medium" else "chl")
            rank="\U0001f947" if i==0 else ("\U0001f948" if i==1 else ("\U0001f949" if i==2 else f"#{i+1}"))
            tags_h="".join(f'<span class="tag">{f}</span>' for f in c.get("flags",[])[:2])
            is_exp = st.session_state.exp_row==i
            st.markdown(f"""<table class="ptable" style="margin-bottom:0"><tbody>
            <tr class="{'xp' if is_exp else ''}">
              <td class="prank">{rank}</td>
              <td><div class="pname">{c.get("name","\u2014")}</div><div class="psub">{c.get("goal","\u2014")} \u00b7 Age {c.get("age","\u2014")}</div></td>
              <td style="font-family:\'JetBrains Mono\',monospace;font-size:12px">{_fi(c.get("portfolio",0))}</td>
              <td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#6e7681">{c.get("tenure","\u2014")}</td>
              <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
                <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
              <td><span class="chip {cc2}">{pr}</span></td>
              <td style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{chcol}">{ch}%</td>
              <td style="font-size:11px">{tags_h}</td>
            </tr></tbody></table>""", unsafe_allow_html=True)
            _,bc = st.columns([11,1])
            with bc:
                if st.button("\u25b2" if is_exp else "\u25bc", key=f"er_{i}"):
                    st.session_state.exp_row = None if is_exp else i; st.rerun()
                    
            if is_exp:
                insight = c.get("feature_importance", "No insight available")
                tag = "🧠 AI Insight" if c.get("ml_powered") else ""
                act = f"{c.get('name','')} → {insight}"
            
                st.markdown(f"""<div class="xin" style="margin-bottom:8px">
                  <div class="xlbl">{tag} Strategic Recommendation</div>
                  <div class="xtxt">{act}</div>
                  <div style="margin-top:8px;font-size:11px;color:#6e7681;font-family:'JetBrains Mono',monospace">
                    📅 Since {c.get("tenure","—")} · 📋 Nominee: {c.get("nominee","—")} · 💰 SIP: {_fi(c.get("sip",0)) if _num(c.get("sip",0))>0 else "None"}
                  </div>
                </div>""", unsafe_allow_html=True)

    # TAB 2
    with tab2:
        top_c = clients[0] if clients else {}
        risk2 = ", ".join(c.get("name","") for c in at_risk[:3]) or "\u2014"
        sip2 = ", ".join(c.get("name","") for c in no_sip[:3]) or "\u2014"
        nom2 = ", ".join(c.get("name","") for c in no_nom[:2]) or "\u2014"
        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mhd"><div class="mic mam">\u26a1</div><div><div class="mtitle">Smart Next Best Action</div><div class="msub">Decision engine \u00b7 Impact scoring \u00b7 Automated outreach templates</div></div></div>', unsafe_allow_html=True)
        tn = top_c.get("name","Top client")
        actions_data = [
            ("HIGH","bhi","WhatsApp + Follow-up call",
             f"{tn} \u2014 Deploy personalised value proposition",
             f"Gradient boosting classifier places {tn} at health score {top_c.get('score',0)}/100. Feature importance analysis shows portfolio sensitivity at {round(_num(top_c.get('portfolio',0))/(max(aum,1)/max(len(clients),1))*100,0):.0f}% of cohort mean. A targeted intervention could shift classification to high-priority with ~{top_c.get('conv',60)}% confidence.",
             f"+{_fi(_num(top_c.get('portfolio',0))*0.05)} expected revenue impact",
             f"Hi {tn.split()[0] if tn else 'there'}! We have prepared something based on your profile. Can I share the details?",
             top_c.get("phone","")),
            ("HIGH","bhi","WhatsApp + Email Sequence",
             f"{len(at_risk)} clients \u2014 Urgent churn prevention",
             f"ML model flags {risk2} as high-risk for advisor switch within 60 days. Inactivity beyond 6 months triples withdrawal probability. A portfolio health-check call \u2014 not a sales call \u2014 is the highest-impact intervention at this stage.",
             f"~{_fi(risk_aum*0.12)} recoverable if re-engaged this month",
             "Hi! I wanted to personally check in \u2014 it has been a while and I want to make sure your portfolio is well positioned. Can we connect briefly?",""),
            ("GROWTH","bgr","SIP upsell sequence",
             f"{len(no_sip)} clients \u2014 SIP conversion drive",
             f"{sip2} hold significant portfolio but have no systematic investment plan. Showing a Rs 5,000/month compound projection closing at Rs 30L over 15 years converts 60%+ of these profiles in one sitting.",
             f"+{_fi(sum(_num(c.get('portfolio',0)) for c in no_sip)*0.03)} annual SIP commission potential",
             "Hi! I have prepared a personalised growth projection based on your portfolio. The numbers are quite compelling \u2014 can I share them with you?",""),
            ("COMPLIANCE","bbl","Compliance outreach",
             f"{len(no_nom)} clients \u2014 Nominee form drive",
             f"{nom2} have not filed nominee forms \u2014 legal risk for families, compliance exposure for your practice. A 10-minute call positions you as the advisor who cares beyond commissions.",
             "Compliance risk mitigated \u00b7 High trust impact",
             "Hi! As part of our annual client care review, I noticed your nominee details may need updating. This protects your family \u2014 can we sort this quickly?",""),
        ]
        for badge,bcls,channel,title,reason,impact,wa_msg,ph in actions_data:
            wl = get_whatsapp_link(ph, wa_msg) if WA_OK and ph else f"https://wa.me/?text={urllib.parse.quote(wa_msg)}"
            st.markdown(f"""<div class="acard">
              <div class="atop"><span class="abadge {bcls}">{badge}</span>
              <div style="flex:1"><div class="achan">\U0001f4f2 {channel}</div>
              <div class="atitle">{title}</div></div></div>
              <div class="areason">{reason}</div>
              <div class="aimpact">Expected impact: {impact}</div>
              <div class="waq"><span class="waql">\U0001f4f1 WhatsApp quick send</span>
              <div class="waqm">"{wa_msg}"</div></div>
              <div class="abtns"><a class="btn-wa" href="{wl}" target="_blank">Open WhatsApp \u2197</a></div>
            </div>""", unsafe_allow_html=True)

    # TAB 3
    with tab3:
        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mhd"><div class="mic mam">\u2726</div><div><div class="mtitle">Event Intelligence</div><div class="msub">Data-driven event recommendations \u00b7 ROI projections</div></div></div>', unsafe_allow_html=True)
        hni_n = ", ".join(c.get("name","") for c in hni[:3])
        senior = [c for c in clients if int(float(c.get("age") or 0))>=55]
        mid = [c for c in clients if c.get("priority")=="Medium"]
        mid_n = ", ".join(c.get("name","") for c in mid[:3])
        senior_n = ", ".join(c.get("name","") for c in senior[:3])
        evs = [
            ("high impact","#d29922","Conversion Accelerator Workshop",
             f"{len(mid)} mid-funnel clients identified near the decision boundary. Deploy a value-demonstration format targeting {mid_n} who are closest to the high-priority threshold. Regression model suggests 22-31% probability of tier upgrade post-event.",
             f"ROI: ~{_fi(sum(_num(c.get('portfolio',0)) for c in mid)*0.08)} uplift",
             f"Workshop \u00b7 {len(mid)} mid-funnel \u00b7 This month"),
            ("medium impact","#58a6ff","HNI Portfolio Deep-Dive",
             f"Your top segment contributes {round(sum(_num(c.get('portfolio',0)) for c in hni)/max(aum,1)*100,1)}% of total AUM across {len(hni)} accounts ({_fi(sum(_num(c.get('portfolio',0)) for c in hni))}). An industry-specific event drives 2.1x engagement vs generic format. Targets: {hni_n}.",
             f"ROI: ~{_fi(sum(_num(c.get('portfolio',0)) for c in hni)*0.04)} incremental",
             f"Industry Event \u00b7 {len(hni)} HNI contacts \u00b7 This quarter"),
            ("medium impact","#58a6ff","Portfolio Intelligence Summit",
             f"With {len(clients)} accounts totalling {_fi(aum)} in pipeline, host a data-driven review combining live dashboards with predictive insights. Focus: identify dormant accounts showing reactivation signals. Targets: {senior_n}.",
             "ROI: Strategic \u2014 long-term LTV impact",
             f"Summit \u00b7 Full portfolio \u00b7 Quarterly"),
        ]
        rows_e = ""
        for tag,tc,title,body,roi,meta_str in evs:
            m_parts = meta_str.split("\u00b7")
            rows_e += f"""<div class="evcard">
              <div class="evtop"><div class="evtitle">{title}</div>
                <span class="chip" style="background:{tc}18;color:{tc};border:1px solid {tc}44;font-size:10px;padding:2px 8px;border-radius:10px;font-family:\'JetBrains Mono\',monospace;font-weight:600">{tag}</span></div>
              <div class="evbody">{body}</div>
              <div class="evroi">{roi}</div>
              <div class="evmeta">{"".join(f"<span>{p.strip()}</span>" for p in m_parts)}</div>
            </div>"""
        st.markdown(f'<div class="evgrid">{rows_e}</div>', unsafe_allow_html=True)

    # TAB 4
    with tab4:
        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mhd"><div class="mic mpu">\u25c8</div><div><div class="mtitle">ML Prediction Engine</div><div class="msub">Ensemble model \u00b7 Churn prediction \u00b7 Revenue forecasting \u00b7 Confidence intervals</div></div></div>', unsafe_allow_html=True)
        if "ml_exp" not in st.session_state: st.session_state.ml_exp = None
        st.markdown('<div class="mlhdr"><span>Account</span><span>Score \u0394</span><span>Churn risk</span><span>Conv. prob</span><span>Predicted rev.</span><span>Trend</span></div>', unsafe_allow_html=True)
        for i,c in enumerate(clients[:15]):
            sc=c.get("score",0); ch=c.get("churn",0); cv=c.get("conv",50)
            dlo,dhi = max(0,sc-3),min(100,sc+3)
            trend = "Ascending" if sc>=60 else ("Stable" if sc>=45 else "Declining")
            tcls = "tup" if trend=="Ascending" else ("tdn" if trend=="Declining" else "tsb")
            conf = random.randint(85,94)
            pred_rev = round(_num(c.get("portfolio",0))*1.12)
            chcol = "#f85149" if ch>50 else ("#d29922" if ch>25 else "#3fb950")
            cvcol = "#3fb950" if cv>60 else ("#d29922" if cv>40 else "#f85149")
            is_me = st.session_state.ml_exp==i
            st.markdown(f"""<div class="mlrow {'mlex' if is_me else ''}">
              <span style="font-weight:600;font-size:13px">{c.get("name","\u2014")}</span>
              <span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#3fb950">{dlo}\u2192{dhi}</span>
              <span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{chcol}">{ch}%</span>
              <span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{cvcol}">{cv}%</span>
              <span style="font-family:\'JetBrains Mono\',monospace;font-size:12px">\u20b9{pred_rev:,}</span>
              <span><span class="{tcls}">\u2197 {trend}</span>
                <span class="cbr" style="margin-left:6px">
                <span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#8b949e">{conf}%</span>
                <span class="cbar"><span class="cfill" style="width:{conf}%"></span></span></span>
              </span>
            </div>""", unsafe_allow_html=True)
            _,bc = st.columns([11,1])
            with bc:
                if st.button("\u25b2" if is_me else "\u25bc", key=f"me_{i}"):
                    st.session_state.ml_exp = None if is_me else i; st.rerun()
            if is_me:
                feat = c.get("feature_importance", "No insight available")
            
                # 👇 Python logic outside
                tag = ""
                if c.get("ml_powered"):
                    tag = '<div style="font-size:10px;color:#3fb950;font-family:JetBrains Mono,monospace">🧠 AI Insight</div>'
            
                # 👇 HTML inside
                st.markdown(f"""
                <div class="mlxpand">
                  {tag}
                  <div class="mlfl">⊕ Model feature importance</div>
                  <div class="mlft">↳ {feat}</div>
                </div>
                """, unsafe_allow_html=True)
                

        if len(clients)>15:
            st.markdown(f"<div style='text-align:center;padding:1rem;font-size:12px;color:#58a6ff;font-family:JetBrains Mono,monospace'>View all {len(clients)} predictions \u2192</div>", unsafe_allow_html=True)

    # TAB 5
    with tab5:
        st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mhd"><div class="mic mgr">\U0001f4f1</div><div><div class="mtitle">WhatsApp Drafts</div><div class="msub">Personalised templates \u00b7 Direct send links</div></div></div>', unsafe_allow_html=True)
        names = [c.get("name","") for c in clients if c.get("name")]
        seln = st.selectbox("Select client", names, label_visibility="collapsed")
        sel = next((c for c in clients if c.get("name")==seln), None)
        if sel:
            sc2=sel.get("score",0); ch2=sel.get("churn",0)
            scc="#3fb950" if sc2>=70 else ("#d29922" if sc2>=45 else "#f85149")
            chc="#f85149" if ch2>50 else "#3fb950"
            un = st.session_state.get("user_name","Your Advisor")
            uc = st.session_state.get("user_company","")
            ca,cb = st.columns([1,1])
            with ca:
                st.markdown(f"""<div class="wprof">
                  <div class="wpname">{sel.get("name","")}</div>
                  <div class="wprow"><span>Portfolio</span><span class="wpval">{_fi(sel.get("portfolio",0))}</span></div>
                  <div class="wprow"><span>Monthly SIP</span><span class="wpval">{_fi(sel.get("sip",0)) if _num(sel.get("sip",0))>0 else "Not started"}</span></div>
                  <div class="wprow"><span>Health score</span><span class="wpval" style="color:{scc}">{sc2}/100</span></div>
                  <div class="wprow"><span>Leaving risk</span><span class="wpval" style="color:{chc}">{ch2}%</span></div>
                  <div class="wprow"><span>Product</span><span class="wpval">{sel.get("goal","\u2014")}</span></div>
                </div>""", unsafe_allow_html=True)
                mt = st.radio("Type", ["Check-in call","SIP proposal","Portfolio review","Nominee update"], label_visibility="visible")
            with cb:
                tmpls = {
                    "Check-in call": f"Dear {sel.get('name','')},\n\nI have been reviewing your portfolio and there are a few developments I would like to walk you through personally.\n\nCould we schedule a quick 20-minute call this week?\n\nWarm regards,\n{un}\n{uc}",
                    "SIP proposal": f"Dear {sel.get('name','')},\n\nBased on your portfolio of {_fi(sel.get('portfolio',0))}, I have prepared a personalised SIP projection that could significantly grow your wealth.\n\nCan we find 15 minutes to walk through it?\n\nWarm regards,\n{un}\n{uc}",
                    "Portfolio review": f"Dear {sel.get('name','')},\n\nYour portfolio review is due. I want to ensure your investments are optimally positioned for the year ahead.\n\nWhen works best for a quick call?\n\nWarm regards,\n{un}\n{uc}",
                    "Nominee update": f"Dear {sel.get('name','')},\n\nAs part of our annual client care review, I noticed your nominee details may need updating. This is critical for your family.\n\nIt takes under 10 minutes. Can I help?\n\nWarm regards,\n{un}\n{uc}",
                }
                edited = st.text_area("Edit before sending", tmpls[mt], height=220, label_visibility="collapsed")
                ph = sel.get("phone","")
                wl2 = get_whatsapp_link(ph, edited) if WA_OK and ph else f"https://wa.me/?text={edited.replace(chr(10),'%0A').replace(' ','%20')}"
                st.markdown(f'<br><a class="btn-wa" href="{wl2}" target="_blank">\U0001f4f1 Open in WhatsApp \u2197</a>', unsafe_allow_html=True)

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    mc = st.session_state.get("merged_count",0)
    ms2 = f" \u00b7 {mc} duplicates merged" if mc else ""
    ml_status = " \u00b7 ML powered" if ML_OK else " \u00b7 rule-based scoring"
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#21262d;font-family:JetBrains Mono,monospace'>AdvisorIQ \u00b7 {len(clients)} clients \u00b7 {_fi(aum)} AUM{ms2}{ml_status}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"**{st.session_state.get('user_name','')}**")
        st.caption(st.session_state.get("user_company",""))
        if st.button("Upload new data"): st.session_state.screen="upload"; st.rerun()
        if st.button("Settings"): st.session_state.screen="settings"; st.rerun()
        if st.button("Sign out"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if "screen" not in st.session_state: st.session_state.screen = "login"
    if "user_id" not in st.session_state and st.session_state.screen not in ("login",):
        st.session_state.screen = "login"
    screen = st.session_state.screen

    if screen == "login":   show_login();  return
    if screen == "settings": show_settings(); return

    if screen == "upload":
        up = show_upload()
        if up:
            try:
                df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                st.session_state.upload_df = df
                st.session_state.screen = "map"; st.rerun()
            except Exception as e: st.error(f"Could not read file: {e}")
        return

    if screen == "map":
        if st.session_state.get("use_demo"):
            with st.spinner("Loading demo data..."):
                clients = prep_demo()
                from ml_model import predict_batch

                clients = predict_batch(clients)
            st.session_state.clients = clients
            if DB_OK: db.save_clients(st.session_state.user_id, clients)
            st.session_state.use_demo = False
            st.session_state.screen = "dashboard"; st.rerun()
        elif st.session_state.get("use_sheets") and st.session_state.get("sheets_url"):
            with st.spinner("Fetching Google Sheets data..."):
                from sheets_sync import fetch_sheet_data
                df, err = fetch_sheet_data(st.session_state.sheets_url)
            if err or df is None:
                st.error(f"Could not fetch sheet: {err}")
                st.session_state.screen = "upload"; st.rerun()
            else:
                st.session_state.upload_df = df
                st.session_state.use_sheets = False
        if "upload_df" in st.session_state:
            show_mapping(st.session_state.upload_df)
        else:
            st.session_state.screen = "upload"; st.rerun()
        return

    if screen == "dashboard":
        clients = st.session_state.get("clients", [])
        if not clients: st.session_state.screen = "upload"; st.rerun(); return
        show_dashboard(clients); return

if __name__ == "__main__":
    main()
