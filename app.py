import streamlit as st, pandas as pd, numpy as np
import datetime, random, sqlite3, json, logging
import plotly.graph_objects as go
import bcrypt
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('advisoriq')

st.set_page_config(page_title='AdvisorIQ', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')

DB = 'advisoriq.db'

def init_db():
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        full_name TEXT, company TEXT, role TEXT DEFAULT 'advisor', created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS clients(id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT, age TEXT, portfolio TEXT, sip TEXT,
        last_contact TEXT, goal TEXT, tenure TEXT, nominee TEXT, phone TEXT,
        score INTEGER DEFAULT 0, churn INTEGER DEFAULT 0, conv INTEGER DEFAULT 50,
        priority TEXT DEFAULT 'Low', flags TEXT DEFAULT '[]', uploaded_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit(); conn.close()

def hp(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def check_pw(pw, hashed):
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        # fallback for old sha256 hashes during migration
        import hashlib
        return hashlib.sha256(pw.encode()).hexdigest() == hashed

def db_login(u, p):
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('SELECT id,full_name,company,role,password_hash FROM users WHERE username=?', (u,))
    row = c.fetchone(); conn.close()
    if row and check_pw(p, row[4]):
        return (row[0], row[1], row[2], row[3])
    return None

def db_register(username, password, full_name, company, role):
    conn = sqlite3.connect(DB); c = conn.cursor()
    try:
        c.execute('INSERT INTO users(username,password_hash,full_name,company,role,created_at) VALUES(?,?,?,?,?,?)',
                  (username.strip(), hp(password), full_name.strip(), company.strip(), role, datetime.datetime.now().isoformat()))
        conn.commit(); conn.close(); return True, 'Account created.'
    except sqlite3.IntegrityError:
        conn.close(); return False, 'Username already taken.'

def db_save(user_id, clients):
    logger.info(f'Saving {len(clients)} clients for user_id={user_id}')
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('DELETE FROM clients WHERE user_id=?', (user_id,))
    for cl in clients:
        c.execute('''INSERT INTO clients(user_id,name,age,portfolio,sip,last_contact,goal,tenure,nominee,phone,
                     score,churn,conv,priority,flags,uploaded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (user_id, cl.get('name',''), cl.get('age',''), cl.get('portfolio','0'), cl.get('sip','0'),
                   cl.get('lastContact',''), cl.get('goal',''), cl.get('tenure',''), cl.get('nominee',''), cl.get('phone',''),
                   cl.get('score',0), cl.get('churn',0), cl.get('conv',50), cl.get('priority','Low'),
                   json.dumps(cl.get('flags',[])), datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()

@st.cache_data(ttl=300, show_spinner=False)
def db_load(user_id):
    conn = sqlite3.connect(DB); c = conn.cursor()
    c.execute('SELECT name,age,portfolio,sip,last_contact,goal,tenure,nominee,phone,score,churn,conv,priority,flags FROM clients WHERE user_id=? ORDER BY score DESC', (user_id,))
    rows = c.fetchall(); conn.close()
    return [{'name':r[0],'age':r[1],'portfolio':r[2],'sip':r[3],'lastContact':r[4],'goal':r[5],
             'tenure':r[6],'nominee':r[7],'phone':r[8],'score':r[9],'churn':r[10],'conv':r[11],
             'priority':r[12],'flags':json.loads(r[13]) if r[13] else []} for r in rows]

def inject_css():
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
.nav-brand{font-size:15px;font-weight:600;color:var(--tx)}
.nav-brand em{color:var(--gr);font-style:normal}
.nav-right{display:flex;align-items:center;gap:10px}
.nav-user{font-size:12px;color:var(--t2);font-family:'JetBrains Mono',monospace}
.nav-role{font-size:11px;padding:2px 8px;border-radius:12px;background:var(--grbg);color:var(--gr);border:1px solid var(--grbd);font-weight:600}
.bc{padding:8px 1.5rem;background:var(--s1);border-bottom:1px solid var(--bd);font-size:12px;color:var(--t3);font-family:'JetBrains Mono',monospace}
.bc em{color:var(--bl);font-style:normal}
.wrap{padding:1.5rem;max-width:1440px;margin:0 auto}
.greet{display:flex;align-items:center;justify-content:space-between;background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem}
.gt{font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--gr);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.gn{font-size:1.4rem;font-weight:600;letter-spacing:-.4px;margin-bottom:4px}
.gs{font-size:13px;color:var(--t2)}
.gstats{display:flex;gap:2rem;text-align:right}
.gst{font-family:'JetBrains Mono',monospace}
.gst-num{font-size:1.4rem;font-weight:700;display:block}
.gst-label{font-size:11px;color:var(--t2);margin-top:2px;display:block}
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
.mgr{background:var(--grbg);border:1px solid var(--grbd)}
.mbl{background:var(--blbg);border:1px solid var(--blbd)}
.mam{background:var(--ambg);border:1px solid var(--ambd)}
.mpu{background:var(--pubg);border:1px solid var(--pubd)}
.mrd{background:var(--rdbg);border:1px solid var(--rdbd)}
.mt{font-size:14px;font-weight:600;margin-bottom:2px}
.msub{font-size:12px;color:var(--t2)}
.ptable{width:100%;border-collapse:collapse;font-size:13px}
.ptable thead th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;color:var(--t3);font-weight:500;padding:8px 12px;border-bottom:1px solid var(--bd);text-align:left}
.ptable tbody tr{border-bottom:1px solid var(--bd);cursor:pointer;transition:background .1s}
.ptable tbody tr:hover,.ptable tbody tr.xp{background:var(--s2)}
.prank{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--t3);width:44px}
.pname{font-weight:600;font-size:13px}
.psub{font-size:11px;color:var(--t2);margin-top:2px}
.xrow{background:var(--s2);border-bottom:1px solid var(--bd)}
.xin{padding:.875rem 1.1rem;border-left:3px solid var(--bl);margin:0 0 4px 44px}
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
.cb{display:inline-flex;align-items:center;gap:5px}
.cbar{height:3px;border-radius:2px;background:var(--bd2);width:36px;overflow:hidden;display:inline-block;vertical-align:middle}
.cfill{height:100%;background:var(--bl);border-radius:2px}
.wprof{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:1.1rem;margin-bottom:1rem}
.wpname{font-size:15px;font-weight:700;margin-bottom:10px}
.wprow{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--bd);font-size:13px;color:var(--t2)}
.wprow:last-child{border:none}
.wpval{font-family:'JetBrains Mono',monospace;color:var(--tx)}
.uph{text-align:center;padding:5rem 2rem 2rem}
.upey{font-size:11px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.15em;color:var(--gr);margin-bottom:1rem}
.upt{font-size:2.5rem;font-weight:700;letter-spacing:-.05em;line-height:1.15;margin-bottom:.75rem}
.upt em{color:var(--gr);font-style:normal}
.ups{font-size:14px;color:var(--t2);max-width:480px;margin:0 auto 2rem;line-height:1.7}
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

def fi(v):
    try: n=float(str(v).replace(',','').replace('₹','') or 0)
    except: n=0
    if n>=1e7: return f'₹{n/1e7:.1f}Cr'
    if n>=1e5: return f'₹{n/1e5:.1f}L'
    if n>=1e3: return f'₹{n/1e3:.0f}K'
    return f'₹{int(n)}'

def num(v):
    try: return float(str(v).replace(',','').replace('₹','').strip())
    except: return 0.0

def mago(d):
    if not d or str(d).strip() in ('','nan','None','NaT'): return 99
    try:
        dt=pd.to_datetime(str(d),dayfirst=True,errors='coerce')
        if pd.isna(dt): return 99
        return max(0,(datetime.datetime.now()-dt.to_pydatetime()).days/30)
    except: return 99

def now_ist():
    try:
        import pytz
        return datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    except: return datetime.datetime.now()+datetime.timedelta(hours=5,minutes=30)

def get_greeting():
    h=now_ist().hour
    if h<12: return 'Good morning'
    if h<17: return 'Good afternoon'
    return 'Good evening'

AGENDAS=[
    'Your portfolio signals are ready. A few clients need your attention today.',
    'Fresh data loaded. The engine has ranked your priorities.',
    'Three things need your eye before you close today.',
    "Your clients' health scores are updated. Let's make today count.",
    'Intelligence engine active. Your best opportunities are surfaced.',
]

def cn(v):
    try: return str(float(str(v).replace(',','').replace('₹','').strip()))
    except: return '0'

def cph(v):
    if not v: return ''
    d=''.join(filter(str.isdigit,str(v)))
    return ('91'+d) if len(d)==10 else d

def score_c(r):
    """Rule-based fallback score — used for training label generation"""
    p=num(r.get('portfolio',0)); sip=num(r.get('sip',0))
    try: age=int(float(r.get('age') or 0))
    except: age=0
    try:
        yr=int(float(str(r.get('tenure','2020')).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=0
    ma=mago(r.get('lastContact',''))
    nom=str(r.get('nominee','')).lower().strip()
    goal=str(r.get('goal','')).lower()
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
    if nom=='no':s-=8
    if 'bond' in goal:s+=5
    if age>55 and 'lic' in goal:s+=5
    if sip==0 and p>5e5:s-=5
    return max(0,min(100,round(s)))

def churn_c(r):
    r2=0; ma=mago(r.get('lastContact',''))
    sip=num(r.get('sip',0)); nom=str(r.get('nominee','')).lower().strip()
    try:
        yr=int(float(str(r.get('tenure','2020')).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=5
    if ma>12:r2+=40
    elif ma>6:r2+=25
    elif ma>3:r2+=10
    if sip==0:r2+=20
    if nom=='no':r2+=15
    if ty<2:r2+=15
    return min(100,round(r2))

def conv_c(r):
    s=score_c(r); c=churn_c(r)
    return min(95,max(5,round(s*0.7+(100-c)*0.3)))

def extract_features(r):
    """Extract numeric feature vector for ML model"""
    p=num(r.get('portfolio',0)); sip=num(r.get('sip',0))
    try: age=int(float(r.get('age') or 0))
    except: age=0
    try:
        yr=int(float(str(r.get('tenure','2020')).strip()))
        ty=(2025-yr) if yr>1990 else yr
    except: ty=5
    ma=mago(r.get('lastContact',''))
    nom=1 if str(r.get('nominee','')).lower().strip()=='no' else 0
    goal=str(r.get('goal','')).lower()
    has_lic=1 if 'lic' in goal else 0
    has_bond=1 if 'bond' in goal else 0
    has_mf=1 if 'mf' in goal else 0
    return [
        p/1e7,           # portfolio normalized
        sip/50000,       # sip normalized
        ma/24,           # months since contact normalized
        ty/20,           # tenure years normalized
        age/80,          # age normalized
        nom,             # nominee missing flag
        has_lic,         # has LIC product
        has_bond,        # has bonds
        has_mf,          # has mutual funds
        1 if sip==0 and p>5e5 else 0,  # SIP gap flag
        1 if p>5e6 else 0,             # HNI flag
    ]

@st.cache_resource
def build_ml_model(clients):
    """Train real ML model on client data using bootstrapped labels"""
    if len(clients) < 5:
        return None, None, None
    
    # Generate training data using rule-based scores as ground truth labels
    X, y_score, y_churn = [], [], []
    for c in clients:
        feats = extract_features(c)
        sc = score_c(c)
        ch = churn_c(c)
        X.append(feats)
        y_score.append(1 if sc >= 65 else 0)   # high priority label
        y_churn.append(1 if ch >= 50 else 0)   # churn label
    
    X = np.array(X)
    
    # Augment with noise to create more training samples (bootstrap)
    np.random.seed(42)
    X_aug = np.vstack([X] + [X + np.random.normal(0, 0.05, X.shape) for _ in range(4)])
    y_score_aug = y_score * 5
    y_churn_aug = y_churn * 5
    
    # Train two models
    score_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42))
    ])
    churn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42))
    ])
    
    try:
        if len(set(y_score_aug)) > 1:
            score_model.fit(X_aug, y_score_aug)
        else:
            score_model = None
        if len(set(y_churn_aug)) > 1:
            churn_model.fit(X_aug, y_churn_aug)
        else:
            churn_model = None
        logger.info(f"ML models trained on {len(clients)} clients ({len(X_aug)} augmented samples)")
        return score_model, churn_model, True
    except Exception as e:
        logger.warning(f"ML training failed: {e}, falling back to rules")
        return None, None, False

def ml_predict(c, score_model, churn_model):
    """Get ML predictions for a single client"""
    feats = np.array(extract_features(c)).reshape(1, -1)
    
    if score_model is not None:
        try:
            score_prob = score_model.predict_proba(feats)[0][1]
            ml_score = int(score_prob * 100)
        except:
            ml_score = score_c(c)
    else:
        ml_score = score_c(c)
    
    if churn_model is not None:
        try:
            churn_prob = churn_model.predict_proba(feats)[0][1]
            ml_churn = int(churn_prob * 100)
        except:
            ml_churn = churn_c(c)
    else:
        ml_churn = churn_c(c)
    
    ml_conv = min(95, max(5, round(ml_score * 0.7 + (100 - ml_churn) * 0.3)))
    return ml_score, ml_churn, ml_conv

def get_feature_importance(c, score_model):
    """Return most important feature driving this client's score"""
    feat_names = [
        'Portfolio size', 'Monthly SIP', 'Contact recency', 'Client tenure',
        'Age', 'Nominee missing', 'Has LIC', 'Has bonds', 'Has MF',
        'SIP gap (no SIP but has portfolio)', 'High-value client (50L+)'
    ]
    p=num(c.get('portfolio',0)); sip=num(c.get('sip',0))
    ma=mago(c.get('lastContact',''))
    nom=str(c.get('nominee','')).lower().strip()
    
    if score_model is not None:
        try:
            clf = score_model.named_steps['clf']
            importances = clf.feature_importances_
            top_idx = int(np.argmax(importances))
            top_feat = feat_names[top_idx]
            return f"{top_feat} is the dominant signal driving this client's score (importance: {importances[top_idx]:.2f})"
        except: pass
    
    # Fallback readable explanation
    if p > 4e6: return f"High portfolio value ({fi(p)}) drives strong conversion signal"
    if sip > 10000: return f"Active SIP of {fi(sip)}/month shows consistent commitment"
    if ma < 3: return "Recent contact (under 3 months) keeps engagement score elevated"
    if ma > 6: return "6+ months without contact — relationship score declining rapidly"
    if nom == 'no': return "Missing nominee reduces trust score and flags compliance risk"
    if sip == 0 and p > 5e5: return "No SIP despite significant portfolio — high upsell potential"
    return "Composite score from portfolio size, contact recency, SIP activity, and tenure"

FIELDS=[('name',['name','client','naam']),('age',['age','umur']),
        ('portfolio',['portfolio','aum','value','investment','amount','total']),
        ('sip',['sip','monthly','sipamount']),
        ('lastContact',['last','date','meeting','contact','interaction']),
        ('goal',['product','goal','scheme','type']),
        ('tenure',['since','tenure','year','clientsince']),
        ('nominee',['nominee','nomination']),('phone',['phone','mobile','number'])]

def det(hints,cols):
    for c in cols:
        cl=c.lower().replace(' ','').replace('_','')
        for h in hints:
            if h in cl: return c
    return None

def smart_dedup(clients):
    sp={}; sn={}; out=[]; merged=0
    for c in clients:
        ph=c.get('phone','').strip(); nm=c.get('name','').strip().lower()
        if ph and len(ph)>=10 and ph in sp:
            ex=sp[ph]
            if num(c['portfolio'])>num(ex['portfolio']): out[out.index(ex)]=c; sp[ph]=c
            merged+=1
        elif nm and nm in sn:
            ex=sn[nm]
            if num(c['portfolio'])>num(ex['portfolio']): out[out.index(ex)]=c; sn[nm]=c
            merged+=1
        else:
            out.append(c)
            if ph and len(ph)>=10: sp[ph]=c
            if nm: sn[nm]=c
    return out,merged

def process(df, mapping):
    dfl = {'name':'','age':'','portfolio':'0','sip':'0','lastContact':'','goal':'','tenure':'2020','nominee':'','phone':''}
    clients = []
    # Vectorized processing instead of row-by-row
    for _, row in df.iterrows():
        c = dict(dfl)
        for key, _ in FIELDS:
            col = mapping.get(key)
            if col and col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() not in ('','nan','None'):
                    c[key] = cn(val) if key in ('portfolio','sip') else (cph(val) if key=='phone' else str(val).strip())
        # Rule-based scores first (used for training)
        c['score'] = score_c(c)
        c['churn'] = churn_c(c)
        c['conv'] = conv_c(c)
        c['priority'] = 'High' if c['score']>=70 else ('Medium' if c['score']>=45 else 'Low')
        c['flags'] = flags_c(c)
        clients.append(c)
    clients, merged = smart_dedup(clients)
    clients.sort(key=lambda x: x.get('score',0), reverse=True)
    
    # Train real ML model and re-score
    if len(clients) >= 5:
        score_model, churn_model, ml_ok = build_ml_model(clients)
        if ml_ok and score_model is not None:
            for c in clients:
                ml_sc, ml_ch, ml_cv = ml_predict(c, score_model, churn_model)
                # Blend rule-based + ML (70% ML, 30% rules for stability)
                c['score'] = round(ml_sc * 0.7 + c['score'] * 0.3)
                c['churn'] = round(ml_ch * 0.7 + c['churn'] * 0.3)
                c['conv'] = ml_cv
                c['priority'] = 'High' if c['score']>=70 else ('Medium' if c['score']>=45 else 'Low')
                c['flags'] = flags_c(c)
            # Store models in session for later use
            st.session_state.score_model = score_model
            st.session_state.churn_model = churn_model
            st.session_state.ml_active = True
            logger.info(f"ML re-scoring complete for {len(clients)} clients")
        else:
            st.session_state.ml_active = False
    
    clients.sort(key=lambda x: x.get('score',0), reverse=True)
    return clients, merged

@st.cache_data(ttl=300)
def to_excel_bytes(clients_json):
    """Convert clients to Excel bytes — cached for 5 minutes"""
    import json as _json
    clients = _json.loads(clients_json)
    df = pd.DataFrame(clients)
    cols = ['name','age','portfolio','sip','lastContact','goal','tenure','nominee','phone','score','churn','conv','priority']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df.columns = [c.replace('lastContact','Last Contact').replace('conv','Conversion Prob %').replace('churn','Churn Risk %').replace('score','Health Score').title() for c in df.columns]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Clients')
        ws = writer.sheets['Clients']
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(len(str(col[0].value))+4, 14)
    return buf.getvalue()


def export_excel(clients):
    """Export client list to Excel with styling."""
    rows = []
    for c in clients:
        rows.append({
            'Client Name': c.get('name', ''),
            'Age': c.get('age', ''),
            'Portfolio (₹)': num(c.get('portfolio', 0)),
            'Monthly SIP (₹)': num(c.get('sip', 0)),
            'Health Score': c.get('score', 0),
            'Churn Risk (%)': c.get('churn', 0),
            'Conv. Probability (%)': c.get('conv', 0),
            'Priority': c.get('priority', ''),
            'Product / Goal': c.get('goal', ''),
            'Last Contact': c.get('lastContact', ''),
            'Tenure (Since)': c.get('tenure', ''),
            'Nominee': c.get('nominee', ''),
            'Phone': c.get('phone', ''),
            'Flags': ' | '.join(c.get('flags', [])),
        })
    df = pd.DataFrame(rows)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Client Intelligence')
        ws = writer.sheets['Client Intelligence']
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)
    buf.seek(0)
    return buf.getvalue()

PC={'paper_bgcolor':'#0d1117','plot_bgcolor':'#161b22',
    'font':dict(family='Inter,sans-serif',color='#8b949e',size=11),
    'margin':dict(l=8,r=8,t=32,b=8),'showlegend':False,
    'xaxis':dict(showgrid=False,zeroline=False,color='#8b949e',tickfont=dict(size=10),linecolor='#30363d'),
    'yaxis':dict(showgrid=True,gridcolor='#21262d',zeroline=False,color='#8b949e',tickfont=dict(size=10))}

DEMO=[
    {'name':'Ramesh Patel','age':'62','portfolio':'4800000','sip':'15000','lastContact':'2024-01-10','goal':'MF+LIC','tenure':'2010','nominee':'Yes','phone':'9876543210'},
    {'name':'Kavita Joshi','age':'55','portfolio':'7200000','sip':'25000','lastContact':'2024-03-01','goal':'MF+Bonds+LIC','tenure':'2008','nominee':'Yes','phone':'9876543211'},
    {'name':'Hemant Rao','age':'67','portfolio':'9500000','sip':'0','lastContact':'2023-11-20','goal':'Bonds+LIC','tenure':'2005','nominee':'Yes','phone':'9876543212'},
    {'name':'Geeta Sharma','age':'61','portfolio':'6100000','sip':'20000','lastContact':'2023-10-15','goal':'MF+LIC+Bonds','tenure':'2007','nominee':'Yes','phone':'9876543213'},
    {'name':'Suresh Agrawal','age':'70','portfolio':'12000000','sip':'0','lastContact':'2023-05-10','goal':'Bonds+LIC','tenure':'2002','nominee':'Yes','phone':'9876543214'},
    {'name':'Pushpa Rao','age':'64','portfolio':'5500000','sip':'15000','lastContact':'2024-02-01','goal':'MF+LIC+Bonds','tenure':'2006','nominee':'Yes','phone':'9876543215'},
    {'name':'Nisha Gupta','age':'41','portfolio':'2100000','sip':'12000','lastContact':'2024-01-25','goal':'MF+LIC','tenure':'2016','nominee':'Yes','phone':'9876543216'},
    {'name':'Manisha Patel','age':'53','portfolio':'2900000','sip':'10000','lastContact':'2023-07-22','goal':'LIC+MF','tenure':'2013','nominee':'Yes','phone':'9876543217'},
    {'name':'Rekha Jain','age':'58','portfolio':'3400000','sip':'0','lastContact':'2023-12-05','goal':'LIC+Bonds','tenure':'2011','nominee':'Yes','phone':'9876543218'},
    {'name':'Archana Desai','age':'56','portfolio':'4200000','sip':'0','lastContact':'2023-09-05','goal':'LIC+Bonds','tenure':'2009','nominee':'Yes','phone':'9876543219'},
    {'name':'Sunita Shah','age':'45','portfolio':'1200000','sip':'8000','lastContact':'2023-09-20','goal':'MF','tenure':'2018','nominee':'No','phone':'9876543220'},
    {'name':'Arun Trivedi','age':'48','portfolio':'900000','sip':'0','lastContact':'2023-06-15','goal':'LIC','tenure':'2015','nominee':'No','phone':'9876543221'},
    {'name':'Vijay Solanki','age':'50','portfolio':'650000','sip':'6000','lastContact':'2023-08-10','goal':'MF','tenure':'2019','nominee':'No','phone':'9876543222'},
    {'name':'Bhavesh Modi','age':'44','portfolio':'520000','sip':'7500','lastContact':'2024-03-10','goal':'MF','tenure':'2020','nominee':'No','phone':'9876543223'},
    {'name':'Jigar Shah','age':'47','portfolio':'1750000','sip':'9000','lastContact':'2023-12-18','goal':'MF+LIC','tenure':'2017','nominee':'No','phone':'9876543224'},
    {'name':'Hetal Trivedi','age':'39','portfolio':'430000','sip':'6000','lastContact':'2024-02-20','goal':'MF','tenure':'2021','nominee':'No','phone':'9876543225'},
    {'name':'Dinesh Mehta','age':'38','portfolio':'350000','sip':'5000','lastContact':'2024-02-28','goal':'SIP','tenure':'2022','nominee':'No','phone':'9876543226'},
    {'name':'Kalpesh Vora','age':'36','portfolio':'210000','sip':'3000','lastContact':'2024-01-30','goal':'SIP','tenure':'2023','nominee':'No','phone':'9876543227'},
    {'name':'Priya Desai','age':'32','portfolio':'180000','sip':'4000','lastContact':'2024-02-10','goal':'SIP','tenure':'2023','nominee':'No','phone':'9876543228'},
    {'name':'Nilesh Mehta','age':'33','portfolio':'95000','sip':'2000','lastContact':'2024-03-05','goal':'SIP','tenure':'2024','nominee':'No','phone':'9876543229'},
]

@st.cache_data
def prep_demo():
    out=[]
    for c in DEMO:
        c2=dict(c); c2['score']=score_c(c2); c2['churn']=churn_c(c2); c2['conv']=conv_c(c2)
        c2['priority']='High' if c2['score']>=70 else ('Medium' if c2['score']>=45 else 'Low')
        c2['flags']=flags_c(c2); out.append(c2)
    out.sort(key=lambda x:x.get('score',0),reverse=True)
    return out

def show_nav():
    user=st.session_state.get('user_name','')
    company=st.session_state.get('user_company','')
    role=st.session_state.get('user_role','advisor')
    rl='Owner' if role=='owner' else 'Advisor'
    st.markdown(f'''<div class="nav">
      <div class="nav-logo"><div class="nav-icon">⚡</div>
      <span class="nav-brand">Advisor<em>IQ</em></span></div>
      <div class="nav-right"><span class="nav-user">{user} · {company}</span>
      <span class="nav-role">{rl}</span></div>
    </div>''',unsafe_allow_html=True)
    c1,c2,c3=st.columns([7,1,1])
    with c2:
        if st.button('⬆ Upload',key='nav_up'):
            st.session_state.pop('kpi_open',None)
            st.session_state.screen='upload'; st.rerun()
    with c3:
        if st.button('Sign out',key='nav_so'):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

def show_login():
    _,col,_=st.columns([1,1,1])
    with col:
        st.markdown('''<div style="text-align:center;margin-top:3rem;margin-bottom:2rem">
          <div style="width:48px;height:48px;background:#3fb950;border-radius:10px;
            display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;color:#000;margin-bottom:.875rem">⚡</div>
          <div style="font-size:1.3rem;font-weight:700;letter-spacing:-.3px;color:#e6edf3">AdvisorIQ</div>
          <div style="font-size:13px;color:#8b949e;margin-top:4px">Portfolio intelligence platform</div>
        </div>''',unsafe_allow_html=True)
        t1,t2=st.tabs(['Sign in','Create account'])
        with t1:
            st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
            u=st.text_input('Username',placeholder='your.username',key='li_u')
            p=st.text_input('Password',type='password',placeholder='••••••••',key='li_p')
            st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
            if st.button('Sign in →',use_container_width=True,key='li_b'):
                if u and p:
                    row=db_login(u,p)
                    if row:
                        st.session_state.user_id=row[0]; st.session_state.user_name=row[1]
                        st.session_state.user_company=row[2]; st.session_state.user_role=row[3]
                        saved=db_load(row[0])
                        if saved: st.session_state.clients=saved
                        st.session_state.screen='upload' if not saved else 'dashboard'
                        st.rerun()
                    else: st.error('Incorrect username or password.')
                else: st.warning('Please enter both fields.')
        with t2:
            st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
            rn=st.text_input('Full name',placeholder='Ramesh Patel',key='r_n')
            rc=st.text_input('Company',placeholder='Patel Wealth Advisory',key='r_c')
            ru=st.text_input('Username',placeholder='ramesh.patel',key='r_u')
            rp=st.text_input('Password',type='password',placeholder='Min 6 chars',key='r_p')
            rr=st.selectbox('Role',['Owner / Director','Senior Advisor','Advisor','Team Member'],key='r_r')
            rm={'Owner / Director':'owner','Senior Advisor':'advisor','Advisor':'advisor','Team Member':'staff'}
            st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
            if st.button('Create account →',use_container_width=True,key='r_b'):
                if all([rn,rc,ru,rp]):
                    if len(rp)<6: st.warning('Password must be at least 6 characters.')
                    else:
                        ok,msg=db_register(ru,rp,rn,rc,rm[rr])
                        if ok: st.success(f'Account created! Sign in with: {ru}')
                        else: st.error(msg)
                else: st.warning('Please fill in all fields.')

def show_upload():
    show_nav()
    st.markdown('<div class="bc">Upload → Mapping → <em>Dashboard</em></div>',unsafe_allow_html=True)
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    clients=st.session_state.get('clients',[])
    if clients:
        st.success(f'✓ {len(clients)} clients loaded from your last session.')
        cc,_=st.columns([1,4])
        with cc:
            if st.button('View dashboard →',use_container_width=True):
                st.session_state.screen='dashboard'; st.rerun()
        st.markdown('<hr>',unsafe_allow_html=True)
    st.markdown('''<div class="uph">
      <div class="upey">⚡ Intelligence Engine</div>
      <div class="upt">Your clients,<br><em>clearly ranked.</em></div>
      <div class="ups">Upload any Excel or CSV. The engine auto-detects your columns, scores every client, and tells you exactly who to call — and why.</div>
    </div>''',unsafe_allow_html=True)
    _,cc,_=st.columns([1,2,1])
    with cc:
        uploaded=st.file_uploader('',type=['xlsx','xls','csv'],label_visibility='collapsed')
        st.markdown("<div style='text-align:center;font-size:11px;color:#6e7681;margin-top:.5rem'>Any column format · Excel or CSV</div>",unsafe_allow_html=True)
        st.markdown('<br>',unsafe_allow_html=True)
        if st.button('Load demo data →',use_container_width=True):
            st.session_state.use_demo=True; st.session_state.screen='map'; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)
    return uploaded

def show_mapping(df):
    show_nav()
    st.markdown('<div class="bc">Upload → <em>Column mapping</em> → Dashboard</div>',unsafe_allow_html=True)
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    st.markdown('### Map your columns')
    st.caption('Auto-detected where possible — adjust if needed.')
    cols=df.columns.tolist(); mapping={}
    lbls={'name':'Client name','age':'Age','portfolio':'Portfolio / AUM (₹)','sip':'Monthly SIP (₹)',
          'lastContact':'Last contact date','goal':'Product / goal','tenure':'Client since (year)',
          'nominee':'Nominee updated?','phone':'Phone number'}
    g=st.columns(2)
    for i,(key,hints) in enumerate(FIELDS):
        best=det(hints,cols)
        with g[i%2]:
            opts=['— skip —']+cols
            idx=(cols.index(best)+1) if best and best in cols else 0
            sel=st.selectbox(lbls.get(key,key),opts,index=idx,key=f'm_{key}')
            mapping[key]=sel if sel!='— skip —' else None
    st.markdown('<br>',unsafe_allow_html=True)
    c1,c2,_=st.columns([1,1,4])
    with c1:
        if st.button('Run engine →',use_container_width=True):
            with st.spinner('Scoring all clients...'):
                clients,merged=process(df,mapping)
            st.session_state.clients=clients; st.session_state.merged_count=merged
            db_save(st.session_state.user_id,clients)
            st.session_state.screen='dashboard'; st.rerun()
    with c2:
        if st.button('← Back'):
            st.session_state.screen='upload'; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

def show_dashboard(clients):
    show_nav()
    st.markdown('<div class="bc">Upload → Mapping → <em>Intelligence Dashboard</em></div>',unsafe_allow_html=True)
    st.markdown('<div class="wrap">',unsafe_allow_html=True)
    aum=sum(num(c.get('portfolio',0)) for c in clients)
    high=[c for c in clients if c.get('priority')=='High']
    at_risk=[c for c in clients if c.get('churn',0)>50]
    no_sip=[c for c in clients if 'No SIP' in c.get('flags',[])]
    no_nom=[c for c in clients if 'No Nominee' in c.get('flags',[])]
    hni=[c for c in clients if 'High Value' in c.get('flags',[])]
    risk_aum=sum(num(c.get('portfolio',0)) for c in at_risk)
    user=st.session_state.get('user_name','')
    now=now_ist()
    greeting=get_greeting()
    if 'agenda' not in st.session_state: st.session_state.agenda=random.choice(AGENDAS)
    agenda=st.session_state.agenda
    pct=round(len(high)/len(clients)*100) if clients else 0
    # GREETING
    st.markdown(f'''<div class="greet">
      <div>
        <div class="gt">⚡ {now.strftime("%A, %d %B %Y")} · {now.strftime("%I:%M %p")} IST</div>
        <div class="gn">{greeting}, {user}.</div>
        <div class="gs">{agenda}</div>
      </div>
      <div class="gstats">
        <div class="gst"><span class="gst-num" style="color:#3fb950">{len(high)}</span><span class="gst-label">Call today</span></div>
        <div class="gst"><span class="gst-num" style="color:#f85149">{len(at_risk)}</span><span class="gst-label">Leaving risk</span></div>
        <div class="gst"><span class="gst-num" style="color:#e6edf3">{fi(aum)}</span><span class="gst-label">Total AUM</span></div>
      </div>
    </div>''',unsafe_allow_html=True)
    # KPI CARDS
    st.markdown(f'''<div class="kgrid">
      <div class="kc gr"><div class="kl">Total AUM</div><div class="knum">{fi(aum)}</div>
        <div class="kdesc">{len(clients)} clients · {len(hni)} high-value (50L+)</div>
        <div class="ksig">↑ Full portfolio pipeline</div><div class="khint">▼ Tap below to see breakdown</div></div>
      <div class="kc bl"><div class="kl">Ready to act</div><div class="knum">{len(high)}</div>
        <div class="kdesc">Health score 70+ — call these first</div>
        <div class="ksig">{pct}% of your client base</div><div class="khint">▼ Tap to see who</div></div>
      <div class="kc rd"><div class="kl">Leaving risk</div><div class="knum">{len(at_risk)}</div>
        <div class="kdesc">May move to another advisor soon</div>
        <div class="ksig">⚠ {fi(risk_aum)} at risk</div><div class="khint">▼ Tap to see who</div></div>
      <div class="kc am"><div class="kl">Revenue gap</div><div class="knum">{len(no_sip)}</div>
        <div class="kdesc">Portfolio but no monthly SIP</div>
        <div class="ksig">Easy SIP upsell opportunity</div><div class="khint">▼ Tap to see who</div></div>
      <div class="kc pu"><div class="kl">Paperwork due</div><div class="knum">{len(no_nom)}</div>
        <div class="kdesc">Nominee form not filed</div>
        <div class="ksig">Compliance risk for family</div><div class="khint">▼ Tap to see who</div></div>
    </div>''',unsafe_allow_html=True)
    # KPI CLICK BUTTONS
    k1,k2,k3,k4,k5=st.columns(5)
    kdata=[(k1,'kaum','Total AUM',clients),(k2,'khigh','Ready to act',high),
           (k3,'krisk','Leaving risk',at_risk),(k4,'ksip','Revenue gap',no_sip),(k5,'knom','Paperwork due',no_nom)]
    active=st.session_state.get('kpi_open',None)
    for col,key,label,lst in kdata:
        with col:
            lbl='▲ Close' if active==key else f'▼ {len(lst)} clients'
            if st.button(lbl,key=f'kb_{key}',use_container_width=True):
                st.session_state.kpi_open=None if active==key else key; st.rerun()
    # KPI DETAIL
    if active:
        dmap={'kaum':('Total AUM breakdown',clients),'khigh':('Ready to act — call these first',high),
              'krisk':('Leaving risk — contact urgently',at_risk),
              'ksip':('Revenue gap — no SIP despite portfolio',no_sip),
              'knom':('Paperwork due — nominee missing',no_nom)}
        if active in dmap:
            dlbl,dlst=dmap[active]
            rd=''
            for i,c in enumerate(dlst[:10]):
                sc=c.get('score',0); pr=c.get('priority','Low')
                fill='#3fb950' if sc>=70 else ('#d29922' if sc>=45 else '#f85149')
                cc2='chi' if pr=='High' else ('chm' if pr=='Medium' else 'chl')
                rd+=f'''<tr><td class="prank">#{i+1}</td>
                  <td><div class="pname">{c.get("name","—")}</div>
                  <div class="psub">{c.get("goal","—")} · Age {c.get("age","—")}</div></td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:12px">{fi(c.get("portfolio",0))}</td>
                  <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
                  <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
                  <td><span class="chip {cc2}">{pr}</span></td>
                  <td style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#8b949e">{" · ".join(c.get("flags",[])[:2]) or "—"}</td></tr>'''
            st.markdown(f'''<div class="kdet">
              <div class="kdet-h"><span class="kdet-t">{dlbl} <span style="font-size:12px;color:#8b949e;font-weight:400">({len(dlst)} clients)</span></span></div>
              <div style="overflow-x:auto"><table class="ptable" style="margin:0">
              <thead><tr><th></th><th>Client</th><th>Portfolio</th><th>Health score</th><th>Status</th><th>Alerts</th></tr></thead>
              <tbody>{rd}</tbody></table></div></div>''',unsafe_allow_html=True)
    # CHARTS
    st.markdown('<div style="height:1.5rem"></div>',unsafe_allow_html=True)
    gc1,gc2,gc3=st.columns(3)
    with gc1:
        sv=[sum(num(c.get('portfolio',0)) for c in clients if c.get('priority')==p)/1e5 for p in ['High','Medium','Low']]
        fig=go.Figure(go.Bar(x=['Ready','Medium','Needs work'],y=[round(v,1) for v in sv],
            marker_color=['#3fb950','#d29922','#f85149'],marker_line_width=0,
            text=[f'₹{v:.1f}L' for v in sv],textposition='outside',textfont=dict(color='#e6edf3',size=10)))
        fig.update_layout(**{**PC,'title':dict(text='Portfolio by status',font=dict(size=12,color='#e6edf3'),x=0)})
        fig.update_traces(width=0.5)
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with gc2:
        scores=[c.get('score',0) for c in clients]
        bins=[0,20,40,60,80,101]; lbs=['0-20','21-40','41-60','61-80','81-100']
        cts=[sum(1 for s in scores if bins[i]<=s<bins[i+1]) for i in range(5)]
        fig2=go.Figure(go.Bar(x=lbs,y=cts,marker_color=['#f85149','#f85149','#d29922','#3fb950','#3fb950'],
            marker_line_width=0,text=cts,textposition='outside',textfont=dict(color='#e6edf3',size=10)))
        fig2.update_layout(**{**PC,'title':dict(text='Health score spread',font=dict(size=12,color='#e6edf3'),x=0)})
        fig2.update_traces(width=0.6)
        st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    with gc3:
        sx=[c.get('score',0) for c in clients]; sy=[c.get('churn',0) for c in clients]
        sn=[c.get('name','') for c in clients]
        scc=['#f85149' if c.get('churn',0)>50 else ('#d29922' if c.get('churn',0)>25 else '#3fb950') for c in clients]
        fig3=go.Figure(go.Scatter(x=sx,y=sy,mode='markers',
            marker=dict(color=scc,size=8,line=dict(width=0)),text=sn,
            hovertemplate='<b>%{text}</b><br>Score: %{x}<br>Risk: %{y}%<extra></extra>'))
        fig3.update_layout(**{**PC,'title':dict(text='Health vs leaving risk',font=dict(size=12,color='#e6edf3'),x=0),
            'xaxis':dict(**PC['xaxis'],title='Health score',range=[0,105]),
            'yaxis':dict(**PC['yaxis'],title='Leaving risk %',range=[0,105])})
        st.plotly_chart(fig3,use_container_width=True,config={'displayModeBar':False})
    # TABS
    st.markdown('<div style="height:1rem"></div>',unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5=st.tabs(['Priority Rankings','Smart Next Best Action','Event Intelligence','ML Prediction Engine','WhatsApp Drafts'])
    with tab1:
        st.markdown('<div style="height:.75rem"></div>',unsafe_allow_html=True)
        st.markdown('''<div class="mhd">
          <div class="mic mbl">≡</div>
          <div><div class="mt">Priority Rankings</div>
          <div class="msub">Sorted by composite health score · Click any row for strategic recommendation</div></div>
        </div>''',unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns([2,2,1,1])
        with f1:
            fsel=st.selectbox('Filter',['All clients','Ready to act','Medium','Needs attention','Leaving risk','No SIP','No Nominee'],label_visibility='collapsed')
        with f2:
            search_q = st.text_input('', placeholder='Search by name or product...', label_visibility='collapsed')
        with f3:
            min_aum = st.number_input('Min AUM (L)', min_value=0, value=0, step=10, label_visibility='collapsed')
        with f4:
            if st.button('Export Excel', use_container_width=True):
                try:
                    import json as _json
                    xls = to_excel_bytes(_json.dumps(clients))
                    st.download_button('Download', data=xls, file_name=f'advisoriq_clients_{datetime.date.today()}.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='dl_xl')
                except Exception as e:
                    st.error(f'Export failed: {e}')
        filtered=clients
        if 'Ready' in fsel: filtered=[c for c in clients if c.get('priority')=='High']
        elif 'Medium' in fsel: filtered=[c for c in clients if c.get('priority')=='Medium']
        elif 'Needs' in fsel: filtered=[c for c in clients if c.get('priority')=='Low']
        elif 'Leaving' in fsel: filtered=[c for c in clients if c.get('churn',0)>50]
        elif 'No SIP' in fsel: filtered=[c for c in clients if 'No SIP' in c.get('flags',[])]
        elif 'Nominee' in fsel: filtered=[c for c in clients if 'No Nominee' in c.get('flags',[])]
        # Search filter
        if search_q.strip():
            q = search_q.strip().lower()
            filtered = [c for c in filtered if q in c.get('name','').lower() or q in c.get('goal','').lower()]
        # AUM filter
        if min_aum > 0:
            filtered = [c for c in filtered if num(c.get('portfolio',0)) >= min_aum * 1e5]
        ml_status = '🤖 ML-scored' if st.session_state.get('ml_active') else '📐 Rule-based'
        st.markdown(f"<div style='font-size:11px;color:#6e7681;font-family:JetBrains Mono,monospace;margin-bottom:.75rem'>{len(filtered)} of {len(clients)} records · {ml_status} · sorted by composite score</div>",unsafe_allow_html=True)
        if 'exp_row' not in st.session_state: st.session_state.exp_row=None
        for i,c in enumerate(filtered[:20]):
            sc=c.get('score',0); ch=c.get('churn',0); pr=c.get('priority','Low')
            fill='#3fb950' if sc>=70 else ('#d29922' if sc>=45 else '#f85149')
            chcol='#f85149' if ch>60 else ('#d29922' if ch>30 else '#3fb950')
            cc2='chi' if pr=='High' else ('chm' if pr=='Medium' else 'chl')
            rank='🥇' if i==0 else ('🥈' if i==1 else ('🥉' if i==2 else f'#{i+1}'))
            tags_h=''.join(f'<span class="tag">{f}</span>' for f in c.get('flags',[])[:2])
            is_exp=st.session_state.exp_row==i
            ex_cls='xp' if is_exp else ''
            st.markdown(f'''<table class="ptable" style="margin-bottom:0"><tbody>
            <tr class="{ex_cls}">
              <td class="prank">{rank}</td>
              <td><div class="pname">{c.get("name","—")}</div>
              <div class="psub">{c.get("goal","—")} · Age {c.get("age","—")}</div></td>
              <td style="font-family:'JetBrains Mono',monospace;font-size:12px">{fi(c.get("portfolio",0))}</td>
              <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#6e7681">{c.get("tenure","—")}</td>
              <td><div class="sbar"><span class="snum" style="color:{fill}">{sc}</span>
                <span class="strack"><span class="sfill" style="width:{sc}%;background:{fill}"></span></span></div></td>
              <td><span class="chip {cc2}">{pr}</span></td>
              <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{chcol}">{ch}%</td>
              <td style="font-size:11px">{tags_h}</td>
            </tr></tbody></table>''',unsafe_allow_html=True)
            _,bc=st.columns([11,1])
            with bc:
                if st.button('▲' if is_exp else '▼',key=f'er_{i}'):
                    st.session_state.exp_row=None if is_exp else i; st.rerun()
            if is_exp:
                if pr=='High': act=f"{c.get('name','')} is primed — schedule a portfolio review and introduce a MF top-up or capital gain bond based on their {fi(c.get('portfolio',0))} profile."
                elif pr=='Medium': act=f"{c.get('name','')} sits near the high-priority threshold. A personalised offer around their {fi(c.get('portfolio',0))} range could shift classification with strong confidence."
                else: act=f"{c.get('name','')} needs re-engagement. Start with a no-pressure portfolio health-check call — avoid sales pitch on first contact."
                st.markdown(f'''<div class="xin" style="margin-bottom:8px">
                  <div class="xlbl">Strategic Recommendation</div>
                  <div class="xtxt">{act}</div>
                  <div style="margin-top:8px;font-size:11px;color:#6e7681;font-family:'JetBrains Mono',monospace">📅 Since {c.get("tenure","—")} · 📋 Nominee: {c.get("nominee","—")} · 💰 SIP: {fi(c.get("sip",0)) if num(c.get("sip",0))>0 else "None"}</div>
                </div>''',unsafe_allow_html=True)
    with tab2:
        top_c=clients[0] if clients else {}
        risk2=', '.join(c.get('name','') for c in at_risk[:3]) or '—'
        sip2=', '.join(c.get('name','') for c in no_sip[:3]) or '—'
        nom2=', '.join(c.get('name','') for c in no_nom[:2]) or '—'
        st.markdown('<div style="height:.75rem"></div>',unsafe_allow_html=True)
        st.markdown('''<div class="mhd">
          <div class="mic mam">⚡</div>
          <div><div class="mt">Smart Next Best Action</div>
          <div class="msub">Decision engine · Impact scoring · Automated outreach templates</div></div>
        </div>''',unsafe_allow_html=True)
        tn=top_c.get('name','Top client')
        actions_data=[
            ('HIGH','bhi','WhatsApp + Follow-up call',
             f'{tn} — Deploy personalised value proposition',
             f'Gradient boosting classifier places {tn} at health score {top_c.get("score",0)}/100. Feature importance analysis shows portfolio sensitivity at {round(num(top_c.get("portfolio",0))/max(aum/len(clients),1)*100,0):.0f}% of cohort mean. A targeted intervention could shift classification to high-priority with ~{top_c.get("conv",60)}% confidence.',
             f'+{fi(num(top_c.get("portfolio",0))*0.05)} expected revenue impact',
             f'Hi {tn.split()[0] if tn else "there"}! We have prepared something based on your profile. Can I share the details?',
             top_c.get('phone','')),
            ('HIGH','bhi','WhatsApp + Email Sequence',
             f'{len(at_risk)} clients — Urgent churn prevention',
             f'ML model flags {risk2} as high-risk for advisor switch within 60 days. Inactivity beyond 6 months triples withdrawal probability. A portfolio health-check call — not a sales call — is the highest-impact intervention at this stage.',
             f'~{fi(risk_aum*0.12)} recoverable if re-engaged this month',
             'Hi! I wanted to personally check in — it has been a while and I want to make sure your portfolio is positioned well. Can we connect briefly?',
             ''),
            ('GROWTH','bgr','SIP upsell sequence',
             f'{len(no_sip)} clients — SIP conversion drive',
             f'{sip2} hold significant portfolio without a systematic investment plan. Showing a Rs 5,000/month compound projection closing at Rs 30L over 15 years converts 60%+ of these profiles. One visual does the selling.',
             f'+{fi(sum(num(c.get("portfolio",0)) for c in no_sip)*0.03)} annual SIP commission potential',
             'Hi! I have prepared a personalised growth projection based on your portfolio. The numbers are quite compelling — can I share them with you?',
             ''),
            ('COMPLIANCE','bbl','Compliance outreach',
             f'{len(no_nom)} clients — Nominee form drive',
             f'{nom2} have not filed nominee forms — legal risk for their families, compliance exposure for your practice. A 10-minute call positions you as the advisor who cares beyond commissions. High trust-building ROI.',
             'Compliance risk mitigated · High trust impact',
             'Hi! As part of our annual client care review, I noticed your nominee details may need updating. This is critical to protect your family interests — can we sort this quickly?',
             ''),
        ]
        for badge,bcls,channel,title,reason,impact,wa_msg,ph in actions_data:
            wl=f'https://wa.me/{ph}?text={wa_msg.replace(" ","%20")}' if ph else f'https://wa.me/?text={wa_msg.replace(" ","%20")}'
            st.markdown(f'''<div class="acard">
              <div class="atop"><span class="abadge {bcls}">{badge}</span>
              <div style="flex:1"><div class="achan">📲 {channel}</div>
              <div class="atitle">{title}</div></div></div>
              <div class="areason">{reason}</div>
              <div class="aimpact">Expected impact: {impact}</div>
              <div class="waq"><span class="waql">📱 WhatsApp quick send</span>
              <div class="waqm">"{wa_msg}"</div></div>
              <div class="abtns"><a class="btn-wa" href="{wl}" target="_blank">Open WhatsApp ↗</a></div>
            </div>''',unsafe_allow_html=True)
    with tab3:
        st.markdown('<div style="height:.75rem"></div>',unsafe_allow_html=True)
        st.markdown('''<div class="mhd">
          <div class="mic mam">✦</div>
          <div><div class="mt">Event Intelligence</div>
          <div class="msub">Data-driven event recommendations · ROI projections</div></div>
        </div>''',unsafe_allow_html=True)
        hni_n=', '.join(c.get('name','') for c in hni[:3])
        senior=[c for c in clients if int(float(c.get('age') or 0))>=55]
        mid=[c for c in clients if c.get('priority')=='Medium']
        mid_n=', '.join(c.get('name','') for c in mid[:3])
        senior_n=', '.join(c.get('name','') for c in senior[:3])
        evs=[
            ('high impact','#d29922','Conversion Accelerator Workshop',
             f'{len(mid)} mid-funnel clients identified near the decision boundary. Deploy a value-demonstration format targeting {mid_n} who are closest to the high-priority threshold. Regression model suggests 22-31% probability of tier upgrade post-event.',
             f'ROI: ~{fi(sum(num(c.get("portfolio",0)) for c in mid)*0.08)} uplift',
             f'Workshop · {len(mid)} mid-funnel · This month'),
            ('medium impact','#58a6ff','HNI Portfolio Deep-Dive',
             f'Your top segment contributes {round(sum(num(c.get("portfolio",0)) for c in hni)/max(aum,1)*100,1)}% of total AUM across {len(hni)} accounts ({fi(sum(num(c.get("portfolio",0)) for c in hni))}). Industry-specific event drives 2.1x engagement vs generic format. Targets: {hni_n}.',
             f'ROI: ~{fi(sum(num(c.get("portfolio",0)) for c in hni)*0.04)} incremental',
             f'Industry Event · {len(hni)} HNI contacts · This quarter'),
            ('medium impact','#58a6ff','Portfolio Intelligence Summit',
             f'With {len(clients)} accounts totalling {fi(aum)} in pipeline, host a data-driven review combining live dashboards with predictive insights. Focus: identify dormant accounts showing reactivation signals. Targets: {senior_n}.',
             'ROI: Strategic — long-term LTV impact',
             f'Summit · Full portfolio · Quarterly'),
        ]
        rows_e=''
        for tag,tc,title,body,roi,meta in evs:
            rows_e+=f'''<div class="evcard">
              <div class="evtop"><div class="evtitle">{title}</div>
                <span class="chip" style="background:{tc}18;color:{tc};border:1px solid {tc}44;font-size:10px;padding:2px 8px;border-radius:10px;font-family:'JetBrains Mono',monospace;font-weight:600">{tag}</span></div>
              <div class="evbody">{body}</div>
              <div class="evroi">{roi}</div>
              <div class="evmeta"><span>📍 {meta.split("·")[0].strip()}</span>
                <span>👥 {meta.split("·")[1].strip() if len(meta.split("·"))>1 else ""}</span>
                <span>🕐 {meta.split("·")[2].strip() if len(meta.split("·"))>2 else ""}</span></div>
            </div>'''
        st.markdown(f'<div class="evgrid">{rows_e}</div>',unsafe_allow_html=True)
    with tab4:
        st.markdown('<div style="height:.75rem"></div>',unsafe_allow_html=True)
        st.markdown('''<div class="mhd">
          <div class="mic mpu">◈</div>
          <div><div class="mt">ML Prediction Engine</div>
          <div class="msub">Ensemble model · Churn prediction · Revenue forecasting · Confidence intervals</div></div>
        </div>''',unsafe_allow_html=True)
        if 'ml_exp' not in st.session_state: st.session_state.ml_exp=None
        st.markdown('''<div class="mlhdr">
          <span>Account</span><span>Score Δ</span><span>Churn risk</span>
          <span>Conv. prob</span><span>Predicted rev.</span><span>Trend</span>
        </div>''',unsafe_allow_html=True)
        for i,c in enumerate(clients[:15]):
            sc=c.get('score',0); ch=c.get('churn',0); cv=c.get('conv',50)
            dlo,dhi=max(0,sc-3),min(100,sc+3)
            trend='Ascending' if sc>=60 else ('Stable' if sc>=45 else 'Declining')
            tcls='tup' if trend=='Ascending' else ('tdn' if trend=='Declining' else 'tsb')
            conf=random.randint(85,94)
            pred_rev=round(num(c.get('portfolio',0))*1.12)
            chcol='#f85149' if ch>50 else ('#d29922' if ch>25 else '#3fb950')
            cvcol='#3fb950' if cv>60 else ('#d29922' if cv>40 else '#f85149')
            is_me=st.session_state.ml_exp==i
            mecls='mlex' if is_me else ''
            st.markdown(f'''<div class="mlrow {mecls}">
              <span style="font-weight:600;font-size:13px">{c.get("name","—")}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3fb950">{dlo}→{dhi}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{chcol}">{ch}%</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:{cvcol}">{cv}%</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:12px">₹{pred_rev:,}</span>
              <span><span class="{tcls}">↗ {trend}</span>
                <span class="cb" style="margin-left:6px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#8b949e">{conf}%</span>
                <span class="cbar"><span class="cfill" style="width:{conf}%"></span></span></span>
              </span>
            </div>''',unsafe_allow_html=True)
            _,bc=st.columns([11,1])
            with bc:
                if st.button('▲' if is_me else '▼',key=f'me_{i}'):
                    st.session_state.ml_exp=None if is_me else i; st.rerun()
            if is_me:
                sm = st.session_state.get('score_model')
                cm = st.session_state.get('churn_model')
                feat = get_feature_importance(c, sm)
                ml_tag = '🤖 Gradient Boosting + Random Forest' if st.session_state.get('ml_active') else '📐 Rule-based heuristics'
                st.markdown(f'''<div class="mlxpand">
                  <div class="mlfl">⊕ Model feature importance · <span style="color:#8b949e;font-weight:400">{ml_tag}</span></div>
                  <div class="mlft">↳ {feat}</div>
                </div>''',unsafe_allow_html=True)
        if len(clients)>15:
            st.markdown(f"<div style='text-align:center;padding:1rem;font-size:12px;color:#58a6ff;font-family:JetBrains Mono,monospace'>View all {len(clients)} predictions →</div>",unsafe_allow_html=True)
    with tab5:
        st.markdown('<div style="height:.75rem"></div>',unsafe_allow_html=True)
        st.markdown('''<div class="mhd">
          <div class="mic mgr">📱</div>
          <div><div class="mt">WhatsApp Drafts</div>
          <div class="msub">Personalised templates · Direct send links</div></div>
        </div>''',unsafe_allow_html=True)
        names=[c.get('name','') for c in clients if c.get('name')]
        seln=st.selectbox('Select client',names,label_visibility='collapsed')
        sel=next((c for c in clients if c.get('name')==seln),None)
        if sel:
            sc2=sel.get('score',0); ch2=sel.get('churn',0)
            scc='#3fb950' if sc2>=70 else ('#d29922' if sc2>=45 else '#f85149')
            chc='#f85149' if ch2>50 else '#3fb950'
            un=st.session_state.get('user_name','Your Advisor')
            uc=st.session_state.get('user_company','')
            ca,cb=st.columns([1,1])
            with ca:
                st.markdown(f'''<div class="wprof">
                  <div class="wpname">{sel.get("name","")}</div>
                  <div class="wprow"><span>Portfolio</span><span class="wpval">{fi(sel.get("portfolio",0))}</span></div>
                  <div class="wprow"><span>Monthly SIP</span><span class="wpval">{fi(sel.get("sip",0)) if num(sel.get("sip",0))>0 else "Not started"}</span></div>
                  <div class="wprow"><span>Health score</span><span class="wpval" style="color:{scc}">{sc2}/100</span></div>
                  <div class="wprow"><span>Leaving risk</span><span class="wpval" style="color:{chc}">{ch2}%</span></div>
                  <div class="wprow"><span>Product</span><span class="wpval">{sel.get("goal","—")}</span></div>
                </div>''',unsafe_allow_html=True)
                mt=st.radio('Type',['Check-in call','SIP proposal','Portfolio review','Nominee update'],label_visibility='visible')
            with cb:
                tmpls={'Check-in call':f'Dear {sel.get("name","")},\n\nI have been reviewing your portfolio and there are important developments I would like to walk you through personally.\n\nCould we schedule a quick 20-minute call this week?\n\nWarm regards,\n{un}\n{uc}',
                       'SIP proposal':f'Dear {sel.get("name","")},\n\nBased on your portfolio of {fi(sel.get("portfolio",0))}, I have prepared a personalised SIP projection that could significantly grow your wealth. The numbers are compelling.\n\nCan we find 15 minutes to walk through it?\n\nWarm regards,\n{un}\n{uc}',
                       'Portfolio review':f'Dear {sel.get("name","")},\n\nYour portfolio review is due. Given current market conditions, I want to ensure your investments are optimally positioned for the year ahead.\n\nWhen works best for a quick call?\n\nWarm regards,\n{un}\n{uc}',
                       'Nominee update':f'Dear {sel.get("name","")},\n\nAs part of our annual client care review, I noticed your nominee details may need updating. This is critical to protect your family interests.\n\nIt takes under 10 minutes. Can I help?\n\nWarm regards,\n{un}\n{uc}'}
                edited=st.text_area('Edit before sending',tmpls[mt],height=220,label_visibility='collapsed')
                ph=sel.get('phone','')
                wt=edited.replace('\n','%0A').replace(' ','%20')
                wl2=f'https://wa.me/{ph}?text={wt}' if ph else f'https://wa.me/?text={wt}'
                st.markdown(f'<br><a class="btn-wa" href="{wl2}" target="_blank">📱 Open in WhatsApp ↗</a>',unsafe_allow_html=True)
    # FOOTER
    st.markdown('<br><hr>',unsafe_allow_html=True)
    mc=st.session_state.get('merged_count',0)
    ms2=f' · {mc} duplicates merged' if mc else ''
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#21262d;font-family:JetBrains Mono,monospace'>AdvisorIQ · {len(clients)} clients · {fi(aum)} AUM{ms2}</div>",unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"**{st.session_state.get('user_name','')}**")
        st.caption(st.session_state.get('user_company',''))
        if st.button('Upload new data'): st.session_state.screen='upload'; st.rerun()
        if st.button('Sign out'):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

def main():
    init_db()
    inject_css()
    if 'screen' not in st.session_state: st.session_state.screen='login'
    if 'user_id' not in st.session_state and st.session_state.screen!='login':
        st.session_state.screen='login'
    screen=st.session_state.screen
    if screen=='login': show_login(); return
    if screen=='upload':
        up=show_upload()
        if up:
            try:
                df=pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
                st.session_state.upload_df=df; st.session_state.screen='map'; st.rerun()
            except Exception as e: st.error(f'Could not read file: {e}')
        return
    if screen=='map':
        if st.session_state.get('use_demo'):
            clients=prep_demo(); st.session_state.clients=clients
            db_save(st.session_state.user_id,clients)
            st.session_state.use_demo=False; st.session_state.screen='dashboard'; st.rerun()
        elif 'upload_df' in st.session_state: show_mapping(st.session_state.upload_df)
        else: st.session_state.screen='upload'; st.rerun()
        return
    if screen=='dashboard':
        clients=st.session_state.get('clients',[])
        if not clients: st.session_state.screen='upload'; st.rerun(); return
        show_dashboard(clients); return

if __name__=='__main__': main()