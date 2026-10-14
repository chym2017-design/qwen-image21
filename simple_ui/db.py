"""数据层：SQLite（标准库）+ 密码哈希 + 会话 + 积分 + 个人仓库 + 标签。

单文件、零外部依赖（仅用 sqlite3 / hashlib / secrets / os / time）。
库文件：~/qwen-image21/simple_ui/data/app.db
图片库：~/qwen-image21/simple_ui/data/users/<uid>/<kind>/
"""
import base64
import hashlib
import os
import secrets
import sqlite3
import time

ROOT = os.path.expanduser("~/qwen-image21")
DATA_DIR = os.path.join(ROOT, "simple_ui", "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
SESSION_TTL = 30 * 24 * 3600          # 会话有效期 30 天
PBKDF2_ITERS = 200_000

# 管理员可调的运营参数（存 config 表；均为字符串）
DEFAULT_CONFIG = {
    "signup_code": "",            # 注册码；空时 server 启动会用旧 access_token 填充
    "allow_open_signup": "0",     # 1=开放注册（无需码）
    "cost_turbo": "1",            # turbo 出图消耗积分
    "cost_standard": "3",         # 高质量/pro 出图消耗积分
    "signup_bonus": "50",         # 新用户赠送积分
    "member_daily": "100",        # 会员每日刷新积分
    "member_daily_cap": "200",    # 会员积分上限（刷新时不超过）
}

# 衣橱默认标签（系统级，user_id=0，对所有用户只读可见）
DEFAULT_TAGS = [
    ("上装", "类别"), ("下装", "类别"), ("外套", "类别"), ("连衣裙", "类别"),
    ("鞋", "类别"), ("包", "类别"), ("配饰", "类别"), ("帽子", "类别"),
    ("春", "季节"), ("夏", "季节"), ("秋", "季节"), ("冬", "季节"),
    ("休闲", "风格"), ("正式", "风格"), ("运动", "风格"), ("居家", "风格"),
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      pass_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'user',
      credits INTEGER NOT NULL DEFAULT 0,
      is_member INTEGER NOT NULL DEFAULT 0,
      allow_local INTEGER NOT NULL DEFAULT 1,
      allow_cloud INTEGER NOT NULL DEFAULT 1,
      created_at REAL NOT NULL,
      last_refresh TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS sessions(
      sid TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      created_at REAL NOT NULL,
      expires_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS credit_log(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      delta INTEGER NOT NULL,
      reason TEXT,
      at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS items(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      kind TEXT NOT NULL,
      filename TEXT NOT NULL,
      orig_name TEXT,
      width INTEGER,
      height INTEGER,
      created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tags(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      kind TEXT,
      UNIQUE(user_id, name)
    );
    CREATE TABLE IF NOT EXISTS item_tags(
      item_id INTEGER NOT NULL,
      tag_id INTEGER NOT NULL,
      PRIMARY KEY(item_id, tag_id)
    );
    CREATE TABLE IF NOT EXISTS config(
      key TEXT PRIMARY KEY,
      value TEXT
    );
    """)
    for k, v in DEFAULT_CONFIG.items():
        c.execute("INSERT OR IGNORE INTO config(key,value) VALUES(?,?)", (k, v))
    # 系统默认标签（user_id=0）
    for name, kind in DEFAULT_TAGS:
        c.execute("INSERT OR IGNORE INTO tags(user_id,name,kind) VALUES(0,?,?)", (name, kind))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- 密码 / 用户
def _hash(password, salt_b64):
    salt = base64.b64decode(salt_b64)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return h.hex()


def create_user(username, password, signup_bonus):
    """创建用户。若库中无任何用户，自动成为 admin。返回新用户 dict。"""
    username = (username or "").strip()
    if not username or len(username) < 2:
        raise ValueError("用户名至少 2 个字符")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    salt = base64.b64encode(secrets.token_bytes(16)).decode()
    conn = get_conn()
    c = conn.cursor()
    n = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    role = "admin" if n == 0 else "user"
    try:
        cur = c.execute(
            "INSERT INTO users(username,pass_hash,salt,role,credits,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (username, _hash(password, salt), salt, role, int(signup_bonus), time.time()))
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("用户名已存在")
    uid = cur.lastrowid
    c.execute("INSERT INTO credit_log(user_id,delta,reason,at) VALUES(?,?,?,?)",
              (uid, int(signup_bonus), "注册赠送", time.time()))
    conn.commit()
    row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row)


def verify_user(username, password):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    conn.close()
    if not row:
        return None
    if not secrets.compare_digest(_hash(password, row["salt"]), row["pass_hash"]):
        return None
    return dict(row)


def get_user(uid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id,username,role,credits,is_member,allow_local,allow_cloud,created_at "
        "FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user(uid, **fields):
    allowed = {"role", "credits", "is_member", "allow_local", "allow_cloud"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append("%s=?" % k)
            vals.append(int(v) if k not in ("role",) else v)
    if not sets:
        return
    vals.append(uid)
    conn = get_conn()
    conn.execute("UPDATE users SET %s WHERE id=?" % ",".join(sets), vals)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- 会话
def create_session(uid):
    sid = secrets.token_urlsafe(24)
    now = time.time()
    conn = get_conn()
    conn.execute("INSERT INTO sessions(sid,user_id,created_at,expires_at) VALUES(?,?,?,?)",
                 (sid, uid, now, now + SESSION_TTL))
    conn.commit()
    conn.close()
    return sid


def get_user_by_session(sid):
    """按会话取用户；顺带做会员每日积分刷新。sid 无效/过期返回 None。"""
    if not sid:
        return None
    conn = get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE sid=?", (sid,)).fetchone()
    if not row or row["expires_at"] < time.time():
        conn.close()
        return None
    u = conn.execute("SELECT * FROM users WHERE id=?", (row["user_id"],)).fetchone()
    if not u:
        conn.close()
        return None
    u = dict(u)
    # 会员每日刷新
    today = time.strftime("%Y-%m-%d")
    if u["is_member"] and u["last_refresh"] != today:
        daily = int(config_get("member_daily"))
        cap = int(config_get("member_daily_cap"))
        before = u["credits"]
        u["credits"] = min(u["credits"] + daily, cap)
        gained = u["credits"] - before
        conn.execute("UPDATE users SET credits=?, last_refresh=? WHERE id=?",
                     (u["credits"], today, u["id"]))
        if gained > 0:
            conn.execute("INSERT INTO credit_log(user_id,delta,reason,at) VALUES(?,?,?,?)",
                         (u["id"], gained, "会员每日刷新", time.time()))
        conn.commit()
    conn.close()
    return u


def delete_session(sid):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE sid=?", (sid,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- 积分
def spend_credits(uid, amount, reason):
    """扣积分；不足返回 False。"""
    conn = get_conn()
    u = conn.execute("SELECT credits FROM users WHERE id=?", (uid,)).fetchone()
    if not u or u["credits"] < amount:
        conn.close()
        return False
    conn.execute("UPDATE users SET credits=credits-? WHERE id=?", (amount, uid))
    conn.execute("INSERT INTO credit_log(user_id,delta,reason,at) VALUES(?,?,?,?)",
                 (uid, -amount, reason, time.time()))
    conn.commit()
    conn.close()
    return True


def grant_credits(uid, delta, reason):
    conn = get_conn()
    conn.execute("UPDATE users SET credits=credits+? WHERE id=?", (delta, uid))
    conn.execute("INSERT INTO credit_log(user_id,delta,reason,at) VALUES(?,?,?,?)",
                 (uid, delta, reason, time.time()))
    conn.commit()
    conn.close()


def credit_log(uid, limit=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT delta,reason,at FROM credit_log WHERE user_id=? ORDER BY at DESC LIMIT ?",
        (uid, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- 配置
def config_get(key):
    conn = get_conn()
    row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else DEFAULT_CONFIG.get(key, "")


def config_set(key, value):
    conn = get_conn()
    conn.execute("INSERT INTO config(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=?",
                 (key, value, value))
    conn.commit()
    conn.close()


def config_all():
    conn = get_conn()
    rows = conn.execute("SELECT key,value FROM config").fetchall()
    conn.close()
    d = dict(DEFAULT_CONFIG)
    d.update({r["key"]: r["value"] for r in rows})
    return d


# ---------------------------------------------------------------- 个人仓库
KINDS = ("photo", "wardrobe", "tryon", "t2i")


def user_dir(uid, kind):
    d = os.path.join(DATA_DIR, "users", str(uid), kind)
    os.makedirs(d, exist_ok=True)
    return d


def add_item(uid, kind, filename, orig_name="", w=0, h=0):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO items(user_id,kind,filename,orig_name,width,height,created_at) VALUES(?,?,?,?,?,?,?)",
        (uid, kind, filename, orig_name, w, h, time.time()))
    iid = cur.lastrowid
    conn.commit()
    conn.close()
    return iid


def list_items(uid, kind=None):
    conn = get_conn()
    if kind:
        rows = conn.execute(
            "SELECT * FROM items WHERE user_id=? AND kind=? ORDER BY created_at DESC", (uid, kind)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM items WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_item(iid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM items WHERE id=?", (iid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_item(iid, uid):
    """删除一条记录 + 物理文件。仅当属于该用户。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM items WHERE id=? AND user_id=?", (iid, uid)).fetchone()
    if not row:
        conn.close()
        return None
    conn.execute("DELETE FROM item_tags WHERE item_id=?", (iid,))
    conn.execute("DELETE FROM items WHERE id=?", (iid,))
    conn.commit()
    conn.close()
    path = os.path.join(DATA_DIR, "users", str(uid), row["kind"], row["filename"])
    try:
        os.remove(path)
    except OSError:
        pass
    return dict(row)


def rename_item(iid, uid, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("名称不能为空")
    conn = get_conn()
    cur = conn.execute("UPDATE items SET orig_name=? WHERE id=? AND user_id=?", (name, iid, uid))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


# ---------------------------------------------------------------- 标签
def list_tags(uid):
    """系统默认(user_id=0) + 该用户自定义。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM tags WHERE user_id=0 OR user_id=? ORDER BY user_id, kind, name", (uid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_tag(uid, name, kind="自定义"):
    name = (name or "").strip()
    if not name:
        raise ValueError("标签名不能为空")
    conn = get_conn()
    try:
        cur = conn.execute("INSERT INTO tags(user_id,name,kind) VALUES(?,?,?)", (uid, name, kind))
        tid = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("标签已存在")
    conn.close()
    return tid


def delete_tag(uid, tag_id):
    """仅能删自己创建的（user_id=uid）。系统标签(user_id=0)不可删。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM tags WHERE id=? AND user_id=?", (tag_id, uid)).fetchone()
    if not row:
        conn.close()
        return False
    conn.execute("DELETE FROM item_tags WHERE tag_id=?", (tag_id,))
    conn.execute("DELETE FROM tags WHERE id=?", (tag_id,))
    conn.commit()
    conn.close()
    return True


def set_item_tags(item_id, tag_ids):
    conn = get_conn()
    conn.execute("DELETE FROM item_tags WHERE item_id=?", (item_id,))
    for tid in tag_ids:
        conn.execute("INSERT OR IGNORE INTO item_tags(item_id,tag_id) VALUES(?,?)", (item_id, tid))
    conn.commit()
    conn.close()


def item_tag_ids(item_id):
    conn = get_conn()
    rows = conn.execute("SELECT tag_id FROM item_tags WHERE item_id=?", (item_id,)).fetchall()
    conn.close()
    return [r["tag_id"] for r in rows]


def items_with_tags(uid, kind):
    """列表 + 每条的 tag_id 列表，便于前端筛选。"""
    items = list_items(uid, kind)
    conn = get_conn()
    for it in items:
        rows = conn.execute("SELECT tag_id FROM item_tags WHERE item_id=?", (it["id"],)).fetchall()
        it["tags"] = [r["tag_id"] for r in rows]
    conn.close()
    return items
