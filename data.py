"""
data.py — everything related to storing and loading scheme data:
  1. Database engine/session setup (SQLite)
  2. Scheme + ConversationLog table definitions
  3. SEED_SCHEMES — 10 real hand-written schemes, used as a fallback
  4. populate_database() — loads from a local CSV file (e.g. downloaded
     from Kaggle: "Indian Government Schemes" by jainamgada45).
     No AI calls, no rate limits — just reads the file directly.
"""

import os
import sys
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# 1. DATABASE SETUP
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///./rightbridge.db"

# Put the downloaded CSV in your project folder and update this filename
# if it's different (e.g. if Kaggle names it something else on download).
CSV_PATH = "updated_data.csv"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db():
    """Create tables if they don't already exist. Safe to call every startup."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# 2. TABLES
# ---------------------------------------------------------------------------

class Scheme(Base):
    __tablename__ = "schemes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    category = Column(String, index=True)
    eligibility = Column(Text)
    benefits = Column(Text)
    how_to_apply = Column(Text)
    official_link = Column(String)
    documents_required = Column(Text)
    source = Column(String, default="unknown")

    def __repr__(self):
        return f"<Scheme id={self.id} name={self.name!r}>"


class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    user_message = Column(Text)
    bot_reply = Column(Text)
    matched_scheme_names = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<ConversationLog id={self.id} user_id={self.user_id!r}>"


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    scheme_id = Column(Integer, index=True, nullable=True)
    rating = Column(Integer)  # 1-5
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<Feedback id={self.id} rating={self.rating}>"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    password_salt = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"


# ---------------------------------------------------------------------------
# Password hashing — stdlib only (hashlib.pbkdf2_hmac), no extra dependencies
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: str = None) -> tuple:
    """Returns (hash_hex, salt_hex). Pass an existing salt to verify a login."""
    import hashlib
    import secrets

    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    check_hash, _ = hash_password(password, salt)
    import hmac
    return hmac.compare_digest(check_hash, stored_hash)


# ---------------------------------------------------------------------------
# 3. SEED DATA — fallback if the CSV can't be found/read
# ---------------------------------------------------------------------------

SEED_SCHEMES = [
    {
        "name": "e-Shram Card (National Database of Unorganised Workers)",
        "category": "Registration / Social Security",
        "eligibility": "Any unorganised sector worker aged 16-59, including migrant, "
                        "construction, gig, and domestic workers, not a member of EPFO/ESIC.",
        "benefits": "Free unique 12-digit UAN card, access to social security schemes, "
                    "PMSBY accident insurance cover of Rs 2 lakh included free for a year.",
        "how_to_apply": "Register free at eshram.gov.in or nearest Common Service Centre (CSC) "
                         "using Aadhaar and bank account details.",
        "official_link": "https://eshram.gov.in",
        "source": "seed",
    },
    {
        "name": "Ayushman Bharat - PM-JAY",
        "category": "Healthcare",
        "eligibility": "Families identified as poor/vulnerable under SECC 2011 database; "
                        "many unorganised and migrant worker households qualify.",
        "benefits": "Free cashless hospitalization cover up to Rs 5 lakh per family per year "
                    "at empanelled government and private hospitals across India.",
        "how_to_apply": "Check eligibility at pmjay.gov.in or call 14555, then get an "
                         "Ayushman card made at any empanelled hospital or CSC.",
        "official_link": "https://pmjay.gov.in",
        "source": "seed",
    },
    {
        "name": "Pradhan Mantri Suraksha Bima Yojana (PMSBY)",
        "category": "Insurance",
        "eligibility": "Any individual aged 18-70 with a bank account, willing to pay a small "
                        "annual premium.",
        "benefits": "Accidental death and full disability cover of Rs 2 lakh, and Rs 1 lakh "
                     "for partial disability, for an annual premium of around Rs 20.",
        "how_to_apply": "Ask your bank to auto-debit the yearly premium and enroll you, or "
                         "apply through internet/mobile banking.",
        "official_link": "https://jansuraksha.gov.in",
        "source": "seed",
    },
    {
        "name": "Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY)",
        "category": "Insurance",
        "eligibility": "Any individual aged 18-50 with a bank account.",
        "benefits": "Life insurance cover of Rs 2 lakh in case of death of the insured, for a "
                     "low annual premium (around Rs 436, revised periodically).",
        "how_to_apply": "Enroll through your bank branch or net/mobile banking; premium is "
                         "auto-debited yearly.",
        "official_link": "https://jansuraksha.gov.in",
        "source": "seed",
    },
    {
        "name": "Employees' State Insurance (ESI)",
        "category": "Healthcare / Labor Right",
        "eligibility": "Employees earning up to a notified wage ceiling, working in factories "
                        "or establishments covered under the ESI Act with 10+ employees.",
        "benefits": "Free medical treatment for employee and family, sickness benefit, "
                     "maternity benefit, and disablement benefit.",
        "how_to_apply": "Your employer is legally required to register you and deduct a small "
                         "contribution from wages; check status at esic.gov.in.",
        "official_link": "https://esic.gov.in",
        "source": "seed",
    },
    {
        "name": "Employees' Provident Fund (EPF)",
        "category": "Pension / Labor Right",
        "eligibility": "Employees in establishments with 20+ workers; wage ceiling and other "
                        "conditions apply.",
        "benefits": "Retirement savings with employer + employee contribution, withdrawal "
                     "allowed for emergencies, plus linked pension (EPS) and insurance (EDLI).",
        "how_to_apply": "Employer registers you automatically; check your balance via the "
                         "UMANG app or epfindia.gov.in using your UAN.",
        "official_link": "https://www.epfindia.gov.in",
        "source": "seed",
    },
    {
        "name": "Building and Other Construction Workers (BOCW) Welfare Fund",
        "category": "Construction Worker Welfare",
        "eligibility": "Construction workers aged 18-60 who have worked at least 90 days in "
                        "the last year, registered with the state BOCW Welfare Board.",
        "benefits": "Financial assistance for accidents, maternity, education of children, "
                     "housing, pension, and death/disability benefits — rules vary by state.",
        "how_to_apply": "Register at the nearest Labour Department office or CSC with proof "
                         "of construction work (site certificate) and Aadhaar.",
        "official_link": "https://labour.gov.in",
        "source": "seed",
    },
    {
        "name": "One Nation One Ration Card (ONORC)",
        "category": "Food Security",
        "eligibility": "Any ration card holder under the National Food Security Act (NFSA), "
                        "including migrant workers who move between states.",
        "benefits": "Allows access to subsidised food grains from any Fair Price Shop in India, "
                     "not just the state where the ration card was issued.",
        "how_to_apply": "No separate application needed if you already hold an NFSA ration "
                         "card — just use it via biometric authentication at any FPS.",
        "official_link": "https://nfsa.gov.in",
        "source": "seed",
    },
    {
        "name": "Pradhan Mantri Awas Yojana (PMAY)",
        "category": "Housing",
        "eligibility": "Economically weaker sections, low and middle income households without "
                        "a pucca house, based on income category (EWS/LIG/MIG).",
        "benefits": "Financial assistance or interest subsidy on home loans to build or buy a "
                     "house, including a specific track for urban migrant/rental workers.",
        "how_to_apply": "Apply online at pmaymis.gov.in or through the nearest Urban Local "
                         "Body / Common Service Centre.",
        "official_link": "https://pmaymis.gov.in",
        "source": "seed",
    },
    {
        "name": "Minimum Wages Act Protection",
        "category": "Labor Right",
        "eligibility": "All workers, including migrant and informal workers, in scheduled "
                        "employments across every Indian state.",
        "benefits": "Legal right to be paid at least the state-notified minimum wage for your "
                     "category of work, regardless of which state you are working in.",
        "how_to_apply": "Not an application — it is a legal right. If underpaid, file a "
                         "complaint with the local Labour Commissioner's office or call the "
                         "state labour helpline.",
        "official_link": "https://labour.gov.in",
        "source": "seed",
    },
]


# ---------------------------------------------------------------------------
# 4. LOADER — local CSV first, seed data as automatic fallback
# ---------------------------------------------------------------------------

# Possible column names the CSV might use for each field we need.
# We check each list in order (case-insensitive) and use the first match.
FIELD_ALIASES = {
    "name": ["name of the scheme", "scheme_name", "name", "title"],
    "category": ["category", "schemecategory", "sector", "ministry", "department", "scheme_category"],
    "eligibility": ["eligibility", "eligibility_criteria", "who_can_apply"],
    "benefits": ["benefits", "benefit", "scheme_details", "details", "description"],
    "how_to_apply": ["how_to_apply", "application_process", "application", "apply", "how_to_avail"],
    "official_link": ["official_link", "link", "url", "scheme_url", "website"],
    "documents_required": ["documents", "documents_required", "required_documents"],
}


def _find_column(df_columns_lower_map, aliases):
    """df_columns_lower_map: dict of {lowercase_col_name: actual_col_name}"""
    for alias in aliases:
        if alias in df_columns_lower_map:
            return df_columns_lower_map[alias]
    return None


def load_from_csv():
    """Returns a list of mapped scheme dicts, or None if it fails for any reason."""
    if not os.path.exists(CSV_PATH):
        print(f"[data] CSV not found at '{CSV_PATH}'. Place the downloaded file "
              f"in this folder, or update CSV_PATH at the top of data.py.")
        return None

    try:
        import pandas as pd
    except ImportError:
        print("[data] `pandas` not installed. Run: pip install pandas --break-system-packages")
        return None

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[data] Failed to read CSV: {e}")
        return None

    print(f"[data] Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")

    col_map = {c.strip().lower(): c for c in df.columns}
    resolved = {field: _find_column(col_map, aliases) for field, aliases in FIELD_ALIASES.items()}
    print(f"[data] Resolved column mapping: {resolved}")

    if not resolved["name"]:
        print("[data] Could not find a 'name' column in the CSV — check FIELD_ALIASES "
              "in data.py against the printed column list above.")
        return None

    has_slug_col = "slug" in col_map
    slug_col = col_map.get("slug")

    rows = []
    for _, row in df.iterrows():
        name = row.get(resolved["name"])
        if pd.isna(name) or not str(name).strip():
            continue

        def get_field(key):
            col = resolved[key]
            if col is None:
                return None
            val = row.get(col)
            return None if pd.isna(val) else str(val).strip()

        official_link = get_field("official_link")
        if not official_link and has_slug_col:
            slug_val = row.get(slug_col)
            if not pd.isna(slug_val) and str(slug_val).strip():
                official_link = f"https://www.myscheme.gov.in/schemes/{str(slug_val).strip()}"

        rows.append({
            "name": str(name).strip(),
            "category": get_field("category"),
            "eligibility": get_field("eligibility"),
            "benefits": get_field("benefits"),
            "how_to_apply": get_field("how_to_apply"),
            "official_link": official_link,
            "documents_required": get_field("documents_required"),
            "source": "kaggle_csv",
        })

    if not rows:
        print("[data] No usable rows found in CSV.")
        return None

    print(f"[data] Successfully mapped {len(rows)} schemes from CSV.")
    return rows


def populate_database(reset: bool = False):
    if reset:
        Base.metadata.drop_all(bind=engine, tables=[Scheme.__table__])
        Base.metadata.create_all(bind=engine)
        print("[data] Database schema reset (--reset). Starting fresh.")
    else:
        init_db()

    db = SessionLocal()
    try:
        existing_count = db.query(Scheme).count()
        if existing_count > 0 and not reset:
            print(f"[data] Database already has {existing_count} schemes. "
                  f"Skipping load. Run with --reset to reload.")
            return

        rows = load_from_csv()
        if rows is None:
            print("[data] Using built-in seed dataset instead.")
            rows = SEED_SCHEMES

        for r in rows:
            db.add(Scheme(
                name=r.get("name") or "Unnamed scheme",
                category=r.get("category"),
                eligibility=r.get("eligibility"),
                benefits=r.get("benefits"),
                how_to_apply=r.get("how_to_apply"),
                official_link=r.get("official_link"),
                documents_required=r.get("documents_required"),
                source=r.get("source", "unknown"),
            ))

        db.commit()
        total = db.query(Scheme).count()
        print(f"[data] Done. {total} schemes now in the database.")

    finally:
        db.close()


if __name__ == "__main__":
    populate_database(reset="--reset" in sys.argv)