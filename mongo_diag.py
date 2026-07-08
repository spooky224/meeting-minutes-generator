"""
mongo_diag.py — run this directly to isolate the SSL handshake issue from the app.

    python3 mongo_diag.py

It prints your Python/OpenSSL/pymongo versions, then tries three connection
strategies in order so you can see exactly which one (if any) works.
"""
import os
import ssl
import sys

from dotenv import load_dotenv
load_dotenv()

print("── Environment ─────────────────────────────")
print("Python:      ", sys.version.replace("\n", " "))
print("SSL backend: ", ssl.OPENSSL_VERSION)

try:
    import pymongo
    print("pymongo:     ", pymongo.version)
except ImportError:
    print("pymongo:     NOT INSTALLED")
    sys.exit(1)

try:
    import certifi
    print("certifi:     ", certifi.__version__, "->", certifi.where())
except ImportError:
    print("certifi:     NOT INSTALLED  (pip install certifi)")
    certifi = None

uri = os.getenv("MONGO_URI")
if not uri:
    print("\nMONGO_URI is not set in your environment/.env — nothing to test.")
    sys.exit(1)

print(f"\nURI host looks like: {uri.split('@')[-1].split('/')[0] if '@' in uri else uri}")

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def try_connect(label, **kwargs):
    print(f"\n── Attempt: {label} ─────────────────────────────")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, **kwargs)
        client.admin.command("ping")
        print("✅ SUCCESS — this configuration works.")
        return True
    except PyMongoError as e:
        print(f"❌ FAILED: {str(e)[:300]}")
        return False


# 1. Exactly what db.py currently does
ok = try_connect("default (matches current db.py)")

# 2. Force certifi's CA bundle — the most common fix for this exact error
if not ok and certifi:
    ok = try_connect("tlsCAFile=certifi.where()", tlsCAFile=certifi.where())

# 3. Last resort diagnostic ONLY — proves it's a cert/trust issue, not a real fix.
#    Never leave this on in the actual app.
if not ok:
    print("\n(Skipping insecure tlsAllowInvalidCertificates test — see notes below instead.)")

print("\n── Next steps ─────────────────────────────")
if ok:
    print("If attempt 2 worked but attempt 1 didn't: the fix is to pass")
    print("tlsCAFile=certifi.where() in db.py's MongoClient(...) call.")
else:
    print("Both failed — check Atlas Network Access (IP allowlist) and that")
    print("your cluster isn't paused, then re-run this script.")