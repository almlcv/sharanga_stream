import os
from pymongo import MongoClient

# MongoDB Configuration
MONGO_URI = os.getenv(
    'MONGO_URI',
    ''
)

DB_NAME = "Sharanga_Production"

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    print("[✓] MongoDB connected successfully")
except Exception as e:
    print(f"[!] MongoDB connection failed: {e}")
    db = None
