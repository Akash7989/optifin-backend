from pathlib import Path

from dotenv import load_dotenv

# Load optifin-be/.env before any module reads settings. Variables already set in the
# environment take precedence over the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
