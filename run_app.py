#!/usr/bin/env python3
"""
Point d'entrée pour lancer l'application Streamlit.

Usage:
    python run_app.py
    
Ou directement:
    streamlit run app/frontend/main_app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Lance l'application Streamlit."""
    app_path = Path(__file__).parent / "app" / "frontend" / "main_app.py"
    
    if not app_path.exists():
        print(f"Erreur: {app_path} non trouvé")
        sys.exit(1)
    
    print("Lancement de Moulinette Simulator...")
    print(f"   App: {app_path}")
    print("   Ouvrez http://localhost:8501 dans votre navigateur")
    print("-" * 50)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
