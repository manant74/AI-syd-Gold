#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script wrapper per eseguire i test dalla root del progetto.
Delega l'esecuzione al test runner in tests/run_tests.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Esegue il test runner dalla cartella tests/"""
    tests_dir = Path(__file__).parent / "tests"
    runner_script = tests_dir / "run_tests.py"

    if not runner_script.exists():
        print("❌ Test runner non trovato in tests/run_tests.py")
        return 1

    # Passa tutti gli argomenti al test runner
    cmd = [sys.executable, str(runner_script)] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, cwd=str(tests_dir.parent))
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Test interrotti dall'utente")
        return 1
    except Exception as e:
        print(f"❌ Errore nell'esecuzione dei test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())