#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per eseguire i test del progetto AI-syd-Gold.
Fornisce diverse opzioni per eseguire test specifici o completi.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Esegue un comando e gestisce il risultato."""
    if description:
        print(f"\n{'='*60}")
        print(f"🔧 {description}")
        print(f"{'='*60}")

    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ Comando fallito con codice: {result.returncode}")
        return False
    else:
        print("✅ Comando completato con successo")
        return True


def check_dependencies():
    """Verifica che le dipendenze di test siano installate."""
    try:
        import pytest
        print("✅ pytest installato")
    except ImportError:
        print("❌ pytest non installato. Installa con: pip install pytest")
        return False

    return True


def run_unit_tests(verbose=False, coverage=False):
    """Esegue i test unitari."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])

    cmd.extend(["-m", "unit"])

    return run_command(cmd, "Esecuzione Test Unitari")


def run_all_tests(verbose=False, coverage=False):
    """Esegue tutti i test."""
    cmd = ["python", "-m", "pytest", "tests/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])

    return run_command(cmd, "Esecuzione Tutti i Test")


def run_linting():
    """Esegue il linting del codice."""
    success = True

    # Black (formattazione)
    print("\n🔍 Controllo formattazione con Black...")
    black_cmd = ["python", "-m", "black", "--check", "--diff", "."]
    if not run_command(black_cmd, "Controllo Formattazione (Black)"):
        print("💡 Per correggere automaticamente: python -m black .")
        success = False

    # Flake8 (linting)
    print("\n🔍 Controllo linting con Flake8...")
    flake8_cmd = ["python", "-m", "flake8", "app.py", "streamlit_app.py", "build_cache.py"]
    if not run_command(flake8_cmd, "Linting (Flake8)"):
        success = False

    return success


def run_type_checking():
    """Esegue il controllo dei tipi."""
    mypy_cmd = ["python", "-m", "mypy", "app.py", "--ignore-missing-imports"]
    return run_command(mypy_cmd, "Controllo Tipi (MyPy)")


def run_security_check():
    """Esegue controlli di sicurezza."""
    try:
        # Bandit per security
        bandit_cmd = ["python", "-m", "bandit", "-r", ".", "-x", "tests,venv,env"]
        return run_command(bandit_cmd, "Controllo Sicurezza (Bandit)")
    except FileNotFoundError:
        print("⚠️  Bandit non installato. Installa con: pip install bandit")
        return True  # Non fallire se bandit non è installato


def generate_coverage_report():
    """Genera report di coverage dettagliato."""
    print("\n📊 Generazione Report Coverage...")

    # Esegue test con coverage
    cmd = ["python", "-m", "pytest", "tests/", "--cov=app", "--cov-report=html", "--cov-report=term"]
    if run_command(cmd, "Generazione Coverage Report"):
        print("\n✅ Report coverage generato in htmlcov/index.html")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Test runner per AI-syd-Gold")

    parser.add_argument("--unit", action="store_true", help="Esegue solo test unitari")
    parser.add_argument("--all", action="store_true", help="Esegue tutti i test")
    parser.add_argument("--lint", action="store_true", help="Esegue linting")
    parser.add_argument("--type-check", action="store_true", help="Esegue controllo tipi")
    parser.add_argument("--security", action="store_true", help="Esegue controlli sicurezza")
    parser.add_argument("--coverage", action="store_true", help="Genera report coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output verboso")
    parser.add_argument("--quick", action="store_true", help="Quick check (unit tests + linting)")
    parser.add_argument("--full", action="store_true", help="Full check (tutti i controlli)")

    args = parser.parse_args()

    # Verifica che siamo nella directory giusta
    if not os.path.exists("app.py"):
        print("❌ Esegui lo script dalla directory root del progetto")
        sys.exit(1)

    # Verifica dipendenze
    if not check_dependencies():
        sys.exit(1)

    print("🚀 AI-syd-Gold Test Runner")
    print(f"📂 Directory: {os.getcwd()}")

    success = True

    # Quick check
    if args.quick:
        print("\n🏃‍♂️ Quick Check Mode")
        success &= run_unit_tests(verbose=args.verbose)
        success &= run_linting()

    # Full check
    elif args.full:
        print("\n🔍 Full Check Mode")
        success &= run_all_tests(verbose=args.verbose, coverage=True)
        success &= run_linting()
        success &= run_type_checking()
        success &= run_security_check()

    # Controlli specifici
    else:
        if args.unit:
            success &= run_unit_tests(verbose=args.verbose, coverage=args.coverage)

        if args.all:
            success &= run_all_tests(verbose=args.verbose, coverage=args.coverage)

        if args.lint:
            success &= run_linting()

        if args.type_check:
            success &= run_type_checking()

        if args.security:
            success &= run_security_check()

        if args.coverage:
            success &= generate_coverage_report()

        # Se nessuna opzione specifica, esegui quick check
        if not any([args.unit, args.all, args.lint, args.type_check, args.security, args.coverage]):
            print("\n🏃‍♂️ Nessuna opzione specificata, eseguo Quick Check")
            success &= run_unit_tests(verbose=args.verbose)
            success &= run_linting()

    # Risultato finale
    print(f"\n{'='*60}")
    if success:
        print("🎉 Tutti i controlli completati con successo!")
        sys.exit(0)
    else:
        print("❌ Alcuni controlli sono falliti")
        sys.exit(1)


if __name__ == "__main__":
    main()