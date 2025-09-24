#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main index builder script for AI-syd-Gold.
Wrapper script to build various types of vector stores.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_indexer(script_name, description=""):
    """Run an indexer script from the index/ directory."""
    index_dir = Path(__file__).parent / "index"
    script_path = index_dir / script_name

    if not script_path.exists():
        print(f"❌ Indexer script not found: {script_name}")
        return 1

    if description:
        print(f"\n🔧 {description}")
        print("=" * 60)

    try:
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(Path(__file__).parent))
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Indexing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running indexer: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="AI-syd-Gold Index Builder")
    parser.add_argument("--quick", action="store_true", help="Quick vector store build")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid multimodal build (recommended)")
    parser.add_argument("--simple", action="store_true", help="Simple vector store build")
    parser.add_argument("--fixed", action="store_true", help="Fixed parent-document build")
    parser.add_argument("--all", action="store_true", help="Build all index types")

    args = parser.parse_args()

    if not any(vars(args).values()):
        # Default to hybrid build
        return run_indexer("build_hybrid_store.py", "Building Hybrid Multimodal Vector Store (Default)")

    exit_code = 0

    if args.quick or args.all:
        exit_code = max(exit_code, run_indexer("quick_vector_build.py", "Building Quick Vector Store"))

    if args.hybrid or args.all:
        exit_code = max(exit_code, run_indexer("build_hybrid_store.py", "Building Hybrid Multimodal Vector Store"))

    if args.simple or args.all:
        exit_code = max(exit_code, run_indexer("simple_vector_build.py", "Building Simple Vector Store"))

    if args.fixed or args.all:
        exit_code = max(exit_code, run_indexer("fixed_vector_build.py", "Building Fixed Parent-Document Vector Store"))

    return exit_code

if __name__ == "__main__":
    sys.exit(main())