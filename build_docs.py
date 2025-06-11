#!/usr/bin/env python3
"""
Script to build Sphinx documentation for einit
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    """Build the Sphinx documentation"""
    
    # Get project root directory
    project_root = Path(__file__).parent
    docs_dir = project_root / "einit_docs"
    build_dir = docs_dir / "_build"
    
    print("Building einit documentation...")
    print(f"Project root: {project_root}")
    print(f"Docs directory: {docs_dir}")
    
    # Check if docs directory exists
    if not docs_dir.exists():
        print(f"Error: Documentation directory {docs_dir} not found!")
        return 1
    
    # Clean previous builds
    if build_dir.exists():
        print("Cleaning previous build...")
        shutil.rmtree(build_dir)
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    try:
        # Build HTML documentation
        print("Building HTML documentation...")
        result = subprocess.run([
            "sphinx-build", 
            "-b", "html",
            ".", 
            "_build/html"
        ], capture_output=True, text=True, check=True)
        
        print("✓ HTML documentation built successfully!")
        print(f"Output: {docs_dir / '_build' / 'html' / 'index.html'}")
        
        # Optionally build PDF if LaTeX is available
        try:
            print("Building PDF documentation...")
            subprocess.run([
                "sphinx-build",
                "-b", "latex",
                ".",
                "_build/latex"
            ], capture_output=True, text=True, check=True)
            
            # Run pdflatex if available
            latex_dir = docs_dir / "_build" / "latex"
            if (latex_dir / "einit.tex").exists():
                os.chdir(latex_dir)
                subprocess.run(["pdflatex", "einit.tex"], 
                             capture_output=True, text=True)
                subprocess.run(["pdflatex", "einit.tex"], 
                             capture_output=True, text=True)  # Run twice for references
                
                if (latex_dir / "einit.pdf").exists():
                    print("✓ PDF documentation built successfully!")
                    print(f"Output: {latex_dir / 'einit.pdf'}")
        
        except subprocess.CalledProcessError:
            print("⚠ PDF build failed (LaTeX not available)")
        except FileNotFoundError:
            print("⚠ PDF build skipped (LaTeX not found)")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Documentation build failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return 1
    
    except FileNotFoundError:
        print("✗ Sphinx not found! Install with: pip install sphinx sphinx_rtd_theme")
        return 1

if __name__ == "__main__":
    sys.exit(main())