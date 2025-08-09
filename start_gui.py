#!/usr/bin/env python3
"""
Startup script for the Enhanced Dashed Line Detection GUI

This script provides a safe way to start the GUI with proper error handling.
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import streamlit_drawable_canvas
        print("✅ All GUI requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please run: pip install streamlit-drawable-canvas")
        return False

def start_gui():
    """Start the Streamlit GUI."""
    if not check_requirements():
        return False
    
    print("🚀 Starting Enhanced Dashed Line Detection GUI...")
    print("🌐 The GUI will open in your web browser")
    print("📝 Features:")
    print("   • Interactive ROI selection with rectangle drawing")
    print("   • Interactive scale setting with line drawing") 
    print("   • Complete detection pipeline")
    print("   • Multiple export formats")
    print("")
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dash_detection_gui.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 GUI stopped by user")
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_gui()
