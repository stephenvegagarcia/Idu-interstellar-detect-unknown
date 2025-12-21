#!/usr/bin/env python
"""
Simple startup script for the IDU Flask application
"""
import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        print("✓ Flask is installed")
    except ImportError:
        print("✗ Flask is not installed")
        print("  Run: pip install -r requirements.txt")
        return False
    return True

def main():
    import sys
    
    print("=" * 60)
    print("IDU - Interstellar Detect Unknown")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Check if debug mode is requested
    debug_mode = '--debug' in sys.argv
    
    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    if debug_mode:
        print("⚠️  Running in DEBUG mode - not for production use!")
    print("Press CTRL+C to stop the server\n")
    
    # Import and run the Flask app
    from app import app
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
