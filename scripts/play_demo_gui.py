"""
GUI demo launcher - convenient wrapper for gui_train_game.py

Usage:
    python play_demo_gui.py
    
This launches the GUI with the Agent Play feature for visual demonstrations.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to access gui_train_game.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Change to parent directory so relative paths work
os.chdir(parent_dir)

# Import and run GUI
try:
    import tkinter as tk
    from gui_train_game import TrainGameApp
    
    if __name__ == '__main__':
        print("🚆 Launching Train Game GUI...")
        print("   Click 'Agent Play' to watch trained agents")
        print("   Click 'Play' for manual control")
        print()
        
        root = tk.Tk()
        app = TrainGameApp(root)
        root.mainloop()
        
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nMake sure you have tkinter installed:")
    print("  - On macOS: Should be pre-installed with Python")
    print("  - On Ubuntu: sudo apt-get install python3-tk")
    print("  - On Windows: Should be included with Python")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error launching GUI: {e}")
    sys.exit(1)
