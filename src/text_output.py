"""
Text Output Module for Displaying Recognized Text

This module handles:
- Real-time text display
- Text history management
- Export functionality
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import threading
import time
from datetime import datetime
from typing import List, Optional

class TextOutput:
    """
    Text output display and management
    """
    
    def __init__(self):
        """Initialize the text output system"""
        self.current_text = ""
        self.text_history = []
        self.max_history = 100
        
        # GUI components (initialized later)
        self.root = None
        self.text_display = None
        self.history_display = None
        self.status_label = None
        
        # Start GUI in separate thread
        self.gui_thread = threading.Thread(target=self._init_gui, daemon=True)
        self.gui_thread.start()
        
        # Wait for GUI to initialize
        time.sleep(0.5)
    
    def _init_gui(self):
        """Initialize the GUI window"""
        self.root = tk.Tk()
        self.root.title("Gesture-to-Text Output")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(
            self.root,
            text="🧠 Brain-Inspired Gesture Recognition",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        title_label.pack(pady=10)
        
        # Current text display
        current_frame = tk.Frame(self.root, bg='#2b2b2b')
        current_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            current_frame,
            text="Current Text:",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='white'
        ).pack(anchor='w')
        
        self.text_display = tk.Text(
            current_frame,
            height=3,
            font=('Arial', 14),
            bg='#1e1e1e',
            fg='white',
            insertbackground='white'
        )
        self.text_display.pack(fill='x', pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            bg='#ff4444',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="Export",
            command=self.export_text,
            bg='#4444ff',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="Copy",
            command=self.copy_to_clipboard,
            bg='#44ff44',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20
        ).pack(side='left', padx=5)
        
        # History display
        history_frame = tk.Frame(self.root, bg='#2b2b2b')
        history_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(
            history_frame,
            text="History:",
            font=('Arial', 12, 'bold'),
            bg='#2b2b2b',
            fg='white'
        ).pack(anchor='w')
        
        self.history_display = scrolledtext.ScrolledText(
            history_frame,
            height=10,
            font=('Arial', 10),
            bg='#1e1e1e',
            fg='white',
            state='disabled'
        )
        self.history_display.pack(fill='both', expand=True, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        self.status_label.pack(pady=5)
        
        # Start GUI event loop
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def update_text(self, text: str):
        """
        Update the current text display
        
        Args:
            text: New text to display
        """
        self.current_text = text
        
        if self.text_display:
            # Update in GUI thread
            self.root.after(0, self._update_text_display, text)
        
        # Add to history
        self._add_to_history(text)
    
    def _update_text_display(self, text: str):
        """Update text display (must be called in GUI thread)"""
        if self.text_display:
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, text)
            
            # Update status
            char_count = len(text)
            self.status_label.config(text=f"Characters: {char_count}")
    
    def _add_to_history(self, text: str):
        """Add text to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {text}"
        
        self.text_history.append(entry)
        
        # Limit history size
        if len(self.text_history) > self.max_history:
            self.text_history.pop(0)
        
        # Update history display
        if self.history_display:
            self.root.after(0, self._update_history_display)
    
    def _update_history_display(self):
        """Update history display (must be called in GUI thread)"""
        if self.history_display:
            self.history_display.config(state='normal')
            self.history_display.delete(1.0, tk.END)
            
            for entry in self.text_history:
                self.history_display.insert(tk.END, entry + '\n')
            
            self.history_display.config(state='disabled')
            # Scroll to bottom
            self.history_display.see(tk.END)
    
    def clear(self):
        """Clear the current text"""
        self.current_text = ""
        if self.text_display:
            self.root.after(0, self._update_text_display, "")
        
        if self.status_label:
            self.root.after(0, lambda: self.status_label.config(text="Cleared"))
    
    def export_text(self):
        """Export current text to file"""
        if not self.current_text:
            messagebox.showwarning("Export", "No text to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"gesture_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Gesture-to-Text Output\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.current_text)
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("History:\n")
                    for entry in self.text_history:
                        f.write(f"{entry}\n")
                
                messagebox.showinfo("Export", f"Text exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def copy_to_clipboard(self):
        """Copy current text to clipboard"""
        if not self.current_text:
            messagebox.showwarning("Copy", "No text to copy")
            return
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_text)
            messagebox.showinfo("Copy", "Text copied to clipboard")
            
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy: {str(e)}")
    
    def get_current_text(self) -> str:
        """Get the current text"""
        return self.current_text
    
    def get_history(self) -> List[str]:
        """Get the text history"""
        return self.text_history.copy()
    
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the text output window?"):
            self.root.destroy()
    
    def is_gui_ready(self) -> bool:
        """Check if GUI is ready"""
        return self.root is not None and self.text_display is not None
