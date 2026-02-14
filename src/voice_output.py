"""
Voice Output Module for Text-to-Speech

This module handles:
- Text-to-speech conversion
- Voice settings and control
- Optional voice feedback
"""

import pyttsx3
import threading
import queue
import time
from typing import Optional

class VoiceOutput:
    """
    Voice output system using text-to-speech
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize voice output
        
        Args:
            enabled: Whether voice output is enabled by default
        """
        self.enabled = True
        self.engine = None
        self.voice_queue = queue.Queue()
        self.is_speaking = False
        
        # Voice settings
        self.rate = 150  # Words per minute
        self.volume = 0.9
        
        # Initialize TTS engine
        self._init_tts_engine()
        
        # Start worker thread for speech
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
    
    def _init_tts_engine(self):
        """Initialize the text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set voice properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Try to set a female voice (often clearer for assistive applications)
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            print("🔊 Voice output initialized")
            
        except Exception as e:
            print(f"⚠️ Voice output initialization failed: {e}")
            self.enabled = False
    
    def speak(self, text: str, interrupt: bool = True):
        """
        Speak the given text
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        if not self.enabled or not self.engine or not text.strip():
            return
        
        if interrupt and self.is_speaking:
            self.engine.stop()
        
        # Add to queue
        self.voice_queue.put(text)
    
    def speak_character(self, char: str):
        """
        Speak a single character with clear pronunciation
        
        Args:
            char: Character to speak
        """
        if not self.enabled or not char:
            return
        
        # Convert character to spoken form
        spoken_text = self._char_to_spoken_form(char)
        self.speak(spoken_text)
    
    def _char_to_spoken_form(self, char: str) -> str:
        """
        Convert character to its spoken form
        
        Args:
            char: Input character
            
        Returns:
            str: Spoken form of character
        """
        char = char.lower()
        
        # Special cases
        if char == ' ':
            return "space"
        elif char == '.':
            return "period"
        elif char == ',':
            return "comma"
        elif char == '!':
            return "exclamation mark"
        elif char == '?':
            return "question mark"
        elif char.isdigit():
            return self._number_to_words(char)
        else:
            return char.upper()  # Spell out letters
    
    def _number_to_words(self, number: str) -> str:
        """
        Convert number to words
        
        Args:
            number: Number as string
            
        Returns:
            str: Number in words
        """
        num_to_words = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }
        
        return ' '.join(num_to_words.get(d, d) for d in number)
    
    def _speech_worker(self):
        """Worker thread for handling speech synthesis"""
        while True:
            try:
                # Get text from queue
                text = self.voice_queue.get(timeout=0.1)
                
                if text and self.engine:
                    self.is_speaking = True
                    
                    # Speak the text
                    self.engine.say(text)
                    self.engine.runAndWait()
                    
                    self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Speech error: {e}")
                self.is_speaking = False
    
    def toggle_enabled(self):
        """Toggle voice output on/off"""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        print(f"🔊 Voice output {status}")
        
        if self.enabled:
            self.speak("Voice output enabled")
    
    def set_rate(self, rate: int):
        """
        Set speech rate
        
        Args:
            rate: Words per minute (50-400)
        """
        if self.engine:
            self.rate = max(50, min(400, rate))
            self.engine.setProperty('rate', self.rate)
    
    def set_volume(self, volume: float):
        """
        Set speech volume
        
        Args:
            volume: Volume level (0.0-1.0)
        """
        if self.engine:
            self.volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', self.volume)
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            return [(voice.id, voice.name) for voice in voices]
        return []
    
    def set_voice(self, voice_id: str):
        """
        Set the voice to use
        
        Args:
            voice_id: ID of the voice to use
        """
        if self.engine:
            self.engine.setProperty('voice', voice_id)
    
    def stop_speech(self):
        """Stop current speech"""
        if self.engine and self.is_speaking:
            self.engine.stop()
            self.is_speaking = False
            
            # Clear queue
            while not self.voice_queue.empty():
                try:
                    self.voice_queue.get_nowait()
                except queue.Empty:
                    break
    
    def clear_queue(self):
        """Clear the speech queue"""
        while not self.voice_queue.empty():
            try:
                self.voice_queue.get_nowait()
            except queue.Empty:
                break
    
    def is_voice_available(self) -> bool:
        """Check if voice output is available"""
        return self.engine is not None
    
    def get_status(self) -> dict:
        """Get current voice status"""
        return {
            'enabled': self.enabled,
            'available': self.is_voice_available(),
            'speaking': self.is_speaking,
            'rate': self.rate,
            'volume': self.volume,
            'queue_size': self.voice_queue.qsize()
        }
    
    def test_speech(self):
        """Test the speech system"""
        if self.enabled and self.engine:
            test_text = "Voice output test. Hello, this is the gesture to text system."
            self.speak(test_text)
        else:
            print("⚠️ Voice output is not enabled or available")
