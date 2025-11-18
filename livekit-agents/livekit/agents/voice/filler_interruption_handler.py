"""
Filler Word Interruption Handler for LiveKit Agents

This module provides intelligent interruption handling that distinguishes
between filler words (like "uh", "umm", "hmm", "haan") and genuine user
interruptions. It only filters filler words when the agent is currently
speaking, ensuring natural conversation flow.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Callable

from livekit.agents import utils
from livekit.agents.stt import SpeechEvent
from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice.events import UserInputTranscribedEvent


class InterruptionType(str, Enum):
    """Types of interruptions detected by the filler handler."""
    FILLER_ONLY = "filler_only"
    GENUINE = "genuine"
    MIXED = "mixed"  # Contains both filler and meaningful content


@dataclass
class FillerInterruptionEvent:
    """Event emitted when filler word interruption is detected."""
    type: InterruptionType
    transcript: str
    filler_words_detected: List[str]
    agent_was_speaking: bool
    timestamp: float
    should_interrupt: bool


class FillerInterruptionHandler:
    """
    Handles intelligent interruption filtering based on filler words.
    
    This handler monitors transcription events and VAD events to determine
    whether user speech should cause an interruption. When the agent is
    speaking, it filters out filler-only interruptions while allowing
    genuine interruptions to pass through.
    """
    
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        min_speech_duration: float = 0.3,
        debug_mode: bool = False
    ):
        """
        Initialize the filler interruption handler.
        
        Args:
            ignored_words: List of filler words to ignore (default: common fillers)
            confidence_threshold: Minimum ASR confidence to consider speech valid
            min_speech_duration: Minimum duration of speech to process (seconds)
            debug_mode: Enable detailed logging for debugging
        """
        self.ignored_words = set(word.lower().strip() for word in (ignored_words or [
            "uh", "umm", "um", "hmm", "haan", "ah", "er", "like", "you know", 
            "i mean", "well", "so", "basically", "actually", "right"
        ]))
        
        self.confidence_threshold = confidence_threshold
        self.min_speech_duration = min_speech_duration
        self.debug_mode = debug_mode
        
        # State tracking
        self._agent_speaking = False
        self._current_transcript = ""
        self._transcript_start_time: Optional[float] = None
        self._last_interruption_time = 0.0
        self._interruption_cooldown = 0.5  # Prevent rapid-fire interruptions
        
        # Event callbacks
        self._on_interruption_detected: Optional[Callable[[FillerInterruptionEvent], None]] = None
        self._on_agent_speaking_changed: Optional[Callable[[bool], None]] = None
        
        # Pattern matching for filler words
        self._filler_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) 
            for word in self.ignored_words
        ]
        
        if self.debug_mode:
            print(f"[FillerHandler] Initialized with ignored words: {self.ignored_words}")
    
    def set_interruption_callback(
        self, 
        callback: Callable[[FillerInterruptionEvent], None]
    ) -> None:
        """Set callback for interruption events."""
        self._on_interruption_detected = callback
    
    def set_agent_speaking_callback(
        self, 
        callback: Callable[[bool], None]
    ) -> None:
        """Set callback for agent speaking state changes."""
        self._on_agent_speaking_changed = callback
    
    def update_agent_speaking_state(self, is_speaking: bool) -> None:
        """
        Update the agent's speaking state.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        if self._agent_speaking != is_speaking:
            self._agent_speaking = is_speaking
            if self.debug_mode:
                print(f"[FillerHandler] Agent speaking state changed: {is_speaking}")
            
            if self._on_agent_speaking_changed:
                self._on_agent_speaking_changed(is_speaking)
    
    def on_vad_event(self, event: VADEvent) -> None:
        """
        Handle VAD events to track speech activity.
        
        Args:
            event: VAD event from the voice activity detector
        """
        if event.type == VADEventType.START_OF_SPEECH:
            self._transcript_start_time = time.time()
            self._current_transcript = ""
            
        elif event.type == VADEventType.END_OF_SPEECH:
            # Process the collected transcript when speech ends
            if self._transcript_start_time:
                duration = time.time() - self._transcript_start_time
                if duration >= self.min_speech_duration:
                    self._process_transcript(self._current_transcript)
            
            self._transcript_start_time = None
            self._current_transcript = ""
    
    def on_transcript_event(self, event: UserInputTranscribedEvent) -> None:
        """
        Handle transcription events to collect user speech.
        
        Args:
            event: User input transcription event
        """
        if not self._transcript_start_time:
            return  # Ignore transcripts not associated with speech activity
            
        # Collect transcript content
        if event.is_final:
            self._current_transcript = event.transcript.strip()
            
            # Check confidence threshold if available
            if hasattr(event, 'confidence') and event.confidence < self.confidence_threshold:
                if self.debug_mode:
                    print(f"[FillerHandler] Low confidence transcript ignored: {event.confidence}")
                return
            
            # Process immediately for final transcripts
            self._process_transcript(self._current_transcript)
    
    def _process_transcript(self, transcript: str) -> None:
        """
        Process a transcript to determine if it should cause an interruption.
        
        Args:
            transcript: The user's speech transcript
        """
        if not transcript.strip():
            return
            
        current_time = time.time()
        if current_time - self._last_interruption_time < self._interruption_cooldown:
            if self.debug_mode:
                print(f"[FillerHandler] Ignoring transcript due to cooldown: {transcript}")
            return
        
        # Detect filler words in transcript
        detected_fillers = self._detect_filler_words(transcript)
        interruption_type = self._classify_interruption(transcript, detected_fillers)
        
        # Determine if interruption should be allowed
        should_interrupt = self._should_allow_interruption(interruption_type, transcript)
        
        # Create and emit event
        event = FillerInterruptionEvent(
            type=interruption_type,
            transcript=transcript,
            filler_words_detected=detected_fillers,
            agent_was_speaking=self._agent_speaking,
            timestamp=current_time,
            should_interrupt=should_interrupt
        )
        
        if self.debug_mode:
            print(f"[FillerHandler] Processed: {transcript}")
            print(f"  - Type: {interruption_type}")
            print(f"  - Fillers: {detected_fillers}")
            print(f"  - Agent speaking: {self._agent_speaking}")
            print(f"  - Should interrupt: {should_interrupt}")
        
        if self._on_interruption_detected:
            self._on_interruption_detected(event)
        
        self._last_interruption_time = current_time
    
    def _detect_filler_words(self, transcript: str) -> List[str]:
        """
        Detect filler words in the transcript.
        
        Args:
            transcript: The speech transcript to analyze
            
        Returns:
            List of detected filler words
        """
        detected = []
        transcript_lower = transcript.lower()
        
        for word in self.ignored_words:
            if word in transcript_lower:
                # Check if it's a whole word match
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(transcript):
                    detected.append(word)
        
        return detected
    
    def _classify_interruption(self, transcript: str, detected_fillers: List[str]) -> InterruptionType:
        """
        Classify the type of interruption.
        
        Args:
            transcript: The speech transcript
            detected_fillers: List of detected filler words
            
        Returns:
            InterruptionType classification
        """
        if not detected_fillers:
            return InterruptionType.GENUINE
        
        # Check if transcript contains only filler words
        words = re.findall(r'\b\w+\b', transcript.lower())
        non_filler_words = [w for w in words if w not in self.ignored_words]
        
        if not non_filler_words:
            return InterruptionType.FILLER_ONLY
        else:
            return InterruptionType.MIXED
    
    def _should_allow_interruption(
        self, 
        interruption_type: InterruptionType, 
        transcript: str
    ) -> bool:
        """
        Determine whether an interruption should be allowed.
        
        Args:
            interruption_type: The classified interruption type
            transcript: The original transcript
            
        Returns:
            True if interruption should be allowed, False otherwise
        """
        # Always allow genuine interruptions
        if interruption_type == InterruptionType.GENUINE:
            return True
        
        # Allow mixed interruptions (contain both filler and meaningful content)
        if interruption_type == InterruptionType.MIXED:
            return True
        
        # Filter filler-only interruptions only when agent is speaking
        if interruption_type == InterruptionType.FILLER_ONLY:
            return not self._agent_speaking
        
        return True
    
    def update_ignored_words(self, words: List[str]) -> None:
        """
        Update the list of ignored filler words.
        
        Args:
            words: New list of filler words to ignore
        """
        self.ignored_words = set(word.lower().strip() for word in words)
        self._filler_patterns = [
            re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) 
            for word in self.ignored_words
        ]
        
        if self.debug_mode:
            print(f"[FillerHandler] Updated ignored words: {self.ignored_words}")
    
    def get_stats(self) -> dict:
        """
        Get current handler statistics.
        
        Returns:
            Dictionary with handler statistics
        """
        return {
            "ignored_words": list(self.ignored_words),
            "agent_speaking": self._agent_speaking,
            "confidence_threshold": self.confidence_threshold,
            "min_speech_duration": self.min_speech_duration,
            "debug_mode": self.debug_mode
        }
