"""
Enhanced Agent Session with Filler Word Interruption Handling

This module provides a wrapper around the standard AgentSession that
integrates intelligent filler word interruption handling without
modifying the core LiveKit SDK.
"""

import asyncio
import logging
from typing import Callable, Optional, List

from livekit.agents.stt import SpeechEvent
from livekit.agents.vad import VADEvent
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import (
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    UserStateChangedEvent
)
from .filler_interruption_handler import (
    FillerInterruptionHandler,
    FillerInterruptionEvent,
    InterruptionType
)


logger = logging.getLogger(__name__)


class EnhancedAgentSession:
    """
    Enhanced AgentSession with intelligent filler word interruption handling.
    
    This wrapper monitors transcription and VAD events, applies filler word
    filtering, and controls interruption behavior based on agent speaking state.
    """
    
    def __init__(
        self,
        session: AgentSession,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
        min_speech_duration: float = 0.3,
        debug_mode: bool = False,
        enable_logging: bool = True
    ):
        """
        Initialize the enhanced agent session.
        
        Args:
            session: The underlying AgentSession to enhance
            ignored_words: List of filler words to ignore
            confidence_threshold: Minimum ASR confidence threshold
            min_speech_duration: Minimum speech duration to process
            debug_mode: Enable detailed debug logging
            enable_logging: Enable interruption logging
        """
        self.session = session
        self.enable_logging = enable_logging
        
        # Initialize filler interruption handler
        self.filler_handler = FillerInterruptionHandler(
            ignored_words=ignored_words,
            confidence_threshold=confidence_threshold,
            min_speech_duration=min_speech_duration,
            debug_mode=debug_mode
        )
        
        # Set up callbacks
        self.filler_handler.set_interruption_callback(self._on_interruption_detected)
        self.filler_handler.set_agent_speaking_callback(self._on_agent_speaking_changed)
        
        # Statistics tracking
        self.stats = {
            "total_interruptions": 0,
            "filtered_interruptions": 0,
            "allowed_interruptions": 0,
            "filler_only_interruptions": 0,
            "genuine_interruptions": 0,
            "mixed_interruptions": 0
        }
        
        # Register event listeners
        self._setup_event_listeners()
        
        if enable_logging:
            logger.info("Enhanced AgentSession initialized with filler interruption handling")
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners on the underlying session."""
        # Listen to user input transcribed events
        self.session.on("user_input_transcribed", self._on_user_input_transcribed)
        
        # Listen to agent state changes to track speaking state
        self.session.on("agent_state_changed", self._on_agent_state_changed)
        
        # Note: VAD events are handled through the audio recognition module
        # We'll need to hook into that separately
    
    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        """
        Handle user input transcribed events.
        
        Args:
            event: User input transcription event
        """
        # Forward to filler handler for processing
        self.filler_handler.on_transcript_event(event)
    
    def _on_agent_state_changed(self, event) -> None:
        """
        Handle agent state changes to track speaking status.
        
        Args:
            event: Agent state changed event
        """
        is_speaking = event.new_state == "speaking"
        self.filler_handler.update_agent_speaking_state(is_speaking)
    
    def _on_interruption_detected(self, event: FillerInterruptionEvent) -> None:
        """
        Handle interruption detection from filler handler.
        
        Args:
            event: Filler interruption event
        """
        # Update statistics
        self.stats["total_interruptions"] += 1
        
        if event.type == InterruptionType.FILLER_ONLY:
            self.stats["filler_only_interruptions"] += 1
        elif event.type == InterruptionType.GENUINE:
            self.stats["genuine_interruptions"] += 1
        elif event.type == InterruptionType.MIXED:
            self.stats["mixed_interruptions"] += 1
        
        if event.should_interrupt:
            self.stats["allowed_interruptions"] += 1
        else:
            self.stats["filtered_interruptions"] += 1
        
        # Log the interruption
        if self.enable_logging:
            self._log_interruption(event)
        
        # Emit custom event for external handling
        self.session.emit("filler_interruption_detected", event)
    
    def _on_agent_speaking_changed(self, is_speaking: bool) -> None:
        """
        Handle agent speaking state changes.
        
        Args:
            is_speaking: Whether the agent is currently speaking
        """
        if self.enable_logging:
            logger.debug(f"Agent speaking state changed: {is_speaking}")
        
        # Emit custom event for external handling
        self.session.emit("agent_speaking_changed", {"is_speaking": is_speaking})
    
    def _log_interruption(self, event: FillerInterruptionEvent) -> None:
        """
        Log interruption details.
        
        Args:
            event: Filler interruption event
        """
        action = "ALLOWED" if event.should_interrupt else "FILTERED"
        logger.info(f"[FILLER_INTERRUPTION] {action}: '{event.transcript}' "
                   f"(Type: {event.type}, Agent speaking: {event.agent_was_speaking}, "
                   f"Fillers: {event.filler_words_detected})")
    
    def setup_vad_integration(self) -> None:
        """
        Set up VAD event integration.
        
        This method should be called after the session's audio recognition
        is initialized to properly intercept VAD events.
        """
        # This is a placeholder for VAD integration
        # In a real implementation, we'd need to hook into the audio recognition module
        # For now, we'll use the transcript events as the primary source
        if self.enable_logging:
            logger.info("VAD integration setup (using transcript events as primary source)")
    
    def update_ignored_words(self, words: List[str]) -> None:
        """
        Update the list of ignored filler words.
        
        Args:
            words: New list of filler words to ignore
        """
        self.filler_handler.update_ignored_words(words)
        if self.enable_logging:
            logger.info(f"Updated ignored words: {words}")
    
    def get_stats(self) -> dict:
        """
        Get interruption handling statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "filler_handler_stats": self.filler_handler.get_stats()
        }
    
    def reset_stats(self) -> None:
        """Reset interruption statistics."""
        self.stats = {
            "total_interruptions": 0,
            "filtered_interruptions": 0,
            "allowed_interruptions": 0,
            "filler_only_interruptions": 0,
            "genuine_interruptions": 0,
            "mixed_interruptions": 0
        }
        if self.enable_logging:
            logger.info("Interruption statistics reset")
    
    # Delegate methods to underlying session
    def __getattr__(self, name):
        """Delegate attribute access to the underlying session."""
        return getattr(self.session, name)


def create_enhanced_session(
    session: AgentSession,
    ignored_words: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    min_speech_duration: float = 0.3,
    debug_mode: bool = False,
    enable_logging: bool = True
) -> EnhancedAgentSession:
    """
    Create an enhanced agent session with filler interruption handling.
    
    Args:
        session: The base AgentSession to enhance
        ignored_words: List of filler words to ignore
        confidence_threshold: Minimum ASR confidence threshold
        min_speech_duration: Minimum speech duration to process
        debug_mode: Enable detailed debug logging
        enable_logging: Enable interruption logging
    
    Returns:
        EnhancedAgentSession instance
    """
    return EnhancedAgentSession(
        session=session,
        ignored_words=ignored_words,
        confidence_threshold=confidence_threshold,
        min_speech_duration=min_speech_duration,
        debug_mode=debug_mode,
        enable_logging=enable_logging
    )
