#!/usr/bin/env python3
"""
Test Suite for Filler Word Interruption Handler

This script tests the filler word interruption handling functionality
with various scenarios including filler-only speech, genuine interruptions,
and mixed content.
"""

import asyncio
import logging
import time
from typing import List

from livekit.agents.vad import VADEvent, VADEventType
from livekit.agents.voice import UserInputTranscribedEvent
from livekit.agents.voice.filler_interruption_handler import (
    FillerInterruptionHandler,
    FillerInterruptionEvent,
    InterruptionType
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFillerInterruptionHandler:
    """Test suite for filler interruption handler."""
    
    def __init__(self):
        self.handler = FillerInterruptionHandler(
            ignored_words=["uh", "umm", "um", "hmm", "haan", "ah", "er"],
            confidence_threshold=0.7,
            min_speech_duration=0.3,
            debug_mode=True
        )
        
        self.test_results = []
        self.interruption_events = []
        
        # Set up callbacks
        self.handler.set_interruption_callback(self._on_interruption)
        self.handler.set_agent_speaking_callback(self._on_agent_speaking_changed)
    
    def _on_interruption(self, event: FillerInterruptionEvent):
        """Handle interruption events during testing."""
        self.interruption_events.append(event)
        logger.info(f"Test interruption: {event}")
    
    def _on_agent_speaking_changed(self, is_speaking: bool):
        """Handle agent speaking state changes during testing."""
        logger.info(f"Test agent speaking: {is_speaking}")
    
    async def test_filler_only_when_agent_speaking(self):
        """Test filler-only speech when agent is speaking (should be filtered)."""
        logger.info("=== Test: Filler-only when agent speaking ===")
        
        # Set agent as speaking
        self.handler.update_agent_speaking_state(True)
        
        # Simulate VAD start
        vad_start = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start)
        
        # Simulate filler-only transcript
        transcript_event = UserInputTranscribedEvent(
            transcript="uh umm hmm",
            is_final=True
        )
        self.handler.on_transcript_event(transcript_event)
        
        # Simulate VAD end
        vad_end = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=1000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check results
        if self.interruption_events:
            last_event = self.interruption_events[-1]
            success = (
                last_event.type == InterruptionType.FILLER_ONLY and
                not last_event.should_interrupt and
                last_event.agent_was_speaking
            )
        else:
            success = False
        
        self.test_results.append({
            "test": "filler_only_when_agent_speaking",
            "success": success,
            "description": "Filler-only speech should be filtered when agent is speaking"
        })
        
        logger.info(f"Result: {'PASS' if success else 'FAIL'}")
    
    async def test_filler_only_when_agent_silent(self):
        """Test filler-only speech when agent is silent (should be allowed)."""
        logger.info("=== Test: Filler-only when agent silent ===")
        
        # Set agent as silent
        self.handler.update_agent_speaking_state(False)
        
        # Simulate VAD start
        vad_start = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start)
        
        # Simulate filler-only transcript
        transcript_event = UserInputTranscribedEvent(
            transcript="uh umm",
            is_final=True
        )
        self.handler.on_transcript_event(transcript_event)
        
        # Simulate VAD end
        vad_end = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=1000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check results
        if self.interruption_events:
            last_event = self.interruption_events[-1]
            success = (
                last_event.type == InterruptionType.FILLER_ONLY and
                last_event.should_interrupt and
                not last_event.agent_was_speaking
            )
        else:
            success = False
        
        self.test_results.append({
            "test": "filler_only_when_agent_silent",
            "success": success,
            "description": "Filler-only speech should be allowed when agent is silent"
        })
        
        logger.info(f"Result: {'PASS' if success else 'FAIL'}")
    
    async def test_genuine_interruption_when_agent_speaking(self):
        """Test genuine interruption when agent is speaking (should be allowed)."""
        logger.info("=== Test: Genuine interruption when agent speaking ===")
        
        # Set agent as speaking
        self.handler.update_agent_speaking_state(True)
        
        # Simulate VAD start
        vad_start = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start)
        
        # Simulate genuine interruption transcript
        transcript_event = UserInputTranscribedEvent(
            transcript="wait stop",
            is_final=True
        )
        self.handler.on_transcript_event(transcript_event)
        
        # Simulate VAD end
        vad_end = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=1000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check results
        if self.interruption_events:
            last_event = self.interruption_events[-1]
            success = (
                last_event.type == InterruptionType.GENUINE and
                last_event.should_interrupt and
                last_event.agent_was_speaking
            )
        else:
            success = False
        
        self.test_results.append({
            "test": "genuine_interruption_when_agent_speaking",
            "success": success,
            "description": "Genuine interruptions should always be allowed"
        })
        
        logger.info(f"Result: {'PASS' if success else 'FAIL'}")
    
    async def test_mixed_content_when_agent_speaking(self):
        """Test mixed content (filler + meaningful) when agent is speaking."""
        logger.info("=== Test: Mixed content when agent speaking ===")
        
        # Set agent as speaking
        self.handler.update_agent_speaking_state(True)
        
        # Simulate VAD start
        vad_start = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start)
        
        # Simulate mixed content transcript
        transcript_event = UserInputTranscribedEvent(
            transcript="uh wait a minute",
            is_final=True
        )
        self.handler.on_transcript_event(transcript_event)
        
        # Simulate VAD end
        vad_end = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=1000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check results
        if self.interruption_events:
            last_event = self.interruption_events[-1]
            success = (
                last_event.type == InterruptionType.MIXED and
                last_event.should_interrupt and
                last_event.agent_was_speaking
            )
        else:
            success = False
        
        self.test_results.append({
            "test": "mixed_content_when_agent_speaking",
            "success": success,
            "description": "Mixed content should be allowed as interruption"
        })
        
        logger.info(f"Result: {'PASS' if success else 'FAIL'}")
    
    async def test_short_speech_ignored(self):
        """Test that very short speech is ignored."""
        logger.info("=== Test: Short speech ignored ===")
        
        # Set agent as speaking
        self.handler.update_agent_speaking_state(True)
        
        # Simulate very short speech (less than min_speech_duration)
        vad_start = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start)
        
        # Very short delay
        await asyncio.sleep(0.1)
        
        vad_end = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=100,
            timestamp=time.time(),
            speech_duration=0.1,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Should not generate any interruption events
        event_count_before = len(self.interruption_events)
        
        self.test_results.append({
            "test": "short_speech_ignored",
            "success": len(self.interruption_events) == event_count_before,
            "description": "Very short speech should be ignored"
        })
        
        logger.info(f"Result: {'PASS' if len(self.interruption_events) == event_count_before else 'FAIL'}")
    
    async def test_cooldown_prevention(self):
        """Test that rapid-fire interruptions are prevented by cooldown."""
        logger.info("=== Test: Cooldown prevention ===")
        
        # Set agent as speaking
        self.handler.update_agent_speaking_state(True)
        
        # First interruption
        vad_start1 = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=0,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start1)
        
        transcript1 = UserInputTranscribedEvent(
            transcript="wait",
            is_final=True
        )
        self.handler.on_transcript_event(transcript1)
        
        vad_end1 = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=1000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end1)
        
        await asyncio.sleep(0.1)
        
        # Second interruption immediately (should be blocked by cooldown)
        vad_start2 = VADEvent(
            type=VADEventType.START_OF_SPEECH, 
            samples_index=2000,
            timestamp=time.time(),
            speech_duration=0.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_start2)
        
        transcript2 = UserInputTranscribedEvent(
            transcript="stop",
            is_final=True
        )
        self.handler.on_transcript_event(transcript2)
        
        vad_end2 = VADEvent(
            type=VADEventType.END_OF_SPEECH, 
            samples_index=3000,
            timestamp=time.time(),
            speech_duration=1.0,
            silence_duration=0.0
        )
        self.handler.on_vad_event(vad_end2)
        
        await asyncio.sleep(0.1)
        
        # Check that only one interruption was processed
        processed_count = len([e for e in self.interruption_events 
                              if e.transcript in ["wait", "stop"]])
        
        self.test_results.append({
            "test": "cooldown_prevention",
            "success": processed_count == 1,
            "description": "Rapid-fire interruptions should be prevented by cooldown"
        })
        
        logger.info(f"Result: {'PASS' if processed_count == 1 else 'FAIL'}")
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("Starting filler interruption handler tests...")
        
        # Clear previous results
        self.test_results = []
        self.interruption_events = []
        
        # Run tests
        await self.test_filler_only_when_agent_speaking()
        await asyncio.sleep(1.0)
        
        await self.test_filler_only_when_agent_silent()
        await asyncio.sleep(1.0)
        
        await self.test_genuine_interruption_when_agent_speaking()
        await asyncio.sleep(1.0)
        
        await self.test_mixed_content_when_agent_speaking()
        await asyncio.sleep(1.0)
        
        await self.test_short_speech_ignored()
        await asyncio.sleep(1.0)
        
        await self.test_cooldown_prevention()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print a summary of all test results."""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = 0
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "PASS" if result["success"] else "FAIL"
            logger.info(f"{status}: {result['test']}")
            logger.info(f"  {result['description']}")
            if result["success"]:
                passed += 1
        
        logger.info("="*60)
        logger.info(f"Results: {passed}/{total} tests passed")
        logger.info("="*60)
        
        # Print handler statistics
        stats = self.handler.get_stats()
        logger.info(f"Handler stats: {stats}")


async def main():
    """Main test runner."""
    test_suite = TestFillerInterruptionHandler()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
