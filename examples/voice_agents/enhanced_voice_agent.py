#!/usr/bin/env python3
"""
Enhanced Voice Agent with Filler Word Interruption Handling

This example demonstrates how to use the enhanced agent session with
intelligent filler word interruption handling. The agent will:
- Ignore filler words (uh, umm, hmm, haan) when speaking
- Allow genuine interruptions at any time
- Provide detailed logging of interruption decisions
"""

import asyncio
import logging
import os
from typing import Annotated

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.llm import OpenAI, ChatContext
from livekit.agents.stt import DeepgramSTT
from livekit.agents.tts import ElevenLabsTTS
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice import function_tool
from livekit.agents.voice.enhanced_agent_session import create_enhanced_session
from livekit.agents.voice.filler_interruption_handler import FillerInterruptionEvent, InterruptionType


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedVoiceAgent(Agent):
    """
    Voice agent with enhanced interruption handling.
    
    This agent demonstrates intelligent filler word filtering
    while maintaining natural conversation flow.
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Your name is Alex. "
                "You speak naturally and conversationally. "
                "When users speak, listen carefully and respond thoughtfully. "
                "Keep your responses concise but informative."
            )
        )
        
        # Enhanced session will be set during start
        self.enhanced_session = None
        
        # Track conversation for better context
        self.conversation_history = []
    
    async def on_enter(self) -> None:
        """Called when the agent starts in the session."""
        logger.info("Enhanced Voice Agent started")
        
        # Generate initial greeting
        await self.session.generate_reply()
    
    @function_tool
    async def get_weather(
        self, 
        context: RunContext, 
        location: Annotated[str, "The city or location for weather"]
    ) -> str:
        """
        Get weather information for a location.
        
        Args:
            context: Run context
            location: Location to get weather for
            
        Returns:
            Weather information
        """
        # Simulate weather API call
        await asyncio.sleep(1)  # Simulate API delay
        
        weather_info = (
            f"The weather in {location} is currently 72°F (22°C) "
            f"with partly cloudy skies. There's a light breeze from the west. "
            f"It's a pleasant day overall!"
        )
        
        logger.info(f"Weather requested for {location}")
        return weather_info
    
    @function_tool
    async def tell_joke(self, context: RunContext) -> str:
        """
        Tell a random joke to lighten the mood.
        
        Args:
            context: Run context
            
        Returns:
            A joke
        """
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fake noodle? An impasta!",
            "Why did the bicycle fall over? Because it was two-tired!"
        ]
        
        import random
        joke = random.choice(jokes)
        
        logger.info("Joke requested")
        return joke
    
    @function_tool
    async def get_time(self, context: RunContext) -> str:
        """
        Get the current time.
        
        Args:
            context: Run context
            
        Returns:
            Current time
        """
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
        logger.info("Time requested")
        return f"The current time is {current_time}."


async def setup_enhanced_session(session: AgentSession) -> None:
    """
    Set up the enhanced session with filler interruption handling.
    
    Args:
        session: The base agent session to enhance
    """
    # Create enhanced session with custom filler words
    enhanced = create_enhanced_session(
        session=session,
        ignored_words=[
            "uh", "umm", "um", "hmm", "haan", "ah", "er", 
            "like", "you know", "i mean", "well", "so"
        ],
        confidence_threshold=0.7,
        min_speech_duration=0.3,
        debug_mode=True,
        enable_logging=True
    )
    
    # Set up custom interruption handling
    @enhanced.on("filler_interruption_detected")
    def on_filler_interruption(event: FillerInterruptionEvent):
        """Handle filler interruption events."""
        if event.type == InterruptionType.FILLER_ONLY and not event.should_interrupt:
            logger.info(f"🔇 Filler interruption filtered: '{event.transcript}'")
        elif event.should_interrupt:
            logger.info(f"🎯 Valid interruption allowed: '{event.transcript}'")
    
    @enhanced.on("agent_speaking_changed")
    def on_agent_speaking_changed(data):
        """Handle agent speaking state changes."""
        state = "🔊 SPEAKING" if data["is_speaking"] else "🔇 SILENT"
        logger.info(f"Agent state: {state}")
    
    # Set up VAD integration
    enhanced.setup_vad_integration()
    
    logger.info("Enhanced session configured with filler interruption handling")


async def agent_entrypoint(ctx: JobContext) -> None:
    """
    Entry point for the enhanced voice agent.
    
    Args:
        ctx: Job context
    """
    # Configure agent components
    stt = DeepgramSTT()
    tts = ElevenLabsTTS(
        model_id="eleven_monolingual_v1",
        voice_id="rachel"  # Choose an appropriate voice
    )
    llm = OpenAI(model="gpt-4o-mini")
    
    # Create the agent
    agent = EnhancedVoiceAgent()
    
    # Create the session with enhanced interruption handling
    session = AgentSession(
        ctx=ctx,
        agent=agent,
        stt=stt,
        tts=tts,
        llm=llm,
        vad="silero_vad",
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=1
    )
    
    # Set up enhanced session
    await setup_enhanced_session(session)
    
    # Start the session
    await session.start()


def main():
    """Main entry point."""
    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not all([api_key, deepgram_key, elevenlabs_key]):
        logger.error("Missing required environment variables:")
        logger.error("- OPENAI_API_KEY")
        logger.error("- DEEPGRAM_API_KEY") 
        logger.error("- ELEVENLABS_API_KEY")
        return
    
    # Run the agent worker
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=agent_entrypoint,
            agent_name="enhanced-voice-agent",
        )
    )


if __name__ == "__main__":
    main()
