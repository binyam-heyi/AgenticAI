"""
MULTI-AGENT RESEARCH & ANALYSIS SYSTEM
======================================

PURPOSE:
This script creates a two-agent conversation system where a Researcher and an Analyst
discuss topics collaboratively. The Researcher provides factual information while the
Analyst critically examines and provides insights on that information.

ARCHITECTURE:
- Researcher Agent: Gathers and presents facts on topics
- Analyst Agent: Critically examines information and identifies patterns
- Round-Robin Communication: Agents alternate turns speaking
- Message Limit: Conversation stops after 6 messages (3 turns each)

WORKFLOW:
1. Load API credentials from environment variables
2. Configure and initialize Gemini AI model
3. Create two specialized AI agents with distinct roles
4. Organize agents into round-robin discussion team
5. Start collaborative discussion on specified topic
6. Display real-time conversation in console
7. Automatically end after message limit reached
8. Clean up API connections

REQUIREMENTS:
- .env file containing: GEMINI_API_KEY=your_api_key_here
- Python packages: autogen-agentchat, autogen-ext, python-dotenv

USAGE:
python script_name.py

The agents will automatically discuss AI's impact on society.
"""

# ============================================================================
# SECTION 1: IMPORTS - Load Required Libraries
# ============================================================================

# Core AutoGen models for message handling and model configuration
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

# Standard library imports for async operations and environment variables
import asyncio  # Enables asynchronous programming (non-blocking operations)
import os  # Provides access to operating system environment variables
from dotenv import load_dotenv  # Loads variables from .env file into environment

# AutoGen agent and team components
from autogen_agentchat.agents import AssistantAgent  # AI-powered agent class
from autogen_agentchat.ui import Console  # Terminal/console interface for displaying conversations
from autogen_agentchat.teams import RoundRobinGroupChat  # Manages turn-taking between agents
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination  # Conversation stopping rules

# ============================================================================
# SECTION 2: ENVIRONMENT CONFIGURATION
# ============================================================================

# Load environment variables from .env file
# This reads the .env file and makes variables like GEMINI_API_KEY available via os.getenv()
# ⚠️ Security: Never hardcode API keys in source code - always use environment variables
load_dotenv()


# ============================================================================
# SECTION 3: MAIN APPLICATION LOGIC
# ============================================================================

async def main():
    """
    Main asynchronous function that orchestrates the multi-agent conversation.

    This function:
    1. Retrieves API credentials securely
    2. Configures the Gemini AI model
    3. Creates two specialized agents (Researcher and Analyst)
    4. Sets up a round-robin conversation team
    5. Executes the discussion on AI's societal impact
    6. Handles cleanup of resources

    The conversation follows a round-robin pattern:
    Researcher speaks -> Analyst responds -> Researcher speaks -> Analyst responds...
    """

    print("In AI Agent!")

    # -------------------------------------------------------------------------
    # Step 1: Retrieve API Key from Environment
    # -------------------------------------------------------------------------
    # Fetch the Gemini API key from environment variables
    # This keeps sensitive credentials secure and separate from code
    gkey = os.getenv("GEMINI_API_KEY")

    # Optional: Add validation to ensure API key exists
    if not gkey:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please add GEMINI_API_KEY=your_key to your .env file")
        return

    # -------------------------------------------------------------------------
    # Step 2: Configure Model Capabilities
    # -------------------------------------------------------------------------
    # Define what features and capabilities the Gemini model supports
    # This helps AutoGen understand how to interact with the model
    model_info = ModelInfo(
        vision=True,  # Model can process and analyze images
        function_calling=True,  # Model can execute functions/tools
        json_output=False,  # Model doesn't require JSON-formatted responses
        family="unknown",  # Model family classification (for AutoGen internal use)
        structured_output=True  # Model can generate structured data formats
    )

    # -------------------------------------------------------------------------
    # Step 3: Initialize Gemini API Client
    # -------------------------------------------------------------------------
    # Create connection to Google's Gemini AI service
    # This client handles all communication with the Gemini API
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-lite",  # Specific Gemini model version (fast, efficient)
        model_info=model_info,  # Pass the model capability configuration
        api_key=gkey,  # Authentication key for API access
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini API endpoint
    )

    # -------------------------------------------------------------------------
    # Step 4: Create Researcher Agent
    # -------------------------------------------------------------------------
    # The Researcher agent focuses on gathering and presenting factual information
    # Its role is to provide accurate, well-researched data on topics
    researcher = AssistantAgent(
        name="Researcher",  # Unique identifier (must be valid Python identifier - no spaces)
        model_client=model_client,  # Connect this agent to Gemini AI
        system_message="""You are a researcher who gathers facts and information.
            You provide detailed, accurate information on topics.
            Keep responses concise and factual."""  # Defines agent's personality and behavior
    )

    # -------------------------------------------------------------------------
    # Step 5: Create Analyst Agent
    # -------------------------------------------------------------------------
    # The Analyst agent critically examines information provided by the Researcher
    # Its role is to identify patterns, ask questions, and provide insights
    analyst = AssistantAgent(
        name="Analyst",  # Unique identifier
        model_client=model_client,  # Connect this agent to Gemini AI
        system_message="""You are an analyst who examines information critically.
            You analyze the information provided by others and provide insights.
            Ask probing questions and identify patterns."""  # Defines agent's analytical role
    )

    # -------------------------------------------------------------------------
    # Step 6: Create Round-Robin Team
    # -------------------------------------------------------------------------
    # Organize the two agents into a structured conversation team
    # Round-robin means agents take turns speaking in order:
    # Researcher -> Analyst -> Researcher -> Analyst -> ...
    team = RoundRobinGroupChat(
        participants=[researcher, analyst],  # List of agents participating in conversation
        termination_condition=MaxMessageTermination(max_messages=6)  # Stop after 6 messages (3 turns each)
    )
    # Note: With 6 messages, each agent speaks exactly 3 times

    # -------------------------------------------------------------------------
    # Step 7: Display Conversation Header
    # -------------------------------------------------------------------------
    # Print formatted header to indicate conversation start
    print("\n" + "=" * 60)
    print("MULTI-AGENT CONVERSATION - ROUND ROBIN")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Step 8: Execute the Conversation
    # -------------------------------------------------------------------------
    # Start the agent conversation and stream results to console
    # Console() displays each message in real-time as agents respond
    # run_stream() enables streaming mode for better user experience
    result = await Console(
        team.run_stream(
            task="""Discuss the topic: 'The Impact of Artificial Intelligence on Society'.
               Each agent should contribute their perspective based on their role.
               After everyone has spoken twice, the Critic should say TERMINATE."""
            # Note: The "Critic" mention in the task is a typo - there's no Critic agent
            # The conversation will end based on MaxMessageTermination instead
        )
    )

    # -------------------------------------------------------------------------
    # Step 9: Cleanup Resources
    # -------------------------------------------------------------------------
    # Close the API client connection to free resources
    # This is important to prevent memory leaks and ensure proper shutdown
    await model_client.close()
    print("\n🔒 Model client closed successfully")


# ============================================================================
# SECTION 4: APPLICATION ENTRY POINT
# ============================================================================

# Execute the main function using asyncio
# asyncio.run() creates an event loop, runs the async function, and closes the loop
# This is the standard way to run async programs in Python
asyncio.run(main())