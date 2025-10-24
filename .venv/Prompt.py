"""
INTERACTIVE MATH TUTORING SYSTEM
=================================

PURPOSE:
This script creates an interactive math tutoring application using AI. A human student
can ask math questions and receive personalized help from an AI-powered math teacher.
The system demonstrates human-AI collaboration using Google's Gemini AI model.

ARCHITECTURE:
- Student (UserProxyAgent): Represents the human user, allows manual input
- Math Teacher (AssistantAgent): AI-powered tutor using Gemini 2.0
- Round-Robin Communication: Student and teacher alternate turns
- Text-Based Termination: Session ends when student says "DONE"

WORKFLOW:
1. Load API credentials from environment variables
2. Configure Gemini AI model with specific capabilities
3. Create AI math teacher agent with teaching instructions
4. Create student proxy agent for human interaction
5. Set up round-robin conversation team
6. Start interactive tutoring session
7. Allow real-time Q&A between student and teacher
8. End session when student says "DONE"
9. Clean up API resources

REQUIREMENTS:
- .env file containing: GEMINI_API_KEY=your_api_key_here
- Python packages: autogen-agentchat, autogen-ext, python-dotenv

USAGE:
python script_name.py

Then interact by typing math questions when prompted.
Type "DONE" to end the session.

SECURITY:
- API keys stored securely in .env file
- Never commit .env to version control
- Add .env to .gitignore
"""

# ============================================================================
# SECTION 1: IMPORTS - Load Required Libraries and Modules
# ============================================================================

# Core AutoGen components for API communication and message handling
from autogen_core.models import UserMessage  # Represents user messages in conversation
from autogen_ext.models.openai import OpenAIChatCompletionClient  # Gemini API client
from autogen_core.models import ModelInfo  # Defines model capabilities

# Standard Python libraries for async operations and environment management
import asyncio  # Enables asynchronous/concurrent operations
import os  # Provides access to operating system environment variables
from dotenv import load_dotenv  # Loads environment variables from .env file

# AutoGen agent types for different roles
from autogen_agentchat.agents import AssistantAgent  # AI-powered agent
from autogen_agentchat.agents import UserProxyAgent  # Human user proxy agent

# UI and team management components
from autogen_agentchat.ui import Console  # Console interface for displaying conversations
from autogen_agentchat.teams import RoundRobinGroupChat  # Manages turn-taking between agents

# Conversation control conditions
from autogen_agentchat.conditions import TextMentionTermination  # Stops on specific text
from autogen_agentchat.conditions import MaxMessageTermination  # Stops after N messages


# ============================================================================
# SECTION 2: ENVIRONMENT SETUP - Load Sensitive Configuration
# ============================================================================

# Load environment variables from .env file into the system environment
# This makes API keys and other secrets available via os.getenv()
# ⚠️ Security Best Practice: Never hardcode API keys directly in source code
load_dotenv()


# ============================================================================
# SECTION 3: MAIN APPLICATION LOGIC - Core Tutoring System
# ============================================================================

async def main():
    """
    Main asynchronous function that orchestrates the math tutoring session.

    This function creates an interactive learning environment where:
    - A human student (via UserProxyAgent) can ask math questions
    - An AI teacher (via AssistantAgent) provides explanations and guidance
    - Conversation flows naturally in round-robin fashion
    - Session ends when student types "DONE"

    The function handles:
    1. API authentication and model configuration
    2. Agent creation with specific roles and behaviors
    3. Team setup for structured conversation
    4. Real-time interaction management
    5. Resource cleanup after session ends

    Returns:
        None - Function runs the interactive session and displays output to console
    """

    # Display welcome message to indicate session start
    print("🎓 Starting Math Tutoring Session\n")

    # -------------------------------------------------------------------------
    # Step 1: Retrieve and Validate API Credentials
    # -------------------------------------------------------------------------
    # Fetch the Gemini API key from environment variables
    # This keeps sensitive credentials separate from code for security
    gkey = os.getenv("GEMINI_API_KEY")

    # Validate that the API key exists before proceeding
    # This prevents cryptic errors later if the key is missing
    if not gkey:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
        return  # Exit early if no API key is found

    print("✓ API Key loaded successfully\n")

    # -------------------------------------------------------------------------
    # Step 2: Configure Model Capabilities and Features
    # -------------------------------------------------------------------------
    # Define what the Gemini model can and cannot do
    # This helps AutoGen understand how to interact with the model effectively
    model_info = ModelInfo(
        vision=True,              # Model can process and understand images
        function_calling=True,    # Model can execute functions/tools (like calculators)
        json_output=False,        # Model returns natural language, not JSON
        family="unknown",         # Model family identifier (for AutoGen categorization)
        structured_output=True    # Model can generate structured data formats
    )
    # Note: These capabilities are important for math tutoring as the teacher
    # may need to use tools like calculators or display visual diagrams

    # -------------------------------------------------------------------------
    # Step 3: Initialize Gemini API Client Connection
    # -------------------------------------------------------------------------
    # Create a client that communicates with Google's Gemini AI service
    # This client handles all API requests, authentication, and response processing
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-lite",  # Specific Gemini model version (fast and efficient)
        model_info=model_info,           # Pass the model capability configuration
        api_key=gkey,                    # Authentication key for API access
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini API endpoint URL
    )
    # Note: The base_url points to Google's Gemini API, which uses OpenAI-compatible format

    # -------------------------------------------------------------------------
    # Step 4: Create the AI Math Teacher Agent
    # -------------------------------------------------------------------------
    # This agent is powered by Gemini AI and acts as a knowledgeable math tutor
    # It can explain concepts, solve problems, and guide students through learning
    teacher = AssistantAgent(
        name="MathTeacher",  # Unique identifier (must be valid Python identifier - no spaces!)
        model_client=model_client,  # Connect this agent to the Gemini API client
        system_message="""You are a helpful math teacher. Help the user learn math concepts clearly and patiently.
        - Explain step-by-step solutions
        - Use simple language appropriate for the student's level
        - Provide examples when helpful
        - Encourage the student and build confidence
        - When the user says 'DONE', acknowledge their progress and say 'LESSON COMPLETE'."""
        # The system_message defines the agent's personality, role, and behavior guidelines
    )

    # -------------------------------------------------------------------------
    # Step 5: Create the Student Proxy Agent (Human Interface)
    # -------------------------------------------------------------------------
    # UserProxyAgent represents the human user in the conversation
    # It allows manual input and passes human messages to the AI teacher
    student_proxy = UserProxyAgent(
        name="MathStudent"  # Unique identifier for the student
    )
    # Note: UserProxyAgent prompts the user for input during the conversation
    # This creates an interactive experience where the human can ask questions

    # -------------------------------------------------------------------------
    # Step 6: Set Up Conversation Termination Condition
    # -------------------------------------------------------------------------
    # Define when the tutoring session should end
    # The session stops when the teacher says "LESSON COMPLETE"
    # Note: There's a typo in the original - extra quote mark should be removed
    termination = TextMentionTermination("LESSON COMPLETE")
    # The conversation will automatically end when this text appears in any message

    # -------------------------------------------------------------------------
    # Step 7: Create Round-Robin Team for Structured Conversation
    # -------------------------------------------------------------------------
    # Organize the student and teacher into a structured conversation team
    # Round-robin ensures they take turns speaking:
    # Student asks question -> Teacher responds -> Student asks -> Teacher responds...
    team = RoundRobinGroupChat(
        participants=[student_proxy, teacher],  # List of agents in the conversation
        termination_condition=termination  # Condition that ends the conversation
    )
    # The order in participants list determines who speaks first (student_proxy)

    # -------------------------------------------------------------------------
    # Step 8: Display Session Header
    # -------------------------------------------------------------------------
    # Print a formatted header to clearly indicate the tutoring session has begun
    print("\n" + "=" * 60)
    print("MATH TUTORING SESSION - ROUND ROBIN")
    print("=" * 60)
    print("Ask math questions and type 'DONE' when finished\n")

    # -------------------------------------------------------------------------
    # Step 9: Execute the Interactive Tutoring Session
    # -------------------------------------------------------------------------
    # Start the conversation and stream responses in real-time to the console
    # Console() displays each message as it's generated for better UX
    # run_stream() enables streaming mode where responses appear progressively
    result = await Console(
        team.run_stream(
            task="""I need help with algebra. Can you help me understand how to solve linear equations?"""
            # This is the initial prompt that starts the conversation
            # The teacher will respond to this, then the student can ask follow-up questions
        )
    )
    # The conversation continues until the student types "DONE" and teacher says "LESSON COMPLETE"

    # -------------------------------------------------------------------------
    # Step 10: Session Completion and Cleanup
    # -------------------------------------------------------------------------
    # Display completion message
    print("\n" + "=" * 60)
    print("✅ SESSION COMPLETE - Thank you for learning!")
    print("=" * 60)

    # Close the API client connection to free up resources
    # This is critical to prevent memory leaks and ensure proper shutdown
    await model_client.close()
    print("\n🔒 Resources cleaned up successfully")


# ============================================================================
# SECTION 4: APPLICATION ENTRY POINT - Program Execution
# ============================================================================

# Execute the main async function using asyncio
# asyncio.run() does the following:
# 1. Creates a new event loop
# 2. Runs the async main() function in that loop
# 3. Closes the loop when main() completes
# This is the standard pattern for running async Python programs
asyncio.run(main())