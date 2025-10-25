"""
SIMPLE GEMINI AI ASSISTANT WITH FILE SYSTEM ACCESS
==================================================
A basic example of using Google's Gemini AI to answer questions
and save responses to files using MCP (Model Context Protocol).
"""

import asyncio
import os
from pathlib import Path

from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

# Load environment variables from .env file
load_dotenv()

def getFileServerMCP():
    """
    Create and return an MCP workbench for file system operations.
    This allows the AI assistant to read/write files in the same directory as this script.
    """
    # Get the directory where this script is located ( Change location if needed)
    current_dir = Path(__file__).parent.resolve()

    # Convert to string with forward slashes (works on Windows too)
    current_dir_str = str(current_dir).replace("\\", "/")

    print(f"📁 File system access granted to: {current_dir_str}\n")

    fileSystemParameters = StdioServerParams(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            current_dir_str
        ],
        read_timeout_seconds=60
    )

    # Create and return the MCP workbench
    fs_workbench = McpWorkbench(fileSystemParameters)
    return fs_workbench


async def main():
    """
    Simple assistant that answers questions using Gemini AI
    and can save responses to files in the same directory as this script.
    """
    print("Hello World!")

    # Get API key from environment variables ( Vault can be used also)
    gkey = os.getenv("GEMINI_API_KEY")

    # Validate API key exists
    if not gkey:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        return

    # Get the file server MCP workbench
    fcb_workbench = getFileServerMCP()

    # Use async context manager for proper resource management
    async with fcb_workbench as fcb:
        # Configure Gemini model capabilities
        model_info = ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=True
        )

        # Create Gemini model client
        model = OpenAIChatCompletionClient(
            model="gemini-2.0-flash-lite",
            model_info=model_info,
            api_key=gkey,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        # Create assistant agent with file system access
        assistant = AssistantAgent(
            name="assistant",
            model_client=model,
            workbench=fcb  # Gives assistant access to file operations
        )

        # Run the assistant with a question
        await Console(
            assistant.run_stream(
                task="What is the capital city of India? Save the answer in a text file called 'answer.txt'"
            )
        )

        # Clean up resources
        await model.close()
        print("\n✅ Task completed! Check the 'answer.txt' file in the same folder as this script.")


# Run the async main function
asyncio.run(main())