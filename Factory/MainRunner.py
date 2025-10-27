
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination  # NEW IMPORT
from autogen_agentchat.ui import Console
# ------------------------------
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from Factory.Config import MCPConfig


async def main_with_round_robin_chat():
    """
    Implements a Sequential Task Pipeline using RoundRobinGroupChat:
    1. RestApiAgent (Extract)
    2. DatabaseAgent (Transform & Load to DB)
    3. FileAgent (Load to File)

    The chat explicitly terminates when the FileAgent outputs the phrase 'TERMINATE_CHAT'.
    """
    print("🤖 Starting Round-Robin Chat System (REST -> DB -> File)\n")
    print("=" * 60)

    gkey = os.getenv("GEMINI_API_KEY")
    if not gkey:
        print("❌ Error: GEMINI_API_KEY not found. Please set the GEMINI_API_KEY environment variable.")
        return

    try:
        # ✅ Get all three workbenches
        rest_wb = MCPConfig.get_RestApi_ServerMCP()
        mysql_wb = MCPConfig.get_MySQL_ServerMCP()
        file_wb = MCPConfig.get_FileSystem_ServerMCP()
        model_info = ModelInfo(
            vision=True,
            function_calling=True,
            json_output=False,
            family="unknown",
            structured_output=True
        )
        # Use an async context manager for all three
        async with rest_wb as rest, mysql_wb as mysql, file_wb as files:


            model = OpenAIChatCompletionClient(
                model="gemini-2.0-flash-lite",
                model_info=model_info,
                api_key=gkey,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

            # ============================================
            # AGENT 1: REST API Agent (EXTRACT)
            # ============================================
            rest_agent = AssistantAgent(
                name="RestApiAgent",
                model_client=model,
                workbench=rest,
                system_message="""You are an API integration expert.
                Your first and only task is to use the rest:get tool to retrieve 10 users data from the configured API endpoint.
                Provide the raw JSON response. The data will be passed to the next agent."""
            )

            # ============================================
            # AGENT 2: Database Agent (TRANSFORM & LOAD)
            # ============================================
            db_agent = AssistantAgent(
                name="DatabaseAgent",
                model_client=model,
                workbench=mysql,
                system_message="""You are a database expert. 
                Your task is to receive the data from the previous agent.
                1. Create a table named 'users' in the database with the same structure as obtained from the previous agent json.body
                1. Insert the data that is received in the table you created always increase the id by 1.
                2. After insertion, query all student records and format the results as a clean text report.
                The formatted report must be passed to the FileAgent."""
            )

            # ============================================
            # AGENT 3: File System Agent (OUTPUT & TERMINATION)
            # ============================================
            file_agent = AssistantAgent(
                name="FileAgent",
                model_client=model,
                workbench=files,
                # --- CHANGE HERE: ADD EXPLICIT TERMINATION PHRASE ---
                system_message="""You are a file system expert.
                Your final task is to take the formatted report from the DatabaseAgent.
                Save this data to a file named 'round_robin_report.txt' using the filesystem:write_file tool.
                After successfully confirming the file was created, you MUST output the phrase 'TERMINATE_CHAT' to end the conversation."""
            )

            # ============================================
            # CHAT MANAGER SETUP 
            # ============================================

            # Define the order: REST -> DB -> File
            agent_list = [rest_agent, db_agent, file_agent]

            # The chat will stop when the message "TERMINATE_CHAT" appears.
            termination_condition = TextMentionTermination(
                text="TERMINATE_CHAT",
                sources=[file_agent.name]  # Only look for the phrase from the final agent
            )

            chat_manager = RoundRobinGroupChat(
                participants=agent_list,
                termination_condition=termination_condition  # Pass the condition to the manager
            )

            # The initial task is for the first agent (RestApiAgent)
            initial_task = "Begin the data pipeline: Extract user data using the rest:get tool."

            print("\n🚀 Starting Round-Robin Conversation (3 Steps)")
            print("=" * 60)

            # Run the chained conversation
            await Console(
                chat_manager.run_stream(task=initial_task)
            )

            print("\n" + "=" * 60)
            print("✅ ALL TASKS COMPLETED SUCCESSFULLY via Round-Robin Chat!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main_with_round_robin_chat())
