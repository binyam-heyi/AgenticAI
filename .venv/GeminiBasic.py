from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

# Load environment variables from .env file
load_dotenv()
# ⚠️ Don't expose API keys publicly!
async def main():
    print("In AI Agent!")

    gkey = os.getenv("GEMINI_API_KEY")

    model_info=ModelInfo(vision=True, function_calling=True, json_output=False, family="unknown",
                             structured_output=True);

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-lite",
        model_info=model_info,
        api_key=gkey
        # api_key="GEMINI_API_KEY",
    )
    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        system_message="""You are a researcher who gathers facts and information.
            You provide detailed, accurate information on topics.
            Keep responses concise and factual."""
    )

    analyst = AssistantAgent(
        name="Analyst",
        model_client=model_client,
        system_message="""You are an analyst who examines information critically.
            You analyze the information provided by others and provide insights.
            Ask probing questions and identify patterns."""
    )

    team = RoundRobinGroupChat(
        participants=[researcher, analyst],
        termination_condition=MaxMessageTermination(max_messages=6)
    )

    # Run the conversation
    print("\n" + "=" * 60)
    print("MULTI-AGENT CONVERSATION - ROUND ROBIN")
    print("=" * 60 + "\n")

    result = await Console(
        team.run_stream(
            task="""Discuss the topic: 'The Impact of Artificial Intelligence on Society'.
               Each agent should contribute their perspective based on their role.
               After everyone has spoken twice, the Critic should say TERMINATE."""
        )
    )


    # Close the model client
    await model_client.close()




# Run the async main function
asyncio.run(main())
