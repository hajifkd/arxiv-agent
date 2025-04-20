from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, OpenAIChatCompletionsModel, Agent, Runner
from agents.mcp import MCPServerStdio
import dotenv
import os
import asyncio

dotenv.load_dotenv()

set_tracing_disabled(disabled=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")

custom_client = AsyncOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_API_BASE}openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
    default_headers={"api-key": AZURE_OPENAI_API_KEY},
    default_query={"api-version": AZURE_OPENAI_API_VERSION},
)

set_default_openai_client(custom_client, use_for_tracing=False)

async def main_async():
    async with MCPServerStdio(
        name="arXiv search",
        params={
            "command": "uvx",
            "args": ["--from", "git+https://github.com/hajifkd/arxiv-mcp-server", "arxiv-mcp-server"]
        },
    ) as arxiv_mcp, \
    MCPServerStdio(
        name="Slack server",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-slack"],
            "env": {
                "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID"),
            }
        },
    ) as slack_mcp:
        agent = Agent(
            name="Elementary particle physics research assistant",
            instructions="""You are an post-doc level researcher in elementary particle physics theorists.
You are suppose to assist other reseacher in the same fields.
You and your colleagues are basically interested in subjects related in hep-ph and hep-th, but not very much in nuclear physics papers even if some appear on hep-ph or hep-th.
You and your colleagues check all the latest papers appearing on arXiv everyday.
""",
            model=OpenAIChatCompletionsModel(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_client=custom_client,
            ),
            mcp_servers=[arxiv_mcp, slack_mcp],
        )

        result = await Runner.run(
            agent,
            input="chatbot-testチャンネルに挨拶を投稿してください。",
        )
        print(result.final_output)

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
