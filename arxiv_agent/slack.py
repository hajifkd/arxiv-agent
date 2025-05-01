import asyncio
import os
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_TEAM_ID = os.getenv("SLACK_TEAM_ID")

# new slack client
from slack_sdk.errors import SlackApiError
from slack_sdk.web import SlackResponse
from slack_sdk.web.async_client import AsyncWebClient

client = AsyncWebClient(token=SLACK_BOT_TOKEN)

async def get_channel_id(channel_name: str) -> str:
    """
    Get the channel ID for a given channel name.
    """
    try:
        response: SlackResponse = await client.conversations_list()
        channels = response["channels"]
        for channel in channels:
            if channel["name"] == channel_name:
                return channel["id"]
    except SlackApiError as e:
        print(f"Error fetching channels: {e.response['error']}")
    return None

async def post_messages_as_thread(channel_id: str, header: str, texts: list[str]) -> None:
    """
    Post a message as a thread in a given channel.
    """
    try:
        initial_post = await client.chat_postMessage(
            channel=channel_id,
            text=header,
        )
        thread_ts = initial_post["ts"]
        for text in texts:
            await asyncio.sleep(1)  # Rate limit handling
            await client.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=thread_ts,
            )
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

async def post_message(channel_id: str, text: str) -> None:
    """
    Post a message in a given channel.
    """
    try:
        await client.chat_postMessage(
            channel=channel_id,
            text=text,
        )
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None