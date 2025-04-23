from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled, OpenAIChatCompletionsModel, Agent, Runner, ItemHelpers
from agents.mcp import MCPServerStdio
import dotenv
import os
import asyncio
from datetime import datetime
from pydantic import BaseModel

dotenv.load_dotenv()

set_tracing_disabled(disabled=True)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME")
AZURE_OPENAI_POSTDOC_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_POSTDOC_DEPLOYMENT_NAME")
AZURE_OPENAI_STAFF_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_STAFF_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")

def get_custom_client(deployment_name):
    return AsyncOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=f"{AZURE_OPENAI_API_BASE}openai/deployments/{deployment_name}",
        default_headers={"api-key": AZURE_OPENAI_API_KEY},
        default_query={"api-version": AZURE_OPENAI_API_VERSION},
    )

student_client = get_custom_client(AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME)
postdoc_client = get_custom_client(AZURE_OPENAI_POSTDOC_DEPLOYMENT_NAME)
staff_client = get_custom_client(AZURE_OPENAI_STAFF_DEPLOYMENT_NAME)
set_default_openai_client(student_client, use_for_tracing=False)

class InterestingPapers(BaseModel):
    arxiv_ids: list[str]

GRADUATE_STUDENT_PROMPT = """You are a graduate student in elementary particle physics theory.
You are suppose to assist other senior researchers in your group. As a student, you are not expected to know everything, but you are expected to be able to find the information and summarize it."""

POSTDOC_PROMPT = """You are a postdoc in elementary particle physics theory.
As a postdoc, you are expected to know topics related to your main interest well, and you have wide knowledge and interest in elementary particle physics theory."""

STAFF_PROMPT = """You are a staff researcher in elementary particle physics theory.
As a staff researcher, you are expected to know various topics in elementary particle physics theory well, and you have wide knowledge and interest in elementary particle physics theory."""

RESEARCHER_PROMPT = """As a researcher, you are skeptical about the results of other researchers, and you must be honest about your understanding of the results; if you don't understand something, you should say so. """

INTEREST_PAPER_PROMPT = """You and your colleagues are basically interested in subjects related in hep-ph and hep-th.
Subjects which you and your colleagues are interested in include:
- Particle phenomenology
- Quantum field theory
- Cosmology
- Quantum information
- Astrophysics
Subjects which you and your colleagues are not interested in include:
- Nuclear physics
- Meson spectroscopy
- Nuclear structure
- Lattice QCD
- Too technical methods such as machine learning
- Too exotic subjects such as Lorentz violation"""

async def discuss_paper(arxiv_mcp, arxiv_id) -> str:
    await asyncio.sleep(3)

    paper_summarize_agent = Agent(
        name="Elementary particle physics graduate student",
        instructions=f"""{GRADUATE_STUDENT_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}""",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME,
            openai_client=student_client,
        ),
        mcp_servers=[arxiv_mcp],
    )
    print(f"Reading the paper {arxiv_id}...")
    summary_result = await Runner.run(
        paper_summarize_agent,
        input=f"Please read the paper {arxiv_id} and tell me the details of the paper. You should clarify what the paper is about, what the main motivation of the paper is, what the main results of the paper are, and what the main conclusions of the paper are. You should also clarify what is the main new idea of the paper and stress the importance of the paper."
    )
    summary = summary_result.final_output
    paper_and_summary = summary_result.to_input_list()

    critial_thinking_agent = Agent(
        name="Elementary particle physics postdoc",
        instructions=f"""{POSTDOC_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
You are attending a journal club meeting about a new paper. This is your first time to see the paper and listen to the summary of the paper. You are expected to be critical about the paper and the summary of the paper. You should ask questions about the paper. Even naive questions are welcome.""",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_POSTDOC_DEPLOYMENT_NAME,
            openai_client=postdoc_client,
        ),
    )

    print("Criticizing the summary of the paper...")
    critize_result = await Runner.run(
        critial_thinking_agent,
        input=f"""Provide your critical thinking about the following summary of the paper. You should provide questions about the summary of the paper. Be critical and skeptical about the summary of the paper. You are encouraged to ask not only scientific questions, but also naive questions, such as "It is not clear to me what the author means by X". You can also ask questions about the paper itself, such as "I don't understand why the author is interested in this topic.".
        You should also provide your own understanding of the paper and your own opinion about the paper.
Summary of the paper:
{summary}
        """)
    critize = critize_result.final_output
    paper_and_summary.append({"content": f"Feedback: {critize}", "role": "user"})

    help_agent = Agent(
        name="Elementary particle physics staff researcher",
        instructions=f"""{STAFF_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
You are attending a journal club meeting about a new paper. You have already checked the paper and have your own opinion about the paper. Although you are critical about the paper, you are expected to help the summary of the paper by the graduate student and answer the questions and criticism of the postdoc.""",
        #model=OpenAIChatCompletionsModel(
        #    model=AZURE_OPENAI_STAFF_DEPLOYMENT_NAME,
        #    openai_client=staff_client,
        #,
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME,
            openai_client=student_client,
        ),
    )

    print("Answering the questions and feedback...")
    help_result = await Runner.run(
        help_agent,
        input=paper_and_summary,
    )

    help = help_result.final_output

    translator = Agent(
        name="translator",
        instructions="You are a translator. You are going to translate the following text into Japanese.",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME,
            openai_client=student_client,
        ),
    )

    print("Translating into Japanese...")
    translation_result = await Runner.run(
        translator,
        input=f"""Please translate the following summary of the paper, questions and answers to the questions into Japanese. Never mind the length of the text. You should translate all the text I provide below.
Summary of the paper:
{summary}
Questions and feedback:
{critize}
Answers to the questions:
{help}
""",
    )

    translation = translation_result.final_output
    return translation

async def agent_orchestration(arxiv_mcp, slack_mcp):
    paper_pick_agent = Agent(
        name="Elementary particle physics graduate student",
        instructions=f"""{GRADUATE_STUDENT_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
You and your colleagues check all the latest papers appearing on arXiv everyday.
When you and your colleagues check the latest papers, you are not going to check too technical papers, but you are going to check papers that are interesting and have some new ideas. New experimental results are also interesting.
When you check the latest papers, you should choose all the interesting papers. It doesn't matter if the number of papers is very large, small or even zero. You choose the interesting papers based on the title and abstract of the papers. You should not check the whole paper.""",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME,
            openai_client=student_client,
        ),
        mcp_servers=[arxiv_mcp, slack_mcp],
        output_type=InterestingPapers,
    )

    print("Checking the latest papers on arXiv...")
    result = await Runner.run(
        paper_pick_agent,
        input=f"""Please check the latest papers on arXiv in hep-ph and hep-th and choose interesting papers.
        Please post the rough summary of the paper you choose on #chatbot-test channel on Slack in Japanese. The format of the summary is like:
        =====
        タイトル: {{title}}
        著者: {{authors}}
        {{summary}}
        {{url}}
        =====
        Repeat the above format for all the papers you choose.
        In the beginning of the message, please write "{datetime.now().strftime("%Y/%m/%d")}のおすすめ論文" to indicate the date of the message.
        Respond with the arXiv IDs of the interesting papers. The format of the arXiv IDs is like: 2501.00001""",
    )

    async def send_to_slack(arxiv_id):
        summary = await discuss_paper(arxiv_mcp, arxiv_id)
        slack_agent = Agent(
        name="Slack agent",
        instructions="You are an AI assistant. You are going to send the following message to Slack.",
        model=OpenAIChatCompletionsModel(
            model=AZURE_OPENAI_STUDENT_DEPLOYMENT_NAME,
            openai_client=student_client,
        ),
        mcp_servers=[slack_mcp],
        )
        print("Sending message to Slack...")
        await Runner.run(
            slack_agent,
            input=f"""Please send the following message to Slack channel #chatbot-test
You **must** send all information in the message. It is not allowed to send only part of the message. You MUST NOT truncate the information. Your message must include all the information I will provide.
You will be unable to see the message reply on the Slack channel. Therefore, you should not ask for any feedback or comments from the Slack channel.
First, you should post the summary of the whole message I provide below. Then, you should use the thread to post the more detailed summary of the paper and discussion, the comments and feedback.
You must attach the whole message I will provide in the following as an attachment file to the message.
The attachment file should be named "paper_summary.md", and the file should be in markdown format.
The attachment file should include the *whole message* I provide below. The readers never mind the length of the file. They only care whether the file includes all the information.
All the messages should be in Japanese.
Here is the summary and discussion about the paper: 
{summary}"""
        )

    #paper_submissions = [send_to_slack(arxiv_id, index) for index, arxiv_id in enumerate(result.final_output.arxiv_ids)]
    # collect all the results into a list of strings
    #await asyncio.gather(*paper_submissions)

    print(f"Collecting the results of {len(result.final_output.arxiv_ids)} papers...")
    # if I do it in parallel, it's too fast and the API rate limit is exceeded
    for arxiv_id in result.final_output.arxiv_ids:
        print(f"Checking the paper {arxiv_id}...")
        await send_to_slack(arxiv_id)
    print("All done.")

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
        await agent_orchestration(arxiv_mcp, slack_mcp)

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
