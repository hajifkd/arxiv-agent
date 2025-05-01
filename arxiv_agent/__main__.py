from agents import OpenAIChatCompletionsModel, Agent, Runner
from agents.mcp import MCPServerStdio
import asyncio
import os
from datetime import datetime
from pydantic import BaseModel
from .connection import (
    fast_client,
    balanced_client,
    deep_client,
)
from .arxiv import (
    today_arxiv,
    download_arxiv_paper
)
from .slack import (
    get_channel_id,
    post_messages_as_thread,
    post_message,
)


class InterestingPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    reason_en: str
    reason_ja: str
    primary_category: str


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
            model=AZURE_OPENAI_FAST_DEPLOYMENT_NAME,
            openai_client=fast_client,
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
            model=AZURE_OPENAI_BALANCED_DEPLOYMENT_NAME,
            openai_client=balanced_client,
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
            model=AZURE_OPENAI_FAST_DEPLOYMENT_NAME,
            openai_client=fast_client,
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
            model=AZURE_OPENAI_FAST_DEPLOYMENT_NAME,
            openai_client=fast_client,
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

async def agent_orchestration():
    paper_pick_agent = Agent(
        name="Elementary particle physics graduate student",
        instructions=f"""{GRADUATE_STUDENT_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
You and your colleagues check all the latest papers appearing on arXiv everyday.
When you and your colleagues check the latest papers, you are not going to check too technical papers, but you are going to check papers that are interesting and have some new ideas. New experimental results are also interesting. You choose the interesting papers based on the title, abstract and the authors of the papers.""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=fast_client,
        ),
        output_type=list[InterestingPaper],
    )

    print("Fetching the latest papers from arXiv...")
    today_papers = today_arxiv("hep-ph") 

    print("Checking the latest papers on arXiv...")
    result = await Runner.run(
        paper_pick_agent,
        input=f"""from the following list of papers, please choose the interesting papers:
{today_papers}""",)

    print("The following papers are chosen as interesting papers:")
    print(result.final_output)
    return

async def main_async():
    await agent_orchestration()

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
