from agents import OpenAIChatCompletionsModel, Agent, Runner
import asyncio
import os
from datetime import datetime
from pydantic import BaseModel
import logging
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

logging.basicConfig(level=logging.INFO)

class InterestingPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    reason_en: str
    reason_ja: str
    primary_category: str

class PaperDiscussion(BaseModel):
    detailed_summary: str
    criticize: str
    answer: str


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

JAPANESE_TRANSLATION_PROMPT = """When you use Japanese, you need not to translate too technical terms into Japanese. You can use English terms in Japanese text. You should not use Katakana to translate technical terms into Japanese by force."""

async def discuss_paper(arxiv_id) -> PaperDiscussion:
    paper = download_arxiv_paper(arxiv_id)
    paper_summarize_agent = Agent(
        name="Elementary particle physics graduate student",
        instructions=f"""{GRADUATE_STUDENT_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=fast_client,
        )
    )
    logging.info(f"Summarizing {arxiv_id}...")
    summary_result = await Runner.run(
        paper_summarize_agent,
        input=f"""Please read the following paper and tell me the details of the paper. You should clarify what the paper is about, what the main motivation of the paper is, what the main results of the paper are, and what the main conclusions of the paper are. You should also clarify what is the main new idea of the paper and stress the importance of the paper. When you use some technical terms, such as some ideas proposed by some people, you should clarify what the terms mean. The content of the paper is following:
{paper}""",)
    summary = summary_result.final_output
    paper_and_summary = summary_result.to_input_list()

    critial_thinking_agent = Agent(
        name="Elementary particle physics postdoc",
        instructions=f"""{POSTDOC_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=deep_client,
        ),
    )

    logging.info("Criticizing the summary of the paper...")
    criticize_result = await Runner.run(
        critial_thinking_agent,
        input=f"""You are attending a journal club meeting about a new paper. This is your first time to see the paper and listen to the summary of the paper. Provide your critical thinking about the following summary of the paper. You should provide questions about the summary of the paper. Be critical and skeptical about the content and claims of the paper. You are encouraged to ask not only scientific questions, but also naive questions, such as "It is not clear to me what the author means by X". You can also ask questions about the paper itself, such as "I don't understand why the author is interested in this topic.". You should start from elementary and basic questions, and then move to more advanced questions so that other audiences should understand the content better. You should also provide your own understanding of the paper and your own opinion about the paper.
Summary of the paper:
{summary}
        """)
    criticize = criticize_result.final_output
    paper_and_summary.append({"content": f"Criticize: {criticize}", "role": "user"})

    answer_agent = Agent(
        name="Elementary particle physics staff researcher",
        instructions=f"""{STAFF_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
You are attending a journal club meeting about a new paper. You have already checked the paper and have your own opinion about the paper. Although you are critical about the paper, you are expected to help the summary of the paper by the graduate student and answer the questions and criticism of the postdoc. Provide constructive feedback to improve the understanding of the paper. Provide your own opinion and understanding of the paper briefly based on the discussion.""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=fast_client,
        ),
    )

    logging.info("Answering the questions and feedback...")
    answer_result = await Runner.run(
        answer_agent,
        input=paper_and_summary,
    )

    answer = answer_result.final_output

    return PaperDiscussion(
        detailed_summary=summary,
        criticize=criticize,
        answer=answer,
    )

async def translate_paper_discussion(paper_discussion: PaperDiscussion) -> PaperDiscussion:
    translator = Agent(
        name="translator",
        instructions=f"""{JAPANESE_TRANSLATION_PROMPT}
You are a faithful and professional translator. You are going to translate the following text into Japanese.""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=fast_client,
        ),
        output_type=PaperDiscussion
    )

    logging.info("Translating into Japanese...")
    translation_result = await Runner.run(
        translator,
        input=f"""Please translate the following summary of the paper, criticize and answers to the criticize into Japanese. Never mind the length of the text. You must translate all the text I provide below. You must not omit any information.
Detailed summary of the paper:
{paper_discussion.detailed_summary}
Criticize:
{paper_discussion.criticize}
Answers to the criticize:
{paper_discussion.answer}
""",
    )

    translation = translation_result.final_output
    return translation

async def pick_interesting_papers(category: str) -> list[InterestingPaper]:
    paper_pick_agent = Agent(
        name="Elementary particle physics graduate student",
        instructions=f"""{GRADUATE_STUDENT_PROMPT}
{RESEARCHER_PROMPT}
{INTEREST_PAPER_PROMPT}
{JAPANESE_TRANSLATION_PROMPT}
You and your colleagues check all the latest papers appearing on arXiv everyday.
When you and your colleagues check the latest papers, you are not going to check too technical papers, but you are going to check papers that are interesting and have some new ideas. New experimental results are also interesting. You choose the interesting papers based on the title, abstract and the authors of the papers.""",
        model=OpenAIChatCompletionsModel(
            model='',
            openai_client=balanced_client,
        ),
        output_type=list[InterestingPaper],
    )

    logging.info("Fetching the latest papers from arXiv...")
    today_papers = today_arxiv(category)

    logging.info("Evaluating papers...")
    result = await Runner.run(
        paper_pick_agent,
        input=f"""from the following list of papers, please choose the interesting papers:
{today_papers}""",)

    logging.info("Following papers are interesting:")
    for paper in result.final_output:
        logging.info(f"{paper.arxiv_id}: {paper.title}")
        logging.info(f"Reason (Japanese): {paper.reason_ja}")
        
    return result.final_output

async def main_async():
    slack_channel_name = os.getenv("SLACK_CHANNEL_NAME")
    slack_channel_id = await get_channel_id(slack_channel_name)
    if slack_channel_id is None:
        logging.error(f"Channel {slack_channel_name} not found.")
        return
    papers = await pick_interesting_papers("hep-ph")

    await post_message(slack_channel_id, f"""Interesting papers of {datetime.now().strftime('%Y-%m-%d')} :tada:
{datetime.now().strftime('%Y年%m月%d日')}の注目論文 :tada:
(I'm sorry if your paper was missed! It's still under development.)""")
    
    for paper in papers:
        try:
            en = await discuss_paper(paper.arxiv_id)
            ja = await translate_paper_discussion(en)
        except Exception as e:
            logging.error(f"Error processing {paper.arxiv_id}: {e}")
            await post_message(slack_channel_id, f"Error processing {paper.arxiv_id}: {e}")
            continue
        header = f"""{paper.title} ({paper.primary_category})
{", ".join(paper.authors) if len(paper.authors) < 4 else f"{paper.authors[0]} et al."}
https://arxiv.org/abs/{paper.arxiv_id}
- {paper.reason_en}
- {paper.reason_ja}"""
        texts = [
            "English follows Japanese.",
            f"Summary (Japanese):\n{ja.detailed_summary}",
            f"Criticize (Japanese):\n{ja.criticize}",
            f"Answer (Japanese):\n{ja.answer}",
            f"Summary:\n{en.detailed_summary}",
            f"Criticize:\n{en.criticize}",
            f"Answer:\n{en.answer}",
        ]
        await post_messages_as_thread(slack_channel_id, header, texts)

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
