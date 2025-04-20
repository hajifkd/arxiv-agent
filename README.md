# arxiv-agent

# run
First, clone the repository:
```bash
git clone https://github.com/hajifkd/arxiv-agent.git
cd arxiv-agent
```

Then, prepare `.env` file with your OpenAI API key and Slack API key:
```bash
cat <<EOF > .env
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_API_BASE=https://xxx.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=xxx
AZURE_OPENAI_API_VERSION=xxx
SLACK_BOT_TOKEN=xxx
SLACK_TEAM_ID=xxx
SLACK_CHANNEL_NAME=xxx
EOF
```

Finally, run the application:
```bash
uv run python main.py
```

# TODO
- [ ] dockerize 