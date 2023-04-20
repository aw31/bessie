# Bessie

Bessie is a programming assistant chatbot that can help you with your programming tasks. It uses OpenAI's GPT or Anthropic's Claude models to generate responses based on your request and your codebase.

## Installation

To install Bessie using pip, run the following command:

```bash
pip install git+https://github.com/aw31/bessie.git
```

### API Keys

Bessie requires either the `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variable to be set, depending on whose models you want to use. You can set these environment variables in your shell or create a `.env` file in the root directory of your project with the following content:
```
OPENAI_API_KEY=sk-...      # to use OpenAI models
ANTHROPIC_API_KEY=sk-...   # to use Anthropic models
```

## Usage

After installing Bessie and setting up the environment variables, you can use the `bessie` command to interact with the chatbot:
```bash
bessie "Write a README with instructions to install using pip from https://github.com/aw31/bessie and set up environment variables" bessie/*
```

For more information on how to use Bessie, you can run `bessie --help`:
```
usage: bessie [-h] [--basedir BASEDIR] [--model MODEL] [--output OUTPUT] request patterns [patterns ...]

Bessie is a programming assistant chatbot

positional arguments:
  request            a programming request in natural language
  patterns           a list of globs of relevant files

options:
  -h, --help         show this help message and exit
  --basedir BASEDIR  base directory for file globs (default: .)
  --model MODEL      OpenAI chat model to use (default: gpt-4)
  --output OUTPUT    output .md file (default: bessie.md)
```
  
## Contributing

If you'd like to contribute to the development of Bessie, please feel free to submit a pull request or open an issue on the [GitHub repository](https://github.com/aw31/bessie).
