import argparse
import glob

from bessie.backends import Message, OpenAIChat, Request
from bessie.settings import jinja_env


def main():
    parser = argparse.ArgumentParser(description="Bessie is a programming assistant")
    parser.add_argument("request", help="A programming request in natural language")
    parser.add_argument("patterns", nargs="+", help="A list of globs of relevant files")
    parser.add_argument("--model", default="gpt-4", help="OpenAI chat model to use")
    args = parser.parse_args()

    files = {}
    for pattern in args.patterns:
        for file in glob.glob(pattern):
            with open(file, "r") as f:
                files[file] = f.read()

    prompt = jinja_env.get_template("bessie.jinja").render(
        request=args.request, files=files
    )
    print(f"Prompt:\n{prompt}")

    messages = [
        Message("system", "You are a helpful programming assistant."),
        Message("environment", prompt),
    ]
    backend = OpenAIChat(args.model, temperature=0, max_tokens=2000)
    response = backend.run(Request(messages))
    print(f"Response:\n{response}")

    with open("bessie.md", "w") as f:
        f.write(response)


if __name__ == "__main__":
    main()
