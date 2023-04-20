import argparse
import pprint
from pathlib import Path

from bessie.backends import AnthropicChat, DummyChat, OpenAIChat
from bessie.settings import jinja_env
from bessie.wrappers import ChatWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Bessie is a programming assistant chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("request", help="a programming request in natural language")
    parser.add_argument("patterns", nargs="+", help="a list of globs of relevant files")
    parser.add_argument("--basedir", default=".", help="base directory for file globs")
    parser.add_argument("--model", default="gpt-4", help="OpenAI chat model to use")
    parser.add_argument("--output", default="bessie.md", help="output .md file")
    args = parser.parse_args()

    basedir = Path(args.basedir)
    files = {}
    for pattern in args.patterns:
        for path in basedir.glob(pattern):
            if path.is_file():
                with open(path, "r") as f:
                    files[path] = f.read()

    if args.model == "dummy":
        backend = DummyChat()
    elif "gpt" in args.model:
        backend = OpenAIChat(args.model, temperature=0, max_tokens=2000)
    elif "claude" in args.model:
        backend = AnthropicChat(args.model, temperature=0, max_tokens=2000)
    else:
        raise ValueError(f"Unknown model {args.model}")

    wrapper = ChatWrapper(system_message="You are a helpful programming assistant.")
    prompt = jinja_env.get_template("bessie.jinja").render(
        request=args.request, files=files
    )
    with open(args.output, "w") as f:
        print(f"\033[1mPrompt:\033[0m\n{prompt}")
        f.write(f"## Args\n```\n{pprint.pformat(vars(args))}\n```\n")
        while prompt:
            response = wrapper.run(backend, prompt)
            print(f"\n\033[1mBessie:\033[0m\n{response}\n")
            f.write(f"## Bessie\n{response}\n")
            prompt = input("\033[1mYou:\033[0m\n")
            f.write(f"## You\n{prompt}\n")

    print(f"Transcript written to {args.output}")


if __name__ == "__main__":
    main()
