import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import system_prompt


def main():
    load_dotenv()
    
    args = sys.argv[1:]

    if not args:
        print("AI Code Assistant")
        print('\nUsage: python main.py "your prompt here"')
        print('Example: python main.py "How do I build a calculator app?"')
        sys.exit(1)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    verbose = "--verbose" in args
    clean_args = [arg for arg in args if not arg.startswith("--")]

    user_prompt = " ".join(clean_args)

    if verbose:
        print(f"User prompt: {user_prompt}")

    messages = [
            types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    generate_content(client, messages, verbose=verbose)

def generate_content(client, messages, verbose=False):
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    print("Response:")
    print(response.text)
    
    if verbose:
        print(f'Prompt tokens: {response.usage_metadata.prompt_token_count}')
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

if __name__ == "__main__":
    main()
