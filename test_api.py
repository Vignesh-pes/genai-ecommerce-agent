import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

print("Attempting to configure API key...")
try:
    # Get the key from the environment
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file.")
    else:
        genai.configure(api_key=api_key)
        print("API key configured.")

        print("\nAttempting to list models...")
        # List the models your key has access to
        for m in genai.list_models():
            print(f"Model found: {m.name}")

        print("\nSUCCESS! Your API key is working and has access to the models above.")

except Exception as e:
    print("\n--- TEST FAILED ---")
    print(f"An error occurred: {e}")
    print("\nThis is likely an API key or permissions problem.")