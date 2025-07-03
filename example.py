import logging
from gptapi_core import gptapi

def read_prompt_from_file(file_path):
    """
    Read the prompt text from a given file.
    
    Parameters:
    - file_path (str): The path to the input file containing the prompt.

    Returns:
    - str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        logging.error("The file %s does not exist.", file_path)
        raise SystemExit(f"Error: The file {file_path} was not found.")
    except Exception as e:
        logging.error("An error occurred while reading the file %s: %s", file_path, e)
        raise SystemExit(f"Error: Unable to read the file {file_path}.")

def main():
    """
    Main function to run the gptapi with the goalplanner profile, using input from a file.
    """
    try:
        # Define the profile name and input file path
        profile_name = 'cot'  # The YAML profile file name without the extension
        input_file_path = './input.txt'  # Path to the input file containing the prompt

        # Read the prompt from the input file
        prompt = read_prompt_from_file(input_file_path)

        # Run the gptapi function
        result = gptapi(profile_name, prompt)

        # Print the result
        print(result)

    except Exception as e:
        logging.error("An error occurred while running the GPT API: %s", e)
        print("An error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
