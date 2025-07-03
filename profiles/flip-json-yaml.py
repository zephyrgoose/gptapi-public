import json
import yaml

def detect_and_convert(input_string):
    try:
        # Try parsing as JSON
        data = json.loads(input_string)
        # Convert to YAML and return
        return yaml.dump(data, sort_keys=False, default_flow_style=False)
    except json.JSONDecodeError:
        pass

    try:
        # Try parsing as YAML
        data = yaml.safe_load(input_string)
        # Convert to JSON and return
        return json.dumps(data, indent=4)
    except yaml.YAMLError:
        pass

    # If neither JSON nor YAML, raise an error
    raise ValueError("Invalid input: not a valid JSON or YAML string.")

if __name__ == "__main__":
    print("Please enter your JSON or YAML input (end input with an empty line):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    
    input_string = "\n".join(lines)

    try:
        result = detect_and_convert(input_string)
        print(result)
    except ValueError as e:
        print(e)
