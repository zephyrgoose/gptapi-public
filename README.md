## GPT API Wrapper

Small utilities for calling the OpenAI API using YAML profiles.  Profiles define
the model, prompts and structured output schema so you can keep configuration
out of the code.

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `keys.yaml.example` to `keys.yaml` and add your OpenAI API key:

   ```yaml
   openai_api: "sk-..."
   ```

### Usage

- Run the simple example:

  ```bash
  python example.py
  ```

- Run the multi agent demo with a schedule file:

  ```bash
  python multiagent.py --schedule schedules/example-schema.yaml "Your prompt"
  ```

### Development

Run tests with `pytest` (requires the `pytest-cov` plugin) and optionally lint with `flake8` if available. The command below also checks that code coverage stays above 80%:

```bash
pytest --cov=gptapi_core --cov-report=term --cov-fail-under=80
flake8  # optional
```
