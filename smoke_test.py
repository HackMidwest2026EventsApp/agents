from strands import Agent, tool
from strands.models.ollama import OllamaModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


model = OllamaModel(
    host="http://localhost:11434",
    model_id="qwen3:4b",
)

agent = Agent(model=model, tools=[add])
result = agent("Use the add tool to compute 2 + 3.")
print(result)