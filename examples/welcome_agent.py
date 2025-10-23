"""
A welcome agent. Welcomes whoever is visiting the site.
"""
import asyncio
from pydantic import Field

from akd.agents._base import LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema

class WelcomeAgentInputSchema(InputSchema):
  """
  Input Schema that takes in the query with the person name
  """
  query: str = Field(
    ...,
    description="Ask the name of visitor"
  )
  
class WelcomeAgentOutputSchema(OutputSchema):
  """
  Generate a Welcome message from the given query
  """
  welcome_message: str = Field(..., description="message")
  

class WelcomeAgent(LiteLLMInstructorBaseAgent):
  input_schema = WelcomeAgentInputSchema
  output_schema = WelcomeAgentOutputSchema

async def main():
  config = BaseAgentConfig(
    model_name="ollama/qwen2:7b",  # Just the model name
    base_url="http://localhost:11434",
    
    # system_prompt="You're a haiku master.",
    stateless=True,
    max_tokens=1000,
    trim_ratio=0.9,
    debug=True,
    enable_trimming=True,
    # api_key="lol",
    input_hints=True,
    temperature=1,
    description="A haiku agent for the existence\n\n",
  )
  
  agent = WelcomeAgent(config=config, debug=True)
  print(f"Agent Model: {agent.model_name}")
  print(f"Description: {agent.description}")
  print("-" * 50)
  
  test_queries = ["sanjog", "ram", "shyam"]
  
  for query in test_queries:
    print(f"\n {query} Entered")
    input = WelcomeAgentInputSchema(query=query)
    output = await agent.arun(input)
    print(f"Welcome message:\n{output.welcome_message}\n")
    
if __name__ == "__main__":
    asyncio.run(main())
