"""
Minimal Haiku Agent Implementation
Demonstrates AKD agent architecture with input/output schemas
"""
import asyncio
from pydantic import Field

from akd.agents._base import LiteLLMInstructorBaseAgent,BaseAgentConfig
from akd._base import InputSchema, OutputSchema


# 1. Define Input Schema
class HaikuAgentInputSchema(InputSchema):
    """Input schema that takes in query"""

    query: str = Field(
        ...,
        description="Query to the haiku agent.",
    )

    topic_steer: str = Field(
        default="5-7-5 format",
        description="Steer in 5-7-5 syllable format. Use Sanskrit words in the middle line.",
    )


# 2. Define Output Schema
class HaikuAgentOutputSchema(OutputSchema):
    """Generate a haiku from given query"""

    haiku: str = Field(..., description="Haiku content")


# 3. Create Agent Class
class HaikuAgent(LiteLLMInstructorBaseAgent):
    input_schema = HaikuAgentInputSchema
    output_schema = HaikuAgentOutputSchema


# 4. Usage Example
async def main():
    # Configure the agent
    # config = BaseAgentConfig(
    #     stateless=True,
    #     max_tokens=1000,
    #     temperature=1.0,
    #     debug=True,
    #     description="A haiku agent for generating poetic responses",
    # )
    
    config = BaseAgentConfig(
      model_name="ollama/qwen3:4b",  # Just the model name
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

    # Initialize agent
    agent = HaikuAgent(config, debug=True)

    print(f"Agent Model: {agent.model_name}")
    print(f"Description: {agent.description}")
    print("-" * 50)

    # Run the agent
    test_queries = ["love", "mountains", "code"]

    for query in test_queries:
        print(f"\nQuery: {query}")
        output = await agent.arun(HaikuAgentInputSchema(query=query))
        print(f"Haiku:\n{output.haiku}\n")


if __name__ == "__main__":
    asyncio.run(main())
