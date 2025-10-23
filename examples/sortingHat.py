from typing import Literal
from pydantic import Field
from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent, BaseAgentConfig
import asyncio


class SortingHatAgentInputSchema(InputSchema):
    """
    Input schema for the Sorting Hat Agent.
    """

    about_user: str = Field(
        ..., 
        description="User's description of themselves, including their traits, values, and characteristics"
    )


class SortingHatAgentOutputSchema(OutputSchema):
    """
    Output schema for the Sorting Hat's house assignment.

    This schema represents the Sorting Hat's decision on which Hogwarts house
    the user belongs to based on their described traits and values.
    """

    house: Literal["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"] = Field(
        ..., 
        description="The Hogwarts house assigned to the user"
    )
    reasoning: str = Field(
        ..., 
        # description="Structured json output in the format: {message: 'sth', reason: 'sth'} with detailed explanation for why the user was sorted into this house, referencing their traits and house values"
        description="output with detailed explanation for why the user was sorted into this house, referencing their traits and house values"
    )


class SortingHatAgent(
    LiteLLMInstructorBaseAgent[
        SortingHatAgentInputSchema, SortingHatAgentOutputSchema
    ]
):
    """
    Agent that acts as the Hogwarts Sorting Hat, analyzing a user's self-description
    and assigning them to one of the four houses based on their traits and values.

    The agent considers:
    - Gryffindor: Bravery, courage, daring, chivalry
    - Hufflepuff: Loyalty, hard work, patience, fairness
    - Ravenclaw: Intelligence, wisdom, creativity, learning
    - Slytherin: Ambition, cunning, resourcefulness, determination
    """

    input_schema = SortingHatAgentInputSchema
    output_schema = SortingHatAgentOutputSchema

async def main():
    config = BaseAgentConfig(
      model_name="ollama/qwen2:7b",  # Just the model name
      base_url="http://localhost:11434", 
      stateless=True,
      max_tokens=1000,
      trim_ratio=0.9,
      debug=True,
      enable_trimming=True,
      input_hints=True,
      temperature=1,
      description="A haiku agent for the existence\n\n",
    )

    agent = SortingHatAgent(
      config=config
    )
    input_schema = SortingHatAgentInputSchema(about_user="I am a brave and courageous wizard")
    return await agent.arun(input_schema)

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)
