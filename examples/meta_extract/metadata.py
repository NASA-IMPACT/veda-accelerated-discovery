from typing import Literal
from pydantic import BaseModel, Field
from akd._base import InputSchema, OutputSchema
from akd.agents import LiteLLMInstructorBaseAgent, BaseAgentConfig
import asyncio

from stac_pydantic import Collection, Item, ItemCollection
from stac_pydantic.links import Relations
from stac_pydantic.shared import BBox, MimeTypes

from contents import pdf_content, header_content

class MetadataExtractorInputSchema(InputSchema):
  """
  Input schema for the Metadata Extractor Agent.
  """
  # pdf_content: str = Field(
  #   ...,
  #   description="The pdf content about the dataset."
  # )
  # header_content: str = Field(
  #   ...,
  #   description="The header content of the raw datafile"
  # )
  content: str = Field(
    ...,
    description="The content from where the metadata is to be extracted."
  )
  
class Cbbox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

class MetadataExtractorOutputSchema(OutputSchema):
  """
  Output schema for the Metadata Extractor agent after proper metadata are extracted
  from the input content.
  """
  title: str = Field(
    ...,
    description="The title of the dataset."
  ),
  description: str = Field(
    ...,
    description="The description about the dataset."
  ),
  bbox: Cbbox = Field(
    ...,
    description="The spatial bounding box of the dataset as defined in the pydantic schema of bbox"
  )
  
class MetadataExtractorAgent(
  LiteLLMInstructorBaseAgent[
    MetadataExtractorInputSchema,
    MetadataExtractorOutputSchema
  ]
):
  """
  Agent that extracts the metadata as mentioned in the output schema from the input content about dataset.
  """
  input_schema = MetadataExtractorInputSchema
  output_schema = MetadataExtractorOutputSchema
  
async def main():
  config = BaseAgentConfig(
    model_name="ollama/llama3:8b",
    base_url="http://localhost:11434",
  )
  
  agent = MetadataExtractorAgent(
    config=config
  )
  
  content = pdf_content+header_content
  print(">>>", content)
  inputSchema = MetadataExtractorInputSchema(content=content)
  output = await agent.arun(inputSchema)
  return output

if __name__ == "__main__":
  result = asyncio.run(main())
  print("result is:: ", result)
