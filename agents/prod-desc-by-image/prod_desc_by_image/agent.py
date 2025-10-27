# --- Agent Definitions ---
import asyncio
import os
from pydantic import BaseModel, Field

from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.genai import types
from google.adk.runners import InMemoryRunner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.tool_context import ToolContext
from typing import AsyncGenerator, Optional
from google.adk.events import Event, EventActions
import httpx
import base64

# --- Constants ---
APP_NAME = "product_description_writer_app_v3" # New App Name
USER_ID = "dev_user_01"
SESSION_ID_BASE = "loop_exit_tool_session" # New Base Session ID
GEMINI_MODEL = "gemini-2.0-flash"
STATE_PRODUCT_GIVEN= "product_given"

# --- State Keys ---
STATE_PRODUCT_DESCRIPTION = "current_product_description"
STATE_CRITICISM = "criticism"
# Define the exact phrase the Critic should use to signal completion
COMPLETION_PHRASE = "No major issues found."

# Method to load image
def load_image(image_path: str) -> bytes:
    """
    Load an image from a file and return its bytes.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return image_bytes

# ----- Image ----
image_path = "./sourcream.png"
#image_bytes = load_image("./prod_desc_by_image/sourcream.png")
image_instruction = "Describe the product you are seeing in this image"

"""
content = types.Content(
        role="user",
        parts=[
            types.Part(text=image_instruction),
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
    )
"""

full_instruction=f"""You are a highly skilled E-commerce Copywriter specializing in persuasive product descriptions.
                    Your task is to generate a **well-structured, engaging, and benefit-driven product description** for an online store.

                    **Product Information:**
                    * **Product Name:** "{{current_product_description}}"
                    Thoroughly analyze the image to identify visual features, materials, design, and overall aesthetic. Integrate these visual details seamlessly into the description.
                    
                    **Structure Requirements (Aim for 100-200 words total):**

                    1.  **Compelling Opening (1-2 sentences):** Start with a hook that addresses a customer pain point or immediately highlights a primary benefit.
                    2.  **Key Features & Benefits (1-2 paragraphs OR 3-5 bullet points):**
                        * Detail the most important features.
                        * For each feature, explain the direct benefit to the customer ("what it does for them").
                        * Use strong, descriptive adjectives.
                    3.  **Ideal User/Scenario (1-2 sentences):** Clearly state who this product is perfect for, or describe a scenario where it excels.
                    4.  **Concluding Call-to-Value (1 sentence):** A final, persuasive statement that reinforces the product's value and encourages purchase.

                    **Tone:** Professional, enticing, slightly elevated, and customer-focused.
                    **Output Format:** Use Markdown (e.g., bolding, bullet points if requested for features) for readability.

                    Output *only* the complete product description text, adhering strictly to the structure and format requested. Do not add any introductory phrases (e.g., "Here is your description:"), explanations, or conversational filler.
                    """
# --- Tool Definition ---
def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}
# --- Structured Output Definition ----

class StructuredOutput(BaseModel):
    name: str = Field(description="Name of the product")
    weight: str = Field(description="Weight of the product")
    attributes: list[str] = Field(description="List of attributes of the product")
    manufacturer: str = Field(description="Name of the manufacturer")
    description: str = Field(description="Description of the product")


# STEP 0: Image Analyzer Agent (Runs ONCE at the beginning)
image_analyzer_agent = LlmAgent(
    name="ImageAnalyzerAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: Ask for a slightly more developed start
    instruction= image_instruction,
    description="Writes the initial description draft based on the image, aiming for some initial substance.",
    output_key=STATE_PRODUCT_DESCRIPTION,
    output_schema=StructuredOutput

)
# STEP 1: Product Descriptor Agent (Runs ONCE at the beginning)
product_description_writer_agent = LlmAgent(
    name="InitialProductDescriptionWriterAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: Ask for a slightly more developed start
    instruction= full_instruction,
    #instruction=f"""You are a Creative Writing Assistant tasked with writing the description of a product.
    #Write the *first draft* of a short product description (aim for 2-4 sentences).
    #Base the content *only* on the product given below. Try to introduce a specific element (like a feature, a benefit, or a scenario) to make it engaging.
    #Product Given: {{shampoo fresh}}
#
    #Output *only* the product description text. Do not add introductions or explanations.
#""",
    description="Writes the initial description draft based on the product, aiming for some initial substance.",
    output_key=STATE_PRODUCT_DESCRIPTION
)


# STEP 2a: Engagement Critic Agent (Inside the Engagement Refinement Loop)
engagement_critic_agent_in_loop = LlmAgent(
    name="EngagementCriticAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: More nuanced completion criteria, look for clear improvement paths.
    instruction=f"""You are a Constructive Critic AI reviewing a short product description draft (typically 2-6 sentences). Your goal is balanced feedback based on user engagement for the given product description focused on user engagement for the content.
        Key metrics to measure the product description:
        - Clarity and Readability
        - Completeness/Accuracy
        - Value Proposition & Benefits Focus
        - Tone of Voice & Brand Consistency
        - Uniqueness & Differentiation
        - Emotional Appeal/Storytelling
        - Call to Action (Implicit/Explicit)
        - Customer Reviews and Feedback
    **Product Description to Review:**
    ```
    {{current_product_description}}
    ```

    **Task:**
    Review the product description for clarity, engagement, and basic coherence according to the product given (if known).

    IF you identify 1-2 *clear and actionable* ways the product description could be improved to better capture the topic or enhance reader engagement:
    Provide feedback in bullet points, suggesting concrete improvements. Output *only* the critique text.

    ELSE IF the document is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
    Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

    Do not add explanations. Output only the critique OR the exact completion phrase.
""",
    description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
    output_key=STATE_CRITICISM
)


# STEP 2b: Refiner/Exiter Agent (Inside the Refinement Loop)
engagement_refiner_agent_in_loop = LlmAgent(
    name="EngagementRefinerAgent",
    model=GEMINI_MODEL,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are a Creative Product Description Writing Assistant refining a document based on feedback OR exiting the process.
    **Current Description:**
    ```
    {{current_product_description}}
    ```
    **Critique/Suggestions:**
    {{criticism}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "{COMPLETION_PHRASE}":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the 'Current Document'. Output *only* the refined document text.

    Do not add explanations. Either output the refined document OR call the exit_loop function.
""",
    description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key=STATE_PRODUCT_DESCRIPTION # Overwrites state['current_document'] with the refined version
)




# STEP 2: Engagement Refinement Loop Agent
engagement_refinement_loop = LoopAgent(
    name="EngagementRefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        engagement_critic_agent_in_loop,
        engagement_refiner_agent_in_loop,
    ],
    max_iterations=3 # Limit loops
)


# STEP 3a: SEO Critic Agent (Inside the SEO Refinement Loop)
seo_critic_agent_in_loop = LlmAgent(
    name="SEOCriticAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: More nuanced completion criteria, look for clear improvement paths.
    instruction=f"""You are a Constructive Critic AI reviewing a short product description draft (typically 2-6 sentences) for search engine effectiveness. Your goal is balanced feedback based on search engine optimization techniques for the given product description.

    **Product Description to Review:**
    ```
    {{current_product_description}}
    ```

    **Task:**
    Review the product description for search engine alignment according to the product given (if known).
    Consider the following:
    - **Keyword Relevance:** Does the description use keywords naturally related to the product name?
    - **Clarity & Conciseness:** Is the description easy to understand and to the point?
    - **Call to Action/Engagement:** Does it encourage a purchase or further interest?
    - **Readability:** Is it well-structured?

    IF you identify 1-2 *clear and actionable* ways the product description could be improved to improve search engine effectiveness: 
    Provide feedback in bullet points, suggesting concrete improvements. Output *only* the critique text.

    ELSE IF the document is coherent, addresses the topic adequately for its length, and has no glaring errors or obvious omissions:
    Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

    Do not add explanations. Output only the critique OR the exact completion phrase.
""",
    description="Reviews the current draft, providing critique if clear improvements are needed, otherwise signals completion.",
    output_key=STATE_CRITICISM
    )


# STEP 3b: SEO Refiner/Exiter Agent (Inside the SEO Refinement Loop)
seo_refiner_agent_in_loop = LlmAgent(
    name="SEORefinerAgent",
    model=GEMINI_MODEL,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are a Creative Product Description Writing Assistant focused on search engine optimization refining a document based on feedback OR exiting the process.
    **Current Description:**
    ```
    {{current_product_description}}
    ```
    **Critique/Suggestions:**
    {{criticism}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "{COMPLETION_PHRASE}":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the 'Current Document'. Output *only* the refined document text.

    Do not add explanations. Either output the refined document OR call the exit_loop function.
""",
    description="Refines the document based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key=STATE_PRODUCT_DESCRIPTION # Overwrites state['current_document'] with the refined version
    )

# STEP 3: SEO Refinement Loop Agent
seo_refinement_loop = LoopAgent(
    name="SEORefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        seo_critic_agent_in_loop,
        seo_refiner_agent_in_loop,
    ],
    max_iterations=3 # Limit loops
)


# STEP 4: Overall Sequential Pipeline
# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = SequentialAgent(
    name="ProductDescriptor",
    sub_agents=[
        image_analyzer_agent, # Generate product description based on image provided
        product_description_writer_agent, # Run first to create initial product description
        engagement_refinement_loop,       # Then run the engagement critique/refine loop
        seo_refinement_loop       # Then run the seo critique/refine loop
    ],
    description="Writes the **final** product description output exactly in bullet points."
)
