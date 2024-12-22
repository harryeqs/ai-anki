import logging
import os

from camel.storages import Neo4jGraph
from camel.loaders import UnstructuredIO
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.configs import QwenConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

uio = UnstructuredIO()

from dotenv import load_dotenv
load_dotenv()

REFINE_SYSTEM_PROMPT = """"
    Objective:
        Generate a comprehensive and detailed summary of the provided input text.
        The summary should capture all key points, essential details, and significant
        information without omitting or altering the original meaning.

    Instructions:
        1. Read Thoroughly:
            Carefully read the entire input text to fully understand its content, context,
            and nuances.

        2. Identify Key Elements:
            Highlight and note all major themes, arguments, facts, data, and any critical
            information presented in the text.

        3. Maintain Accuracy:
            Ensure that all important information from the original text is accurately
            represented in the summary. Do not introduce any new information or personal
            interpretations.

        4. Structured Summary:
            - Introduction:
                Begin with a brief overview of the main topic or purpose of the original text.
            - Main Points:
                Elaborate on each key point or section of the input text in a clear and organized
                manner.
            - Conclusion:
                Summarize any final thoughts, conclusions, or implications presented in the original text.

        5. Clarity and Coherence:
            Write the summary in clear, concise language. Ensure that the summary flows logically
            and is easy to understand.

        6. Length:
            Aim for a summary that is approximately [specify desired length, e.g., 20% of the original
            text length] unless otherwise instructed. Adjust the length to maintain all essential
            information without unnecessary verbosity.

        7. Preserve Original Meaning:
            Do not alter the intent, tone, or meaning of the original text. The summary should
            reflect the author's original message accurately.

        8. Review:
            After drafting the summary, review it to ensure completeness and accuracy. Verify that
            no critical information has been omitted and that all key points are adequately covered.

    Input Text:
        [Insert the text to be summarized here]

    Output:
        [A detailed summary of the input text, adhering to the above instructions]

    Example Usage:
        Input Text:
            [Provide sample text]
        Output:
            [Provide corresponding detailed summary]

    Notes:
        - If the input text contains complex terminology or concepts, ensure that they are explained
          clearly in the summary.
        - Avoid personal opinions or subjective statements; the summary should remain objective.
        - If the text includes data or statistics, include the most relevant figures to support the key points.
    """

class KGGenerator:
    def __init__(self):
        self.n4j = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )   
        self.refine_agent = ChatAgent(
            system_message=REFINE_SYSTEM_PROMPT,
            model = ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_TURBO,
            model_config_dict=QwenConfig(temperature=0.2).as_dict(),
        )
    )
        self.kg_agent = KnowledgeGraphAgent(
            model = ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_TURBO,
            model_config_dict=QwenConfig(temperature=0.2).as_dict(),
        )
    )

    @staticmethod
    def _load_txt_files(save_dir: str):
        try:
            # List all files in the directory
            all_files = os.listdir(save_dir)
            # Filter out only .txt files
            txt_files = [file for file in all_files if file.endswith('.txt')]

            if not txt_files:
                logger.error(f"No files found in {save_dir}")
                return {}
            
            txt_contents = {}
            for filename in txt_files:
                file_path = os.path.join(save_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    txt_contents[filename] = file.read()
        
            return txt_contents
        except FileNotFoundError:
            logger.error(f"The directory {save_dir} does not exist.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def generate_kg(self, save_dir: str):
        txt_contents = self._load_txt_files(save_dir)
        for filename, content in txt_contents.items():
            # Create an element from the provided text
            refined_response= self.refine_agent.step(content)
            refined_content = refined_response.msgs[0].content
            element_example = uio.create_element_from_text(text=refined_content, element_id="001")

            # Extract nodes and relationships using the Knowledge Graph Agent
            logger.info(f"Extracting nodes and relationships from {filename}")
            graph_elements = self.kg_agent.run(element_example, parse_graph_elements=True)
            logger.info(f"Extracted {graph_elements} and relationships from {filename}")

            # Add the extracted graph elements to the Neo4j database
            try:
                self.n4j.add_graph_elements(graph_elements=[graph_elements])
                logger.info(f"Added {filename.replace('.txt', '')} to the Neo4j database")
            except Exception as e:
                logger.error(f"An error occurred while adding {filename.replace('.txt', '')} to the Neo4j database: {e}")

if __name__ == "__main__":
    kg_generator = KGGenerator()
    kg_generator.generate_kg("uploads")