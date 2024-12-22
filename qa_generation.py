import json
from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_LONG ,
    model_config_dict=QwenConfig(temperature=0.8).as_dict(),
)

# Define system message
sys_msg = """你是一个善于对用户给出的内容详细思考过后，一步一步去生成高质量的问题答案对的助手，生成的格式如下


{
    "What is the coefficient of $x^2y^6$ in the expansion of $\\left(\\frac{3}{5}x-\\frac{y}{2}\\right)^8$?  Express your answer as a common fraction": "\\frac{63}{400}",
    "how many r in strawberry?": "3"
}


"""

# Set agent
camel_agent = ChatAgent(system_message=sys_msg, model=model)

with open("concatenated.txt", "r") as f:
    user_msg = f.read()

# Get response information
response = camel_agent.step(user_msg)
json.dump(response.msgs[0].content, open("qa_generation.json", "w"))