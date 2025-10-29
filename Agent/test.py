from openai import  OpenAI
from datetime import datetime


# 初始化客户端和模型
client = OpenAI(
    api_key="sk-hcsgmrfghoughqwescijrakrrtdhkumgxiiwyasiavilovug",
    base_url="https://api.siliconflow.cn/v1"
)

model_name = "Qwen/Qwen2.5-32B-Instruct"

# 定义工具函数
def get_current_datetime() -> str:
    '''
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    '''
    get_current_datetime = datetime.now()
    formatted_datetime = get_current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

# 转成 JSON Scheme格式

def function_to_json(func) -> dict:
    pass
    return{
        'type': 'function',
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type":"object",
                "properties":parameters,
                "required":required,
            },
        },
    }

# 构造Agent类
