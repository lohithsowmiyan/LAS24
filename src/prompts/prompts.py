from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows,row

class SimpleChoice():

    def __init__(self):

    def get_prompt_template(self, options : list):

        pref = ",".join(op[0] for op in options) 

        messages = [
        {"role": "system", "content": "You are veteran crime analyst. You will be provided with a case study and a set of preference options to work with. Based on your thoughts pick the most useful one for the following case study."},
        {"role": "user", "content":  "case :" case},
        {"role": "user", "content": "preference options" :pref}
        ]

        return messages

