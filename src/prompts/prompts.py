from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from src.utils.ezr import rows,row

class SimpleChoice():


    def get_prompt_template(self, options : list):
        pref = ",".join(op[0] for op in options) 
        
        case = '''
        The Karina Moskolenkov case is a tragic story that unfolded in December 2002 in New Jersey. Karina Moskolenkov, a 17-year-old girl, was reported missing under mysterious circumstances. Her disappearance immediately drew concern, as there were no clear indications of where she might have gone or what might have happened to her. The situation grew more alarming when her body was discovered in a shallow grave near her home.

        The investigation into Karina's death quickly focused on her stepfather, Andrey Kakhrov. It was revealed that the household had underlying tensions, and Kakhrov had a history of abusive behavior. Evidence gathered during the investigation pointed to Kakhrov's involvement in Karina's murder, leading to his arrest and subsequent trial.

        Kakhrov was charged with Karina's murder and was eventually convicted and sentenced to life in prison. The case highlighted several critical issues, including the challenges law enforcement faces in missing persons cases, the role of domestic violence in such tragedies, and the importance of community and legal interventions in preventing similar outcomes.

        The Moskolenkov case also sparked conversations about the need for better protective measures for vulnerable individuals, particularly within families where domestic violence is present. It serves as a reminder of the complexities and dangers that can arise behind closed doors and the critical role of vigilance and support systems in safeguarding those at risk.
        '''
        messages = [
        {"role": "system", "content": "You are veteran crime analyst. You will be provided with a case study and a set of preference options to work with. Based on your thoughts pick the most useful one for the following case study."},
        {"role": "user", "content":  "case :" + case},
        {"role": "user", "content": "preference options:" + pref}
        ]

        return messages

