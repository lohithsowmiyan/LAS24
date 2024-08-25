from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts.prompts import SimpleChoice
#from graph import visualize 
from src.language.llms import unload_model
import warnings
import time

pool = [
    'preference',
    'Premortem Analysis',
    'Structured Self-Critique',
    'Starbusting',
    'Chronology, Timeline and Map',
    'Getting Started Checklist',
    'Key Assumption Check',
    'Devils Advocacy',
    'Force Field Analysis',
    'Deception Detection',
    'Chronologies and Timeliness',
    'Quadrant Hypothesis Generation',
    'Analysis of Completing Hypothesis',
    'SWOT',
    'Simple Hypothesis',
    'Multiple Hypothesis Generator',
    'Mind Map',
    'Analysis and Completing Hypothesis',
    'Structured Brainstorming',
    'Pros-Cons Faults and Fixes',
    'Classic Quadrant Crunching',
    'Red Hat Analysis',
    'Multiple Scenarios Generation',
    'Indicators Validator',
    'What if Analysis',
    'Foresight Quadrant Crunching',
    'Morphological Analysis',
    'Outside-in Thinking',
    'Simple Scenarios',
    'Decision Matrix',
]

def vanilla(args, save_results = False, model = None):
    warnings.filterwarnings("ignore")
    loaded_here = False
    if model == None:
        loaded_here = True
        (model, dir) =  load_model(args).get_pipeline()
    #random.seed(args.seed)
    records = []

    # def _tile(lst, curd2h, budget):
    #     num = adds(NUM(),lst)
    #     n=100
    #     print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
    #     sd=int(num.sd*n/2)
    #     mu=int(num.mu*n)
    #     print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")
    #     record = o(the = "result", N = len(lst), Mu = format(num.mu,".3f"), Sd = format(num.sd, ".3f"), Var = " "*(mu-sd) + "-"*sd + "+"*sd, Curd2h = format(curd2h, ".3f"), Budget = budget)
    #     records.append(record)

    def learner(pool: list, callBack=lambda x,y,z:x):
        """
        
        """
        # def _ranked(lst:rows, cur:row = None, budget:int = 0) -> rows:
        #     "Sort `lst` by distance to heaven. Called by `_smo1()`."
        #     lst = sorted(lst, key = lambda r:d2h(i,r))
        #     callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur), budget)
        #     # print(d2h of the best row)
        #     return lst

        def llm_guesser(todo: rows, done: rows) -> row:
            # cut = int(.5 + len(done) ** 0.5)
            # best = clone(i, done[:cut]).rows
            # rest = clone(i, done[cut:]).rows
            #best = [b[:len(i.cols.x)] for b in best]
            #rest = [r[:len(i.cols.x)] for r in rest]
            messages = SimpleChoice().get_prompt_template(todo)
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model.model.config.pad_token_id = model.model.config.eos_token_id
            outputs = model(prompt, max_new_tokens=256,  do_sample=True, temperature=0.5, top_p=0.9) #eos_token_id=terminators,
            btw(outputs[0]['generated_text']) #if args.intermediate else None
            for opt in todo:
                if opt[0] in outputs[0]['generated_text'][len(prompt) :]: 
                    return opt

            
        
        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            if len(done) >= 1: return
            top = llm_guesser(todo, done)
            todo.remove([top])
            btw(top)
            done += [top]
            return done

        i_sampled = DATA([[p] for p in pool]).rows
        return _smo1(i_sampled[args.label:], []) 

    if(save_results):
        results = learner(pool)
        #save_results_txt(model = args.model + "_" + args.llm, dataset = args.dataset, records =  records)
        #time.sleep(5)
        #visualize(dataset = args.dataset[args.dataset.rfind('/')+1:-4], show = 'All', save_fig= True, display = False)
    else: results = learner(pool)
    if loaded_here:
        unload_model(model, dir)
    return result


if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    vanilla(args)
    