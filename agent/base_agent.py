from typing import List, Set, TypedDict, Literal
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END 
from langchain_core.messages import HumanMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv() # at the top and not in main because llm called before the 
    

class Calculation(BaseModel):
    operation: Literal["add","subtract", "multiply", "divide", "power"]
    args: List[str|float|int] =  Field(min_length = 2)

class CalculationPlan(BaseModel):
    operations: List[Calculation]

class CalucationState(TypedDict):
    question: str 
    table:str
    plan: CalculationPlan
    results: list[float]
    final_answer: str
    

SYSTEM_PROMPT = """
You are a financial calculation planner.

You will receive:
1. A financial question.
2. A financial table.

Your task is to generate a sequence of mathematical
operations needed to answer the question.

Available operations:
- add(a, b): Add two numbers.
- subtract(a, b): Subtract b from a.
- multiply(a, b): Multiply two numbers.
- divide(a, b): Divide a by b.
- power(a, b): Raise a to the power of b.

Rules:
1. Extract numerical values from the provided table.
2. Select only operations needed to answer the question.
3. Do not perform calculations yourself.
4. Each operation must contain exactly two arguments.
5. To reuse a previous result, reference it as "step_1",
   "step_2", etc.
6. Reference only steps that occur earlier in the plan.
7. Preserve the correct order of operations.
8. Do not invent values missing from the table.
9. If the question cannot be answered, return an empty
   list of operations.

Example:
Question: What is the percentage increase from 100 to 120?

Operations:
- subtract(120, 100)
- divide(step_1, 100)
- multiply(step_2, 100)
"""

#model from llm
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
#defining nodes:

#structured output
structured_llm = llm.with_structured_output(CalculationPlan)
    

def plan_calculation(state:CalucationState):
    #reading input
    question = state["question"]
    table = state["table"]
    
    message = f"""
    Financial table:
    {table}

    Question:
    {question}
    """

    #Systemprompt = how the system should behave and HumanMessage = input/information the system needs to complete the task
    plan = structured_llm.invoke([SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content=message)])
    
    #in langrapha, a node normally returns a dictionary that contains the state field it wants to updated, so only return what we want to change 
    return {"plan": plan}
    
    
    
def calculate(state:CalucationState):
    step_results = {} #dict to store result from each step. 
    plan = state["plan"]
    for i, operation in enumerate(plan.operations, start=1):
        func = operation.operation 
        arg1,arg2 = operation.args
        
        for i, operation in enumerate(plan.operations, start=1):
            func = operation.operation
            arg1, arg2 = operation.args

            arg1 = resolve_arg(arg1, step_results)
            arg2 = resolve_arg(arg2, step_results)
        
            step_results[i] = perform_operations(func,arg1=arg1,arg2=arg2)
    
    return {"results": list(step_results.values()), "final_answer": step_results[len(plan.operations)]}
        

def resolve_arg(arg, step_results):

    if not isinstance(arg, str):
        return arg

    if not arg.startswith("step_"):
        raise ValueError(f"Invalid reference: {arg}")

    index_num = arg.find("_") + 1
    number_text = arg[index_num:]

    if not number_text.isdecimal():
        raise ValueError(f"Invalid reference: {arg}")

    step_num = int(number_text)

    if step_num not in step_results:
        raise ValueError(f"Step {step_num} not available")

    return step_results[step_num]          
            
      
    
def perform_operations(func:str, arg1:float,arg2:float):
    
    if func == "add":
        return arg1 + arg2
    
    if func == "subtract":
        return arg1 - arg2
    
    if func == "divide":
        if arg2 == 0:
            raise ValueError("Cannot divide by zero")
        return arg1/arg2
    
    if func == "multiply":
        return arg1 * arg2
    
    if func == "power":
        return arg1 ** arg2

if __name__ == "__main__":
    vanilla_financial_agent = StateGraph(CalucationState)
    vanilla_financial_agent.add_node("plan_calculation",plan_calculation)
    vanilla_financial_agent.add_node("calculate", calculate)
    
    vanilla_financial_agent.add_edge(START,"plan_calculation")
    vanilla_financial_agent.add_edge("plan_calculation", "calculate")
    vanilla_financial_agent.add_edge("calculate", END)
    
    compiled_graph = vanilla_financial_agent.compile()
    
    question = "what was the percentage change in net sales from 2000 to 2001?"
    table =  """| Metric | 2002 | 2001 | 2000 |
                |---|---:|---:|---:|
                | Net Sales | $5,742 | $5,363 | $7,983 |
                | Cost of Sales | 4,139 | 4,128 | 5,817 |
                | Gross Margin | $1,603 | $1,235 | $2,166 |
                | Gross Margin Percentage | 28% | 23% | 27% |"""
    
    
    test_state = {
            "question" : question,
            "table": table,
            "plan": None,
            "results": None,
            "final_answer":None
            
        }
    final_state = compiled_graph.invoke(test_state)
    
    print("Plan")
    for i, op in enumerate(final_state["plan"].operations,start=1):
        print(f"  step_{i}: {op.operation}({', '.join(map(str, op.args))})")
    
    # 2. The result of each step
    print("Step results:", final_state["results"])

    # 3. The final answer
    print("Final answer:", final_state["final_answer"])
        
    
    
    
    
    
    