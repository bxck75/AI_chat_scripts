import os
import uuid
#
from typing import Annotated, TypedDict, List,Optional,Any,Mapping
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
import json
from langchain_core.runnables.base import Runnable
from hugging_chat_wrapper import HuggingChatWrapper,HuggingFaceEmbeddings,ConversationManager
from langchain_core.language_models.llms import LLM
from rich import print as rp
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

system_prompt="""You are a helpful AI assistant that generates Python code. 
Response should be in JSON. Please provide a solution to the following task:"""
model=3
email, password, cookie_folder=os.getenv('EMAIL'), os.getenv('PASSWD'), 'cookies'
wrapper = HuggingChatWrapper('langgraph_experimental')
chatbot = ConversationManager(
                email, 
                password, 
                cookie_folder, 
                system_prompt, 
                model
            ).chatbot

class HuggingChatLLM(LLM):
    project_name: str = Field(default="AgenticDashboard")
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    model_id: int = Field(default=0)
    
    _chatbot: Optional[HuggingChatWrapper] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chatbot = chatbot
       # self._chatbot.set_bot(system_prompt=self.system_prompt, model_id=self.model_id)

    @property
    def _llm_type(self) -> str:
        return "huggingchat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        response = self._chatbot.chat(prompt)
        if stop:
            for stop_sequence in stop:
                if stop_sequence in response:
                    response = response[:response.index(stop_sequence)]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "project_name": self.project_name,
            "system_prompt": self.system_prompt,
            "model_id": self.model_id,
        }

class RunnableChatBot(Runnable):
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def invoke(self, input, config=None, **kwargs):
        #rp(f"input:{input}")
        out = self.chatbot.chat(input)
        return out.wait_until_done()

'''Runnable llm'''
rllm = RunnableChatBot(chatbot=chatbot)

# Ensure that API key for Hugging Face is set
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

class CodeSchema(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class GraphState(TypedDict):
    error: str
    messages: Annotated[List[AnyMessage], add_messages]
    generation: CodeSchema
    iterations: int

class LLMCodeGen:
    def __init__(self, max_iterations: int = 10):
        self.rllm = rllm
        self.max_iterations = max_iterations
        self.memory = MemorySaver()
        self.builder = StateGraph(GraphState)
        self._initialize_graph()

    def _initialize_graph(self):
        self.builder.add_node("generate", self.generate)
        self.builder.add_node("check_code", self.code_check)
        self.builder.add_edge(START, "generate")
        self.builder.add_edge("generate", "check_code")
        self.builder.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {"end": END, "generate": "generate"}
        )
        self.graph = self.builder.compile(checkpointer=self.memory)

    def generate(self, state: GraphState):
        messages = state["messages"]
        iterations = state["iterations"]

        print("---GENERATING CODE SOLUTION---")
        last_message = messages[-1].content if messages else ""
        prompt = f"""
        You are a helpful AI assistant that generates Python code. 
        Please provide a solution to the following task:

        {last_message}
        You response must be separated in 4 parts prefix,imports, the code and a docstring.
        Your response must be in JSON format with the following structure:
        {{
            "prefix": "Description of the problem and approach",
            "imports": "Code block import statements",
            "code": "Code block not including import statements"
            "docstring": "Explain the changes and usage"
        }}
        """
        # set the rllm as response generator
        response = rllm.invoke(input=prompt)

        try:
            code_solution = CodeSchema(**json.loads(response))
        except json.JSONDecodeError:
            # If the model doesn't return valid JSON, we'll try to extract the relevant parts
            import re
            prefix = re.search(r'"prefix":\s*"(.*?)"', response, re.DOTALL)
            imports = re.search(r'"imports":\s*"(.*?)"', response, re.DOTALL)
            code = re.search(r'"code":\s*"(.*?)"', response, re.DOTALL)
            
            code_solution = CodeSchema(
                prefix=prefix.group(1) if prefix else "No prefix provided",
                imports=imports.group(1).replace("\\n", "\n") if imports else "",
                code=code.group(1).replace("\\n", "\n") if code else "# No code generated"
            )

        messages.append(
            AIMessage(content=f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}")
        )
        return {"generation": code_solution, "messages": messages, "iterations": iterations + 1}

    def code_check(self, state: GraphState):
        messages = state["messages"]
        code_solution = state["generation"]

        try:
            exec(code_solution.imports)
        except Exception as e:
            error_message = HumanMessage(content=f"Import error: {e}. Reflect and try again with corrections.")
            messages.append(error_message)
            return {"generation": code_solution, "messages": messages, "error": "yes"}

        try:
            exec(f"{code_solution.imports}\n{code_solution.code}")
        except Exception as e:
            error_message = HumanMessage(content=f"Execution error: {e}. Reflect and try again with corrections.")
            messages.append(error_message)
            return {"generation": code_solution, "messages": messages, "error": "yes"}

        print("---CODE EXECUTION SUCCESSFUL---")
        return {"generation": code_solution, "messages": messages, "error": "no"}

    def decide_to_finish(self, state: GraphState):
        if state["error"] == "no" or state["iterations"] >= self.max_iterations:
            return "end"
        return "generate"

    def run_question(self, question: str):
        thread_id = str(uuid.uuid4())
        events = self.graph.stream(
            {"messages": [HumanMessage(content=question)], "iterations": 0, "error": "no"},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        )
        final_result = None
        for event in events:
            self._print_event(event)
            final_result = event
        return final_result

    def _print_event(self, event):
        message = event.get("messages", [])
        if message and isinstance(message, list):
            print(f"Message: {message[-1].content}")

# Example Usage
if __name__ == "__main__":
    code_gen = LLMCodeGen()
    result = code_gen.run_question("""
                                   1 Improve the following script,
                                   2 Add some new features, 
                                   3 test/debug the script, 
                                   4 itterate untill the bugs are solved.

                                   Here is the script:
                                        from langchain import hub
                                        from langchain_community.chat_models import ChatAnthropic
                                        from langchain.agents import (
                                            AgentExecutor, create_self_ask_with_search_agent
                                        )

                                        prompt = hub.pull("hwchase17/self-ask-with-search")
                                        model = ChatAnthropic(model="claude-3-haiku-20240307")
                                        tools = [...]  # Should just be one tool with name `Intermediate Answer`

                                        agent = create_self_ask_with_search_agent(model, tools, prompt)
                                        agent_executor = AgentExecutor(agent=agent, tools=tools)

                                        agent_executor.invoke({"input": "hi"})
                                   """)
    
    if result and 'generation' in result:
        final_code = result['generation']
        print("\nFinal Generated Code:")
        print(f"Imports:\n{final_code.imports}")
        print(f"\nCode:\n{final_code.code}")
        
        print("\nExecuting the generated code:")
        exec(f"{final_code.imports}\n{final_code.code}")