import os,uuid,json
from typing import List, Dict, Literal, Annotated, TypedDict
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END,START

# Ensure that API key for Hugging Face is set
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

class ProjectStructure(BaseModel):
    files: Dict[str, str] = Field(description="Dictionary of filenames and their content")
    dependencies: List[str] = Field(description="List of project dependencies")
    entry_point: str = Field(description="The main entry point file for the project")

class ProjectState(BaseModel):
    messages: List[Dict[str, str]]
    project_structure: ProjectStructure
    current_role: Literal["architect", "coder", "tester", "integrator"]
    iterations: int
    error: str

class CodeProjectGenerator:
    def __init__(self, max_iterations: int = 10):
        self.client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=os.getenv("HUGGINGFACE_API_KEY"))
        self.max_iterations = max_iterations
        self.builder = StateGraph(ProjectState)
        self._initialize_graph()

    def _initialize_graph(self):
        self.builder.add_node("process", self.process_step)
        self.builder.add_edge("process", "process")
        self.builder.add_conditional_edges(
            "process",
            self.decide_next_step,
            {
                "architect": "process",
                "coder": "process",
                "tester": "process",
                "integrator": "process",
                "end": END
            }
        )

    def process_step(self, state: ProjectState):
        role = state.current_role
        prompt = self._get_role_prompt(role, state)
        
        response = self.client.text_generation(prompt, max_new_tokens=500, temperature=0.2)
        
        # Process the response based on the current role
        if role == "architect":
            # Update project structure based on architectural decisions
            pass
        elif role == "coder":
            # Generate or update file content
            pass
        elif role == "tester":
            # Perform tests and update error state
            pass
        elif role == "integrator":
            # Integrate components and resolve conflicts
            pass

        # Update state based on the response
        state.messages.append({"role": "assistant", "content": response})
        state.iterations += 1
        
        return state

    def _get_role_prompt(self, role: str, state: ProjectState) -> str:
        base_prompt = f"You are now acting as the {role} for this project. "
        
        if role == "architect":
            return base_prompt + f"Design the overall structure for: {state.messages[0]['content']}"
        elif role == "coder":
            return base_prompt + f"Generate code for the file: {self._get_next_file_to_code(state)}"
        elif role == "tester":
            return base_prompt + "Test the following code and report any issues: " + self._get_code_to_test(state)
        elif role == "integrator":
            return base_prompt + "Integrate the following components and resolve any conflicts: " + self._get_components_to_integrate(state)
        
        return "Error: Unknown role"

    def decide_next_step(self, state: ProjectState) -> str:
        if state.iterations >= self.max_iterations or state.error == "no":
            return "end"
        
        if not state.project_structure.files:
            return "architect"
        if any(not content for content in state.project_structure.files.values()):
            return "coder"
        if state.error == "yes":
            return "tester"
        return "integrator"

    def _get_next_file_to_code(self, state: ProjectState) -> str:
        # Logic to determine the next file that needs coding
        pass

    def _get_code_to_test(self, state: ProjectState) -> str:
        # Logic to get the code that needs testing
        pass

    def _get_components_to_integrate(self, state: ProjectState) -> str:
        # Logic to get components that need integration
        pass

    def run_project(self, project_description: str):
        initial_state = ProjectState(
            messages=[{"role": "user", "content": project_description}],
            project_structure=ProjectStructure(files={}, dependencies=[], entry_point=""),
            current_role="architect",
            iterations=0,
            error="no"
        )

        events = self.builder.stream(initial_state)
        final_result = None
        for event in events:
            self._print_event(event)
            final_result = event
        return final_result

    def _print_event(self, event: ProjectState):
        print(f"Current Role: {event.current_role}")
        print(f"Iteration: {event.iterations}")
        print(f"Last Message: {event.messages[-1]['content'][:100]}...")
        print("---")

# Example usage
if __name__ == "__main__":
    project_gen = CodeProjectGenerator()
    result = project_gen.run_project("Create a simple web scraping project with BeautifulSoup")
    
    if result:
        print("\nFinal Project Structure:")
        for filename, content in result.project_structure.files.items():
            print(f"\nFile: {filename}")
            print(content[:100] + "..." if len(content) > 100 else content)
        
        print("\nProject Dependencies:")
        print(result.project_structure.dependencies)
        
        print(f"\nEntry Point: {result.project_structure.entry_point}")