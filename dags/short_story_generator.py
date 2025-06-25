from airflow.sdk import dag, task, Variable
from airflow.providers.openai.hooks.openai import OpenAIHook
from pendulum import duration
import logging

STORY_IDEA_MAX_TOKENS = 300
STORY_IDEA_TEMPERATURE = 1.0
STORY_PLOT_MAX_TOKENS = 1000
STORY_PLOT_TEMPERATURE = 0.8

MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"
GENERATE_STORY_IDEA_SYSTEM_PROMPT = "You are a creative writing assistant that generates engaging short story ideas. Return only the story idea, no other text. Make sure the story idea is not too long or too short, maximum 100 words."
GENERATE_STORY_IDEA_PROMPT_TEMPLATE = "Generate a creative short story idea based on this description: {description} and the following themes: {tags}. Make sure the story idea is different from the following stories: {stories}"
GENERATE_STORY_PLOT_SYSTEM_PROMPT = """You are a creative writing assistant that generates engaging short story plots. 
Return only the story plot, no other text. 
Make sure the story plot has the following structure: 
1. Introduction: Presentation of characters, setting, and initial situation.
2. Inciting Incident: The event that triggers the main action or conflict.
3. Development: The series of events or challenges the characters face.
4. Climax: The moment of greatest tension or turning point in the story.
5. Resolution: The conflict’s outcome and the story’s conclusion.
Make sure the story plot is not too long or too short, maximum 1000 words."""
GENERATE_STORY_PLOT_PROMPT_TEMPLATE = "Generate a creative short story plot based on this story idea: {story_idea}"

@dag(
    params={
        "description": "A story about a cat",
        "tags": "fiction, creative, storytelling", 
        "quantity": 3
    },
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)
def short_story_generator():
    """A DAG for generating short stories books."""

    @task(retries=3, retry_delay=duration(seconds=30))
    def generate_story_ideas(**context) -> list[str]:
        """Generate story ideas based on the description and tags."""
        description = context["params"]["description"]
        tags = context["params"]["tags"]
        quantity = context["params"]["quantity"]
        openai_hook = OpenAIHook(conn_id="my_sambanova_conn")
        client = openai_hook.get_conn()
        
        story_ideas = []
        for _ in range(quantity):
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": GENERATE_STORY_IDEA_SYSTEM_PROMPT},
                    {"role": "user", "content": GENERATE_STORY_IDEA_PROMPT_TEMPLATE.format(description=description, tags=tags, stories="\n".join(story_ideas))}
                ],
                max_tokens=STORY_IDEA_MAX_TOKENS,
                temperature=STORY_IDEA_TEMPERATURE
            )

            story_idea = response.choices[0].message.content.strip()
            logging.info(f"Generated story idea: {story_idea}")
            story_ideas.append(story_idea)
        return story_ideas
    
    _generated_story_ideas = generate_story_ideas()

    @task
    def generate_story_plot(story_idea: str) -> str:
        """Generate a story plot based on the story idea."""
        openai_hook = OpenAIHook(conn_id="my_sambanova_conn")
        client = openai_hook.get_conn()
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": GENERATE_STORY_PLOT_SYSTEM_PROMPT},
                {"role": "user", "content": GENERATE_STORY_PLOT_PROMPT_TEMPLATE.format(story_idea=story_idea)}
            ],
            max_tokens=STORY_PLOT_MAX_TOKENS,
            temperature=STORY_PLOT_TEMPERATURE
        )
        return response.choices[0].message.content.strip()

    _generated_story_plots = generate_story_plot.expand(story_idea=_generated_story_ideas)

_short_story_generator_dag = short_story_generator()
