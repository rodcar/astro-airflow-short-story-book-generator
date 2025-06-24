from airflow.sdk import dag, task, Variable
from airflow.providers.openai.hooks.openai import OpenAIHook
from pendulum import duration
import logging

STORY_IDEA_MAX_TOKENS = 200
STORY_IDEA_TEMPERATURE = 0.8
MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"

@dag(
    params={
        "description": "A story about a cat",
        "tags": "fiction, creative, storytelling", 
        "quantity": 1
    },
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)
def short_story_generator():
    """A DAG for generating short stories books."""

    @task
    def init(**context) -> list[dict]:
        """Initialize the DAG."""
        return [context["params"]] * context["params"]["quantity"]
    
    _init = init()

    @task(retries=3, retry_delay=duration(seconds=30))
    def generate_story_idea(config: dict) -> list[str]:
        """Generate story ideas based on the description and tags."""
        description = config["description"]
        tags = config["tags"]
        
        openai_hook = OpenAIHook(conn_id="my_sambanova_conn")
        client = openai_hook.get_conn()
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a creative writing assistant that generates engaging short story ideas."},
                {"role": "user", "content": f"Generate a creative short story idea based on this description: {description} and the following themes: {tags}"}
            ],
            max_tokens=STORY_IDEA_MAX_TOKENS,
            temperature=STORY_IDEA_TEMPERATURE
        )

        story_idea = response.choices[0].message.content.strip()
        logging.info(f"Generated story idea: {story_idea}")
        return story_idea
    
    _generate_story_ideas = generate_story_idea.expand(config=_init)

_short_story_generator_dag = short_story_generator()
