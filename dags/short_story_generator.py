from airflow.sdk import dag, task, Variable
from airflow.providers.openai.hooks.openai import OpenAIHook
from pendulum import duration
import logging
import os
import subprocess

STORY_IDEA_MAX_TOKENS = 300
STORY_IDEA_TEMPERATURE = 1.0
STORY_PLOT_MAX_TOKENS = 1000
STORY_PLOT_TEMPERATURE = 0.8

# OpenAI API
#MODEL_ID = "gpt-4.1"

# Sambanova API
#MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"

# TogetherAI API
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

GENERATE_STORY_IDEA_SYSTEM_PROMPT = "You are a creative writing assistant that generates engaging short story ideas. Return only the story idea, no other text. Make sure the story idea is not too long or too short, maximum 100 words."
GENERATE_STORY_IDEA_PROMPT_TEMPLATE = "Generate a creative short story idea based on this description: {description} and the following themes: {tags}. Make sure the story idea is different from the following stories: {stories}"
GENERATE_STORY_PLOT_SYSTEM_PROMPT = """You are a creative writing assistant that generates engaging short story plots. 
Return only the story plot, no other text. 
Make sure the story plot has the following structure: 
1. Introduction: Presentation of characters, setting, and initial situation.
2. Inciting Incident: The event that triggers the main action or conflict.
3. Development: The series of events or challenges the characters face.
4. Climax: The moment of greatest tension or turning point in the story.
5. Resolution: The conflict's outcome and the story's conclusion.
Make sure the story plot is not too long or too short, maximum 1000 words."""
GENERATE_STORY_PLOT_PROMPT_TEMPLATE = "Generate a creative short story plot based on this story idea: {story_idea}"
STORY_SECTIONS = [
    "Introduction",
    "Inciting Incident",
    "Development",
    "Climax",
    "Resolution"
]
GENERATE_STORY_CONTENT_SYSTEM_PROMPT = """You are a creative writing assistant that generates engaging short story content section by section. 
You will be asked to write a specific section of a story while maintaining continuity with what has been written so far and the overall plot.
Return ONLY a valid JSON object with "title" and "content" properties. No other text or formatting.
Make sure the content flows naturally from previous sections and sets up future sections appropriately.
Keep the content engaging and well-written, with a length appropriate for the section (typically 1000-1600 words per section)."""
GENERATE_STORY_CONTENT_PROMPT_TEMPLATE = """Write the '{current_section}' section of a short story.

Story Idea: {story_idea}

Plot Outline: {story_plot}

Story Written So Far:
{story_content}

Instructions:
- Return ONLY a JSON object with "title" and "content" properties
- Include a creative title for this section in the "title" field (limit 32 characters)
- Write specifically the '{current_section}' section content in the "content" field
- Ensure it flows naturally from what has been written so far
- Follow the plot outline to maintain story coherence
- Set up the narrative appropriately for the sections that will follow
- Focus on advancing the story according to the plot structure
- Maintain consistent character voices and story tone

Example response format:
{{"title": "The Mysterious Beginning", "content": "Once upon a time, in a small village..."}}

Return only the JSON object, no other text."""
STORY_CONTENT_MAX_TOKENS = 6000
STORY_CONTENT_TEMPERATURE = 0.8


@dag(
    params={
        "description": "A story about a cat",
        "tags": "fiction, creative, storytelling", 
        "quantity": 1,
        "author": "Ivan Rodriguez"
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
        openai_hook = OpenAIHook(conn_id="my_togetherai_conn")
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

    @task(retries=3, retry_delay=duration(seconds=60))
    def generate_story_plot(story_idea: str) -> str:
        """Generate a story plot based on the story idea."""
        sambanova_hook = OpenAIHook(conn_id="my_togetherai_conn")
        client = sambanova_hook.get_conn()
        
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

    @task(retries=3, retry_delay=duration(seconds=60))
    def generate_story_image_cover(story_idea: str) -> str:
        """Generate an image cover for the story based on the story idea."""
        openai_hook = OpenAIHook(conn_id="my_openai_conn")
        client = openai_hook.get_conn()
        import base64
        
        # Create an illustration prompt based on the story idea
        image_prompt = f"An illustration that represents: {story_idea}. Style: illustration."
        
        response = client.images.generate(
            model="gpt-image-1",
            prompt=image_prompt,
            n=1,
            size="1024x1024"
        )
        
        # Return the base64 encoded image
        image_base64 = response.data[0].b64_json
        logging.info(f"Generated image cover for story idea: {story_idea[:50]}...")

        # Save the image to a file on include folder
        # Truncate story_idea for filename to avoid "File name too long" error
        safe_filename = story_idea.replace(" ", "_").replace("/", "_").replace("\\", "_")[:10]
        filename = f"story_image_cover_{safe_filename}.png"
        with open(f"include/{filename}", "wb") as f:
            f.write(base64.b64decode(image_base64))
        return filename

    _generated_story_image_covers = generate_story_image_cover.expand(story_idea=_generated_story_ideas)

    @task(retries=3, retry_delay=duration(seconds=60))
    def generate_story_content(story_data) -> str:
        """Generate the story content based on the story idea and plot."""
        import time
        story_idea, story_plot = story_data  # Unpack zipped data
        
        # connect to sambanova
        sambanova_hook = OpenAIHook(conn_id="my_togetherai_conn")
        client = sambanova_hook.get_conn()

        story_sections = []

        for section in STORY_SECTIONS:
            story_content = "\n".join([f"{sec['title']}: {sec['content']}" for sec in story_sections]) if story_sections else "This is the beginning of the story."

            # generate the story content for the current section
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": GENERATE_STORY_CONTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": GENERATE_STORY_CONTENT_PROMPT_TEMPLATE.format(
                        current_section=section,
                        story_idea=story_idea, 
                        story_plot=story_plot, 
                        story_content=story_content
                    )}
                ],
                max_tokens=STORY_CONTENT_MAX_TOKENS,
                temperature=STORY_CONTENT_TEMPERATURE
            )
            
            # Parse JSON response
            raw_content = response.choices[0].message.content.strip()
            
            try:
                import json
                section_data = json.loads(raw_content)
                extracted_title = section_data.get("title", section)
                clean_content = section_data.get("content", "")
            except json.JSONDecodeError:
                # Fallback to section name if JSON parsing fails
                logging.warning(f"Failed to parse JSON response for section '{section}'. Using raw content.")
                extracted_title = section
                clean_content = raw_content
            
            story_sections.append({
                "title": extracted_title,
                "content": clean_content
            })
        return story_sections

    _generated_story_contents = generate_story_content.expand(story_data=_generated_story_ideas.zip(_generated_story_plots))

    @task(retries=5, retry_delay=duration(seconds=20))
    def generate_book(book_data, **context) -> str:
        """Generate PDF from story content using LaTeX template."""
        import datetime
        
        story_sections, cover_image = book_data  # Unpack story sections and cover image
        
        # Format story content
        formatted_content = ""
        story_title = "Generated Story"
        
        for section in story_sections:
            formatted_content += f"\\begin{{center}}\n\\section*{{{section['title']}}}\n\\end{{center}}\n"
            formatted_content += f"{section['content']}\n\n"
            formatted_content += "\\newpage\n"
            
        # Get the first section title as story title
        if story_sections:
            story_title = story_sections[0]['title']
            
        # Read template
        template_path = "include/story_template.tex"
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Replace placeholders
        tex_content = template.replace("{{STORY_TITLE}}", story_title)
        tex_content = tex_content.replace("{{STORY_CONTENT}}", formatted_content)
        tex_content = tex_content.replace("{{COVER_IMAGE}}", cover_image)

        # Replace author
        tex_content = tex_content.replace("{{AUTHOR}}", context["params"]["author"])

        # Replace year (dynamic)
        tex_content = tex_content.replace("{{YEAR}}", str(datetime.datetime.now().year))
        
        # Write temp tex file
        # Sanitize story_title for filename - remove problematic characters
        safe_story_title = story_title.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("'", "").replace('"', "").replace(":", "").replace("?", "").replace("*", "").replace("<", "").replace(">", "").replace("|", "")
        tex_filename = f"include/temp_story_{safe_story_title[:10]}.tex"
        with open(tex_filename, 'w') as f:
            f.write(tex_content)
            
        # Generate PDF (run twice for TikZ overlays)
        pdf_filename = tex_filename.replace('.tex', '.pdf')
        pdflatex_cmd = [
            'pdflatex',
            '-interaction=nonstopmode',
            '-output-directory=include', 
            tex_filename
        ]
        try:
            # First pass
            subprocess.run(pdflatex_cmd, check=True, capture_output=True, text=True, timeout=60)
            # Second pass for TikZ overlays
            result = subprocess.run(pdflatex_cmd, check=True, capture_output=True, text=True, timeout=60)
            logging.info(f"PDF generation successful: {result.stdout}")
        except subprocess.TimeoutExpired:
            logging.error("PDF generation timed out")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"PDF generation failed: {e.stderr}")
            raise
        
        # Clean up temp files
        os.remove(tex_filename)
        aux_file = tex_filename.replace('.tex', '.aux')
        log_file = tex_filename.replace('.tex', '.log')
        if os.path.exists(aux_file):
            os.remove(aux_file)
        if os.path.exists(log_file):
            os.remove(log_file)
            
        return pdf_filename

    _generated_pdfs = generate_book.expand(book_data=_generated_story_contents.zip(_generated_story_image_covers))

_short_story_generator_dag = short_story_generator()
