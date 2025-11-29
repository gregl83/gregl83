import argparse
from datetime import date
from enum import Enum
import json
import os
from pathlib import Path
import re
from typing import Any, Optional, Protocol, runtime_checkable

from google import genai as google_genai
from google.genai import types as google_genai_types
from jinja2 import Environment, FileSystemLoader
from PIL import Image
import xai_sdk
from xai_sdk.chat import system, user


class LanguageModel(str, Enum):
    """Available language generation models."""
    GROK = "grok"
    GEMINI = "gemini"

@runtime_checkable
class LanguageGenerator(Protocol):
    """Protocol for language generation."""
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

class GrokLanguageGenerator:
    """Grok language generator."""
    def __init__(self, client: xai_sdk.Client):
        self.client = client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        chat = self.client.chat.create(
            model="grok-3",
            messages=[system(system_prompt)]
        )
        chat.append(user(user_prompt))
        response = chat.sample()
        return response.content


class GeminiLanguageGenerator:
    """Gemini language generator."""
    def __init__(self, client: google_genai.Client):
        self.client = client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=user_prompt,
            config=google_genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        return response.text

class ImageModel(str, Enum):
    """Available image generation models."""
    GROK = "grok"
    GEMINI = "gemini"

@runtime_checkable
class ImageGenerator(Protocol):
    """Protocol for image generation."""
    def generate(self, prompt: str, output_path: Path) -> None: ...


class GrokImageGenerator:
    """Grok image generator."""
    def __init__(self, client: xai_sdk.Client):
        self.client = client

    def generate(self, prompt: str, output_path: Path) -> None:
        response = self.client.image.sample(
            model="grok-2-image",
            prompt=prompt,
            image_format="url",
        )

        with open(output_path, "wb") as f:
            f.write(response.image)


class GeminiImageGenerator:
    """Gemini image generator."""
    def __init__(self, client: google_genai.Client):
        self.client = client

    def generate(self, prompt: str, output_path: Path) -> None:
        response = self.client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                prompt,
            ],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(f"{output_path.absolute()}")

def crop_to_square(
        image_path: Path,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
) -> Image.Image:
    """
    Crops an image to a square and resizes it if dimensions are provided.
    Always overwrites the original image.

    Args:
        image_path: Path to the source image
        target_width: Width to resize to after cropping (optional)
        target_height: Height to resize to after cropping (optional)

    Returns:
        The processed image object
    """
    img = Image.open(image_path)
    width, height = img.size
    size = min(width, height)

    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    cropped = img.crop((left, top, right, bottom))

    # resize if dimensions are provided
    if target_width and target_height:
        cropped = cropped.resize((target_width, target_height))

    # always save to the original path
    cropped.save(image_path)

    return cropped

def parse_json_response(response: str) -> Any:
    """Parse JSON response, handling Markdown code blocks (minor cleanup)"""
    # remove Markdown code blocks if present;  ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if match:
        json_str = match.group(1)
    else:
        json_str = response
    # strip whitespace
    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"error parsing JSON: {e}")
        print(f"attempted to parse: {json_str[:200]}...")
        raise

def generate_images(
        language_generator: LanguageGenerator,
        image_generator: ImageGenerator,
        output_dir: Path
) -> list[Path]:
    """Generates images using language generated prompts."""
    template_loader = FileSystemLoader('.')
    template_env = Environment(loader=template_loader)
    context = {
        "current_date": date.today().isoformat()
    }

    sys_template = template_env.get_template('prompts/daily-imagine-sys-prompt.j2')
    system_prompt = sys_template.render(context)
    user_template = template_env.get_template('prompts/daily-imagine-user-prompt.j2')
    user_prompt = user_template.render(context)

    print(f"\n🪄  Generating image generation prompts")
    lang_res = parse_json_response(
        language_generator.generate(system_prompt, user_prompt)
    )

    res: list[Path] = []
    filenames: list[str] = ["alpha.png", "bravo.png", "charlie.png", "delta.png"]
    for i, image in enumerate(lang_res["images"]):
        print(f"\n🎨  Generating image {i + 1}/4...")
        print(f"📅  Date relevance: {image['date_relevance']}")
        print(f"🎭  Style: {image['style']}")
        print(f"🎨  Colors: {image['colors']}")

        output_path: Path = output_dir.joinpath(filenames[i])
        image_generator.generate(prompt=image["prompt"], output_path=output_path)
        res.append(output_path)

    return res

def create_generators(
        language_model: LanguageModel,
        image_model: ImageModel
) -> tuple[LanguageGenerator,ImageGenerator]:
    """Create language and image generators based on Model arguments.

    Raises ValueError if api key env vars aren't set or a model is unsupported.
    """
    clients = {}

    if language_model == LanguageModel.GROK or image_model == ImageModel.GROK:
        api_key: str = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        clients["xai"] = xai_sdk.Client(api_key=api_key)

    if language_model == LanguageModel.GEMINI or image_model == ImageModel.GEMINI:
        api_key: str = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        clients["google"] = google_genai.client.Client(api_key=api_key)

    language_generator: LanguageGenerator
    if language_model == LanguageModel.GROK:
        language_generator = GrokLanguageGenerator(client=clients["xai"])
    elif language_model == LanguageModel.GEMINI:
        language_generator = GeminiLanguageGenerator(client=clients["google"])
    else:
        raise ValueError(f"{language_model.value} API unsupported")

    image_generator: ImageGenerator
    if image_model == ImageModel.GROK:
        image_generator = GrokImageGenerator(client=clients["xai"])
    elif image_model == ImageModel.GEMINI:
        image_generator = GeminiImageGenerator(client=clients["google"])
    else:
        raise ValueError(f"{language_model.value} API unsupported")

    return language_generator, image_generator

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Models can be selected using optional arguments --language and --image.
    """

    parser = argparse.ArgumentParser(
        description="Generate images using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --language grok --image grok       # Use Grok for both
  %(prog)s --language gemini --image gemini   # Use Google models
  %(prog)s -l grok -i gemini                  # Mix and match
        """
    )

    parser.add_argument(
        "-l", "--language",
        type=str,
        choices=[m.value for m in LanguageModel],
        default=LanguageModel.GROK.value,
        help="Language generation model (default: grok)"
    )

    parser.add_argument(
        "-i", "--image",
        type=str,
        choices=[m.value for m in ImageModel],
        default=ImageModel.GEMINI.value,
        help="Image generation model (default: gemini)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=150,
        help="Output image width (default: 150)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=150,
        help="Output image height (default: 150)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./out"),
        help="Output directory for generated images (default: ./out)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without generating images"
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    language_model = LanguageModel(args.language)
    image_model = ImageModel(args.image)

    print(f"\n🔧  Configuration")
    print(f"      Language model:  {language_model.value}")
    print(f"      Image model: {image_model.value}")
    print(f"      Output dir:  {args.output}")

    if args.dry_run:
        print("\n⏭️  Dry run - skipping generation")
        return

    # ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # generate images
    generated_images = generate_images(
        *create_generators(language_model, image_model),
        args.output
    )

    # resize / crop images
    for generated_image in generated_images:
        crop_to_square(generated_image, args.width, args.height)

if __name__ == "__main__":
    main()
