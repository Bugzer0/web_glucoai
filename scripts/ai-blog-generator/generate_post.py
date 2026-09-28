#!/usr/bin/env python3
"""
AI Blog Post Generator for GlucoAI — Glucose & Diabetes Health Blog.
Generates MDX blog posts with AI-generated images for a Next.js blog.

Flow:
  Step 0: Scan existing blog posts to build context (avoid duplicates)
  Step 1: Auto-generate a unique topic (or use BLOG_TOPIC env if provided)
  Step 2: Generate metadata (title, summary, tags, image prompts) as JSON
  Step 3: Generate full blog content as plain markdown
  Step 4: Generate images
  Step 5: Create MDX file
"""

import os
import sys
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from slugify import slugify

from google import genai
from google.genai import types
from PIL import Image
import io
import requests

# Load environment variables from .env.local for local development
try:
    from dotenv import load_dotenv
    # Look for .env.local in project root
    env_path = Path(__file__).resolve().parent.parent.parent / ".env.local"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # dotenv not installed, skip (GitHub Actions uses secrets directly)
    pass

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Project root — resolve relative to this script's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BLOG_DIR = PROJECT_ROOT / "data" / "blog"


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required")
        sys.stdout.flush()
        sys.exit(1)
    return genai.Client(api_key=api_key)


def load_skill():
    """Load the blog writing skill/prompt from skill file."""
    skill_path = Path(__file__).parent / "SKILL.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return ""


def sanitize_yaml_string(s: str) -> str:
    """Sanitize a string for safe use in YAML double-quoted scalar."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return s.strip()


def retry_api_call(func):
    """Retry an API call with exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"  ⚠ API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            sys.stdout.flush()
            print(f"  ⏳ Retrying in {wait}s...")
            sys.stdout.flush()
            time.sleep(wait)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 0: Scan existing posts
# ---------------------------------------------------------------------------

def scan_existing_posts() -> List[Dict]:
    """Scan all .mdx files in data/blog/ and extract title, tags, summary."""
    posts = []
    if not BLOG_DIR.exists():
        return posts

    for mdx_file in sorted(BLOG_DIR.glob("*.mdx")):
        try:
            text = mdx_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Extract YAML frontmatter between --- markers
        fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if not fm_match:
            continue

        fm_text = fm_match.group(1)

        title = ""
        summary = ""
        tags_list = []

        # Parse title
        title_match = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        # Parse summary (handle multiline YAML |)
        summary_match = re.search(r'^summary:\s*\|?\s*\n?\s*(.*?)(?:\n\w|\Z)', fm_text, re.MULTILINE | re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip().replace("\n", " ")
        else:
            summary_match = re.search(r'^summary:\s*["\']?(.*?)["\']?\s*$', fm_text, re.MULTILINE)
            if summary_match:
                summary = summary_match.group(1).strip()

        # Parse tags
        tags_match = re.search(r'^tags:\s*(\[.*?\])', fm_text, re.MULTILINE)
        if tags_match:
            try:
                tags_list = json.loads(tags_match.group(1))
            except json.JSONDecodeError:
                tags_list = []

        posts.append({
            "file": mdx_file.name,
            "title": title,
            "summary": summary[:200],
            "tags": tags_list,
        })

    return posts


def build_existing_posts_context(posts: list[dict]) -> str:
    """Build a concise summary of existing posts for the AI to avoid duplicates."""
    if not posts:
        return "No existing blog posts found."

    lines = [f"There are {len(posts)} existing blog posts:"]
    for i, p in enumerate(posts, 1):
        tags_str = ", ".join(p["tags"]) if p["tags"] else "none"
        lines.append(f"{i}. \"{p['title']}\" [tags: {tags_str}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Auto-generate topic
# ---------------------------------------------------------------------------

def generate_topic(client, existing_context: str, skill_prompt: str) -> str:
    """Use AI to generate a unique blog topic about glucose/diabetes."""
    response = retry_api_call(lambda: client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""Generate ONE new blog post topic for an English glucose/diabetes health blog.

{existing_context}

IMPORTANT: The new topic MUST be different from ALL existing posts listed above.
Choose a topic that would attract readers interested in glucose monitoring, diabetes management, metabolic health, CGM technology, or healthy lifestyle for blood sugar control.""",
        config=types.GenerateContentConfig(
            system_instruction=f"""You are a content strategist for an English health blog about glucose and diabetes.

Reference the content strategy below to pick topics from the right content pillars:
{skill_prompt}

Return ONLY a JSON object:
{{
  "topic": "A specific blog topic description in English (1-2 sentences)",
  "content_type": "how-to|listicle|myth-busting|case-study|beginners-guide|device-review",
  "primary_keyword": "main SEO keyword in English",
  "reasoning": "Brief explanation of why this topic is valuable and how it differs from existing posts"
}}

Rules:
- Topic MUST be in English
- Topic should be specific enough to write a focused 800-2500 word article
- Pick topics that naturally lead to discussing glucose monitoring and diabetes management
- Vary content types — don't always pick the same type
- Consider seasonal relevance and trending health topics""",
            temperature=0.9,
            response_mime_type="application/json",
        ),
    ))

    if not response.text:
        print("Error: Gemini returned empty topic response")
        sys.exit(1)

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("Error: Failed to parse topic JSON")
        print(f"Raw: {response.text[:500]}")
        sys.exit(1)

    topic = data.get("topic", "").strip()
    if not topic:
        print("Error: AI returned empty topic")
        sys.exit(1)

    content_type = data.get("content_type", "unknown")
    keyword = data.get("primary_keyword", "")
    reasoning = data.get("reasoning", "")

    print(f"  ✓ Topic: {topic}")
    print(f"  ✓ Type: {content_type}")
    print(f"  ✓ Keyword: {keyword}")
    if reasoning:
        print(f"  ✓ Reasoning: {reasoning}")

    return topic


# ---------------------------------------------------------------------------
# Step 2: Generate metadata
# ---------------------------------------------------------------------------

def generate_metadata(client, topic: str, existing_context: str) -> dict:
    """Generate blog metadata as structured JSON."""
    response = retry_api_call(lambda: client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"Generate blog post metadata for the topic: {topic}",
        config=types.GenerateContentConfig(
            system_instruction=f"""Return ONLY a JSON object with blog post metadata.
Do NOT include the blog content itself.

JSON structure:
{{
  "title": "A compelling blog post title in English",
  "summary": "A 1-2 sentence SEO summary in English (single line, no newlines)",
  "tags": ["tag1", "tag2"],
  "image_prompts": {{
    "banner": "Detailed prompt for a square thumbnail image directly related to the blog topic",
    "inline1": "Detailed prompt for first inline illustration (appears after first section)",
    "inline2": "Detailed prompt for second inline illustration (appears mid-article)"
  }}
}}

{existing_context}

Rules:
- Title and summary MUST be in English
- Title should contain the primary SEO keyword, under 60 characters
- Tags should be lowercase English from this list: glucose, diabetes, cgm, a1c, insulin, blood-sugar, nutrition, diet, tips, health, metabolic-health, type-1, type-2, prediabetes
- image_prompts: MUST include banner, inline1, and inline2 (3 images total)
- banner: MUST directly represent the specific topic of the blog post (e.g., if the post is about A1C, the image should depict A1C testing or results). Do NOT use generic health images.
- All image prompts should describe clean, modern, health-themed illustrations (no text in images, no faces)
- inline1: illustration for early in article (after first main section), related to the section content
- inline2: illustration for mid-article (around 50-60% through content), related to the section content
- Keep summary under 200 characters
- 1-3 tags maximum
- Title MUST be different from all existing posts""",
            temperature=0.7,
            response_mime_type="application/json",
        ),
    ))

    if not response.text:
        print("Error: Gemini returned empty metadata response")
        sys.exit(1)

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("Error: Failed to parse metadata JSON")
        print(f"Raw: {response.text[:500]}")
        sys.exit(1)

    for field in ["title", "summary"]:
        if not data.get(field):
            print(f"Error: Missing required field '{field}'")
            sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Step 3: Generate content
# ---------------------------------------------------------------------------

def generate_content(client, topic: str, title: str, skill_prompt: str, existing_context: str) -> str:
    """Generate blog content as plain markdown."""
    response = retry_api_call(lambda: client.models.generate_content(
        model="gemini-flash-latest",
        contents=f'Write a blog post titled "{title}" about: {topic}',
        config=types.GenerateContentConfig(
            system_instruction=f"""You are an English health blog writer specializing in glucose monitoring and diabetes management.

{skill_prompt}

{existing_context}
If any existing posts are related, you may reference them with relative links like [related article](/blog/slug-name) to build internal linking.

Write the blog post content in PLAIN MARKDOWN format.

Rules:
- Write ENTIRELY in English
- Do NOT include any frontmatter (no --- block)
- Do NOT wrap in JSON or code fences
- Use ## for main sections and ### for subsections
- MUST include `<!-- INLINE_IMAGE_1 -->` placeholder after the first main section
- MUST include `<!-- INLINE_IMAGE_2 -->` placeholder around 50-60% through the article
- Start directly with the introduction paragraph (no title heading)
- Minimum 800 words, maximum 2500 words
- Follow the CTA strategy: naturally guide readers toward better health habits and GlucoAI
- Do NOT mention any specific third-party app names. Use generic terms like "glucose tracking apps" or "diabetes management tools" instead. Only GlucoAI may be mentioned by name.
- End with a "Summary" or "Conclusion" section with key takeaways
- Include a natural call-to-action at the end""",
            temperature=0.8,
        ),
    ))

    if not response.text:
        print("Error: Gemini returned empty content response")
        sys.exit(1)

    content = response.text.strip()

    # Strip wrapping code fences if Gemini added them
    if content.startswith("```markdown"):
        content = content[len("```markdown"):].strip()
    if content.startswith("```md"):
        content = content[len("```md"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content


# ---------------------------------------------------------------------------
# Step 4: Generate images
# ---------------------------------------------------------------------------

def generate_image(client, prompt: str, output_path: Path, aspect_ratio: str = "1:1") -> bool:
    """Generate an image using Gemini's image generation model.
    
    Args:
        client: Gemini API client
        prompt: Image generation prompt
        output_path: Path to save the image
        aspect_ratio: Aspect ratio for the generated image (e.g., "1:1", "16:9")
    """
    print(f"  → Starting image generation: {output_path.name} (aspect_ratio={aspect_ratio})")
    sys.stdout.flush()
    try:
        response = retry_api_call(lambda: client.models.generate_content(
            model="gemini-3.1-flash-image",
            # model="gemini-2.5-flash-image",
            contents=f"Generate an image: {prompt}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        ))

        if not response.candidates:
            print("  ✗ No candidates in response (safety filter?)")
            return False

        content = response.candidates[0].content
        if not content or not content.parts:
            print("  ✗ Response has no content parts")
            return False

        for part in content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
                image = Image.open(io.BytesIO(image_data))
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path), "JPEG", quality=85)
                print(f"  ✓ Image saved: {output_path}")
                print(f"  ✓ Image size: {image.size}")
                sys.stdout.flush()
                return True

        print(f"  ✗ No image data in response for: {prompt[:80]}...")
        return False

    except Exception as e:
        print(f"  ✗ Image generation failed: {e}")
        sys.stdout.flush()
        return False


# ---------------------------------------------------------------------------
# Step 5: Create MDX file
# ---------------------------------------------------------------------------

def send_telegram_notification(title: str, slug: str, tags: list = None, word_count: int = 0) -> bool:
    """Send a Telegram notification when a new blog post is created.
    
    Args:
        title: Blog post title
        slug: Blog post slug
        tags: List of tags (optional)
        word_count: Word count of the post (optional)
        
    Returns:
        True if notification sent successfully, False otherwise
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    
    if not bot_token or not chat_id:
        print("  ⚠ Telegram credentials not configured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing)")
        return False
    
    try:
        # Build message with additional info
        message_parts = [
            "📝 *New blog post created!*",
            "",
            f"📌 *Title:* {title}",
            f"🔗 *Slug:* `{slug}`"
        ]
        
        if word_count > 0:
            message_parts.append(f"📊 *Words:* {word_count}")
        
        if tags:
            tags_str = ", ".join(tags)
            message_parts.append(f"🏷 *Tags:* {tags_str}")
        
        message = "\n".join(message_parts)
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("  ✓ Telegram notification sent successfully")
            sys.stdout.flush()
            return True
        else:
            print(f"  ✗ Failed to send Telegram notification: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            sys.stdout.flush()
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Network error sending Telegram notification: {e}")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"  ✗ Error sending Telegram notification: {e}")
        sys.stdout.flush()
        return False


def create_blog_post(metadata: dict, content: str, slug_name: str, images_generated: dict):
    """Create the MDX file with frontmatter and content."""
    today = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    image_dir = f"/static/images/{slug_name}"
    alt_base = metadata.get("title", "illustration")

    # Build images list for frontmatter
    images_list = []
    if images_generated.get("banner"):
        images_list.append(f"{image_dir}/banner.jpg")

    # Process content - replace inline image placeholders
    if images_generated.get("inline1"):
        if "<!-- INLINE_IMAGE_1 -->" in content:
            inline1_img = f"\n![{alt_base} - illustration 1]({image_dir}/inline1.jpg)\n"
            content = content.replace("<!-- INLINE_IMAGE_1 -->", inline1_img)
        else:
            # Fallback: insert after first ## section
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("## ") and i > 0:
                    inline1_img = f"\n![{alt_base} - illustration 1]({image_dir}/inline1.jpg)\n"
                    lines.insert(i + 1, inline1_img)
                    break
            content = "\n".join(lines)

    if images_generated.get("inline2"):
        if "<!-- INLINE_IMAGE_2 -->" in content:
            inline2_img = f"\n![{alt_base} - illustration 2]({image_dir}/inline2.jpg)\n"
            content = content.replace("<!-- INLINE_IMAGE_2 -->", inline2_img)
        else:
            # Fallback: insert at middle of content
            lines = content.split("\n")
            mid_point = len(lines) // 2
            inserted = False
            for i in range(mid_point, len(lines)):
                if lines[i].startswith("## "):
                    inline2_img = f"\n![{alt_base} - illustration 2]({image_dir}/inline2.jpg)\n"
                    lines.insert(i + 1, inline2_img)
                    inserted = True
                    break
            if inserted:
                content = "\n".join(lines)

    # Always remove any leftover placeholders to prevent MDX/JSX parse errors
    content = content.replace("<!-- INLINE_IMAGE_1 -->", "")
    content = content.replace("<!-- INLINE_IMAGE_2 -->", "")
    content = content.replace("<!-- INLINE_IMAGE -->", "")
    content = content.replace("{INLINE_IMAGE}", "")
    
    # Fix spacing issues: ensure only 1 blank line before inline images (not 2+)
    # This prevents "cannot end with a space or control character" errors in Next.js
    content = re.sub(r'\n\n\n+(!\[.*?\]\(/static/images/.*?\))', r'\n\n\1', content)

    safe_title = sanitize_yaml_string(metadata["title"])
    safe_summary = sanitize_yaml_string(metadata["summary"])
    tags_str = json.dumps(metadata.get("tags", ["ai-generated"]))
    images_list = [f"/static/images/{slug_name}/banner.jpg"]
    # Use YAML array format instead of JSON to avoid parsing errors
    images_yaml = "\n".join([f"  - {img}" for img in images_list])

    frontmatter = f'''---
title: "{safe_title}"
summary: |
  {safe_summary}
tags: {tags_str}
date: {today}
draft: false
images:
{images_yaml}
---'''

    mdx_content = f"{frontmatter}\n\n{content}\n"

    mdx_path = BLOG_DIR / f"{slug_name}.mdx"
    mdx_path.parent.mkdir(parents=True, exist_ok=True)
    mdx_path.write_text(mdx_content, encoding="utf-8", newline='\n')
    print(f"✓ Blog post created: {mdx_path}")
    sys.stdout.flush()
    return mdx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🩺 GlucoAI Blog Generator — Glucose & Diabetes Health")
    print("=" * 60)
    sys.stdout.flush()

    client = get_client()
    skill_prompt = load_skill()

    # Step 0: Scan existing posts
    print("\n🔍 Scanning existing blog posts...")
    sys.stdout.flush()
    existing_posts = scan_existing_posts()
    existing_context = build_existing_posts_context(existing_posts)
    print(f"  ✓ Found {len(existing_posts)} existing posts")
    sys.stdout.flush()
    for p in existing_posts:
        print(f"    • {p['title']}")
    sys.stdout.flush()

    # Step 1: Get or generate topic
    topic = os.environ.get("BLOG_TOPIC", "").strip()
    if topic:
        print(f"\n📌 Using provided topic: {topic}")
        sys.stdout.flush()
    else:
        print("\n🧠 Auto-generating a new unique topic...")
        sys.stdout.flush()
        topic = generate_topic(client, existing_context, skill_prompt)
        sys.stdout.flush()

    # Step 2: Generate metadata
    print("\n📋 Generating metadata...")
    sys.stdout.flush()
    metadata = generate_metadata(client, topic, existing_context)
    print(f"  ✓ Title: {metadata['title']}")
    print(f"  ✓ Summary: {metadata['summary'][:100]}...")
    print(f"  ✓ Tags: {metadata.get('tags', [])}")
    sys.stdout.flush()

    # Step 3: Generate content
    print("\n📝 Generating content...")
    sys.stdout.flush()
    content = generate_content(client, topic, metadata["title"], skill_prompt, existing_context)
    word_count = len(content.split())
    print(f"  ✓ Content generated ({word_count} words)")
    sys.stdout.flush()

    # Step 4: Create slug & check duplicates
    slug_name = slugify(metadata["title"], max_length=60)
    if not slug_name:
        slug_name = slugify(topic, max_length=60) or f"post-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"  ✓ Slug: {slug_name}")
    sys.stdout.flush()

    mdx_path = BLOG_DIR / f"{slug_name}.mdx"
    if mdx_path.exists():
        slug_name = f"{slug_name}-{datetime.now().strftime('%Y%m%d')}"
        print(f"  ⚠ Slug already exists, using: {slug_name}")
        sys.stdout.flush()

    # Step 5: Generate images
    images_generated = {}
    image_prompts = metadata.get("image_prompts", {})
    image_dir = Path(f"public/static/images/{slug_name}")

    if image_prompts.get("banner"):
        print("\n🎨 Generating banner image...")
        sys.stdout.flush()
        images_generated["banner"] = generate_image(
            client, image_prompts["banner"], image_dir / "banner.jpg",
            aspect_ratio="1:1",
        )
        sys.stdout.flush()

    if image_prompts.get("inline1"):
        print("🎨 Generating inline image 1...")
        sys.stdout.flush()
        images_generated["inline1"] = generate_image(
            client, image_prompts["inline1"], image_dir / "inline1.jpg",
            aspect_ratio="16:9",
        )
        sys.stdout.flush()

    if image_prompts.get("inline2"):
        print("🎨 Generating inline image 2...")
        sys.stdout.flush()
        images_generated["inline2"] = generate_image(
            client, image_prompts["inline2"], image_dir / "inline2.jpg",
            aspect_ratio="16:9",
        )
        sys.stdout.flush()

    # Validate: warn if missing inline images
    inline_count = sum([bool(images_generated.get("inline1")), bool(images_generated.get("inline2"))])
    if inline_count < 2:
        print(f"  ⚠ Warning: Only {inline_count}/2 inline images generated successfully")
        sys.stdout.flush()

    # Step 6: Create MDX file
    print("\n📄 Creating MDX file...")
    sys.stdout.flush()
    created_path = create_blog_post(metadata, content, slug_name, images_generated)
    sys.stdout.flush()

    # Output for GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"slug={slug_name}\n")
            safe_title = metadata["title"].replace('"', "").replace("'", "")
            safe_title = safe_title.replace("\n", " ").replace("\r", "")[:80]
            f.write(f"title={safe_title}\n")
            f.write(f"mdx_path={created_path}\n")

    # Step 7: Send Telegram notification
    print("\n📱 Sending Telegram notification...")
    sys.stdout.flush()
    send_telegram_notification(
        title=metadata["title"],
        slug=slug_name,
        tags=metadata.get("tags", []),
        word_count=word_count
    )
    
    print("\n" + "=" * 60)
    print("✅ Done! New glucose/diabetes blog post created successfully.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
