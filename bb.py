import re
import PyPDF2
from pathlib import Path
from openai import OpenAI
import os
from typing import List, Tuple


class BulletPointExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Need an OpenAI API key!")

        self.client = OpenAI(
            api_key = "")
        self.pattern = r'^\s*[•\-\*∙◦⚬⦁⦾⦿→▪︎▸◆◇○●■□]'

    def extract_text_from_file(self, file_path: str) -> str:
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return '\n'.join(page.extract_text() for page in pdf_reader.pages)

        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Only PDF or TXT files are supported!")

    def regex_extract_bullets(self, text: str) -> List[str]:
        return [line.strip() for line in text.split('\n')
                if re.match(self.pattern, line)]

    async def llm_verify_and_enhance(self, text: str, regex_bullets: List[str]) -> Tuple[List[str], List[str]]:
        prompt = f"""Analyze this text and help with two tasks:

1. Verify if these extracted bullet points are correct (respond with VALID or INVALID for each):
{chr(10).join(f"- {bullet}" for bullet in regex_bullets)}

2. Find any additional bullet points or list items that were missed.

Format your response as:
VERIFICATION:
[bullet point]: VALID/INVALID
...

ADDITIONAL BULLETS:
- [new bullet point 1]
- [new bullet point 2]
...

Original text:
{text}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that verifies bullet points and finds missing ones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            result = response.choices[0].message.content
            verified_bullets = []
            new_bullets = []

            verification_section = result.split('VERIFICATION:')[1].split('ADDITIONAL BULLETS:')[0].strip()
            additional_section = result.split('ADDITIONAL BULLETS:')[1].strip()

            for line in verification_section.split('\n'):
                if ':' in line:
                    bullet, status = line.split(':', 1)
                    if 'VALID' in status.upper():
                        verified_bullets.append(bullet.strip())

            for line in additional_section.split('\n'):
                if line.strip().startswith('-'):
                    new_bullets.append(line.strip('- ').strip())

            return verified_bullets, new_bullets

        except Exception as e:
            print(f"LLM verification failed: {str(e)}")
            return regex_bullets, []

    async def extract_bullet_points(self, file_path: str) -> Tuple[List[str], List[str]]:
        text = self.extract_text_from_file(file_path)
        bullets = self.regex_extract_bullets(text)
        return await self.llm_verify_and_enhance(text, bullets)


async def main():
    try:
        extractor = BulletPointExtractor()
        file_path = input("File path: ")
        verified_bullets, new_bullets = await extractor.extract_bullet_points(file_path)

        print("\nVerified bullet points:")
        print("-" * 50)
        for idx, point in enumerate(verified_bullets, 1):
            print(f"{idx}. {point}")

        if new_bullets:
            print("\nAdditional bullet points found:")
            print("-" * 50)
            for idx, point in enumerate(new_bullets, 1):
                print(f"{idx}. {point}")

        if input("\nSave results to file? (y/n): ").lower() == 'y':
            output_path = Path(file_path).stem + '_enhanced_bullets.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== VERIFIED BULLET POINTS ===\n\n")
                for point in verified_bullets:
                    f.write(f"• {point}\n")
                if new_bullets:
                    f.write("\n=== ADDITIONAL BULLET POINTS ===\n\n")
                    for point in new_bullets:
                        f.write(f"• {point}\n")
            print(f"\nSaved to: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
