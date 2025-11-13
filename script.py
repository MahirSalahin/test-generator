from __future__ import annotations

import os
import asyncio
import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SystemPrompt(SystemMessage):
    """System prompt for AI task generation."""

    def __init__(self) -> None:
        """System prompt for AI task generation."""
        content = """You are an expert QA engineer. Generate comprehensive test cases in exact JSON format.

ALWAYS output this exact JSON structure:
{
  "test_cases": [
    {
      "title": "Descriptive title", 
      "preconditions": "Prerequisites",
      "steps": ["step1", "step2"],
      "expected_result": "Clear outcome"
    }
  ]
}

Coverage requirements:
- 1-2 happy path scenarios
- 2-3 negative/edge cases  
- 1-2 security/validation cases
- Total: 4-8 test cases based on complexity

Output ONLY raw JSON, no other text.

Focus on creating tasks that are:
- Clear and actionable
- Include all necessary context
- Properly structured for automation

Remember: 
- Return ONLY the JSON object
- Use EXACTLY the field names shown above (title, preconditions, steps, expected_result)
- No additional text or explanation"""
        super().__init__(content=content)

async def main() -> None:
    """Main entry point for the script."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=SecretStr(api_key),
    )
    system_prompt = SystemPrompt()   
    user_message = input("Enter the test case description: ") 
    # user_message = """User should be able to log in, log out, and reset password."""
    
    messages = [system_prompt, HumanMessage(
                content=f'Please convert this test case into a task:\n\n{user_message}')]

    try:
        response = await llm.ainvoke(messages)
        content = response.content
        print(content)
    except Exception as e:
        pass

if __name__ == '__main__':
    asyncio.run(main())
