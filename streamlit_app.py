from __future__ import annotations

import os
import asyncio
import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import streamlit as st

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

async def generate_test_cases(user_message: str) -> str:
    """Generate test cases based on user input."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=SecretStr(api_key),
    )
    system_prompt = SystemPrompt()   
    
    messages = [system_prompt, HumanMessage(
                content=f'Please convert this test case into a task:\n\n{user_message}')]

    try:
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as e:
        raise e

def format_test_cases(json_content: str) -> None:
    """Format and display test cases in Streamlit."""
    try:
        # Parse the JSON response
        test_data = json.loads(json_content)
        test_cases = test_data.get("test_cases", [])
        
        if not test_cases:
            st.warning("No test cases found in the response.")
            return
        
        st.success(f"Generated {len(test_cases)} test cases:")
        
        for i, test_case in enumerate(test_cases, 1):
            with st.expander(f"Test Case {i}: {test_case.get('title', 'Untitled')}"):
                st.markdown(f"**Title:** {test_case.get('title', 'N/A')}")
                st.markdown(f"**Preconditions:** {test_case.get('preconditions', 'N/A')}")
                
                steps = test_case.get('steps', [])
                if steps:
                    st.markdown("**Steps:**")
                    for j, step in enumerate(steps, 1):
                        st.markdown(f"{j}. {step}")
                
                st.markdown(f"**Expected Result:** {test_case.get('expected_result', 'N/A')}")
                
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {str(e)}")
        st.text("Raw response:")
        st.code(json_content, language="json")
    except Exception as e:
        st.error(f"Error formatting test cases: {str(e)}")

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Test Case Generator",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Test Case Generator")
    st.markdown("Generate comprehensive test cases using AI")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        st.stop()
    
    # User input
    st.header("Input")
    user_input = st.text_area(
        "Enter your test case description:",
        placeholder="e.g., User should be able to log in, log out, and reset password.",
        height=100
    )
    
    # Generate button
    if st.button("Generate Test Cases", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a test case description.")
        else:
            with st.spinner("Generating test cases..."):
                try:
                    # Run the async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(generate_test_cases(user_input))
                    loop.close()
                    
                    st.header("Generated Test Cases")
                    format_test_cases(response)
                    
                    # Show raw JSON in expandable section
                    with st.expander("View Raw JSON"):
                        st.code(response, language="json")
                        
                except ValueError as e:
                    st.error(f"Configuration error: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error generating test cases: {str(e)}")

if __name__ == '__main__':
    main()