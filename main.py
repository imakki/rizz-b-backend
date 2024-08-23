from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAI, OpenAIError,RateLimitError
from typing import List
import json
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

openai_api_key=os.getenv('OPENAI_API_KEY')
model=os.getenv('GPT_MODEL')

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

class ChatRequest(BaseModel):
    chat_history: str
    user_profile: str

class ChatResponse(BaseModel):
    conversation_starter: str

@app.post("/generate-starter/", response_model=List[ChatResponse])
async def generate_starter(request: ChatRequest):
    # Create the prompt using the provided chat history and user profile
    prompt = f"""
    ### Instructions ###
    You are a conversational assistant helping a user navigate early-stage dating conversations. Use the provided chat history and user profile to generate engaging and personalized conversation starters. Ensure the tone is friendly, respectful, and natural.

    ### User Profile ###
    {request.user_profile}

    ### Chat History ###
    {request.chat_history}

    ### Example Output ###
    Based on the above context, provide several conversation starters that continue the flow naturally and spark interest:
    """

    try:
        # Call the OpenAI API to generate multiple conversation starters
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            n=3  # Generate three different suggestions
        )
        # Extract the content from the response
        starters = [ChatResponse(conversation_starter=choice.message.content.strip()) for choice in response.choices]
        print(json.dumps([starter.dict() for starter in starters], indent=2))
        return starters
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    except OpenAIError as e:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again later.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)