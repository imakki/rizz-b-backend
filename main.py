from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import openai
from openai import OpenAI, OpenAIError,RateLimitError
from typing import List
import json
import os
from dotenv import load_dotenv
import random
from datetime import datetime
from sqlalchemy.orm import Session
from models import Feedback
from db import get_db

# Load environment variables from the .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

openai_api_key=os.getenv('OPENAI_API_KEY')
model=os.getenv('GPT_MODEL')

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)


class UserProfile(BaseModel):
    name: str
    age: str
    about: str
    education: str
    location: str
    badges: List[str]
    profileSectionResponses: dict

class ChatRequest(BaseModel):
    chat_history: str
    user_profile: dict[str, object]


class ChatResponse(BaseModel):
    conversation_starter: str
    
with open('goodOpeners.json', 'r') as f:
    GOOD_OPENERS = json.load(f)

with open('highResponseLines.json', 'r') as f:
    HIGH_RESPONSE_LINES = json.load(f)

def select_examples(source_list, n=3):
    return random.sample(source_list, min(n, len(source_list)))

        
class FeedbackRequest(BaseModel):
    is_good: bool
    message: str
    
class FeedbackResponse(BaseModel):
    id: int
    message: str
    is_good: bool
    created_at: datetime
    updated_at: datetime
    
@app.post("/feedback/", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    db_feedback = Feedback(is_good=feedback.is_good, message=feedback.message)
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

@app.get("/feedback/", response_model=List[FeedbackResponse])
def get_feedback(db: Session = Depends(get_db)):
    feedback = db.query(Feedback).all()
    return feedback


@app.post("/generate-starter/", response_model=List[ChatResponse])
async def generate_starter(request: ChatRequest):
    
    # Select example openers and response lines
    example_openers = select_examples(GOOD_OPENERS)
    example_responses = select_examples(HIGH_RESPONSE_LINES)
    
    # Create the prompt using the provided user profile only
    prompt = f"""
    ### Instructions ###
    You are a highly engaging conversational assistant specializing in helping users navigate early-stage dating conversations with humor, flirtation, and authenticity. Your goal is to craft personalized and natural conversation starters that resonate with the other person, based solely on the provided profile. Ensure the tone is friendly, respectful, with a touch of playful flirtation, and make sure the lines feel genuine and relatable.

    ### User Profile ###
    The user profile includes information such as interests, personality traits, and other relevant details. Use this data to tailor the conversation starters:
    - **Name:** {request.user_profile['name']}
    - **Age:** {request.user_profile['age']}
    - **About:** {request.user_profile['about']}
    - **Education:** {request.user_profile['education']}
    - **Location:** {request.user_profile['location']}
    - **Badges:** {", ".join(request.user_profile['badges'])}
    - **Interests/Personality:** {", ".join(f"{key}: {value}" for key, value in request.user_profile['profileSectionResponses'].items())}
    
    Here are some examples of good openers:
    {json.dumps(example_openers)}
        
    And here are some examples of high-response lines:
    {json.dumps(example_responses)}
        
    Use these examples as inspiration, but create original starters that are personalized, 
    humorous, and relevant to the user's interests. You can occasionally adapt or incorporate 
    elements from the provided examples.
    also you can just return the starters without the predecessor that here are some starters ...
    """

    try:    
        response = client.chat.completions.create(
            model=model,
            messages=[
               {"role": "system", "content": "You are a helpful assistant that generates engaging conversation starters for dating scenarios."},
               {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7,
            n=3  # Generate three different suggestions
        )

        # Process the response
        starters = [ChatResponse(conversation_starter=choice.message.content) for choice in response.choices]        
        
        # Occasionally replace a generated starter with a good opener or high response line
        for i in range(len(starters)):
            if random.random() < 0.2:  # 20% chance for each starter
                if random.random() < 0.5:  # 50% chance for good opener, 50% for high response line
                    starters[i] = ChatResponse(conversation_starter=random.choice(GOOD_OPENERS))
                else:
                    starters[i] = ChatResponse(conversation_starter=random.choice(HIGH_RESPONSE_LINES))
                            
        return starters


    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded")
    except OpenAIError as e:
        raise HTTPException(status_code=401, detail="OpenAI API authentication error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
