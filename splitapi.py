import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
from exercises_string import exercises

exercises_dict=json.loads(exercises)
exercises_names=exercises_dict.keys()
genai.configure(api_key="AIzaSyAGT8ojwDtHKuV5HGYbhDg4QNVM0OfXKl8")

app = FastAPI(
    title="AI Fitness Coach API",
    description="A unified API to generate personalized diet plans and workout splits with premium tiers.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models without enums or constraints ---

class UserInput(BaseModel):
    age: int
    gender: str
    height_cm: float
    fitness_goal: str
    dietary_preference: str
    cuisine: str
    allergies: List[str] = []
    meals_per_day: int
    is_premium: bool = False
    current_weight: int
    target_weight: int
    language: str
    time_span: str

class MealDetail(BaseModel):
    food_items: List[str]
    calories: int
    protein_g: int
    carbs_g: int
    fats_g: int

class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail]
    daily_totals: Dict[str, int]

class DietPlanResponse(BaseModel):
    plan_summary: Dict[str, str]
    weekly_plan: Dict[str, DailyPlan]
    general_tips: List[str]

class WorkoutRequest(BaseModel):
    days_per_week: int
    experience_level: str
    goal: str
    focus: Optional[str] = None
    is_premium: bool = False

class ExerciseDetail(BaseModel):
    name: str
    sets: str
    reps: str
    url: Optional[str] = None

class DailyWorkout(BaseModel):
    day: str
    focus: str
    exercises: List[ExerciseDetail]

class WorkoutSplitResponse(BaseModel):
    workout_plan: List[DailyWorkout]

# --- Prompt builders ---

def create_diet_prompt(cuisine: str,language:str) -> str:
    return f"""You are an expert nutritionist generating a diet plan.

**CRITICAL RULES:**
1.  **CUISINE:** The plan MUST be based on **{cuisine}** food items.
2.  **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text, markdown, or explanations outside of the JSON brackets.
3.  **STRICT SCHEMA:** The JSON object must have exactly three top-level keys: `plan_summary`, `weekly_plan`, and `general_tips`.
4.  **DATA TYPES:** All calorie and macronutrient values (`calories`, `protein_g`, `carbs_g`, `fats_g`) MUST be integers, NOT strings.
5. **Include all kind of Fruits and Juices in Diet plan**
6. The diet should be able to achive the target in given Time span.
Make diet plan in {language}
If the target is weightloss the diet must be calorie deficient and if gain then surplus.
**EXAMPLE JSON STRUCTURE TO FOLLOW:**
{{
  "plan_summary": {{
    "estimated_daily_calories": "Approx. 2200-2400 kcal",
    "estimated_daily_protein": "Approx. 150-160 g"
  }},
  "weekly_plan": {{
    "Monday": {{
      "meals": {{
        "Breakfast": {{
          "food_items": ["Oats with whey protein", "Handful of almonds"],
          "calories": 400, "protein_g": 30, "carbs_g": 50, "fats_g": 10
        }}
      }},
      "daily_totals": {{ "total_calories": 2250, "total_protein_g": 155, "total_carbs_g": 220, "total_fats_g": 65 }}
    }},
    "Tuesday": "..."  
  }},
  "general_tips": [
    "Drink 3-4 liters of water daily.",
    "Get 7-8 hours of sleep for recovery."
  ]
}}"""

@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    try:
        model_to_use = "gemini-2.5-flash" if user_input.is_premium else "gemini-2.0-flash"
        print(f"Diet plan request for {user_input.cuisine} cuisine. Using model: {model_to_use}")
        system_prompt = create_diet_prompt(user_input.cuisine,user_input.language)

        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        user_prompt_data = f"Here are my details, generate my diet plan:\n{json.dumps(user_input.model_dump(), indent=2)}"
        response = await model.generate_content_async([user_prompt_data])
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in /generate-diet-plan: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")

def create_workout_prompt(request: WorkoutRequest) -> str:
    return f"""You are an expert fitness coach. Generate a detailed, day-wise workout split based on:
- Days per Week: {request.days_per_week}
- Experience Level: {request.experience_level}
- Primary Goal: {request.goal}
{'- Specific Focus: ' + request.focus if request.focus else ''}

INSTRUCTIONS:
1. Your response MUST be a valid JSON object with a single root key: "workout_plan".
2. "workout_plan" is an array of objects, one for each day (including rest days).
3. Each day object has keys: "day", "focus", and "exercises" (an array of exercise objects).
4. Each exercise object has keys: "name", "sets", "reps".
5. Do NOT include any text, markdown, or explanations outside the JSON object.
The split should strictly include these exercises only:
{exercises_names}

exercise names must be lowered.
"""
@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    """
    Generates a weekly workout split based on user preferences.
    - Uses **Gemini 2.5 Flash** for premium requests.
    - Uses **Gemini 2.0 Flash** for standard requests.
    """
    try:
        prompt = create_workout_prompt(request)
        model_to_use = "gemini-2.5-flash" if request.is_premium else "gemini-2.0-flash"
        print(f"Workout split request. Using model: {model_to_use}")
        
        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        
        response = await model.generate_content_async(prompt)
        workout_json = json.loads(response.text)  # Parsed AI output
        
        for day_plan in workout_json["workout_plan"]:
            for exercise in day_plan["exercises"]:
                # Normalize sets/reps to strings
                if "sets" in exercise:
                    exercise["sets"] = str(exercise["sets"])
                if "reps" in exercise:
                    exercise["reps"] = str(exercise["reps"])
                
                # Add URL if found
                name = exercise["name"].lower().strip()
                if name in exercises_dict:
                    exercise["url"] = exercises_dict.get(name)
        
        return workout_json  # Must be inside try

    except Exception as e:
        print(f"Error in /generate-workout-split: {e}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


@app.get("/", tags=["Status"])
def read_root():
    return {"status": "AI Fitness Coach API is running. Go to /docs for all endpoints."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
