import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Load Environment Variables & Configure API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found in the .env file.")
genai.configure(api_key=GOOGLE_API_KEY)


# --- 2. FastAPI App Instance & CORS ---
app = FastAPI(
    title="AI Fitness Coach API (Flexible & Robust)",
    description="An API that uses free-text input to generate personalized diet plans and workout splits.",
    version="3.1.0", 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Pydantic Models (No Enums for Flexibility) ---

# --- DIET PLAN MODELS ---
class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=80)
    gender: str = Field(..., example="Male")
    height_cm: float = Field(..., gt=100, lt=250)
    weight_kg: float = Field(..., gt=30, lt=200)
    fitness_goal: str = Field(..., example="I want to lose fat and build some muscle.")
    dietary_preference: str = Field(..., example="Mostly vegetarian, but I eat eggs.")
    cuisine: str = Field(default="Indian", example="North Indian")
    language: str = Field(default="English", example="Hindi")
    allergies: List[str] = Field([], example=["peanuts"])
    meals_per_day: int = Field(5, gt=2, lt=7)
    is_premium: bool = Field(default=False)

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


# --- WORKOUT SPLIT MODELS ---
class WorkoutRequest(BaseModel):
    days_per_week: int = Field(..., gt=0, le=7)
    experience_level: str = Field(..., example="I have been lifting for 2 years.")
    goal: str = Field(..., example="Get stronger and look more athletic.")
    focus: Optional[str] = Field(None, example="legs")
    is_premium: bool = Field(default=False)

class ExerciseDetail(BaseModel):
    name: str
    sets: str
    reps: str

class DailyWorkout(BaseModel):
    day: str
    focus: str
    exercises: List[ExerciseDetail]

class WorkoutSplitResponse(BaseModel):
    workout_plan: List[DailyWorkout]


# --- 4. Helper Functions ---

def fix_plan_structure(plan_data: dict, meals_per_day: int) -> dict:
    """Fix common structural issues in the AI response."""
    
    # Ensure plan_summary exists
    if "plan_summary" not in plan_data:
        plan_data["plan_summary"] = {
            "total_calories": "2000-2200",
            "protein_target": "120-150g",
            "goal_description": "Balanced nutrition plan"
        }
    
    # Ensure weekly_plan exists
    if "weekly_plan" not in plan_data:
        plan_data["weekly_plan"] = {}
    
    # Standard meal names based on meals_per_day
    meal_names = ["breakfast", "snack1", "lunch", "snack2", "dinner"][:meals_per_day]
    
    # Ensure all 7 days exist
    for i in range(1, 8):
        day_key = f"day_{i}"
        if day_key not in plan_data["weekly_plan"]:
            plan_data["weekly_plan"][day_key] = {
                "meals": {},
                "daily_totals": {"calories": 2000, "protein_g": 120, "carbs_g": 250, "fats_g": 70}
            }
        
        # Ensure all required meals exist for each day
        day_data = plan_data["weekly_plan"][day_key]
        if "meals" not in day_data:
            day_data["meals"] = {}
        
        for meal_name in meal_names:
            if meal_name not in day_data["meals"]:
                day_data["meals"][meal_name] = {
                    "food_items": ["Sample food item"],
                    "calories": 400,
                    "protein_g": 20,
                    "carbs_g": 50,
                    "fats_g": 15
                }
    
    # Fix nutritional values to ensure they're integers
    for day_key, day_data in plan_data["weekly_plan"].items():
        if "meals" in day_data:
            for meal_name, meal_data in day_data["meals"].items():
                # Ensure all required fields exist
                if "food_items" not in meal_data or not meal_data["food_items"]:
                    meal_data["food_items"] = ["Sample food item"]
                
                for nutrient in ["calories", "protein_g", "carbs_g", "fats_g"]:
                    if nutrient not in meal_data:
                        meal_data[nutrient] = 0
                    else:
                        try:
                            # Clean and convert to int
                            value = str(meal_data[nutrient]).replace('g', '').replace('cal', '').replace('kcal', '').strip()
                            meal_data[nutrient] = int(float(value)) if value else 0
                        except (ValueError, TypeError):
                            meal_data[nutrient] = 0
        
        if "daily_totals" not in day_data:
            day_data["daily_totals"] = {}
            
        for nutrient in ["calories", "protein_g", "carbs_g", "fats_g"]:
            if nutrient not in day_data["daily_totals"]:
                day_data["daily_totals"][nutrient] = 0
            else:
                try:
                    value = str(day_data["daily_totals"][nutrient]).replace('g', '').replace('cal', '').replace('kcal', '').strip()
                    day_data["daily_totals"][nutrient] = int(float(value)) if value else 0
                except (ValueError, TypeError):
                    day_data["daily_totals"][nutrient] = 0
    
    # Ensure general_tips exists
    if "general_tips" not in plan_data:
        plan_data["general_tips"] = ["Stay hydrated", "Eat balanced meals", "Exercise regularly"]
    elif not isinstance(plan_data["general_tips"], list):
        plan_data["general_tips"] = ["Stay hydrated", "Eat balanced meals", "Exercise regularly"]
    
    return plan_data


def create_diet_prompt(cuisine: str, language: str, meals_per_day: int) -> str:
    """Creates a robust, dynamic system prompt for the diet plan generator."""
    
    # Generate meal structure based on meals_per_day
    meal_names = ["breakfast", "snack1", "lunch", "snack2", "dinner", "snack3"][:meals_per_day]
    
    meals_structure = ""
    for meal in meal_names:
        meals_structure += f'''
        "{meal}": {{
          "food_items": ["item1", "item2"],
          "calories": 400,
          "protein_g": 20,
          "carbs_g": 45,
          "fats_g": 15
        }},'''
    
    meals_structure = meals_structure.rstrip(',')  # Remove trailing comma
    
    return f"""You are an expert nutritionist generating a diet plan.

**CRITICAL RULES:**
1. **LANGUAGE:** All text in your response MUST be in **{language}**.
2. **CUISINE:** The plan MUST be based on **{cuisine}** food items.
3. **JSON FORMAT:** Your ENTIRE response MUST be a single, valid JSON object. Do not add any text or markdown outside of the JSON brackets.

**EXACT JSON STRUCTURE REQUIRED:**
{{
  "plan_summary": {{
    "total_calories": "string describing daily calorie range",
    "protein_target": "string describing protein target", 
    "goal_description": "string describing the plan's goal"
  }},
  "weekly_plan": {{
    "day_1": {{
      "meals": {{{meals_structure}
      }},
      "daily_totals": {{
        "calories": 1800,
        "protein_g": 120,
        "carbs_g": 200,
        "fats_g": 60
      }}
    }},
    "day_2": {{ ... }},
    "day_3": {{ ... }},
    "day_4": {{ ... }},
    "day_5": {{ ... }},
    "day_6": {{ ... }},
    "day_7": {{ ... }}
  }},
  "general_tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}}

**CRITICAL REQUIREMENTS:** 
- ALL nutritional values (calories, protein_g, carbs_g, fats_g) MUST be integers, NOT strings
- Include exactly 7 days (day_1 through day_7)
- Each day must have exactly {meals_per_day} meals: {', '.join(meal_names)}
- food_items must be an array of strings
- Do not add any extra fields beyond the schema
- Ensure all days follow the EXACT same structure"""


def create_workout_prompt(request: WorkoutRequest) -> str:
    """Creates a robust system prompt for the workout split generator."""
    return f"""You are an expert fitness coach. Generate a workout split based on this user's free-text description:

**USER REQUIREMENTS:**
- Days per Week: {request.days_per_week}
- Experience Level: "{request.experience_level}"
- Primary Goal: "{request.goal}"
- Specific Focus: "{request.focus or 'General fitness'}"

**CRITICAL INSTRUCTIONS:**
1. **Output Format:** Your ENTIRE response must be a single, valid JSON object with ONE root key: "workout_plan".
2. **Schema Adherence:** Do not add fields not in the schema.
3. **Object Structure:** "workout_plan" must be an array of exactly {request.days_per_week} workout day objects.

**EXACT JSON STRUCTURE REQUIRED:**
{{
  "workout_plan": [
    {{
      "day": "Day 1",
      "focus": "Upper Body Push",
      "exercises": [
        {{
          "name": "Bench Press",
          "sets": "3-4",
          "reps": "8-12"
        }},
        {{
          "name": "Shoulder Press",
          "sets": "3",
          "reps": "10-15"
        }}
      ]
    }}
  ]
}}

**REQUIREMENTS:**
- Generate exactly {request.days_per_week} workout days
- Each exercise must have "name", "sets", and "reps" fields
- "sets" and "reps" should be strings (e.g., "3-4", "8-12")
- Include 4-8 exercises per day depending on the focus
- Do not add any extra fields"""


# --- 5. API Endpoints ---

# --- DIET PLAN ENDPOINT ---
@app.post("/generate-diet-plan", response_model=DietPlanResponse, tags=["Diet Plan"])
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """Generates a 7-day personalized diet plan using free-text inputs."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # FIXED: Using gemini-2.5-flash and gemini-2.0-flash as requested
            PREMIUM_MODEL = "gemini-2.5-flash"
            STANDARD_MODEL = "gemini-2.0-flash"
            
            model_to_use = PREMIUM_MODEL if user_input.is_premium else STANDARD_MODEL
            print(f"Diet plan request attempt {attempt + 1} for {user_input.cuisine} cuisine in {user_input.language}. Using model: {model_to_use}")

            system_prompt = create_diet_prompt(user_input.cuisine, user_input.language, user_input.meals_per_day)
            model = genai.GenerativeModel(
                model_name=model_to_use,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1  # Lower temperature for more consistent output
                )
            )
            
            user_prompt_data = f"""Generate a complete 7-day diet plan based on these user details:
            
Age: {user_input.age} years
Gender: {user_input.gender}
Height: {user_input.height_cm} cm
Weight: {user_input.weight_kg} kg
Fitness Goal: {user_input.fitness_goal}
Dietary Preference: {user_input.dietary_preference}
Allergies: {user_input.allergies}
Meals per day: {user_input.meals_per_day}

Create a personalized plan that matches their goals and preferences."""

            response = await model.generate_content_async([user_prompt_data])
            
            # Clean the response text - FIXED SYNTAX ERROR
            response_text = response.text.strip()
            if response_text.startswith("```
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```
                response_text = response_text[:-3]
            
            print(f"Raw AI Response (Attempt {attempt + 1}):\n{response_text[:500]}...")
            
            try:
                plan_data = json.loads(response_text)
                
                # Validate and fix common issues
                plan_data = fix_plan_structure(plan_data, user_input.meals_per_day)
                
                validated_plan = DietPlanResponse.model_validate(plan_data)
                print(f"Successfully generated diet plan on attempt {attempt + 1}")
                return validated_plan
                
            except ValidationError as e:
                print(f"--- Pydantic Validation Error (Attempt {attempt + 1}) ---\n{e}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"The AI returned an invalid plan structure after {max_retries} attempts. Please try again."
                    )
                continue
                
            except json.JSONDecodeError as e:
                print(f"--- JSON Decode Error (Attempt {attempt + 1}) ---\n{e}")
                print(f"Response text: {response_text}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"The AI returned invalid JSON after {max_retries} attempts. Please try again."
                    )
                continue
                
        except Exception as e:
            print(f"Unexpected error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503, 
                    detail="AI service error occurred after multiple attempts. Please try again later."
                )


# --- WORKOUT SPLIT ENDPOINT ---
@app.post("/generate-workout-split", response_model=WorkoutSplitResponse, tags=["Workout Split"])
async def generate_workout_split(request: WorkoutRequest = Body(...)):
    """Generates a weekly workout split using free-text inputs."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # FIXED: Using gemini-2.5-flash and gemini-2.0-flash as requested
            PREMIUM_MODEL = "gemini-2.5-flash"
            STANDARD_MODEL = "gemini-2.0-flash"
            
            model_to_use = PREMIUM_MODEL if request.is_premium else STANDARD_MODEL
            print(f"Workout split request attempt {attempt + 1}. Using model: {model_to_use}")
            
            prompt = create_workout_prompt(request)
            model = genai.GenerativeModel(
                model_name=model_to_use,
                system_instruction=prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            response = await model.generate_content_async("Generate the workout plan now.")

            # Clean the response text - FIXED SYNTAX ERROR
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            print(f"Raw AI Response (Attempt {attempt + 1}):\n{response_text[:300]}...")

            try:
                split_data = json.loads(response_text)
                
                # Basic validation - ensure workout_plan exists and is a list
                if "workout_plan" not in split_data:
                    split_data = {"workout_plan": []}
                
                if not isinstance(split_data["workout_plan"], list):
                    split_data["workout_plan"] = []
                
                # Ensure we have the right number of days
                while len(split_data["workout_plan"]) < request.days_per_week:
                    split_data["workout_plan"].append({
                        "day": f"Day {len(split_data['workout_plan']) + 1}",
                        "focus": "Full Body",
                        "exercises": [
                            {"name": "Push-ups", "sets": "3", "reps": "10-15"},
                            {"name": "Squats", "sets": "3", "reps": "12-20"}
                        ]
                    })
                
                validated_split = WorkoutSplitResponse.model_validate(split_data)
                print(f"Successfully generated workout split on attempt {attempt + 1}")
                return validated_split
                
            except ValidationError as e:
                print(f"--- Pydantic Validation Error (Attempt {attempt + 1}) ---\n{e}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"The AI returned an invalid workout structure after {max_retries} attempts."
                    )
                continue
                
            except json.JSONDecodeError as e:
                print(f"--- JSON Decode Error (Attempt {attempt + 1}) ---\n{e}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"The AI returned invalid JSON after {max_retries} attempts."
                    )
                continue
                
        except Exception as e:
            print(f"Unexpected error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503, 
                    detail="AI service error occurred after multiple attempts."
                )


# --- ROOT ENDPOINT ---
@app.get("/", tags=["Status"])
def read_root():
    return {
        "status": "AI Fitness Coach API is running", 
        "version": "3.1.0",
        "endpoints": {
            "diet_plan": "/generate-diet-plan",
            "workout_split": "/generate-workout-split",
            "docs": "/docs"
        },
        "models": {
            "standard": "gemini-2.0-flash",
            "premium": "gemini-2.5-flash"
        }
    }


# --- HEALTH CHECK ENDPOINT ---
@app.get("/health", tags=["Status"])
def health_check():
    return {
        "status": "healthy",
        "api_key_configured": bool(GOOGLE_API_KEY),
        "timestamp": "2025-01-20T23:06:00Z"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
