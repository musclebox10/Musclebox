import os
import json
from fastapi import FastAPI, HTTPException, Body
from enum import Enum  # <-- CORRECT: Import Enum from Python's standard library
from pydantic import BaseModel, Field
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware




# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# --- Configure the Gemini API ---
genai.configure(api_key=GOOGLE_API_KEY)


# --- Pydantic Models for Structured Input/Output ---

class FitnessGoal(str, Enum):
    FAT_LOSS = "Fat Loss"
    MUSCLE_GAIN = "Muscle Gain"
    MAINTENANCE = "Maintenance"

class DietaryPreference(str, Enum):
    VEGETARIAN = "Vegetarian"
    NON_VEGETARIAN = "Non-Vegetarian"
    VEGAN = "Vegan"
    EGGITARIAN = "Eggitarian"

class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=80, description="User's age in years")
    gender: str = Field(..., example="Male", description="User's gender (Male/Female)")
    height_cm: float = Field(..., gt=100, lt=250, description="User's height in centimeters")
    weight_kg: float = Field(..., gt=30, lt=200, description="User's weight in kilograms")
    fitness_goal: FitnessGoal = Field(..., description="Primary fitness goal")
    dietary_preference: DietaryPreference = Field(..., description="Dietary preference")
    allergies: List[str] = Field([], example=["peanuts", "gluten"], description="List of food allergies")
    meals_per_day: int = Field(5, gt=2, lt=7, description="Number of meals per day (e.g., 3 main, 2 snacks)")


class MealDetail(BaseModel):
    food_items: List[str] = Field(..., description="List of food items for the meal")
    calories: int = Field(..., description="Estimated calories for the meal")
    protein_g: int = Field(..., description="Estimated protein in grams")
    carbs_g: int = Field(..., description="Estimated carbohydrates in grams")
    fats_g: int = Field(..., description="Estimated fats in grams")

class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail] = Field(..., description="Dictionary of meals for the day (e.g., 'Breakfast', 'Lunch')")
    daily_totals: Dict[str, int] = Field(..., description="Total macronutrients for the day")

class DietPlanResponse(BaseModel):
    plan_summary: Dict[str, str] = Field(..., description="A summary of the generated plan")
    weekly_plan: Dict[str, DailyPlan] = Field(..., description="The full 7-day diet plan")
    general_tips: List[str] = Field(..., description="General health and hydration tips")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Gym Diet Plan Generator API",
    description="An API to generate personalized Indian diet plans for gym-goers using Google's Gemini Pro.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Prompt for the Gemini Model ---
# This is the most crucial part. It sets the context, role, rules, and output format.
SYSTEM_PROMPT = """
You are an expert nutritionist and fitness coach specializing in Indian cuisine for gym-goers.
Your task is to generate a detailed, 7-day diet plan based on the user's specifications.

**CRITICAL INSTRUCTIONS:**
1.  **Cuisine:** The entire diet plan MUST be based on common, easily available Indian food items.
2.  **Target Audience:** The plan is for individuals who are active and go to the gym. Protein intake should be prioritized according to their goal (higher for muscle gain, moderate-high for fat loss).
3.  **Calculations:** Based on the user's details, first estimate their daily calorie and macronutrient needs (Protein, Carbs, Fats). The final plan's daily totals should be close to these estimates.
4.  **Structure:** The output MUST be a valid JSON object that strictly adheres to the provided schema. Do NOT include any text, explanations, or markdown formatting like ```json before or after the JSON object.
5.  **Variety:** Provide a varied plan for all 7 days to ensure a range of nutrients and prevent monotony.
6.  **Practicality:** Suggest realistic portion sizes (e.g., "1 katori of dal", "100g of chicken breast", "2 whole eggs", "1 scoop of whey protein").
7.  **Meal Naming:** Use clear meal names like "Early Morning", "Breakfast", "Mid-Morning Snack", "Lunch", "Pre-Workout", "Post-Workout / Dinner", "Bedtime". Adjust the number of meals based on the user's request.
8.  **Hydration:** Always include a general tip about staying hydrated.

**OUTPUT JSON SCHEMA:**
{
  "plan_summary": {
    "estimated_daily_calories": "A string like 'Approx. 2200-2400 kcal'",
    "estimated_daily_protein": "A string like 'Approx. 150-160 g'",
    "estimated_daily_carbs": "A string like 'Approx. 200-220 g'",
    "estimated_daily_fats": "A string like 'Approx. 60-70 g'"
  },
  "weekly_plan": {
    "Monday": {
      "meals": {
        "Meal Name 1": {
          "food_items": ["Food item 1 with portion", "Food item 2 with portion"],
          "calories": 250, "protein_g": 20, "carbs_g": 30, "fats_g": 5
        }
      },
      "daily_totals": { "total_calories": 2300, "total_protein_g": 155, "total_carbs_g": 210, "total_fats_g": 65 }
    },
    "... other days ..."
  },
  "general_tips": [
    "Drink 3-4 liters of water throughout the day.",
    "Another relevant tip for the user's goal."
  ]
}
"""

# --- API Endpoint ---

@app.post("/generate-diet-plan", response_model=DietPlanResponse)
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """
    Generates a 7-day personalized Indian diet plan for gym-goers.

    Provide the user's details in the request body to receive a structured JSON diet plan.
    """
    try:
        # Initialize the Gemini Pro model with JSON output configuration
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        # Create the user-specific prompt
        user_prompt = f"Here are my details, please generate my diet plan:\n{user_input.model_dump_json(indent=2)}"

        # Generate content
        response = await model.generate_content_async([user_prompt])

        # The response.text should be a valid JSON string due to the response_mime_type config
        # We parse it to ensure it's valid and to let Pydantic re-validate it on return
        plan_json = json.loads(response.text)
        return plan_json

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid JSON format. Please try again."
        )
    except Exception as e:
        # Catch-all for other potential errors (API issues, etc.)
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"An error occurred while communicating with the AI service: {str(e)}"
        )

# Optional: Add a root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "API is running. Go to /docs for documentation."}