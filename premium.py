import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Load Environment Variables from .env file ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found in the .env file.")

# --- 2. Configure the Gemini API with your key ---
genai.configure(api_key=GOOGLE_API_KEY)


# --- 3. Pydantic Models for Robust Input/Output Validation ---

# Enums for predefined choices in the API
class FitnessGoal(str, Enum):
    FAT_LOSS = "Fat Loss"
    MUSCLE_GAIN = "Muscle Gain"
    MAINTENANCE = "Maintenance"

class DietaryPreference(str, Enum):
    VEGETARIAN = "Vegetarian"
    NON_VEGETARIAN = "Non-Vegetarian"
    VEGAN = "Vegan"
    EGGITARIAN = "Eggitarian"

# Input model: Defines the structure of the request body
class UserInput(BaseModel):
    age: int = Field(..., gt=15, lt=80, description="User's age in years")
    gender: str = Field(..., example="Male", description="User's gender (Male/Female)")
    height_cm: float = Field(..., gt=100, lt=250, description="User's height in centimeters")
    weight_kg: float = Field(..., gt=30, lt=200, description="User's weight in kilograms")
    fitness_goal: FitnessGoal = Field(..., description="Primary fitness goal")
    dietary_preference: DietaryPreference = Field(..., description="Dietary preference")
    allergies: List[str] = Field([], example=["peanuts", "gluten"], description="List of food allergies")
    meals_per_day: int = Field(5, gt=2, lt=7, description="Number of meals per day")
    is_premium: bool = Field(default=False, description="Set to true to use the higher-quality Gemini Pro model.")

# Output models: Defines the structure of the JSON response
class MealDetail(BaseModel):
    food_items: List[str] = Field(..., description="List of food items for the meal")
    calories: int = Field(..., description="Estimated calories for the meal")
    protein_g: int = Field(..., description="Estimated protein in grams")
    carbs_g: int = Field(..., description="Estimated carbohydrates in grams")
    fats_g: int = Field(..., description="Estimated fats in grams")

class DailyPlan(BaseModel):
    meals: Dict[str, MealDetail] = Field(..., description="Dictionary of meals for the day")
    daily_totals: Dict[str, int] = Field(..., description="Total macronutrients for the day")

class DietPlanResponse(BaseModel):
    plan_summary: Dict[str, str] = Field(..., description="A summary of the generated plan")
    weekly_plan: Dict[str, DailyPlan] = Field(..., description="The full 7-day diet plan")
    general_tips: List[str] = Field(..., description="General health and hydration tips")


# --- 4. Create and Configure the FastAPI App Instance ---
app = FastAPI(
    title="Gym Diet Plan Generator API",
    description="An API to generate personalized Indian diet plans for gym-goers using Google's Gemini Models.",
    version="1.1.0",
)

# Add CORS Middleware to allow requests from the frontend
# The "*" allows all origins, which is fine for development.
# For production, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 5. Define the System Prompt for the AI Model ---
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
    "Tuesday": {}, "Wednesday": {}, "Thursday": {}, "Friday": {}, "Saturday": {}, "Sunday": {}
  },
  "general_tips": [
    "Drink 3-4 liters of water throughout the day.",
    "Another relevant tip for the user's goal."
  ]
}
"""

# --- 6. Define the API Endpoints ---

@app.post("/generate-diet-plan", response_model=DietPlanResponse)
async def generate_diet_plan(user_input: UserInput = Body(...)):
    """
    Generates a 7-day personalized Indian diet plan for gym-goers.
    - Uses **Gemini 1.5 Pro** for premium requests.
    - Uses **Gemini 1.5 Flash** for standard requests.
    """
    try:
        # Define model names
        PREMIUM_MODEL = "gemini-2.5-flash"
        STANDARD_MODEL = "gemini-2.0-flash"

        # Select the model based on the 'is_premium' flag from the user input
        model_to_use = PREMIUM_MODEL if user_input.is_premium else STANDARD_MODEL
        print(f"Request received. Using model: {model_to_use}") # Helpful for server-side logging

        # Initialize the chosen generative model
        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        # Create the user-specific prompt
        user_prompt = f"Here are my details, please generate my diet plan:\n{user_input.model_dump_json(indent=2)}"

        # Asynchronously generate the content
        response = await model.generate_content_async([user_prompt])

        # Load the JSON text response into a Python dictionary
        plan_json = json.loads(response.text)

        # FastAPI will automatically validate this dictionary against the DietPlanResponse model
        return plan_json

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="The model returned an invalid JSON format. Please try again."
        )
    except Exception as e:
        # Catch any other exceptions (e.g., API key issues, network problems)
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"An error occurred while communicating with the AI service: {str(e)}"
        )

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Diet Plan API is running. Go to /docs for documentation."}