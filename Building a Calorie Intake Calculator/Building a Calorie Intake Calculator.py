import json  # Import the json module to work with JSON files

# Open the nutrition.json file in read mode and load its content into a dictionary
with open('nutrition.json', 'r') as json_file:
    nutrition_dict = json.load(json_file)  # Load the JSON content into a dictionary
    
# Display the first 3 items of the nutrition dictionary
list(nutrition_dict.items())[:3]

def nutritional_summary(foods):
    total_nutrition = {
    'calories': 0,
    'total_fat': 0,
    'protein': 0,
    'carbohydrate': 0,
    'sugars': 0
}

    for name, grams in foods.items():
        if name in nutrition_dict:
            total_nutrition["calories"] += grams * nutrition_dict[name]["calories"] / 100
            total_nutrition["total_fat"] += grams * nutrition_dict[name]["total_fat"] / 100
            total_nutrition["protein"] += grams * nutrition_dict[name]["protein"] / 100
            total_nutrition["carbohydrate"] += grams * nutrition_dict[name]["carbohydrate"] / 100
            total_nutrition["sugars"] += grams * nutrition_dict[name]["sugars"] / 100
        else:
            return name
    return total_nutrition