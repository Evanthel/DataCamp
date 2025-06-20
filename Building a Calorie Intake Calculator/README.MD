As a Software Engineer in a Health and Leisure company, your task is to add a new feature to the app: a calorie and nutrition calculator. This tool will calculate and display total calories, sugars, fats, and other nutritional values for different foods based on user input.

You have been provided with the `nutrition.json` dataset, which contains the necessary nutritional information for various foods. Each value in the dataset is per **100 grams** of the food item. The dataset has already been read and loaded for you as the dictionary `nutrition_dict`.

## Dataset Summary

`nutrition.json`

| Column        | Description                                             |
|---------------|---------------------------------------------------------|
| `food` | The name of the food.                                   |
| `calories`  | The amount of energy provided by the food, measured in kilocalories (kcal) per 100 grams. |
| `total_fat` | The total fat content in grams per 100 grams.                         |
| `protein`   | The protein content in grams per 100 grams.                           |
| `carbohydrate` | The total carbohydrate content in grams per 100 grams.             |
| `sugars`    | The amount of sugars in grams per 100 grams.                          |