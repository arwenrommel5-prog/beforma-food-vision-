"""
BeForma V1 Enhanced — calorie_lookup.json Generator
Generates a per-class calorie/nutrition database for all 251 FoodX classes.

Run once, then commit calorie_lookup.json alongside your model files.
Usage:
    python generate_calorie_lookup.py --class_list class_list.txt --out calorie_lookup.json
"""

import json, argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Nutrition table: {class_name: (kcal/100g, default_g, small_g, medium_g, large_g,
#                                protein_g/100g, carbs_g/100g, fat_g/100g)}
# Sources: USDA FoodData Central, FDA, standard food composition tables.
# All values are per 100 g of edible portion unless noted.
# ---------------------------------------------------------------------------
NUTRITION_DB: dict[str, tuple] = {
    # name                       kcal  def  sm   med  lg    prot  carb  fat
    "apple_pie":                (237,  200, 100, 200, 300,   2.0, 34.0,  11.0),
    "baby_back_ribs":           (297,  250, 150, 250, 400,  24.0,  0.0,  22.0),
    "baklava":                  (428,   80,  50,  80, 120,   5.0, 45.0,  28.0),
    "beef_carpaccio":           (180,  100,  60, 100, 160,  20.0,  0.5,  10.0),
    "beef_tartare":             (196,  150,  80, 150, 220,  20.0,  1.0,  13.0),
    "beet_salad":               ( 65,  150, 100, 150, 250,   2.5, 12.0,   1.5),
    "beignets":                 (320,  100,  60, 100, 180,   4.0, 40.0,  18.0),
    "bibimbap":                 (130,  350, 250, 350, 500,   7.0, 20.0,   3.0),
    "bread_pudding":            (250,  200, 120, 200, 300,   6.5, 35.0,  10.0),
    "bruschetta":               (180,  120,  80, 120, 200,   5.0, 25.0,   6.5),
    "caesar_salad":             (100,  200, 150, 200, 300,   5.0,  6.0,   8.0),
    "cannoli":                  (368,   80,  50,  80, 130,   7.0, 38.0,  20.0),
    "caprese_salad":            (150,  200, 120, 200, 300,   8.0,  4.0,  11.0),
    "carrot_cake":              (415,  120,  80, 120, 180,   4.0, 55.0,  21.0),
    "ceviche":                  ( 90,  200, 130, 200, 300,  14.0,  6.0,   2.0),
    "cheese_plate":             (350,  100,  60, 100, 150,  20.0,  1.0,  29.0),
    "cheesecake":               (321,  150,  80, 150, 230,   5.5, 30.0,  22.0),
    "chicken_curry":            (165,  300, 200, 300, 450,  15.0, 10.0,   8.0),
    "chicken_quesadilla":       (222,  250, 150, 250, 380,  16.0, 20.0,  10.0),
    "chicken_wings":            (290,  200, 120, 200, 350,  27.0,  0.0,  20.0),
    "chocolate_cake":           (371,  120,  80, 120, 180,   5.0, 52.0,  17.0),
    "chocolate_mousse":         (220,  150,  80, 150, 250,   5.0, 20.0,  14.0),
    "churros":                  (364,  100,  60, 100, 170,   5.0, 48.0,  17.0),
    "clam_chowder":             (109,  350, 250, 350, 500,   6.0, 12.0,   4.5),
    "club_sandwich":            (250,  280, 180, 280, 380,  18.0, 25.0,  10.0),
    "crab_cakes":               (220,  200, 120, 200, 300,  16.0, 12.0,  12.0),
    "creme_brulee":             (280,  150, 100, 150, 230,   5.0, 24.0,  18.0),
    "croque_madame":            (315,  200, 130, 200, 300,  18.0, 20.0,  18.0),
    "cup_cakes":                (370,   80,  50,  80, 130,   4.0, 52.0,  17.0),
    "deviled_eggs":             (190,   80,  40,  80, 120,  10.0,  2.0,  15.0),
    "donuts":                   (452,   65,  40,  65, 100,   5.0, 52.0,  25.0),
    "dumplings":                (155,  150, 100, 150, 250,   8.0, 20.0,   5.0),
    "edamame":                  (122,  150, 100, 150, 250,  11.0,  9.0,   5.0),
    "eggs_benedict":            (240,  250, 160, 250, 380,  14.0, 17.0,  14.0),
    "escargots":                (180,  150, 100, 150, 200,  16.0,  2.0,  11.0),
    "falafel":                  (333,  150, 100, 150, 250,  13.0, 31.0,  18.0),
    "filet_mignon":             (271,  200, 130, 200, 300,  26.0,  0.0,  18.0),
    "fish_and_chips":           (250,  350, 200, 350, 500,  15.0, 25.0,  12.0),
    "foie_gras":                (462,   80,  40,  80, 120,  11.0,  4.0,  44.0),
    "french_fries":             (312,  150,  80, 150, 250,   3.5, 41.0,  15.0),
    "french_onion_soup":        (100,  350, 250, 350, 500,   5.0, 12.0,   4.0),
    "french_toast":             (229,  200, 130, 200, 300,   8.0, 28.0,  10.0),
    "fried_calamari":           (175,  180, 100, 180, 280,  15.0, 13.0,   7.0),
    "fried_rice":               (163,  300, 200, 300, 450,   4.5, 30.0,   4.0),
    "frozen_yogurt":            (127,  180, 100, 180, 280,   4.0, 26.0,   2.0),
    "garlic_bread":             (330,  100,  60, 100, 180,   7.0, 42.0,  15.0),
    "gnocchi":                  (131,  250, 150, 250, 380,   3.0, 27.0,   1.0),
    "greek_salad":              ( 95,  200, 150, 200, 300,   3.0,  7.0,   7.0),
    "grilled_cheese_sandwich":  (380,  200, 130, 200, 300,  16.0, 30.0,  22.0),
    "grilled_salmon":           (206,  200, 130, 200, 300,  25.0,  0.0,  12.0),
    "guacamole":                (160,  100,  60, 100, 180,   2.0,  9.0,  15.0),
    "gyoza":                    (180,  150, 100, 150, 250,   8.0, 20.0,   7.0),
    "hamburger":                (295,  250, 180, 250, 380,  17.0, 24.0,  14.0),
    "hot_and_sour_soup":        ( 50,  350, 250, 350, 500,   4.0,  6.0,   1.5),
    "hot_dog":                  (290,  150, 100, 150, 250,  11.0, 23.0,  17.0),
    "huevos_rancheros":         (150,  300, 200, 300, 450,  10.0, 13.0,   8.0),
    "hummus":                   (166,  100,  60, 100, 180,   8.0, 14.0,   9.5),
    "ice_cream":                (207,  150, 100, 150, 250,   3.5, 24.0,  11.0),
    "lobster_bisque":           (110,  350, 250, 350, 500,   7.0,  9.0,   5.0),
    "lobster_roll_sandwich":    (220,  280, 180, 280, 400,  18.0, 20.0,   8.0),
    "macaroni_and_cheese":      (164,  350, 250, 350, 500,   7.5, 20.0,   6.5),
    "macarons":                 (440,   50,  30,  50,  80,   5.0, 66.0,  18.0),
    "miso_soup":                ( 40,  350, 250, 350, 500,   3.0,  5.0,   1.0),
    "mussels":                  ( 86,  200, 130, 200, 300,  12.0,  4.0,   2.0),
    "nachos":                   (306,  200, 120, 200, 350,   9.0, 36.0,  16.0),
    "omelette":                 (154,  200, 130, 200, 300,  11.0,  0.5,  12.0),
    "onion_rings":              (411,  150, 100, 150, 250,   5.0, 48.0,  22.0),
    "oysters":                  ( 68,  150, 100, 150, 250,   7.0,  3.7,   2.5),
    "pad_thai":                 (195,  300, 200, 300, 450,  11.0, 26.0,   5.5),
    "paella":                   (190,  350, 250, 350, 500,  14.0, 22.0,   5.0),
    "pancakes":                 (227,  200, 130, 200, 350,   6.0, 35.0,   7.0),
    "panna_cotta":              (185,  150, 100, 150, 250,   4.0, 20.0,  10.0),
    "peking_duck":              (338,  250, 150, 250, 380,  19.0,  0.0,  29.0),
    "pho":                      ( 50,  500, 350, 500, 700,   5.0,  8.0,   1.0),
    "pizza":                    (266,  200, 100, 200, 350,  11.0, 33.0,  10.0),
    "pork_chop":                (231,  250, 150, 250, 380,  26.0,  0.0,  14.0),
    "poutine":                  (223,  300, 200, 300, 450,   8.0, 27.0,  10.0),
    "prime_rib":                (300,  300, 200, 300, 450,  27.0,  0.0,  21.0),
    "pulled_pork_sandwich":     (258,  280, 180, 280, 400,  19.0, 24.0,  10.0),
    "ramen":                    (100,  500, 350, 500, 700,   7.0, 13.0,   3.0),
    "ravioli":                  (170,  250, 150, 250, 380,   8.0, 23.0,   5.0),
    "red_velvet_cake":          (367,  120,  80, 120, 180,   4.0, 52.0,  17.0),
    "risotto":                  (145,  300, 200, 300, 450,   5.0, 26.0,   3.5),
    "samosa":                   (308,  100,  60, 100, 160,   6.0, 36.0,  16.0),
    "sashimi":                  (130,  150, 100, 150, 250,  22.0,  0.0,   5.0),
    "scallops":                 ( 88,  200, 130, 200, 300,  17.0,  5.0,   1.0),
    "seaweed_salad":            ( 45,  150, 100, 150, 250,   2.0,  7.0,   0.5),
    "shrimp_and_grits":         (160,  300, 200, 300, 450,  12.0, 16.0,   5.5),
    "spaghetti_bolognese":      (175,  350, 250, 350, 500,  11.0, 23.0,   5.0),
    "spaghetti_carbonara":      (237,  350, 250, 350, 500,  11.0, 27.0,  11.0),
    "spring_rolls":             (200,  150, 100, 150, 250,   5.0, 28.0,   8.0),
    "steak":                    (270,  250, 150, 250, 380,  26.0,  0.0,  17.0),
    "strawberry_shortcake":     (255,  180, 100, 180, 280,   4.0, 36.0,  11.0),
    "sushi":                    (145,  200, 130, 200, 300,  10.0, 26.0,   2.0),
    "tacos":                    (218,  200, 130, 200, 350,  12.0, 22.0,   9.0),
    "takoyaki":                 (200,  150, 100, 150, 250,   8.0, 22.0,   9.0),
    "tiramisu":                 (283,  150, 100, 150, 250,   5.0, 28.0,  17.0),
    "tuna_tartare":             (160,  150, 100, 150, 250,  24.0,  2.0,   6.0),
    "waffles":                  (291,  200, 130, 200, 350,   7.0, 38.0,  13.0),
    # Additional FoodX-251 classes
    "beignet":                  (320,  100,  60, 100, 180,   4.0, 40.0,  18.0),
    "cruller":                  (380,   70,  45,  70, 110,   5.0, 45.0,  22.0),
    "cockle_food":              ( 79,  150, 100, 150, 250,  14.0,  3.0,   1.5),
    "tostada":                  (235,  180, 120, 180, 280,   9.0, 25.0,  11.0),
    "moussaka":                 (178,  300, 200, 300, 450,  11.0, 13.0,   9.0),
    "dumpling":                 (155,  150, 100, 150, 250,   8.0, 20.0,   5.0),
    "macaron":                  (440,   50,  30,  50,  80,   5.0, 66.0,  18.0),
    "sashimi":                  (130,  150, 100, 150, 250,  22.0,  0.0,   5.0),
    "wonton_soup":              ( 80,  350, 250, 350, 500,   5.0, 10.0,   2.5),
    "pork_belly":               (518,  200, 120, 200, 300,  14.0,  0.0,  53.0),
    "laksa":                    (155,  400, 300, 400, 600,   9.0, 18.0,   6.0),
    "beef_stew":                (140,  350, 250, 350, 500,  14.0, 10.0,   5.0),
    "lamb_chops":               (294,  250, 150, 250, 380,  24.0,  0.0,  22.0),
    "risotto":                  (145,  300, 200, 300, 450,   5.0, 26.0,   3.5),
    "shawarma":                 (220,  280, 180, 280, 400,  18.0, 18.0,  10.0),
    "kebab":                    (195,  250, 150, 250, 380,  20.0,  5.0,  11.0),
    "fajitas":                  (185,  300, 200, 300, 450,  15.0, 17.0,   7.0),
    "carne_asada":              (230,  250, 150, 250, 380,  25.0,  1.0,  14.0),
    "bibimbap":                 (130,  350, 250, 350, 500,   7.0, 20.0,   3.0),
    "bulgogi":                  (200,  250, 150, 250, 380,  21.0,  8.0,  10.0),
    "katsu_curry":              (220,  400, 300, 400, 600,  16.0, 25.0,   7.0),
    "tonkatsu":                 (280,  200, 130, 200, 300,  20.0, 14.0,  16.0),
    "tempura":                  (220,  200, 130, 200, 300,  10.0, 20.0,  12.0),
    "okonomiyaki":              (195,  250, 150, 250, 380,  10.0, 22.0,   8.0),
    "teriyaki_chicken":         (180,  250, 150, 250, 380,  22.0,  8.0,   7.0),
    "mochi":                    (230,  100,  60, 100, 150,   3.0, 50.0,   1.0),
    "matcha_cake":              (320,  120,  80, 120, 180,   5.0, 45.0,  14.0),
    "dim_sum":                  (190,  200, 130, 200, 300,  10.0, 22.0,   8.0),
    "peking_duck":              (338,  250, 150, 250, 380,  19.0,  0.0,  29.0),
    "noodle_soup":              ( 70,  500, 350, 500, 700,   5.0, 11.0,   1.5),
    "daal":                     (116,  300, 200, 300, 450,   8.0, 18.0,   2.5),
    "butter_chicken":           (185,  300, 200, 300, 450,  16.0,  8.0,   9.0),
    "biryani":                  (180,  400, 300, 400, 600,  12.0, 26.0,   5.0),
    "naan":                     (317,  100,  60, 100, 180,   9.0, 56.0,   7.0),
    "chapati":                  (297,  100,  60, 100, 180,   9.0, 55.0,   4.0),
    "idli":                     ( 58,  150, 100, 150, 250,   3.5, 12.0,   0.3),
    "dosa":                     (165,  150, 100, 150, 280,   3.5, 30.0,   3.5),
    "sambar":                   ( 55,  250, 150, 250, 400,   3.5,  9.0,   1.5),
    "chutney":                  ( 80,   30,  15,  30,  60,   1.0, 18.0,   0.5),
    "korma":                    (200,  300, 200, 300, 450,  15.0,  9.0,  12.0),
    "vindaloo":                 (175,  300, 200, 300, 450,  16.0,  7.0,  10.0),
    "tandoori_chicken":         (175,  250, 150, 250, 380,  22.0,  4.0,   8.0),
    "palak_paneer":             (180,  300, 200, 300, 450,   9.0,  8.0,  12.0),
    "matar_paneer":             (165,  300, 200, 300, 450,   9.0, 10.0,  10.0),
    "chole":                    (164,  300, 200, 300, 450,   9.0, 24.0,   5.0),
    "rajma":                    (127,  300, 200, 300, 450,   9.0, 22.0,   0.5),
    "khichdi":                  (120,  300, 200, 300, 450,   5.0, 22.0,   2.0),
    "pav_bhaji":                (175,  300, 200, 300, 450,   5.0, 28.0,   5.0),
    "vada_pav":                 (290,  150, 100, 150, 250,   7.0, 40.0,  12.0),
    "kachori":                  (360,  100,  60, 100, 180,   7.0, 44.0,  18.0),
    "jalebi":                   (400,  100,  60, 100, 180,   1.5, 77.0,  10.0),
    "gulab_jamun":              (382,  100,  60, 100, 180,   5.0, 62.0,  14.0),
    "halwa":                    (340,  150, 100, 150, 250,   4.0, 55.0,  12.0),
    "kheer":                    (155,  200, 130, 200, 300,   4.0, 25.0,   5.0),
    "rasmalai":                 (215,  150, 100, 150, 250,   6.0, 25.0,  10.0),
    "flan":                     (150,  150, 100, 150, 250,   4.5, 23.0,   5.0),
    "churro":                   (364,  100,  60, 100, 170,   5.0, 48.0,  17.0),
    "empanada":                 (292,  150, 100, 150, 250,   9.0, 28.0,  16.0),
    "arepa":                    (230,  180, 120, 180, 280,   5.5, 38.0,   5.5),
    "pupusa":                   (210,  180, 120, 180, 280,   7.0, 33.0,   6.5),
    "ceviche":                  ( 90,  200, 130, 200, 300,  14.0,  6.0,   2.0),
    "lomo_saltado":             (215,  350, 250, 350, 500,  14.0, 20.0,   8.0),
    "aji_de_gallina":           (180,  300, 200, 300, 450,  15.0, 14.0,   7.0),
    "bandeja_paisa":            (400,  500, 350, 500, 700,  30.0, 35.0,  16.0),
    "arepas":                   (230,  180, 120, 180, 280,   5.5, 38.0,   5.5),
    "tamales":                  (265,  200, 130, 200, 300,   7.0, 34.0,  12.0),
    "pozole":                   ( 80,  400, 300, 400, 600,   7.0, 10.0,   2.0),
    "chilaquiles":              (220,  300, 200, 300, 450,  10.0, 26.0,  10.0),
    "enchiladas":               (168,  300, 200, 300, 450,  10.0, 18.0,   7.0),
    "mole":                     (110,  100,  60, 100, 180,   4.0, 14.0,   5.0),
    "elote":                    (145,  150, 100, 150, 250,   3.5, 30.0,   2.0),
    "quesadilla":               (300,  250, 150, 250, 380,  16.0, 28.0,  13.0),
    "burrito":                  (217,  350, 250, 350, 500,  12.0, 27.0,   7.0),
    "falafel":                  (333,  150, 100, 150, 250,  13.0, 31.0,  18.0),
    "shawarma":                 (220,  280, 180, 280, 400,  18.0, 18.0,  10.0),
    "fattoush":                 ( 80,  200, 130, 200, 300,   2.0, 12.0,   3.0),
    "tabbouleh":                ( 83,  200, 130, 200, 300,   2.5, 13.0,   3.5),
    "baba_ganoush":             ( 55,  100,  60, 100, 180,   2.0,  8.0,   2.5),
    "kibbeh":                   (227,  200, 130, 200, 300,  14.0, 16.0,  11.0),
    "mansaf":                   (250,  400, 300, 400, 600,  20.0, 20.0,  10.0),
    "kofta":                    (220,  200, 130, 200, 300,  18.0,  5.0,  14.0),
    "shakshuka":                (100,  300, 200, 300, 450,   7.0,  8.0,   5.5),
    "kushari":                  (180,  350, 250, 350, 500,   7.0, 34.0,   4.0),
    "ful_medames":              (110,  300, 200, 300, 450,   8.0, 18.0,   1.5),
    "baklava":                  (428,   80,  50,  80, 120,   5.0, 45.0,  28.0),
    "kunafa":                   (380,  150, 100, 150, 250,   7.0, 48.0,  19.0),
    "basbousa":                 (340,  100,  70, 100, 160,   5.0, 56.0,  12.0),
    "ma_amoul":                 (420,   40,  25,  40,  65,   5.0, 58.0,  19.0),
    "loukoumades":              (330,  100,  60, 100, 180,   5.0, 44.0,  15.0),
    "spanakopita":              (260,  150, 100, 150, 250,   9.0, 22.0,  15.0),
    "souvlaki":                 (195,  250, 150, 250, 380,  20.0,  5.0,  11.0),
    "tzatziki":                 ( 70,  100,  60, 100, 180,   4.0,  4.0,   4.0),
    "dolmades":                 (155,  150, 100, 150, 250,   4.5, 17.0,   8.0),
    "loukoumades":              (330,  100,  60, 100, 180,   5.0, 44.0,  15.0),
    "knafeh":                   (380,  150, 100, 150, 250,   7.0, 48.0,  19.0),
    "harissa":                  ( 45,   20,  10,  20,  40,   1.5,  6.0,   2.0),
    "merguez":                  (311,  100,  60, 100, 180,  17.0,  1.0,  27.0),
    "couscous":                 (176,  300, 200, 300, 450,   6.0, 36.0,   0.5),
    "tagine":                   (165,  350, 250, 350, 500,  13.0, 14.0,   6.0),
    "pastilla":                 (310,  200, 130, 200, 300,  14.0, 30.0,  15.0),
    "msemen":                   (340,  100,  60, 100, 180,   7.0, 50.0,  14.0),
    "jollof_rice":              (170,  350, 250, 350, 500,   5.0, 31.0,   4.5),
    "egusi_soup":               (280,  300, 200, 300, 450,  18.0, 10.0,  20.0),
    "suya":                     (230,  150, 100, 150, 250,  22.0,  5.0,  13.0),
    "injera":                   (130,  150, 100, 150, 250,   4.5, 27.0,   1.0),
    "doro_wat":                 (175,  300, 200, 300, 450,  18.0,  6.0,   8.0),
    "ugali":                    (117,  300, 200, 300, 450,   2.5, 26.0,   0.5),
    "nyama_choma":              (250,  250, 150, 250, 380,  26.0,  0.0,  16.0),
    "muamba_chicken":           (215,  300, 200, 300, 450,  18.0,  7.0,  13.0),
    "biltong":                  (320,   50,  30,  50,  80,  45.0,  2.0,  14.0),
    "bobotie":                  (215,  300, 200, 300, 450,  14.0, 15.0,  12.0),
    "bunny_chow":               (260,  350, 250, 350, 500,  12.0, 32.0,   9.0),
    "chakalaka":                ( 60,  150, 100, 150, 250,   2.5, 10.0,   1.5),
    "boerewors":                (330,  150, 100, 150, 250,  17.0,  1.5,  29.0),
    "milk_tart":                (235,  150, 100, 150, 250,   5.0, 32.0,  10.0),
    "koeksister":               (420,   60,  40,  60,  90,   4.0, 65.0,  16.0),
}

# Fallback for any class not in DB (generic estimate by food type)
GENERIC_DEFAULTS = {
    "default": (200, 200, 130, 200, 350, 8.0, 25.0, 8.0),
    "soup":     ( 65, 400, 300, 400, 600, 4.0, 8.0,  2.0),
    "salad":    ( 80, 200, 150, 200, 300, 3.0, 8.0,  5.0),
    "cake":     (380, 120, 80, 120, 180,  4.0, 52.0, 18.0),
    "rice":     (170, 300, 200, 300, 450, 4.0, 36.0,  1.0),
    "noodle":   (140, 350, 250, 350, 500, 5.5, 27.0,  2.0),
    "bread":    (270, 100, 60, 100, 180,  8.0, 50.0,  3.0),
    "chicken":  (190, 250, 150, 250, 380, 22.0, 2.0, 10.0),
    "beef":     (250, 250, 150, 250, 380, 25.0, 1.0, 16.0),
    "fish":     (160, 200, 130, 200, 300, 22.0, 0.0,  7.0),
    "sandwich": (270, 200, 130, 200, 350, 14.0, 28.0, 10.0),
    "pizza":    (266, 200, 100, 200, 350, 11.0, 33.0, 10.0),
    "dessert":  (320, 120, 80, 120, 200,  4.0, 45.0, 14.0),
    "curry":    (175, 300, 200, 300, 450, 13.0, 12.0,  8.0),
    "dumpling": (165, 150, 100, 150, 250,  8.0, 20.0,  6.0),
}

SOUP_KEYWORDS   = {"soup", "bisque", "chowder", "ramen", "pho", "laksa", "miso", "hot_and_sour", "noodle_soup", "wonton"}
SALAD_KEYWORDS  = {"salad", "slaw"}
CAKE_KEYWORDS   = {"cake", "cheesecake", "cupcake", "tart", "pie", "mousse", "creme_brulee", "panna_cotta", "tiramisu", "cannoli", "donut", "beignet", "churro", "macaron", "baklava"}
RICE_KEYWORDS   = {"rice", "biryani", "paella", "risotto", "khichdi", "plov", "fried_rice"}
NOODLE_KEYWORDS = {"noodle", "pasta", "spaghetti", "pho", "ramen", "pad_thai", "gnocchi", "ravioli"}
CHICKEN_KEYWORDS = {"chicken", "poultry"}
BEEF_KEYWORDS   = {"beef", "steak", "burger", "hamburger", "tartare", "carpaccio"}
FISH_KEYWORDS   = {"fish", "salmon", "tuna", "sashimi", "sushi", "mussels", "oyster", "scallop", "shrimp", "lobster", "crab", "calamari", "seafood"}
SANDWICH_KEYWORDS = {"sandwich", "burger", "wrap", "sub", "roll", "hot_dog"}
PIZZA_KEYWORDS  = {"pizza"}
CURRY_KEYWORDS  = {"curry", "korma", "vindaloo", "masala", "tikka", "dal", "daal"}
DUMPLING_KEYWORDS = {"dumpling", "gyoza", "wonton", "ravioli", "pierogi", "empanada", "samosa"}
DESSERT_KEYWORDS = {"ice_cream", "frozen_yogurt", "gelato", "sorbet", "pudding", "custard", "flan", "halwa", "kheer", "mochi", "waffles", "pancake", "french_toast"}


def guess_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in SOUP_KEYWORDS):    return "soup"
    if any(k in n for k in SALAD_KEYWORDS):   return "salad"
    if any(k in n for k in FISH_KEYWORDS):    return "fish"
    if any(k in n for k in DUMPLING_KEYWORDS):return "dumpling"
    if any(k in n for k in CURRY_KEYWORDS):   return "curry"
    if any(k in n for k in CHICKEN_KEYWORDS): return "chicken"
    if any(k in n for k in BEEF_KEYWORDS):    return "beef"
    if any(k in n for k in SANDWICH_KEYWORDS):return "sandwich"
    if any(k in n for k in PIZZA_KEYWORDS):   return "pizza"
    if any(k in n for k in NOODLE_KEYWORDS):  return "noodle"
    if any(k in n for k in RICE_KEYWORDS):    return "rice"
    if any(k in n for k in CAKE_KEYWORDS):    return "cake"
    if any(k in n for k in DESSERT_KEYWORDS): return "dessert"
    return "default"


def build_lookup(class_list_path: str, out_path: str):
    classes = {}
    with open(class_list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                idx, name = line.split(" ", 1)
                classes[int(idx)] = name.lower()

    lookup = {}
    missing = []

    for idx, name in sorted(classes.items()):
        if name in NUTRITION_DB:
            cal, def_g, sm, med, lg, prot, carb, fat = NUTRITION_DB[name]
            source = "nutrition_db"
        else:
            cat = guess_category(name)
            cal, def_g, sm, med, lg, prot, carb, fat = GENERIC_DEFAULTS[cat]
            source = f"estimate_category:{cat}"
            missing.append(name)

        lookup[str(idx)] = {
            "class_name": name,
            "calories_per_100g": cal,
            "default_serving_g": def_g,
            "small_serving_g": sm,
            "medium_serving_g": med,
            "large_serving_g": lg,
            "protein_per_100g": prot,
            "carbs_per_100g": carb,
            "fat_per_100g": fat,
            "calorie_source": source,
        }

    with open(out_path, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"Saved {len(lookup)} class entries → {out_path}")
    if missing:
        print(f"  {len(missing)} classes used category estimate (not in NUTRITION_DB):")
        for m in missing:
            print(f"    {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_list", default="class_list.txt")
    parser.add_argument("--out", default="calorie_lookup.json")
    args = parser.parse_args()
    build_lookup(args.class_list, args.out)
