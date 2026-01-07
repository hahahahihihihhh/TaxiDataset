# POI
CATES = ['scenic_spots', 'medical_and_health', 'domestic_services', 'residential_area',
             'finance', 'sports_and_leisure', 'culture_and_education', 'shopping', 'governments_and_organizations',
             'corporations', 'catering', 'transportation', 'public_services',
             'parking_area', 'place_of_worship', 'others']
# CATES_LABELS = {
#         'scenic_spots': ['terrace', 'grandstand', 'pavilion', 'tower', 'triumphal_arch',
#                          'farm', 'grave_yard', 'tourism', 'fountain', 'shelter',
#                          'bench', 'greenhouse', 'bunker', 'surface'],
#         'medical_and_health': ['hospital', 'clinic', 'doctors', 'dentist', 'pharmacy',
#                                'veterinary', 'nursing_home', 'childcare', 'Health_Center',
#                                'ambulance', 'first_aid'],
#         'domestic_services': ['service', 'ventilation_shaft', 'stable', 'slipway', 'kindergarten',
#                               'Castle_Hill_Electrical_Supply', 'dressing_room'],
#         'residential_area': ['apartments', 'house', 'hotel', 'residential',
#                              'dormitory', 'terrace', 'pavilion', 'mansion',
#                              'hut', 'boathouse', 'apartment', 'townhall', 'semidetached_house',
#                              'cabin', 'houseboat', 'gatehouse', 'security_booth'],
#         'finance': ['bank', 'stock_exchange', 'money_transfer', 'payment_centre'],
#         'sports_and_leisure': ['theatre', 'sports_centre', 'museum', 'historical', 'stadium',
#                                'rest', 'cinema', 'arts_centre', 'detached',
#                                'brewery', 'pub', 'food_and_drink', 'cafe',
#                                'bar', 'nightclub', 'stripclub', 'biergarten', 'music_venue', 'dojo',
#                                'shooting_stand', 'stage', 'karaoke_box'],
#         'culture_and_education': ['school', 'university', 'college', 'historical', 'museum',
#                                   'library', 'arts_centre', 'studio', 'research_institute',
#                                   'music_school', 'social_centre', 'prep_school', 'school;place_of_worship',
#                                   'yes;school', 'stage'],
#         'shopping': ['retail', 'commercial', 'supermarket', 'kiosk', 'marketplace'],
#         'governments_and_organizations': ['fire_station', 'storage_tank',
#                                           'civic', 'government', 'data_center',
#                                           'prison', 'post_office', 'police',
#                                           'courthouse', 'events_venue',
#                                           'community_centre', 'social_facility', 'recycling',
#                                           'mortuary', 'crematorium', 'coworking_space',
#                                           'ranger_station', 'waste_disposal', 'security_booth'],
#         'corporations': ['office', 'industrial', 'warehouse', 'works', 'car_wash',
#                          'car_rental', 'slaughterhouse', 'company', 'manufacture',
#                          'coworking_space', 'barn'],
#         'catering': ['detached', 'brewery', 'food_and_drink', 'restaurant',
#                      'fast_food', 'pub', 'cafe', 'bar', 'food_court', 'biergarten', 'ice_cream', 'plaza',
#                      'drinking_water', 'water_tower'],
#         'transportation': ['train_station', 'transportation', 'ship', 'hangar',
#                            'bridge', 'subway_entrance', 'bicycle_rental', 'boat_rental',
#                            'post_depot', 'waste_transfer_station', 'fuel', 'bus_station', 'animal_boarding',
#                            'container', 'taxi', 'bus_depot', 'static_caravan', 'houseboat'],
#         'public_services': ['construction', 'roof', 'shed', 'public', 'toilets', 'triumphal_arch',
#                             'mixed_use', 'mixed use', 'ferry_terminal', 'fountain',
#                             'shelter', 'post_depot', 'waste_transfer_station',
#                             'fuel', 'animal_shelter', 'recycling', 'public_building', 'canopy',
#                             'tent', 'lighthouse', 'outdoor_seating'],
#         'parking_area': ['garage', 'garages', 'parking', 'ruins', 'derelict,_uninhabited',
#                          'carport', 'bicycle_parking', 'parking_space', 'motorcycle_parking',
#                          'parking_entrance'],
#         'place_of_worship': ['church', 'religious', 'mosque',
#                              'synagogue', 'chapel', 'cathedral', 'convent', 'place of worship',
#                              'place_of_worship', 'monastery', 'temple', 'school;place_of_worship'],
#         'others': []
# }
CATES_LABELS = {
    "scenic_spots": [
        "terrace",
        "grandstand",
        "pavilion",
        "tower",
        "triumphal_arch",
        "farm",
        "grave_yard",
        "tourism",
        "fountain",
        "shelter",
        "bench",
        "greenhouse",
        "bunker",
        "surface",
    ],
    "medical_and_health": [
        "hospital",
        "clinic",
        "doctors",
        "dentist",
        "pharmacy",
        "veterinary",
        "nursing_home",
        "childcare",
        "ambulance",
        "first_aid",
    ],
    "domestic_services": [
        "service",
        "ventilation_shaft",
        "stable",
        "slipway",
        "kindergarten",
        "dressing_room",
    ],
    "residential_area": [
        "apartments",
        "house",
        "hotel",
        "residential",
        "dormitory",
        "mansion",
        "hut",
        "boathouse",
        "apartment",
        "townhall",
        "semidetached_house",
        "cabin",
        "houseboat",
        "gatehouse",
        "security_booth",
    ],
    "finance": [
        "bank",
        "stock_exchange",
        "money_transfer",
        "payment_centre",
    ],
    "sports_and_leisure": [
        "theatre",
        "sports_centre",
        "stadium",
        "cinema",
        "arts_centre",
        "brewery",
        "pub",
        "food_and_drink",
        "cafe",
        "bar",
        "nightclub",
        "stripclub",
        "biergarten",
        "music_venue",
        "dojo",
        "shooting_stand",
        "stage",
        "karaoke_box",
    ],
    "culture_and_education": [
        "school",
        "university",
        "college",
        "museum",
        "library",
        "arts_centre",
        "studio",
        "research_institute",
        "music_school",
        "social_centre",
        "prep_school",
        "place_of_worship",  # 由 school;place_of_worship 拆分得到
    ],
    "shopping": [
        "retail",
        "commercial",
        "supermarket",
        "kiosk",
        "marketplace",
    ],
    "governments_and_organizations": [
        "fire_station",
        "civic",
        "government",
        "data_center",
        "prison",
        "post_office",
        "police",
        "courthouse",
        "events_venue",
        "community_centre",
        "social_facility",
        "recycling",
        "mortuary",
        "crematorium",
        "coworking_space",
        "ranger_station",
        "waste_disposal",
    ],
    "corporations": [
        "office",
        "industrial",
        "warehouse",
        "works",
        "car_wash",
        "car_rental",
        "slaughterhouse",
        "company",
        "manufacture",
        "barn",
    ],
    "catering": [
        "brewery",
        "food_and_drink",
        "restaurant",
        "fast_food",
        "pub",
        "cafe",
        "bar",
        "food_court",
        "biergarten",
        "ice_cream",
        "plaza",
        "drinking_water",
        "water_tower",
    ],
    "transportation": [
        "train_station",
        "transportation",
        "ship",
        "hangar",
        "bridge",
        "subway_entrance",
        "bicycle_rental",
        "boat_rental",
        "post_depot",
        "waste_transfer_station",
        "fuel",
        "bus_station",
        "animal_boarding",
        "container",
        "taxi",
        "bus_depot",
        "static_caravan",
        "houseboat",
    ],
    "public_services": [
        "construction",
        "roof",
        "shed",
        "public",
        "toilets",
        "mixed_use",
        "ferry_terminal",
        "fountain",
        "shelter",
        "fuel",
        "animal_shelter",
        "recycling",
        "public_building",
        "canopy",
        "tent",
        "lighthouse",
        "outdoor_seating",
    ],
    "parking_area": [
        "garage",
        "garages",
        "parking",
        "ruins",
        "carport",
        "bicycle_parking",
        "parking_space",
        "motorcycle_parking",
        "parking_entrance",
    ],
    "place_of_worship": [
        "church",
        "religious",
        "mosque",
        "synagogue",
        "chapel",
        "cathedral",
        "convent",
        "place_of_worship",
        "monastery",
        "temple",
    ],
    "others": [],
}
EXPAND_KEYS = [
    "amenity", "shop", "tourism", "leisure", "office", "building", "landuse",
    "religion", "healthcare",
    # 下面这些有时是“显式功能列”或“展开列”
    "school", "restaurant", "fast_food", "cafe", "bar", "parking", "fuel",
    "police", "post_office", "hospital",
]
IGNORE_VALUE_SET = {"yes", "no", "true", "false", "1", "0"}

# weather
ALL_METEOROLOGICAL_VARS = [
    # 🌡 温度相关
    "temperature_2m",   # 气温
    "apparent_temperature",
    "dewpoint_2m",

    # 💧 降水相关
    "precipitation",    # 降雨量
    "rain",
    "snowfall",
    "snow_depth",

    # ☁️ 云 & 辐射
    "cloud_cover",  # 总云量
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",

    # 💨 风
    "wind_speed_10m",   # 风速
    "wind_direction_10m",
    "wind_gusts_10m",

    # 🌫 能见度 & 湿度
    "relative_humidity_2m", # 相对湿度
    "visibility",   # 能见度

    # 🌬 气压
    "surface_pressure", # 气压
    "pressure_msl"
]
METEOROLOGICAL_VARS = [
    "temperature_2m",   # 气温(C)
    "precipitation",    # 降雨量(mm)
    "cloud_cover",  # 总云量(%)
    "wind_speed_10m",   # 风速(km/s)
    "relative_humidity_2m", # 相对湿度(%)
    "surface_pressure", # 气压(hPa)
]
METEOROLOGICAL_VARS2LABELS = {
    "temperature_2m": "Temperature(°C)",
    "precipitation": "Precipitation(mm)",
    "cloud_cover": "Cloud Cover(%)",
    "wind_speed_10m": "Wind Speed(km/h)",
    "relative_humidity_2m": "Relative Humidity(%)",
    "surface_pressure": "Surface Pressure(hPa)",
}
METEOROLOGICAL_VARS_DIVIDE = {
    "temperature_2m": {  # 单位: °C, 参考 WMO CIMO Guide
        -10: "extreme_cold",
        -5: "very_cold",
        0: "cold",
        5: "cool",
        10: "mild",
        15: "comfortable",
        20: "warm",
        25: "hot",
        30: "very_hot",
        35: "extreme_hot",
        40: "scorching"
    },
    "precipitation": {  # 单位: mm, 参考 WMO FM12 SYNOP Manual / NOAA Precipitation Intensity
        0: "no_rain",
        0.1: "very_light_rain",
        2.5: "light_rain",
        7.6: "moderate_rain",
        50: "heavy_rain",
        100: "very_heavy_rain",
    },
    "cloud_cover": {
        # Unit: %
        # Reference:
        # - WMO FM-12 SYNOP Manual (Cloud amount classification)
        # - NOAA sky condition terminology
        0: "clear_sky",
        25: "few_clouds",
        50: "partly_cloudy",
        75: "mostly_cloudy",
        100: "overcast",
    },
    "wind_speed_10m": {  # 单位: km/h, Beaufort 风力等级 0-12, WMO Guide
        0.9: "calm",
        5.9: "light_air",
        11.9: "light_breeze",
        19.9: "gentle_breeze",
        28.9: "moderate_breeze",
        38.9: "fresh_breeze",
        49.9: "strong_breeze",
        61.9: "near_gale",
        74.9: "gale",
        88.9: "strong_gale",
        102.9: "violent_storm",
        117.9: "storm",
        133.0: "hurricane_force"
    },
    "relative_humidity_2m": {
        # Unit: %
        # Reference:
        # - WMO CIMO Guide
        # - Biometeorological humidity comfort ranges
        20: "very_low",
        40: "low",
        60: "medium",
        80: "high",
        100: "very_high",
    },
    "surface_pressure": {  # 单位: hPa, 参考 WMO 标准大气压分类
        950: "very_low_pressure",
        980: "low_pressure",
        1010: "normal_pressure",
        1030: "high_pressure",
        1060: "very_high_pressure",
        1100: "extremely_high_pressure"
    }
}

