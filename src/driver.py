import pandas as pd

target = "status"

categorical_features = [
    "template_id",
    #    "city",
    "brake_type_code",
    "frame_material_code",
    "shifting_code",
    "condition_code",
    "sales_country_id",
    "bike_type_id",
    "bike_category_id",
    "mileage_code",
    "motor",
    "bike_component_id",
    "family_model_id",
    "family_id",
    "brand_id",
    "color",
    #    "currency_id",
    #    "seller_id",
    "is_ebike",
    "is_frameset",
]

numerical_features = [
    "msrp",
    "sales_price",
    "bike_created_at_month",
    "bike_created_at_year",
    "bike_year",
    "rider_height_min",
    "rider_height_max",
    "up_duration",
    "quality_score",
]


main_query = """
                SELECT

                bikes.status as status,

                bikes.id as id,
                bikes.bike_template_id as template_id,

                bike_price_currency as sales_price,

                -- prices


                bikes.msrp as msrp,


                -- temporal

                year(bikes.created_at) as bike_created_at_year,
                month(bikes.created_at) as bike_created_at_month,
                bikes.year as bike_year,



                -- take if succeed_at is not null, else take updated_at difference to created_at
                CASE
                    WHEN bookings.succeed_at IS NOT NULL THEN DATEDIFF(bookings.succeed_at,bikes.created_at)
                    ELSE DATEDIFF(bookings.updated_at,bikes.created_at)

                END as up_duration,



                -- spatial

                bikes.country_id as sales_country_id,

                -- categorizing
                bikes.bike_type_id as bike_type_id,
                bikes.bike_category_id as bike_category_id,
                bikes.mileage_code as mileage_code,
                bikes.motor as motor,

                bikes.condition_code as condition_code,



                bike_additional_infos.rider_height_min as rider_height_min,
                bike_additional_infos.rider_height_max as rider_height_max,


                bikes.brake_type_code as brake_type_code,
                bikes.frame_material_code as frame_material_code,
                bikes.shifting_code as shifting_code,






                bikes.bike_component_id as bike_component_id,

                -- find similarity between hex codes
                bikes.color as color,

                -- quite specific
                bikes.family_model_id as family_model_id,
                bikes.family_id as  family_id,
                bikes.brand_id as brand_id,

                -- quality score
                quality_scores.score AS quality_score,



                -- is_frameset
                COALESCE(bike_template_additional_infos.is_ebike, 0) as is_ebike,
                COALESCE(bike_template_additional_infos.is_frameset, 0) as is_frameset




                FROM bikes

                join bookings on bikes.id = bookings.bike_id
                join booking_accountings on bookings.id = booking_accountings.booking_id


                join quality_scores on bikes.id = quality_scores.bike_id


                left join bike_template_additional_infos on bikes.bike_template_id = bike_template_additional_infos.bike_template_id

                join bike_additional_infos on bikes.id = bike_additional_infos.bike_id


                WHERE bikes.status = 'sold' or bikes.status = 'deleted' or bikes.status = 'active'


             """
