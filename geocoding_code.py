# Open Toronto License data
path = os.path.join('data', 'toronto_license_data.csv')
licenses = pd.read_csv(path)

# 1. Initialize the geocoder
geolocator = Nominatim(user_agent="toronto_address_geocoder")

# 2. Add a rate limiter to avoid getting blocked
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

licenses['address_for_geocode'] = licenses['address'] + ', Toronto, ON, Canada'

# 3. Apply geocoding
licenses['location'] = licenses['address_for_geocode'].progress_apply(geocode)
licenses['latitude'] = licenses['location'].apply(lambda loc: loc.latitude if loc else None)
licenses['longitude'] = licenses['location'].apply(lambda loc: loc.longitude if loc else None)

# Save results to CSV
licenses.to_csv('data/toronto_license_data_geocoded.csv', index=False)

# Followed by manually filling in the missing lat/lon values in the CSV file using Google Maps.


# After manually filling in the missing values as coordinates, the following code reorganizes the data
# For all rows that have a value in the coordinates column, split the coordinates into latitude and longitude columns. Make sure not to overwrite existing latitude and longitude columns.
# Split 'coordinates' into 'latitude' and 'longitude' only for rows where 'coordinates' is not NaN
valid_coordinates = licenses['coordinates'].notna()

# Split the coordinates into two columns
split_coords = licenses.loc[valid_coordinates, 'coordinates'].str.split(',', expand=True)

# Assign the split values to the latitude and longitude columns
licenses.loc[valid_coordinates, 'latitude'] = split_coords[0].astype(float)
licenses.loc[valid_coordinates, 'longitude'] = split_coords[1].astype(float)