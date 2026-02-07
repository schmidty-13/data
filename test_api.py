"""
Test script for the API endpoints.
"""

import sys
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
from main import app, initialize_data

def main():
    # Initialize data
    print('Initializing data for API test...')
    initialize_data('..')

    # Test API
    client = TestClient(app)

    print('\nTesting API endpoints...')

    # Health check
    response = client.get('/api/health')
    print(f'  /api/health: {response.status_code}')
    print(f'    Response: {response.json()}')

    # Get buildings
    response = client.get('/api/buildings?limit=3')
    print(f'  /api/buildings: {response.status_code}')
    data = response.json()
    print(f'    Total: {data["total"]} buildings')
    print(f'    First 3 buildings:')
    for b in data['buildings'][:3]:
        name = b["building_name"][:35] if b.get("building_name") else "Unknown"
        score = b.get("overall_efficiency_score", 0)
        print(f'      - {name}: Score={score}')

    # Get single building
    response = client.get('/api/buildings/50')
    print(f'  /api/buildings/50: {response.status_code}')
    if response.status_code == 200:
        b = response.json()
        print(f'    Building: {b.get("building_name", "Unknown")}')
        print(f'    Score: {b.get("overall_efficiency_score", 0)}')
        print(f'    Meters: {b.get("num_meters", 0)}')

    # Get summary
    response = client.get('/api/summary')
    print(f'  /api/summary: {response.status_code}')
    if response.status_code == 200:
        s = response.json()
        print(f'    Total buildings: {s.get("total_buildings", 0)}')
        print(f'    Total meters: {s.get("total_meters", 0)}')

    # Get rankings
    response = client.get('/api/rankings?limit=5')
    print(f'  /api/rankings: {response.status_code}')
    if response.status_code == 200:
        r = response.json()
        print(f'    Top 5 buildings returned')

    print('\nAPI tests passed!')

if __name__ == "__main__":
    main()
