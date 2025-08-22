import requests

# Change this to your deployed endpoint
ENDPOINT = 'https://ollama.havenify.ai/vision-to-json'

# Path to a small test image or PDF
TEST_FILE = '/Users/mohanpraneeth/Downloads/180.pdf'  # Change to a small PNG/JPG for faster test if needed

# Optional: minimal schema for testing
SCHEMA = {
    "test_field": "string",
    "another_field": "number"
}

def test_vision_to_json():
    with open(TEST_FILE, 'rb') as f:
        files = {'file': f}
        data = {'schema': str(SCHEMA).replace("'", '"')}  # send as JSON string
        response = requests.post(ENDPOINT, files=files, data=data)
        print('Status:', response.status_code)
        print('Response:', response.text)

if __name__ == '__main__':
    test_vision_to_json()
