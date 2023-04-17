import requests

# Define a sample PGN for testing
sample_pgn = """
    [Event "Casual Game"]
    [Site "Chess.com"]
    [Date "2023.04.15"]
    [Round "?"]
    [White "Player1"]
    [Black "Player2"]
    [Result "*"]
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 *
    """
def test_evaluate_moves():
    # Define the URL of the endpoint (assuming the FastAPI server is running on port 5555)
    url = "http://localhost:5555/evaluate_moves"

    

    # Define the request body
    data = {
        "pgn": sample_pgn
    }

    # Send the POST request to the endpoint
    response = requests.post(url, json=data)

    # Check the response status code and print the result
    if response.status_code == 200:
        print("Response:")
        print(response.json())
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        raise Exception("Request failed")




def test_gif_creation():
    url = "http://127.0.0.1:5555/generate-gif-game/"

    response = requests.post(url, json={"pgn_data": sample_pgn})

    if response.status_code == 200:
        presigned_url = response.json()["url"]
        print("Presigned URL:", presigned_url)
    else:
        print("Error:", response.status_code, response.text)
        raise Exception("Request failed")


if __name__ == "__main__":
    test_evaluate_moves()
    test_gif_creation()
