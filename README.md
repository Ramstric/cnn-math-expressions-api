# CNN Powered Math Equation Translator

A web application (Project for Sistemas Inteligentes at UPM) that uses a Convolutional Neural Network (CNN) to translate handwritten math equations into LaTeX and Python function code.

## Table of Contents

- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

<!--
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourproject.git
    ```
2. Navigate to the project directory:
    ```sh
    cd yourproject
    ```
3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```sh
    python app.py
    ```
2. Access the application at `http://localhost:8080`.
-->

## API Endpoints

### Upload File

- **URL:** `/upload`
- **Method:** `POST`
- **Description:** Upload a file to the server.
- **Request:**
    - `file`: The file to upload.
- **Response:**
    ```json
    {
        "message": "File uploaded successfully!"
    }
    ```

### Process Image

- **URL:** `/process`
- **Method:** `GET`
- **Description:** Process the uploaded image.
- **Response:**
    ```json
    {
        "message": "Image processed successfully!",
        "num_segments": "<number_of_segments>"
    }
    ```

### Download Image

- **URL:** `/download`
- **Method:** `GET`
- **Description:** Download processed images.
- **Parameters:**
    - `image`: The type of image to download (`processed`, `contours`, `segmented`).
    - `n`: The segment number (for `segmented` images).
- **Response:** The requested image file.

### Predict

- **URL:** `/predict`
- **Method:** `GET`
- **Description:** Get the LaTeX and Python function code for the predicted equation.
- **Response:**
    ```json
    {
        "latex": "<latex_code>",
        "python": "<python_function_code>"
    }
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
