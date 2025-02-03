import os
import inspect

import torch
from torchvision import transforms
from PIL import Image
from sympy import symbols, lambdify, latex
from sympy.parsing.sympy_parser import parse_expr

from data.cnn_model import SimpleCNN as MyCNN


def predict():

    # Path to segmented images
    segments_dir = "../data/equation_segmented"

    # List all the segmented images
    images = []
    for image_name in os.listdir(segments_dir):
        image_path = os.path.join(segments_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    # Transformations required by the model
    transform  = transforms.Compose([
        transforms.Resize(size=(180, 180)),
        transforms.ToTensor()
    ])

    # Load the classes (loaded onto the CPU)
    classes = torch.load("../data/cnn_model/class_names.pth", weights_only=True, map_location=torch.device('cpu'))

    # Load the model (loaded onto the CPU)
    model = MyCNN.SimpleCNN()
    model.load_state_dict(torch.load("../data/cnn_model/model.pth", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    # Predict the class of each image
    classifications = []
    scores = []
    for image in images:
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)

        all_classes = torch.softmax(output, 1)

        classifications.append(classes[predicted.item()])
        scores.append(all_classes)


    # The characters recognized in the image
    eq_characters = classifications.copy()

    # Insert multiplication sign between numbers and symbols
    for i in reversed(range(len(eq_characters) - 1)):
        if eq_characters[i+1].isalpha() and eq_characters[i].isnumeric():
            eq_characters.insert(i + 1, "*")

    equation_str = "".join(eq_characters)

    equation = parse_expr(equation_str)                         # Convert the string to a sympy expression

    latex_equation = latex(equation)                            # LaTeX code for the equation

    x, y, z = symbols('x y z')
    equation_lambda = lambdify(x, equation)

    py_function_equation = inspect.getsource(equation_lambda)   # Python function for the equation

    return latex_equation, py_function_equation