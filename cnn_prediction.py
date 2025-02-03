import torch
from torchvision import transforms
import os
from PIL import Image
from sympy import symbols, lambdify, latex
from sympy.parsing.sympy_parser import parse_expr
from data.cnn_model import SimpleCNN as MyCNN
import inspect


def predict():
    # Path to segmented images
    image_dir = "data/equation_segmented"

    # Create a list of images
    images = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    # Display the segmented images to classify
    #plt.figure(figsize=(4, 2), facecolor='#282C34')
    #for i, image in enumerate(images):
    #    plt.subplot(1, len(images), i + 1)
    #    plt.imshow(image)
    #    plt.axis('off')
    #plt.show()



    # Define the transformations
    transform  = transforms.Compose([
        transforms.Resize(size=(180, 180)),
        transforms.ToTensor()
    ])

    # Load the classes
    classes = torch.load("data/cnn_model/class_names.pth", weights_only=True, map_location=torch.device('cpu'))

    # Load the model
    model = MyCNN.SimpleCNN()
    model.load_state_dict(torch.load("data/cnn_model/model.pth", weights_only=True, map_location=torch.device('cpu')))
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
    #print(f"Equation: {equation_str}")

    equation = parse_expr(equation_str)

    latex_equation = latex(equation)

    x, y, z = symbols('x y z')
    equation_lambda = lambdify(x, equation)

    py_function_equation = inspect.getsource(equation_lambda)

    return latex_equation, py_function_equation