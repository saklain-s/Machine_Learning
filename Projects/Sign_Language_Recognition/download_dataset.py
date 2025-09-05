import kagglehub

# Download latest version of Sign Language MNIST
dataset_path = kagglehub.dataset_download("datamunge/sign-language-mnist")
print("Path to dataset files:", dataset_path)
