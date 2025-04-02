import kagglehub

# Download latest version
path = kagglehub.dataset_download("hakurei/citypersons")

print("Path to dataset files:", path)