from huggingface_hub import create_repo, upload_folder

create_repo("diffusion-test", repo_type="model", exist_ok=True)

upload_folder(
    repo_id="baldyouth/diffusion-test",
    folder_path="diffusion-test",
    repo_type="model"
)
