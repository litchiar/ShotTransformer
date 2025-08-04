from usage import process_video,clear_model

if __name__ == '__main__':
    process_video(r"test_video\0bce4c24-4809-4cc6-9577-acc3169a5988.mp4", use_cn=False)
    process_video(r"test_video\1c8c4e21-8b89-4fd1-9219-d4794bf71c6b.mp4", use_cn=False)
    clear_model()
