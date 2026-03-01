import os, cv2

gallery = os.path.join(os.path.dirname(os.path.realpath(__file__)), "face_gallery")
print("Gallery:", gallery)

for f in sorted(os.listdir(gallery)):
    p = os.path.join(gallery, f)
    if os.path.isdir(p):
        continue
    img = cv2.imread(p)
    print(f, "->", "OK" if img is not None else "FAILED")