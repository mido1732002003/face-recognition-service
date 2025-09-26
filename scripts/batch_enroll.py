import os
import json
import httpx

API_URL = "http://localhost:8000/api/v1"

# نجيب المسار المطلق للجذر (المجلد اللي فيه dataset)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # مكان scripts/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # نرجع للجذر
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")  # نضيف dataset/

def load_labels(dataset_dir):
    labels_path = os.path.join(dataset_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def create_person(person_id, name=None):
    payload = {
        "id": person_id,
        "name": name or person_id,
        "metadata": {}
    }
    r = httpx.post(f"{API_URL}/persons", json=payload)
    if r.status_code not in [200, 201]:
        print(f"[!] Failed to create person {person_id}: {r.text}")
    else:
        print(f"[+] Person {person_id} ({name}) created/exists")

def enroll_person_images(person_id, image_paths):
    files = [("images", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths]
    r = httpx.post(f"{API_URL}/enroll/{person_id}", files=files)
    for f in files:
        f[1][1].close()

    if r.status_code == 200:
        return r.json()
    else:
        return {"status": "failed", "error": r.text}

def batch_enroll(dataset_dir=DATASET_DIR):
    labels = load_labels(dataset_dir)
    print("Loaded labels:", labels)  # Debug عشان تتأكد إنه شاف الملف
    summary = {}

    for person_id in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_id)
        if not os.path.isdir(person_dir):
            continue

        # 1) Get name from labels.json (if available)
        name = labels.get(person_id, person_id)

        # 2) Create person
        create_person(person_id, name=name)

        # 3) Collect all images
        image_paths = [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not image_paths:
            print(f"[!] No images found for {person_id}")
            continue

        # 4) Enroll images
        result = enroll_person_images(person_id, image_paths)

        faces_enrolled = result.get("faces_enrolled", 0)
        status = result.get("status", "failed")
        errors = result.get("errors")

        summary[person_id] = {
            "name": name,
            "total_images": len(image_paths),
            "faces_enrolled": faces_enrolled,
            "status": status,
            "errors": errors,
        }

    # === Print final summary ===
    print("\n=== Enrollment Summary ===")
    for pid, data in summary.items():
        print(f"- {pid} ({data['name']}):")
        print(f"   Total images: {data['total_images']}")
        print(f"   Faces enrolled: {data['faces_enrolled']}")
        print(f"   Status: {data['status']}")
        if data["errors"]:
            print(f"   Errors: {data['errors']}")
    print("==========================")

if __name__ == "__main__":
    batch_enroll(DATASET_DIR)
