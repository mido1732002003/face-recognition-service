#!/usr/bin/env python3
"""
Face Recognition Service Python Client Example

Usage:
    python example.py enroll <person_id> <image_path> [<image_path>...]
    python example.py identify <image_path>
    python example.py verify <person_id> <image_path>
    python example.py stats
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import httpx
from httpx import AsyncClient


class FaceRecognitionClient:
    """Client for Face Recognition Service"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = httpx.Timeout(30.0, connect=5.0)

    async def enroll(
        self,
        person_id: str,
        image_paths: List[str],
        quality_threshold: float = 0.5,
    ) -> dict:
        """Enroll face images for a person"""
        async with AsyncClient(timeout=self.timeout) as client:
            # Prepare files
            files = []
            for path in image_paths:
                if not Path(path).exists():
                    print(f"Warning: File not found: {path}")
                    continue
                
                with open(path, "rb") as f:
                    files.append(
                        ("images", (Path(path).name, f.read(), "image/jpeg"))
                    )

            if not files:
                return {"error": "No valid images found"}

            # Make request
            response = await client.post(
                f"{self.base_url}/api/v1/enroll/{person_id}",
                files=files,
                data={"quality_threshold": str(quality_threshold)},
                headers=self.headers,
            )

            return response.json()

    async def identify(
        self,
        image_path: str,
        similarity_threshold: float = 0.65,
        top_k: int = 5,
    ) -> dict:
        """Identify a person from face image"""
        async with AsyncClient(timeout=self.timeout) as client:
            if not Path(image_path).exists():
                return {"error": f"File not found: {image_path}"}

            with open(image_path, "rb") as f:
                files = [("image", (Path(image_path).name, f.read(), "image/jpeg"))]

            response = await client.post(
                f"{self.base_url}/api/v1/identify",
                files=files,
                data={
                    "similarity_threshold": str(similarity_threshold),
                    "top_k": str(top_k),
                    "return_face_data": "false",
                },
                headers=self.headers,
            )

            return response.json()

    async def verify(
        self,
        person_id: str,
        image_path: str,
        similarity_threshold: float = 0.65,
    ) -> dict:
        """Verify if face belongs to specific person"""
        async with AsyncClient(timeout=self.timeout) as client:
            if not Path(image_path).exists():
                return {"error": f"File not found: {image_path}"}

            with open(image_path, "rb") as f:
                files = [("image", (Path(image_path).name, f.read(), "image/jpeg"))]

            response = await client.post(
                f"{self.base_url}/api/v1/verify/{person_id}",
                files=files,
                data={"similarity_threshold": str(similarity_threshold)},
                headers=self.headers,
            )

            return response.json()

    async def get_person(self, person_id: str) -> dict:
        """Get person details"""
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/persons/{person_id}",
                headers=self.headers,
            )
            return response.json()

    async def list_persons(self, offset: int = 0, limit: int = 100) -> dict:
        """List all persons"""
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/persons",
                params={"offset": offset, "limit": limit},
                headers=self.headers,
            )
            return response.json()

    async def get_stats(self) -> dict:
        """Get system statistics"""
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/stats",
                headers=self.headers,
            )
            return response.json()

    async def health_check(self) -> dict:
        """Check service health"""
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/health",
                headers=self.headers,
            )
            return response.json()


async def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Initialize client
    base_url = "http://localhost:8000"
    api_key = None  # Set if using API key authentication
    
    client = FaceRecognitionClient(base_url, api_key)
    
    command = sys.argv[1]

    try:
        if command == "enroll" and len(sys.argv) >= 4:
            person_id = sys.argv[2]
            image_paths = sys.argv[3:]
            
            print(f"Enrolling {len(image_paths)} image(s) for {person_id}...")
            result = await client.enroll(person_id, image_paths)
            
            if "faces_enrolled" in result:
                print(f"✅ Successfully enrolled {result['faces_enrolled']} face(s)")
            else:
                print(f"❌ Enrollment failed: {result}")

        elif command == "identify" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            
            print(f"Identifying face in {image_path}...")
            result = await client.identify(image_path)
            
            if "matches" in result and result["matches"]:
                print(f"✅ Identified as: {result['matches'][0]['person_id']}")
                print(f"   Similarity: {result['matches'][0]['similarity']:.3f}")
            elif "matches" in result:
                print("❌ No match found (Unknown person)")
            else:
                print(f"❌ Identification failed: {result}")

        elif command == "verify" and len(sys.argv) >= 4:
            person_id = sys.argv[2]
            image_path = sys.argv[3]
            
            print(f"Verifying if face belongs to {person_id}...")
            result = await client.verify(person_id, image_path)
            
            if "verified" in result:
                if result["verified"]:
                    print(f"✅ Verified: Face belongs to {person_id}")
                    print(f"   Similarity: {result['similarity']:.3f}")
                else:
                    print(f"❌ Not verified: Face does not belong to {person_id}")
            else:
                print(f"❌ Verification failed: {result}")

        elif command == "stats":
            print("Getting system statistics...")
            result = await client.get_stats()
            print(json.dumps(result, indent=2))

        elif command == "health":
            print("Checking service health...")
            result = await client.health_check()
            print(json.dumps(result, indent=2))

        else:
            print(__doc__)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())