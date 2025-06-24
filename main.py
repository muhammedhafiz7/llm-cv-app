import sys
import os
from object_detection import ObjectDetection
from text_generation import Textgenerator

def main():
    print("=== Computer Vision + LLM Text Generation App ===")
    image_path = input("Enter the path to the image file: ").strip()
    if not os.path.isfile(image_path):
        print("Error: Image file does not exist.")
        return
    prompt = input("Enter your text prompt: ").strip()
    if not prompt:
        print("Error: Text prompt cannot be empty.")
        return
    # Object Detection
    try:
        detector = ObjectDetection()
        detections = detector.detect(image_path)
        if not detections:
            print("No objects detected with sufficient confidence.")
        else:
            print("\nObjects detected:")
            for obj in detections:
                print(f"- {obj['label']} (confidence: {obj['score']:.2f})")
            # Prepare analysis summary for LLM
            analysis = ", ".join([f"{obj['label']} ({obj['score']:.2f})" for obj in detections])
    except Exception as e:
        print(f"Object detection failed: {e}")
        return
    # Text Generation
    try:
        textgen = Textgenerator()
        combined_prompt = f"Image analysis: {analysis if detections else 'No objects detected.'} Prompt: {prompt}"
        response = textgen.generate(combined_prompt)
        print("\nGenerated response:")
        print(response)
    except Exception as e:
        print(f"Text generation failed: {e}")
        return

if __name__ == "__main__":
    main()