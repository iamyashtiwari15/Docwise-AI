# test_image_analysis.py

from image_analysis_agent import analyze_image

if __name__ == "__main__":
    test_image_path = "test_img.jpeg"  # Replace with your image path
    print("Analyzing image:", test_image_path)

    try:
        result = analyze_image(test_image_path)
        print("\n--- Analysis Report ---\n")
        print(result)
    except Exception as e:
        print("Error during analysis:", e)
