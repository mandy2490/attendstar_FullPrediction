import joblib
FEATURE_ORDER_PATH = "C:/Users/ManideepM/Downloads/final_feature_names.pkl"

# Load and print feature names
feature_names = joblib.load(FEATURE_ORDER_PATH)
print("\n🚀 Final Training Feature Order:")
for i, feature in enumerate(feature_names):
    print(f"{i+1}. {feature}")
