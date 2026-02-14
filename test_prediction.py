from utils.prediction import predict_risk

# Test input
test_data = {
    "Age": 70,
    "Gender": "Male",
    "Blood_Pressure": 160,
    "Heart_Rate": 110,
    "Temperature": 101.5,
    "Symptoms": ["Chest Pain", "Shortness of Breath"],
    "Pre_Existing_Conditions": "Heart Disease"
}

result = predict_risk(test_data)

print("\n🏥 PREDICTION RESULT:")
print(f"Risk Level: {result['risk_label']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Department: {result['recommended_department']}")
print(f"\nTop Features:")
for feature, value in result['top_features']:
    print(f"  - {feature}: {value:.4f}")
