import tensorflow as tf

# Load old model (from backend folder)
model = tf.keras.models.load_model("backend/model.keras", compile=False)

# Save fixed model
model.save("backend/model_fixed.keras")

print("✅ Model fixed and saved!")
