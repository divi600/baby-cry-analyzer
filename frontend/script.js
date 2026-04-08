async function predict() {
  const fileInput = document.getElementById("audioFile");
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  if (!fileInput.files.length) {
    alert("Please upload an audio file");
    return;
  }

  loading.classList.remove("hidden");
  result.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    const alertMap = {
      "laugh": "😊 Baby is happy and laughing",
      "fatigue cry": "😴 Baby is tired. Try to put to sleep",
      "lonely and scared": "🤗 Baby feels lonely or scared. Comfort needed",
      "hunger cry": "🍼 Baby is hungry. Please feed",
      "general discomfort": "😣 Baby is uncomfortable. Check diaper/position",
      "thermal discomfort": "🌡️ Baby feels too hot or cold. Adjust temperature",
      "digestive pain": "🤒 Baby may have stomach pain. Check feeding"
    };

    let pred = data.prediction.toLowerCase().trim();
    let alertMessage = alertMap[pred] || "⚠️ Unable to detect properly. Please check baby";

    document.getElementById("predictionText").innerText =
      "Prediction: " + data.prediction;

    document.getElementById("alertText").innerText = alertMessage;

    loading.classList.add("hidden");
    result.classList.remove("hidden");

    // ❌ removed popup

  } catch (error) {
    loading.classList.add("hidden");
    alert("Backend not running or error occurred");
    console.error(error);
  }
}
