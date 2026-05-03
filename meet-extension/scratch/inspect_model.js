const fs = require('fs');
const ort = require('onnxruntime-node');

async function inspectModel() {
  const modelPath = './assets/arabic_sign_video_model_small.onnx';
  try {
    const session = await ort.InferenceSession.create(modelPath);
    console.log("Input Names:", session.inputNames);
    console.log("Output Names:", session.outputNames);
    
    // Unfortunately, onnxruntime-node doesn't easily expose input shapes in standard API without running.
    // Let's just catch the error from a dummy run.
    const dummyTensor = new ort.Tensor("float32", new Float32Array(1 * 30 * 126), [1, 30, 126]);
    try {
      await session.run({ [session.inputNames[0]]: dummyTensor });
      console.log("SUCCESS with shape [1, 30, 126]");
    } catch (e) {
      console.log("Error with shape [1, 30, 126]:", e.message);
    }
  } catch (e) {
    console.error("Failed to load model:", e);
  }
}

inspectModel();
