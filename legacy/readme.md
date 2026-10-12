# Baseline Model Experiments

This repository contains the **Vaseline model experiments** for this study.  
The experiments are organized by model type:

- **CNN**: 5 models (`C1`, `C2`, `C3`, `C4`, `C5`)  
- **DNN**: 5 models (`D1`, `D2`, `D3`, `D4`, `D5`)  
- **RNN**: 5 models (`R1`, `R2`, `R3`, `R4`, `R5`)  

---

## Backend Selection

To select a backend in TensorFlow.js, update the code as follows:
```javascript
await tf.setBackend('backend'); 
// Replace 'backend' with one of: 'cpu', 'webgl', 'wasm', 'webgpu'


// For WebGL & WASM
await tf.browser.toPixels(imageTensor, canvas);

// For WebGPU & CPU
await tf.browser.draw(imageTensor, canvas);

⚠️ Note: Comment out the unused function depending on the backend.



Special Notes for WASM
For only WASM, use the standard wasm backend.
For WASM + SIMD + Threads support, uncomment the following lines:
javascript// tf.wasm.setThreadsCount(8);
// await tf.ready();

Running the Experiments
Step 1: Select a Model
Edit index.html and include the desired model script.
For example, to run RNN Model 4:
html<script src="./RNN/R4.js" type="module"></script>
Step 2: Start the Server
In your terminal, run:
bashnode server.js
Step 3: Access the Webpage
Open the following URL in your browser:
http://localhost:5500
Step 4: Collect Results
After running the experiment, a .json file containing all the metrics will be automatically saved in your Downloads folder.

Repository Structure
├── CNN/
│   ├── C1.js
│   ├── C2.js
│   ├── C3.js
│   ├── C4.js
│   └── C5.js
├── DNN/
│   ├── D1.js
│   ├── D2.js
│   ├── D3.js
│   ├── D4.js
│   └── D5.js
├── RNN/
│   ├── R1.js
│   ├── R2.js
│   ├── R3.js
│   ├── R4.js
│   └── R5.js
├── index.html
├── server.js
└── README.md