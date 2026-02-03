/**
 * Teachable Machine Clone Logic
 * Handles camera, data collection, training, and inference.
 */

// --- Configuration ---
const CONFIG = {
  imageSize: 224,
  mobilenetVersion: 2,
  mobilenetAlpha: 1.0,
  topK: 3, // Show top 3 predictions
};

// --- State ---
const state = {
  mobilenet: null,
  model: null,
  classes: [], // { name: string, samples: number }
  data: {
    xs: null, // Tensor
    ys: null  // Tensor
  },
  isTraining: false,
  isPredicting: false,
  webcam: null,
  webcamElement: null,
  animationId: null
};

// --- DOM Elements ---
const UI = {
  classList: document.getElementById('class-list'),
  inputClassName: document.getElementById('input-class-name'),
  btnAddClass: document.getElementById('btn-add-class'),
  btnResetAll: document.getElementById('btn-reset-all'),

  btnTrain: document.getElementById('btn-train'),
  trainProgress: document.getElementById('train-progress'),
  trainStatus: document.getElementById('train-status'),
  settingEpochs: document.getElementById('setting-epochs'),
  settingLr: document.getElementById('setting-lr'),
  settingBatchSize: document.getElementById('setting-batch-size'),
  btnExport: document.getElementById('btn-export'),

  video: document.getElementById('webcam'),
  cameraPlaceholder: document.getElementById('camera-placeholder'),
  btnEnableCam: document.getElementById('btn-enable-cam'),
  btnPredict: document.getElementById('btn-predict'),
  predictionResults: document.getElementById('prediction-results'),

  toast: document.getElementById('toast')
};

// --- Initialization ---
async function init() {
  showToast('Loading MobileNet...', 'info');
  try {
    state.mobilenet = await mobilenet.load({
      version: CONFIG.mobilenetVersion,
      alpha: CONFIG.mobilenetAlpha
    });

    // Warm up
    tf.tidy(() => state.mobilenet.infer(tf.zeros([1, 224, 224, 3]), true));

    showToast('MobileNet loaded!', 'success');
    UI.btnEnableCam.disabled = false;
  } catch (err) {
    console.error(err);
    showToast('Failed to load MobileNet', 'error');
  }
}

// --- Camera Handling ---
async function setupCamera() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false
      });
      UI.video.srcObject = stream;
      state.webcamElement = UI.video;

      // Wait for video metadata to load (handle race condition)
      await new Promise((resolve) => {
        if (UI.video.readyState >= 1) {
          resolve();
        } else {
          UI.video.onloadedmetadata = resolve;
        }
      });

      UI.video.play();
      UI.cameraPlaceholder.style.display = 'none';
    } catch (err) {
      console.error(err);
      showToast('Camera access denied', 'error');
    }
  }
}

// --- Data Management ---
function addClass(name) {
  if (!name) return;
  const id = state.classes.length;
  state.classes.push({ id, name, samples: 0 });
  renderClassList();
  UI.inputClassName.value = '';
  updateTrainButton();
}

function renderClassList() {
  UI.classList.innerHTML = '';
  state.classes.forEach((cls, index) => {
    const el = document.createElement('div');
    el.className = 'class-item';

    const header = document.createElement('div');
    header.className = 'class-header';
    header.innerHTML = `<span class="class-name">${cls.name}</span><span class="sample-count">${cls.samples} samples</span>`;

    const actions = document.createElement('div');
    actions.className = 'class-actions';

    const btn = document.createElement('button');
    btn.className = 'btn btn-primary btn-sm';
    btn.textContent = 'Hold to Record';

    // Attach listeners safely
    const start = (e) => {
      if (e.cancelable) e.preventDefault(); // Prevent ghost clicks / scrolling
      startCollecting(index, btn);
    };
    const stop = (e) => {
      if (e.cancelable) e.preventDefault();
      stopCollecting(btn);
    };

    btn.addEventListener('mousedown', start);
    btn.addEventListener('touchstart', start);

    btn.addEventListener('mouseup', stop);
    btn.addEventListener('mouseleave', stop);
    btn.addEventListener('touchend', stop);
    btn.addEventListener('contextmenu', (e) => e.preventDefault()); // No right click

    actions.appendChild(btn);
    el.appendChild(header);
    el.appendChild(actions);
    UI.classList.appendChild(el);
  });

  if (state.classes.length > 0) {
    document.getElementById('initial-help').style.display = 'none';
  }
}

let isCollecting = false;

function startCollecting(classId, btn) {
  if (!state.webcamElement) {
    showToast('Enable camera first!', 'error');
    return;
  }

  if (isCollecting) return;
  isCollecting = true;
  btn.classList.add('active');

  // Loop
  const loop = async () => {
    if (!isCollecting) return;
    try {
      await addSample(classId);
      if (isCollecting) setTimeout(loop, 100); // Schedule next
    } catch (err) {
      console.error(err);
      showToast('Error capturing: ' + err.message, 'error');
      stopCollecting(btn);
    }
  };

  loop();
}

function stopCollecting(btn) {
  isCollecting = false;
  if (btn) btn.classList.remove('active');
}

async function addSample(classId) {
  if (state.isTraining) return;

  // 1. Capture image
  const img = tf.browser.fromPixels(state.webcamElement);

  // 2. Get embedding
  const embedding = tf.tidy(() => {
    const emb = state.mobilenet.infer(img, true); // Should be [1, 1024]
    // Ensure it is 2D [1, N]
    if (emb.shape.length === 1) return emb.expandDims(0);
    return emb;
  });

  // 3. Add to dataset
  if (state.data.xs == null) {
    state.data.xs = tf.keep(embedding);
    state.data.ys = tf.keep(tf.tensor1d([classId], 'int32'));
  } else {
    const oldX = state.data.xs;
    const oldY = state.data.ys;

    const newX = oldX.concat(embedding, 0);
    const newY = oldY.concat(tf.tensor1d([classId], 'int32'), 0);

    state.data.xs = tf.keep(newX);
    state.data.ys = tf.keep(newY);

    oldX.dispose();
    oldY.dispose();
    embedding.dispose(); // Dispose the temp embedding (since we kept the concat result)
  }

  img.dispose();

  // 4. Update UI
  state.classes[classId].samples++;
  // Efficiently update just the text if possible, or re-render
  // For now, full re-render is safe but maybe slow. Let's try to just update the text.
  const classItems = document.querySelectorAll('.class-item');
  if (classItems[classId]) {
    const countSpan = classItems[classId].querySelector('.sample-count');
    if (countSpan) countSpan.textContent = `${state.classes[classId].samples} samples`;
  }

  updateTrainButton();
}

function updateTrainButton() {
  const hasEnough = state.classes.length >= 2 && state.classes.every(c => c.samples > 0);
  UI.btnTrain.disabled = !hasEnough || state.isTraining;
}

// --- Training ---
async function train() {
  if (state.isTraining) return;

  // --- 1. Validation ---
  if (state.classes.length < 2) {
    showToast('Add at least 2 classes', 'error');
    return;
  }
  if (!state.data.xs || !state.data.ys) {
    showToast('Collect data first', 'error');
    return;
  }

  // --- 2. Setup ---
  state.isTraining = true;
  UI.btnTrain.disabled = true;
  UI.btnTrain.textContent = 'Training...';

  try {
    UI.trainStatus.textContent = 'Preparing data...';
    await tf.nextFrame();

    // Get actual input shape from data
    const inputShape = state.data.xs.shape.slice(1); // e.g. [1280]
    const numClasses = state.classes.length;

    // --- 3. Model Creation ---
    // Dispose old model if exists
    if (state.model) {
      state.model.dispose();
    }

    state.model = tf.sequential();
    state.model.add(tf.layers.dense({
      inputShape: inputShape,
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
      useBias: true
    }));
    state.model.add(tf.layers.dense({
      units: numClasses,
      kernelInitializer: 'varianceScaling',
      useBias: false,
      activation: 'softmax'
    }));

    UI.trainStatus.textContent = 'Compiling...';
    await tf.nextFrame();

    const optimizer = tf.train.adam(parseFloat(UI.settingLr.value));
    state.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    // --- 4. Training ---
    const batchSize = parseInt(UI.settingBatchSize.value);
    const epochs = parseInt(UI.settingEpochs.value);

    UI.trainStatus.textContent = 'Encoding labels...';
    const yOneHot = tf.oneHot(state.data.ys, numClasses);

    UI.trainStatus.textContent = 'Starting training...';
    await tf.nextFrame();

    await state.model.fit(state.data.xs, yOneHot, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const progress = ((epoch + 1) / epochs) * 100;
          UI.trainProgress.style.width = `${progress}%`;

          const acc = logs.acc || logs.accuracy || 0;
          const loss = logs.loss || 0;

          UI.trainStatus.textContent = `Epoch ${epoch + 1}/${epochs} | Loss: ${loss.toFixed(4)} | Acc: ${(acc * 100).toFixed(1)}%`;
          await tf.nextFrame();
        }
      }
    });

    // --- 5. Cleanup & Finish ---
    yOneHot.dispose();

    state.isTraining = false;
    UI.btnTrain.textContent = 'Model Trained';
    UI.btnTrain.classList.remove('btn-success');
    UI.btnTrain.classList.add('btn-secondary');
    UI.btnPredict.disabled = false;
    UI.btnExport.disabled = false;
    showToast('Training complete!', 'success');

    togglePrediction();

  } catch (err) {
    console.error(err);
    state.isTraining = false;
    UI.btnTrain.disabled = false;
    UI.btnTrain.textContent = 'Train Model';
    UI.trainStatus.textContent = 'Error: ' + err.message;
    showToast('Training failed: ' + err.message, 'error');
  }
}

// --- Inference ---
async function predict() {
  if (!state.isPredicting || !state.model) return;

  const img = tf.browser.fromPixels(state.webcamElement);
  const embedding = state.mobilenet.infer(img, true);
  const predictions = state.model.predict(embedding);

  const data = await predictions.data();

  img.dispose();
  embedding.dispose();
  predictions.dispose();

  showPredictions(data);

  state.animationId = requestAnimationFrame(predict);
}

function showPredictions(probs) {
  UI.predictionResults.innerHTML = '';

  // Find max probability index for highlighting
  let maxIndex = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIndex]) maxIndex = i;
  }

  // Iterate through classes in original order
  state.classes.forEach((cls, i) => {
    const p = probs[i];
    const percent = (p * 100).toFixed(1);
    const isWinner = i === maxIndex;

    const el = document.createElement('div');
    el.className = `prediction-item ${isWinner ? 'winner' : ''}`;
    // Add extra style for winner if class not present in CSS yet
    if (isWinner) {
      el.style.backgroundColor = '#eff6ff';
      // el.style.border = '1px solid var(--primary)'; // Removed frame
      el.style.borderRadius = '8px';
      el.style.padding = '8px';
      el.style.marginBottom = '4px';
    }

    el.innerHTML = `
      <span class="pred-label" style="${isWinner ? 'font-weight:bold; color:var(--primary);' : ''}">${cls.name}</span>
      <div class="pred-bar-bg">
        <div class="pred-bar-fill" style="width: ${percent}%; ${isWinner ? 'background-color:var(--primary);' : 'background-color:var(--secondary); opacity:0.7;'}"></div>
      </div>
      <span class="pred-value" style="${isWinner ? 'font-weight:bold;' : ''}">${percent}%</span>
    `;
    UI.predictionResults.appendChild(el);
  });
}

function togglePrediction() {
  state.isPredicting = !state.isPredicting;
  if (state.isPredicting) {
    UI.btnPredict.textContent = 'Stop Predicting';
    UI.btnPredict.classList.replace('btn-primary', 'btn-danger');
    predict();
  } else {
    UI.btnPredict.textContent = 'Start Predicting';
    UI.btnPredict.classList.replace('btn-danger', 'btn-primary');
    cancelAnimationFrame(state.animationId);
  }
}

// --- Utilities ---
function showToast(msg, type = 'info') {
  UI.toast.textContent = msg;
  UI.toast.className = `toast ${type} show`;
  setTimeout(() => {
    UI.toast.classList.remove('show');
  }, 3000);
}

function resetAll() {
  if (!confirm('Are you sure you want to reset everything?')) return;
  location.reload();
}

async function exportModel() {
  if (!state.model) return;

  UI.btnExport.textContent = 'Zipping...';
  UI.btnExport.disabled = true;

  try {
    const zip = new JSZip();

    // 1. Save Model Artifacts
    // Use a custom save handler to get artifacts in memory
    let modelArtifacts = null;
    let weightData = null;

    await state.model.save(tf.io.withSaveHandler(async (artifacts) => {
      modelArtifacts = artifacts;
      weightData = artifacts.weightData;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      };
    }));

    // Add model files to zip
    // Construct proper model.json for tf.loadLayersModel
    const modelJson = {
      modelTopology: modelArtifacts.modelTopology,
      format: modelArtifacts.format,
      generatedBy: modelArtifacts.generatedBy,
      convertedBy: modelArtifacts.convertedBy,
      weightsManifest: [
        {
          paths: ['./weights.bin'],
          weights: modelArtifacts.weightSpecs
        }
      ]
    };

    zip.file('model.json', JSON.stringify(modelJson));

    if (weightData) {
      zip.file('weights.bin', weightData);
    }

    // 2. Save Metadata
    const metadata = {
      classes: state.classes.map(c => c.name),
      imageSize: CONFIG.imageSize
    };
    zip.file('metadata.json', JSON.stringify(metadata, null, 2));

    // 3. Standalone App Files
    const standaloneIndex = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Computer Vision Model</title>
  <style>
    body { font-family: sans-serif; padding: 20px; background: #f0f2f5; display: flex; flex-direction: column; align-items: center; }
    .container { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; width: 100%; }
    h1 { text-align: center; margin-top: 0; color: #333; }
    video { width: 100%; border-radius: 8px; background: black; transform: scaleX(-1); }
    #predictions { margin-top: 20px; }
    .pred-item { display: flex; align-items: center; margin-bottom: 8px; }
    .label { width: 100px; font-weight: bold; }
    .bar-bg { flex: 1; height: 10px; background: #eee; border-radius: 5px; overflow: hidden; margin: 0 10px; }
    .bar-fill { height: 100%; background: #6366f1; width: 0%; transition: width 0.1s; }
    .value { width: 50px; text-align: right; font-size: 0.9em; color: #666; }
    .winner .bar-fill { background: #10b981; }
    .winner .label { color: #10b981; }
    #status { text-align: center; margin-bottom: 10px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>My Model</h1>
    <div id="status">Loading model...</div>
    <div style="position: relative;">
      <video id="webcam" autoplay playsinline muted></video>
    </div>
    <div id="predictions"></div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0/dist/mobilenet.min.js"></script>
  <script src="script.js"></script>
</body>
</html>`;

    const standaloneScript = `
const VIDEO = document.getElementById('webcam');
const STATUS = document.getElementById('status');
const PREDS = document.getElementById('predictions');

let mobilenetModel;
let myModel;
let classes = [];

async function init() {
  try {
    // Load Metadata
    const metaRes = await fetch('metadata.json');
    const meta = await metaRes.json();
    classes = meta.classes;

    // Load Models
    STATUS.textContent = 'Loading MobileNet...';
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });

    STATUS.textContent = 'Loading Custom Model...';
    myModel = await tf.loadLayersModel('model.json');

    // Setup Camera
    STATUS.textContent = 'Starting Camera...';
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    VIDEO.srcObject = stream;
    await new Promise(r => VIDEO.onloadedmetadata = r);
    
    STATUS.textContent = 'Running...';
    predictLoop();
  } catch (err) {
    STATUS.textContent = 'Error: ' + err.message;
    console.error(err);
  }
}

async function predictLoop() {
  tf.tidy(() => {
    const img = tf.browser.fromPixels(VIDEO);
    const emb = mobilenetModel.infer(img, true);
    const prediction = myModel.predict(emb);
    const probs = prediction.dataSync();
    showPredictions(probs);
  });
  requestAnimationFrame(predictLoop);
}

function showPredictions(probs) {
  PREDS.innerHTML = '';
  let maxIdx = 0;
  for(let i=1; i<probs.length; i++) if(probs[i] > probs[maxIdx]) maxIdx = i;

  classes.forEach((cls, i) => {
    const p = probs[i];
    const pct = (p*100).toFixed(1);
    const isWinner = i === maxIdx;
    
    const div = document.createElement('div');
    div.className = \`pred-item \${isWinner ? 'winner' : ''}\`;
    div.innerHTML = \`
      <div class="label">\${cls}</div>
      <div class="bar-bg"><div class="bar-fill" style="width:\${pct}%"></div></div>
      <div class="value">\${pct}%</div>
    \`;
    PREDS.appendChild(div);
  });
}

init();
`;

    const readme = `# How to run this model

1. Because of browser security restrictions (CORS), you cannot simply double-click \`index.html\`.
2. You need to run a local web server.
3. If you have Python installed, open a terminal in this folder and run:
   \`python -m http.server\`
   Then open \`http://localhost:8000\` in your browser.
4. Alternatively, use the "Web Server for Chrome" extension or VS Code's "Live Server".
`;

    zip.file('index.html', standaloneIndex);
    zip.file('script.js', standaloneScript);
    zip.file('README.md', readme);

    // 4. Generate Zip
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'my-computer-vision-model.zip';
    a.click();
    URL.revokeObjectURL(url);

    showToast('Export successful!', 'success');
  } catch (err) {
    console.error(err);
    showToast('Export failed: ' + err.message, 'error');
  } finally {
    UI.btnExport.textContent = 'Download Model';
    UI.btnExport.disabled = false;
  }
}

// --- Event Listeners ---
UI.btnEnableCam.addEventListener('click', setupCamera);
UI.btnAddClass.addEventListener('click', () => addClass(UI.inputClassName.value));
UI.inputClassName.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') addClass(UI.inputClassName.value);
});
UI.btnTrain.addEventListener('click', train);
UI.btnPredict.addEventListener('click', togglePrediction);
UI.btnResetAll.addEventListener('click', resetAll);
UI.btnExport.addEventListener('click', exportModel);

// Start
init();
