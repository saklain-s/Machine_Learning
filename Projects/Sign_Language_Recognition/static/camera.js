let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let livePredictBtn = document.getElementById('live-predict-btn');
let liveResult = document.getElementById('live-result');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.play();
  })
  .catch(err => {
    console.error('Error accessing camera:', err);
  });

let livePredicting = false;

livePredictBtn.onclick = function() {
  if (!livePredicting) {
    livePredicting = true;
    livePredictBtn.textContent = 'Stop Live Predict';
    liveResult.style.display = 'block';
    livePredictLoop();
  } else {
    livePredicting = false;
    livePredictBtn.textContent = 'Live Predict';
    liveResult.style.display = 'none';
  }
};

function livePredictLoop() {
  if (!livePredicting) return;
  let context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(function(blob) {
    let formData = new FormData();
    formData.append('image', blob, 'frame.png');
    fetch('/live_predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      liveResult.innerHTML = '<strong>Live Prediction:</strong> ' + (data.prediction || '...');
      setTimeout(livePredictLoop, 500); // Predict every 0.5s
    })
    .catch(() => {
      liveResult.innerHTML = '<strong>Live Prediction:</strong> Error';
      setTimeout(livePredictLoop, 1000);
    });
  }, 'image/png');
}
