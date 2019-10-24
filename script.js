const video = document.getElementById('video')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
  faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
  // faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  // faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceExpressionNet.loadFromUri('/models')
]).then(startVideo)

async function startVideo() {
  const labeledFaceDescriptors = await loadLabeledImages()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  // let image
  // let canvas

  navigator.getUserMedia(
    { video: {} },
    stream => video.srcObject = stream,
    err => console.error(err)
  )

  // const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  // let image
  // let canvas
  
  video.addEventListener('play', () => {
    // if (image) image.remove()
    // if (canvas) canvas.remove()
    const canvas = faceapi.createCanvasFromMedia(video, '1000', '1000')
    document.body.append(canvas)

    
    const displaySize = { width: video.width, height: video.height }
    faceapi.matchDimensions(canvas, displaySize)


    setInterval(async () => {
      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors()
      const detections1 = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      const resizedDetections1 = faceapi.resizeResults(detections1, displaySize)
      const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

      
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box
        const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
        faceapi.draw.drawDetections(canvas, resizedDetections)
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections1)
        drawBox.draw(canvas)
      })

    }, 1000)
  })
}


function loadLabeledImages() {
  const labels = ['Cavidan Talibov', 'Narmina Talibova']
  return Promise.all(
    labels.map(async label => {
      const descriptions = []
      for (let i = 1; i <= 1; i++) {
        const img = await faceapi.fetchImage(`${label}/${i}.jpg`)
        console.log(img);
        
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}