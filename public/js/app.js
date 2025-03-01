// Main Application Logic
document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    const drawingCanvas = new DrawingCanvas('drawing-canvas');
    const neuralNetwork = new NeuralNetwork();
    const networkVisualizer = new NetworkVisualizer('network-canvas');
    
    // DOM elements
    const trainModeBtn = document.getElementById('train-mode');
    const recognizeModeBtn = document.getElementById('recognize-mode');
    const clearCanvasBtn = document.getElementById('clear-canvas');
    const actionButton = document.getElementById('action-button');
    const trainButton = document.getElementById('train-button');
    const labelInput = document.getElementById('character-label');
    const labelInputContainer = document.getElementById('label-input-container');
    const trainingDataContainer = document.getElementById('training-data');
    const trainingCountElement = document.getElementById('training-count');
    const accuracyElement = document.getElementById('accuracy');
    const epochsElement = document.getElementById('epochs');
    const predictionElement = document.getElementById('prediction');
    const confidenceBarsElement = document.getElementById('confidence-bars');
    const recognitionResult = document.getElementById('recognition-result');
    
    // Application state
    let isTrainingMode = true;
    let isTraining = false;
    let currentEpoch = 0;
    let trainingHistory = [];
    
    // Initialize UI
    updateUIForMode();
    loadSavedModel();
    
    // Add info about neural network visualization
    addNetworkUsageInstructions();
    
    // Event listeners
    trainModeBtn.addEventListener('click', () => {
        isTrainingMode = true;
        updateUIForMode();
    });
    
    recognizeModeBtn.addEventListener('click', () => {
        isTrainingMode = false;
        updateUIForMode();
    });
    
    clearCanvasBtn.addEventListener('click', () => {
        drawingCanvas.clearCanvas();
    });
    
    actionButton.addEventListener('click', () => {
        if (isTrainingMode) {
            addTrainingExample();
        } else {
            recognizeDrawing();
        }
    });
    
    trainButton.addEventListener('click', async () => {
        if (isTraining) return;
        
        await trainNetwork();
    });
    
    // Register neural network callbacks
    neuralNetwork.on('epochEnd', (epoch, logs) => {
        currentEpoch = epoch;
        trainingHistory.push({
            epoch: epoch,
            accuracy: logs.acc,
            loss: logs.loss
        });
        updateTrainingStats(epoch, logs);
    });
    
    neuralNetwork.on('trainEnd', () => {
        isTraining = false;
        trainButton.textContent = 'Train Network';
        trainButton.disabled = false;
        
        // Update UI after training
        updateTrainingCount();
        
        // Save model after training
        neuralNetwork.saveModel();
    });
    
    // Function to update UI based on current mode
    function updateUIForMode() {
        // Update button styling
        trainModeBtn.classList.toggle('active', isTrainingMode);
        recognizeModeBtn.classList.toggle('active', !isTrainingMode);
        
        // Update label input visibility
        labelInputContainer.style.display = isTrainingMode ? 'flex' : 'none';
        
        // Update action button text
        actionButton.textContent = isTrainingMode ? 'Add to Training' : 'Recognize';
        
        // Show/hide recognition result
        recognitionResult.style.display = isTrainingMode ? 'none' : 'block';
        
        // Clear prediction when switching to training mode
        if (isTrainingMode) {
            predictionElement.textContent = '-';
            confidenceBarsElement.innerHTML = '';
        }
    }
    
    // Add instructions about the neural network visualization
    function addNetworkUsageInstructions() {
        const networkSection = document.querySelector('.network-section');
        const instructionsDiv = document.createElement('div');
        instructionsDiv.className = 'network-instructions';
        instructionsDiv.innerHTML = `
            <p>📌 Click on any neuron to see details about how it works</p>
            <p>🔵 Blue connections represent positive weights</p>
            <p>🔴 Red connections represent negative weights</p>
        `;
        networkSection.appendChild(instructionsDiv);
    }
    
    // Function to add training example
    function addTrainingExample() {
        const label = labelInput.value.trim();
        
        if (!label) {
            alert('Please enter a label for the training example.');
            return;
        }
        
        // Get processed image data
        const processedData = drawingCanvas.getProcessedData();
        
        // Add to neural network
        const success = neuralNetwork.addTrainingExample(processedData, label);
        
        if (success) {
            // Create a thumbnail for display
            const thumbnail = drawingCanvas.createThumbnail(56, 56);
            
            // Create container for the training example
            const exampleContainer = document.createElement('div');
            exampleContainer.className = 'training-data-item';
            
            // Add thumbnail
            exampleContainer.appendChild(thumbnail);
            
            // Add label
            const labelElement = document.createElement('div');
            labelElement.className = 'label';
            labelElement.textContent = label;
            exampleContainer.appendChild(labelElement);
            
            // Add remove button
            const removeButton = document.createElement('div');
            removeButton.className = 'remove';
            removeButton.textContent = 'X';
            removeButton.addEventListener('click', () => {
                const index = Array.from(trainingDataContainer.children).indexOf(exampleContainer);
                if (neuralNetwork.removeTrainingExample(index)) {
                    trainingDataContainer.removeChild(exampleContainer);
                    updateTrainingCount();
                }
            });
            exampleContainer.appendChild(removeButton);
            
            // Add to display container
            trainingDataContainer.appendChild(exampleContainer);
            
            // Clear canvas and label input
            drawingCanvas.clearCanvas();
            labelInput.value = '';
            
            // Update counter
            updateTrainingCount();

            // Update visualization based on new data
            updateVisualization();
            
            // Show simulated input through network
            visualizeInputExample(processedData);
        }
    }
    
    // Function to visualize how input example flows through network
    function visualizeInputExample(inputData) {
        // Create mock activations that show the data flowing through the network
        const mockActivations = createMockActivationsFromInput(inputData);
        
        // Update visualization with the input data
        networkVisualizer.updateNetworkState(null, mockActivations, inputData);
    }
    
    // Create gradual activation levels simulating network processing
    function createMockActivationsFromInput(inputData) {
        const activations = [];
        
        // Input layer uses actual data
        activations.push(Array.from(inputData));
        
        // For hidden and output layers, generate mock activations based on input
        // Hidden layer 1
        const hidden1Size = neuralNetwork.model ? neuralNetwork.model.layers[0].units : 128;
        const hidden1 = new Array(hidden1Size).fill(0).map((_, i) => {
            // Create pattern based on input data regions
            let sum = 0;
            const regionsToSample = 4; // Sample a few regions of input
            for (let r = 0; r < regionsToSample; r++) {
                const startIdx = Math.floor(inputData.length * r / regionsToSample);
                const endIdx = Math.floor(inputData.length * (r + 1) / regionsToSample);
                for (let j = startIdx; j < endIdx; j++) {
                    sum += inputData[j] * 0.5 * (Math.sin(i*j/100) + 1);
                }
            }
            return Math.min(1, Math.max(0, sum / 10));
        });
        activations.push(hidden1);
        
        // Hidden layer 2
        const hidden2Size = neuralNetwork.model ? neuralNetwork.model.layers[1].units : 64;
        const hidden2 = new Array(hidden2Size).fill(0).map((_, i) => {
            // Create pattern based on hidden1
            let sum = 0;
            const nodesToSample = 20; // Sample a subset of hidden1
            for (let j = 0; j < nodesToSample; j++) {
                const idx = Math.floor(hidden1.length * j / nodesToSample);
                sum += hidden1[idx] * 0.5 * (Math.cos(i*j/50) + 1);
            }
            return Math.min(1, Math.max(0, sum / 5));
        });
        activations.push(hidden2);
        
        // Output layer (one-hot-like activation for the selected class)
        const numClasses = neuralNetwork.labels.size || 10;
        const output = new Array(numClasses).fill(0.1);
        
        // Pick a random class to "activate" more strongly
        const activeClass = Math.floor(Math.random() * numClasses);
        output[activeClass] = 0.9;
        
        activations.push(output);
        
        return activations;
    }
    
    // Function to recognize drawing
    async function recognizeDrawing() {
        // Check if we have a trained model
        if (neuralNetwork.getTrainingCount() === 0) {
            alert('Please add training examples and train the model first.');
            return;
        }
        
        // Get processed image data
        const processedData = drawingCanvas.getProcessedData();
        
        // Make prediction
        const predictions = await neuralNetwork.predict(processedData);
        
        if (predictions) {
            // Display prediction
            predictionElement.textContent = predictions[0].label;
            
            // Update confidence bars
            displayConfidenceBars(predictions);
            
            // Visualize the network's decision process
            visualizePrediction(processedData, predictions);
        } else {
            predictionElement.textContent = 'Error';
            confidenceBarsElement.innerHTML = '';
        }
    }
    
    // Function to update training count
    function updateTrainingCount() {
        const count = neuralNetwork.getTrainingCount();
        trainingCountElement.textContent = count;
        
        // Enable/disable train button
        trainButton.disabled = count === 0 || isTraining;
    }
    
    // Function to train the network
    async function trainNetwork() {
        if (neuralNetwork.getTrainingCount() === 0) {
            alert('Please add training examples first.');
            return;
        }
        
        // Update UI
        isTraining = true;
        trainButton.textContent = 'Training...';
        trainButton.disabled = true;
        
        // Reset training stats
        accuracyElement.textContent = '0%';
        epochsElement.textContent = '0';
        currentEpoch = 0;
        trainingHistory = [];
        
        // Train the network
        const epochs = Math.min(100, Math.max(20, neuralNetwork.getTrainingCount() * 5));
        
        // Animate the training process in the visualization
        startTrainingAnimation(epochs);
        
        await neuralNetwork.train(epochs);
    }
    
    // Animate network during training
    function startTrainingAnimation(totalEpochs) {
        const animationInterval = setInterval(() => {
            if (!isTraining) {
                clearInterval(animationInterval);
                return;
            }
            
            // Create activations that simulate the training process
            const progress = currentEpoch / totalEpochs;
            const mockActivations = createTrainingActivations(progress);
            
            // Update visualization
            networkVisualizer.updateNetworkState(null, mockActivations, null);
            
        }, 800); // Update every 800ms
    }
    
    // Generate mock activations that simulate training progress
    function createTrainingActivations(progress) {
        const activations = [];
        
        // Input layer
        const inputSize = neuralNetwork.inputSize;
        const inputLayer = new Array(inputSize).fill(0).map(() => 
            Math.random() * 0.5 // Random activation levels
        );
        activations.push(inputLayer);
        
        // Hidden layer 1
        const hidden1Size = neuralNetwork.model ? neuralNetwork.model.layers[0].units : 128;
        const hidden1 = new Array(hidden1Size).fill(0).map(() => 
            Math.random() * 0.7 * (0.5 + progress * 0.5) // Higher activations as training progresses
        );
        activations.push(hidden1);
        
        // Hidden layer 2
        const hidden2Size = neuralNetwork.model ? neuralNetwork.model.layers[1].units : 64;
        const hidden2 = new Array(hidden2Size).fill(0).map(() => 
            Math.random() * 0.8 * (0.6 + progress * 0.4) // Higher activations as training progresses
        );
        activations.push(hidden2);
        
        // Output layer - becomes more certain (higher contrast) as training progresses
        const numClasses = neuralNetwork.labels.size || 10;
        const output = new Array(numClasses).fill(0).map(() => 
            Math.random() * 0.3 * (1 - progress) // Lower random noise as training progresses
        );
        
        // Make one output gradually stronger to show increasing confidence
        const strongestOutput = Math.floor(Math.random() * numClasses);
        output[strongestOutput] = 0.3 + progress * 0.6; // Gets stronger over time
        
        activations.push(output);
        
        return activations;
    }
    
    // Function to update training statistics
    function updateTrainingStats(epoch, logs) {
        const accuracy = logs.acc * 100;
        epochsElement.textContent = epoch + 1;
        accuracyElement.textContent = accuracy.toFixed(1) + '%';
        
        // Update visualization
        updateVisualization();
    }
    
    // Function to display confidence bars
    function displayConfidenceBars(predictions) {
        confidenceBarsElement.innerHTML = '';
        
        // Get top 5 predictions or all if less than 5
        const topPredictions = predictions.slice(0, Math.min(5, predictions.length));
        
        topPredictions.forEach(prediction => {
            const barContainer = document.createElement('div');
            barContainer.className = 'confidence-bar';
            
            const label = document.createElement('div');
            label.className = 'confidence-bar-label';
            label.textContent = prediction.label;
            
            const outerBar = document.createElement('div');
            outerBar.className = 'confidence-bar-outer';
            
            const innerBar = document.createElement('div');
            innerBar.className = 'confidence-bar-inner';
            innerBar.style.width = Math.round(prediction.confidence * 100) + '%';
            
            const value = document.createElement('div');
            value.className = 'confidence-bar-value';
            value.textContent = Math.round(prediction.confidence * 100) + '%';
            
            outerBar.appendChild(innerBar);
            barContainer.appendChild(label);
            barContainer.appendChild(outerBar);
            barContainer.appendChild(value);
            
            confidenceBarsElement.appendChild(barContainer);
        });
    }
    
    // Function to update the network visualization
    function updateVisualization() {
        // Set layer sizes based on current model
        const labels = neuralNetwork.getLabels();
        const inputSize = 28 * 28; // 28x28 pixel images
        const outputSize = labels.length > 0 ? labels.length : 10; // Default to 10 if no labels yet
        
        // Our model has two hidden layers
        networkVisualizer.setNetworkConfig(inputSize, [128, 64], outputSize);
        
        // For a real app, we would extract actual weights and activations
        // Here we'll just use our improved visualization with default values
        networkVisualizer.drawNetwork();
    }
    
    // Function to visualize the prediction process
    function visualizePrediction(inputData, predictions) {
        // Create activations based on prediction results
        const activations = createPredictionActivations(predictions);
        
        // Update network visualization with input data and activations
        networkVisualizer.updateNetworkState(null, activations, inputData);
        
        // Start animation to show data flowing through network
        networkVisualizer.visualizePrediction(inputData, activations);
    }
    
    // Create activation values based on prediction results
    function createPredictionActivations(predictions) {
        const activations = [];
        
        // Input layer activations will be set from actual input data
        const inputLayer = new Array(28*28).fill(0.1); // Placeholder
        activations.push(inputLayer);
        
        // Generate hidden layer activations that would produce our result
        // These are simulated values showing a pattern that would lead to the predicted output
        
        // Hidden layer 1 - simulate activations that spread from input 
        const hidden1 = new Array(128).fill(0).map((_, i) => {
            // Create patterns that would contribute to our final prediction
            // For visualization purposes, this simulates what might happen in the network
            return 0.2 + 0.6 * Math.abs(Math.sin(i / 10)); // Pattern showing varied activation
        });
        activations.push(hidden1);
        
        // Hidden layer 2 - more focused activations leading to our prediction
        const hidden2 = new Array(64).fill(0).map((_, i) => {
            // Create more focused activations showing specialization in the network
            const group = Math.floor(i / 8); // Group neurons
            return group % 2 === 0 ? 0.2 + 0.7 * Math.abs(Math.sin(i / 5)) : 0.1;
        });
        activations.push(hidden2);
        
        // Output layer - use actual prediction confidences
        const outputActivations = Array(predictions.length).fill(0.05);
        predictions.forEach((prediction, i) => {
            outputActivations[i] = prediction.confidence;
        });
        
        activations.push(outputActivations);
        
        return activations;
    }
    
    // Load saved model if available
    async function loadSavedModel() {
        const success = await neuralNetwork.loadModel();
        
        if (success) {
            // Update UI elements
            updateTrainingCount();
            
            // Load training data thumbnails
            const trainingData = neuralNetwork.getTrainingData();
            
            trainingDataContainer.innerHTML = '';
            trainingData.forEach((example, index) => {
                // Create a small canvas for the thumbnail
                const thumbnailCanvas = document.createElement('canvas');
                thumbnailCanvas.width = 56;
                thumbnailCanvas.height = 56;
                const ctx = thumbnailCanvas.getContext('2d');
                
                // Draw the image data
                const imageData = new ImageData(28, 28);
                for (let i = 0; i < example.data.length; i++) {
                    const value = Math.round(example.data[i] * 255);
                    const idx = i * 4;
                    imageData.data[idx] = value; // R
                    imageData.data[idx + 1] = value; // G
                    imageData.data[idx + 2] = value; // B
                    imageData.data[idx + 3] = 255; // A
                }
                
                // Create temporary canvas for resizing
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28;
                tempCanvas.height = 28;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.putImageData(imageData, 0, 0);
                
                // Resize to thumbnail size
                ctx.drawImage(tempCanvas, 0, 0, 56, 56);
                
                // Create container for the training example
                const exampleContainer = document.createElement('div');
                exampleContainer.className = 'training-data-item';
                
                // Add thumbnail
                exampleContainer.appendChild(thumbnailCanvas);
                
                // Add label
                const labelElement = document.createElement('div');
                labelElement.className = 'label';
                labelElement.textContent = example.label;
                exampleContainer.appendChild(labelElement);
                
                // Add remove button
                const removeButton = document.createElement('div');
                removeButton.className = 'remove';
                removeButton.textContent = 'X';
                removeButton.addEventListener('click', () => {
                    const itemIndex = Array.from(trainingDataContainer.children).indexOf(exampleContainer);
                    if (neuralNetwork.removeTrainingExample(itemIndex)) {
                        trainingDataContainer.removeChild(exampleContainer);
                        updateTrainingCount();
                    }
                });
                exampleContainer.appendChild(removeButton);
                
                // Add to container
                trainingDataContainer.appendChild(exampleContainer);
            });
            
            // Update visualization
            updateVisualization();
        }
    }
});