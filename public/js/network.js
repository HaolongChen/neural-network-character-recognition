// Neural Network Implementation using TensorFlow.js
class NeuralNetwork {
    constructor() {
        this.model = null;
        this.trainingData = [];
        this.labels = new Set(); // Keep track of unique labels
        this.isTraining = false;
        this.inputSize = 28 * 28; // 28x28 pixel images
        this.callbacks = {};
        this.dataAugmentation = true; // Enable data augmentation by default
    }

    // Initialize the neural network model with an improved architecture
    initModel() {
        // Clear existing model from memory if it exists
        if (this.model) {
            this.model.dispose();
        }

        // Get the number of unique labels/classes
        const numClasses = this.labels.size;
        if (numClasses === 0) {
            console.error("No labels defined. Cannot create model.");
            return false;
        }

        // Create a sequential model
        this.model = tf.sequential();

        // Add convolutional layers for better feature extraction
        // First convolutional layer
        this.model.add(tf.layers.reshape({
            targetShape: [28, 28, 1],
            inputShape: [this.inputSize]
        }));
        
        this.model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            kernelSize: 3,
            filters: 32,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        
        this.model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [2, 2]
        }));

        // Second convolutional layer
        this.model.add(tf.layers.conv2d({
            kernelSize: 3,
            filters: 64,
            strides: 1,
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        
        this.model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [2, 2]
        }));

        // Flatten the output from convolutional layers
        this.model.add(tf.layers.flatten());

        // Dense layers for classification
        this.model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));

        // Add dropout to reduce overfitting
        this.model.add(tf.layers.dropout({ rate: 0.3 }));

        // Output layer
        this.model.add(tf.layers.dense({
            units: numClasses,
            activation: 'softmax',
            kernelInitializer: 'varianceScaling'
        }));

        // Compile the model with improved settings
        this.model.compile({
            optimizer: tf.train.adam(0.0005), // Lower learning rate for better convergence
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return true;
    }

    // Toggle data augmentation
    setDataAugmentation(enabled) {
        this.dataAugmentation = enabled;
    }

    // Add a training example
    addTrainingExample(imageData, label) {
        if (!label) {
            console.error("Label is required for training data");
            return false;
        }

        // Add label to set of known labels
        this.labels.add(label);

        // Add data to training set
        this.trainingData.push({
            data: imageData,
            label: label
        });

        return true;
    }

    // Remove a training example by index
    removeTrainingExample(index) {
        if (index >= 0 && index < this.trainingData.length) {
            const removedItem = this.trainingData.splice(index, 1)[0];
            
            // Check if we need to update labels
            const removedLabel = removedItem.label;
            if (!this.trainingData.some(item => item.label === removedLabel)) {
                this.labels.delete(removedLabel);
            }
            
            return true;
        }
        return false;
    }

    // Get all known labels as an array
    getLabels() {
        return Array.from(this.labels);
    }

    // Apply data augmentation to a single example
    augmentExample(example) {
        const result = [];
        // Always add the original example
        result.push({
            data: example.data,
            label: example.label
        });

        if (!this.dataAugmentation) {
            return result;
        }

        // Create a 2D version of the data for easier manipulation
        const imageSize = 28;
        const image2D = [];
        for (let i = 0; i < imageSize; i++) {
            const row = [];
            for (let j = 0; j < imageSize; j++) {
                row.push(example.data[i * imageSize + j]);
            }
            image2D.push(row);
        }

        // Function to convert 2D array back to flat array
        const flatten = (arr2D) => {
            const result = new Float32Array(imageSize * imageSize);
            for (let i = 0; i < imageSize; i++) {
                for (let j = 0; j < imageSize; j++) {
                    result[i * imageSize + j] = arr2D[i][j];
                }
            }
            return result;
        };

        // 1. Add shifted versions (small translations)
        const shifts = [
            { x: 1, y: 0 },  // right
            { x: -1, y: 0 }, // left
            { x: 0, y: 1 },  // down
            { x: 0, y: -1 }, // up
        ];

        for (const shift of shifts) {
            const shiftedImage = Array(imageSize).fill().map(() => Array(imageSize).fill(0));
            
            for (let i = 0; i < imageSize; i++) {
                for (let j = 0; j < imageSize; j++) {
                    const newI = i + shift.y;
                    const newJ = j + shift.x;
                    
                    if (newI >= 0 && newI < imageSize && newJ >= 0 && newJ < imageSize) {
                        shiftedImage[newI][newJ] = image2D[i][j];
                    }
                }
            }
            
            result.push({
                data: flatten(shiftedImage),
                label: example.label
            });
        }

        // 2. Add a slightly rotated version
        // Simple approximation of rotation for small angles
        const rotateImage = (angle) => {
            const rotated = Array(imageSize).fill().map(() => Array(imageSize).fill(0));
            const centerX = imageSize / 2;
            const centerY = imageSize / 2;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            
            for (let y = 0; y < imageSize; y++) {
                for (let x = 0; x < imageSize; x++) {
                    const xDiff = x - centerX;
                    const yDiff = y - centerY;
                    
                    const sourceX = Math.round(xDiff * cos - yDiff * sin + centerX);
                    const sourceY = Math.round(xDiff * sin + yDiff * cos + centerY);
                    
                    if (sourceX >= 0 && sourceX < imageSize && sourceY >= 0 && sourceY < imageSize) {
                        rotated[y][x] = image2D[sourceY][sourceX];
                    }
                }
            }
            
            return rotated;
        };
        
        // Add slightly rotated versions
        result.push({
            data: flatten(rotateImage(0.1)), // ~6 degrees clockwise
            label: example.label
        });
        
        result.push({
            data: flatten(rotateImage(-0.1)), // ~6 degrees counter-clockwise
            label: example.label
        });

        return result;
    }

    // Preprocess data for training with augmentation
    prepareTrainingData() {
        if (this.trainingData.length === 0) {
            console.error("No training data available");
            return null;
        }

        // Get array of all known labels
        const labelArray = this.getLabels();
        
        // Apply data augmentation to increase dataset size
        let augmentedData = [];
        for (const example of this.trainingData) {
            augmentedData = augmentedData.concat(this.augmentExample(example));
        }
        
        // Prepare input data
        const xs = tf.tensor2d(
            augmentedData.map(example => Array.from(example.data)),
            [augmentedData.length, this.inputSize]
        );

        // Prepare output data (one-hot encoded)
        const ys = tf.tensor2d(
            augmentedData.map(example => {
                // Create a one-hot encoded array based on the label
                const labelIndex = labelArray.indexOf(example.label);
                const oneHot = Array(labelArray.length).fill(0);
                oneHot[labelIndex] = 1;
                return oneHot;
            }),
            [augmentedData.length, labelArray.length]
        );

        return { xs, ys };
    }

    // Register a callback function
    on(event, callback) {
        this.callbacks[event] = callback;
    }

    // Train the neural network with improved training parameters
    async train(epochs = 50, batchSize = 32) {
        if (this.isTraining) {
            console.error("Training already in progress");
            return false;
        }

        if (this.trainingData.length === 0) {
            console.error("No training data available");
            return false;
        }

        // Initialize or reinitialize the model
        if (!this.initModel()) {
            return false;
        }

        // Prepare data for training
        const { xs, ys } = this.prepareTrainingData();

        // Set training flag
        this.isTraining = true;

        try {
            // Train the model with improved parameters
            const history = await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: Math.min(batchSize, Math.floor(this.trainingData.length * 0.8)), // Adjust batch size based on data
                shuffle: true,
                validationSplit: 0.1, // Use 10% of data for validation
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        // Call the callback if registered
                        if (this.callbacks.epochEnd) {
                            this.callbacks.epochEnd(epoch, logs);
                        }
                    },
                    onTrainEnd: () => {
                        // Call the callback if registered
                        if (this.callbacks.trainEnd) {
                            this.callbacks.trainEnd();
                        }
                    }
                }
            });

            // Clean up tensors
            xs.dispose();
            ys.dispose();

            // Reset training flag
            this.isTraining = false;

            return history;
        } catch (error) {
            console.error("Error during training:", error);
            this.isTraining = false;
            return false;
        }
    }

    // Predict the label for an input image with improved prediction handling
    async predict(imageData) {
        if (!this.model) {
            console.error("Model not initialized");
            return null;
        }

        try {
            // Convert input data to tensor
            const input = tf.tensor2d([Array.from(imageData)], [1, this.inputSize]);

            // Make prediction
            const output = this.model.predict(input);
            
            // Get the result as an array
            const predictions = await output.array();
            
            // Get array of labels
            const labelArray = this.getLabels();
            
            // Map probabilities to labels
            const result = predictions[0].map((confidence, index) => ({
                label: labelArray[index],
                confidence: confidence
            })).sort((a, b) => b.confidence - a.confidence);
            
            // Clean up tensors
            input.dispose();
            output.dispose();
            
            return result;
        } catch (error) {
            console.error("Error during prediction:", error);
            return null;
        }
    }

    // Save model to local storage with metadata
    async saveModel() {
        if (!this.model) {
            console.error("No model to save");
            return false;
        }

        try {
            // Save model using IndexedDB
            await this.model.save('indexeddb://character-recognition-model');
            
            // Save training data, labels and metadata
            localStorage.setItem('character-recognition-labels', JSON.stringify(Array.from(this.labels)));
            localStorage.setItem('character-recognition-data', JSON.stringify(this.trainingData));
            localStorage.setItem('character-recognition-metadata', JSON.stringify({
                version: '2.0',
                dataAugmentation: this.dataAugmentation,
                inputSize: this.inputSize,
                timestamp: new Date().toISOString()
            }));
            
            return true;
        } catch (error) {
            console.error("Error saving model:", error);
            return false;
        }
    }

    // Load model from local storage with metadata handling
    async loadModel() {
        try {
            // Load model using IndexedDB
            this.model = await tf.loadLayersModel('indexeddb://character-recognition-model');
            
            // Load training data and labels
            const savedLabels = localStorage.getItem('character-recognition-labels');
            const savedData = localStorage.getItem('character-recognition-data');
            const metadataString = localStorage.getItem('character-recognition-metadata');
            
            if (savedLabels && savedData) {
                this.labels = new Set(JSON.parse(savedLabels));
                this.trainingData = JSON.parse(savedData);
                
                // Load metadata if available
                if (metadataString) {
                    const metadata = JSON.parse(metadataString);
                    if (metadata.dataAugmentation !== undefined) {
                        this.dataAugmentation = metadata.dataAugmentation;
                    }
                }
                
                return true;
            }
            
            return false;
        } catch (error) {
            console.error("Error loading model:", error);
            return false;
        }
    }

    // Get the number of training examples
    getTrainingCount() {
        return this.trainingData.length;
    }

    // Get the training data
    getTrainingData() {
        return this.trainingData;
    }

    // Clear all training data
    clearTrainingData() {
        this.trainingData = [];
        this.labels.clear();
        return true;
    }
}