// Neural Network Visualization
class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.layerSizes = [784, 128, 64, 10]; // Default sizes
        this.layerPositions = [];
        this.nodeRadius = 7;
        this.weights = null;
        this.neuronValues = null;
        this.activations = null;
        this.selectedNeuron = null; // Track which neuron is selected for detail view
        this.animating = false;
        this.setupCanvas();

        // Add event listener for neuron selection
        this.canvas.addEventListener('click', this.handleCanvasClick.bind(this));
    }

    setupCanvas() {
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.calculateLayerPositions();
        this.drawNetwork();
    }

    calculateLayerPositions() {
        const padding = 60;
        const width = this.canvas.width - (padding * 2);
        const layerSpacing = width / (this.layerSizes.length - 1);
        
        this.layerPositions = [];
        for (let i = 0; i < this.layerSizes.length; i++) {
            this.layerPositions.push(padding + i * layerSpacing);
        }
    }

    // Update network configuration based on actual model
    setNetworkConfig(inputSize, hiddenLayers, outputSize) {
        this.layerSizes = [inputSize, ...hiddenLayers, outputSize];
        this.calculateLayerPositions();
    }

    // Update network visualization with weights and activation values
    updateNetworkState(weights, activations, inputValues) {
        this.weights = weights;
        this.activations = activations; // Layer activations
        this.neuronValues = inputValues; // Input neuron values (for visualization)
        this.drawNetwork();
    }

    drawNetwork() {
        // Clear canvas
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw connections between nodes
        this.drawConnections();
        
        // Draw nodes
        this.drawNodes();

        // Draw layer labels
        this.drawLabels();

        // If a neuron is selected, show details
        if (this.selectedNeuron) {
            this.drawNeuronDetails(this.selectedNeuron.layer, this.selectedNeuron.index);
        }
    }

    handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if click is on a neuron
        for (let l = 0; l < this.layerSizes.length; l++) {
            const layerX = this.layerPositions[l];
            
            for (let i = 0; i < Math.min(this.layerSizes[l], 20); i++) {
                const nodeY = this.getNodeY(l, i);
                const distance = Math.sqrt(Math.pow(x - layerX, 2) + Math.pow(y - nodeY, 2));
                
                if (distance <= this.nodeRadius + 2) {
                    // Toggle selection: if already selected, deselect it
                    if (this.selectedNeuron && this.selectedNeuron.layer === l && this.selectedNeuron.index === i) {
                        this.selectedNeuron = null;
                    } else {
                        this.selectedNeuron = { layer: l, index: i };
                    }
                    this.drawNetwork();
                    return;
                }
            }
        }
        
        // If click is not on a neuron, clear selection
        if (this.selectedNeuron) {
            this.selectedNeuron = null;
            this.drawNetwork();
        }
    }

    drawConnections() {
        // Fix connections visualization with better visibility
        
        // Clear any previous connections
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // For each layer (except the last one)
        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const sourceX = this.layerPositions[l];
            const targetX = this.layerPositions[l + 1];
            
            // Calculate the maximum number of nodes to show per layer for clarity
            const maxSourceNodesToShow = Math.min(this.layerSizes[l], 15);
            const maxTargetNodesToShow = Math.min(this.layerSizes[l + 1], 15);
            
            // Calculate step sizes to distribute visible connections evenly
            const sourceStep = Math.ceil(this.layerSizes[l] / maxSourceNodesToShow);
            const targetStep = Math.ceil(this.layerSizes[l + 1] / maxTargetNodesToShow);
            
            // Get visible source nodes
            const sourceNodes = [];
            for (let i = 0; i < this.layerSizes[l]; i += sourceStep) {
                const y = this.getNodeY(l, i);
                if (y > 0) {  // Skip nodes that are not visible (y == -100)
                    sourceNodes.push({ index: i, y: y });
                }
            }
            
            // Get visible target nodes
            const targetNodes = [];
            for (let j = 0; j < this.layerSizes[l + 1]; j += targetStep) {
                const y = this.getNodeY(l + 1, j);
                if (y > 0) {  // Skip nodes that are not visible (y == -100)
                    targetNodes.push({ index: j, y: y });
                }
            }
            
            // Check if a neuron is selected to highlight its connections
            const isSourceLayerSelected = this.selectedNeuron && this.selectedNeuron.layer === l;
            const isTargetLayerSelected = this.selectedNeuron && this.selectedNeuron.layer === l + 1;
            
            // Draw connections between visible source and target nodes
            for (const source of sourceNodes) {
                const isSourceSelected = isSourceLayerSelected && this.selectedNeuron.index === source.index;
                
                for (const target of targetNodes) {
                    const isTargetSelected = isTargetLayerSelected && this.selectedNeuron.index === target.index;
                    
                    // Generate a pseudo-random weight for visualization
                    const seed = (source.index * this.layerSizes[l+1] + target.index) % 100;
                    const weight = (Math.sin(seed) + 1) / 2; // Range 0-1
                    
                    // Determine weight sign (positive/negative)
                    const weightSign = Math.cos(seed * 0.5) > 0;
                    
                    // Determine line style based on selection state and weight
                    let lineWidth = 0.8; // Slightly thicker default lines
                    let alpha = 0.4;    // Higher default alpha for better visibility
                    
                    if (isSourceSelected || isTargetSelected) {
                        // Highlight connections for selected neurons
                        lineWidth = 2.5;
                        alpha = 1.0;
                    } else if (this.selectedNeuron) {
                        // Fade other connections when any neuron is selected, but keep them visible
                        alpha = 0.1;
                    }
                    
                    // Color based on weight sign with better visibility
                    // Use more saturated colors with higher opacity
                    const color = weightSign 
                        ? `rgba(50, 100, 255, ${alpha})` // Brighter blue for positive
                        : `rgba(255, 70, 70, ${alpha})`;  // Brighter red for negative
                    
                    // Draw connection line
                    this.ctx.beginPath();
                    this.ctx.moveTo(sourceX, source.y);
                    this.ctx.lineTo(targetX, target.y);
                    this.ctx.strokeStyle = color;
                    this.ctx.lineWidth = lineWidth;
                    this.ctx.stroke();
                }
            }
        }
    }

    drawNodes() {
        // Draw nodes (but don't call drawConnections here as it clears the canvas)
        for (let l = 0; l < this.layerSizes.length; l++) {
            const x = this.layerPositions[l];
            const maxNodesToShow = 15; // Maximum nodes to show per layer
            
            // If layer has too many nodes, show representative sample
            if (this.layerSizes[l] > maxNodesToShow) {
                // Draw visible nodes at the top
                for (let i = 0; i < Math.min(maxNodesToShow - 3, this.layerSizes[l]); i++) {
                    const y = this.getNodeY(l, i);
                    this.drawNode(x, y, l, i);
                }
                
                // Draw ellipsis to indicate hidden nodes
                this.ctx.fillStyle = '#666';
                this.ctx.font = '16px Arial';
                this.ctx.textAlign = 'center';
                const ellipsisY = this.getNodeY(l, maxNodesToShow - 2) + 10;
                this.ctx.fillText('⋮', x, ellipsisY);
                
                // Draw a few more at the bottom
                for (let i = Math.max(0, this.layerSizes[l] - 2); i < this.layerSizes[l]; i++) {
                    const y = this.getNodeY(l, i);
                    this.drawNode(x, y, l, i);
                }
            } else {
                // Draw all nodes
                for (let i = 0; i < this.layerSizes[l]; i++) {
                    const y = this.getNodeY(l, i);
                    this.drawNode(x, y, l, i);
                }
            }
        }
    }

    drawNode(x, y, layer, index) {
        // Determine activation level for this neuron
        let activation = 0.1; // Default low activation

        // If we're visualizing an input, use the actual input value
        if (layer === 0 && this.neuronValues && index < this.neuronValues.length) {
            activation = this.neuronValues[index];
        } 
        // Otherwise use the layer activation if available
        else if (this.activations && this.activations[layer] && 
                 index < this.activations[layer].length) {
            activation = this.activations[layer][index];
        }
        
        // Determine node style based on activation and selection
        const isSelected = this.selectedNeuron && 
            this.selectedNeuron.layer === layer && 
            this.selectedNeuron.index === index;
            
        // Calculate radius - selected neurons are larger
        const radius = isSelected ? this.nodeRadius * 1.5 : this.nodeRadius;
        
        // Calculate fill color based on activation (white to blue gradient)
        const intensity = Math.round(activation * 255);
        let fillColor;
        
        // Different colors for different layer types
        if (layer === 0) {
            // Input layer: gray to black based on input intensity
            fillColor = `rgb(${255-intensity}, ${255-intensity}, ${255-intensity})`;
        } else if (layer === this.layerSizes.length - 1) {
            // Output layer: white to green based on output confidence
            fillColor = `rgb(${255-Math.round(intensity*0.7)}, 255, ${255-intensity})`;
        } else {
            // Hidden layers: white to blue based on activation
            fillColor = `rgb(${255-intensity}, ${255-Math.round(intensity*0.8)}, 255)`;
        }
        
        // Draw node circle with gradient
        const gradient = this.ctx.createRadialGradient(
            x, y, 0,
            x, y, radius
        );
        gradient.addColorStop(0, fillColor);
        gradient.addColorStop(1, `rgba(${fillColor.slice(4, -1)}, 0.8)`);
        
        // Draw the neuron
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        
        // Add border - thicker for selected neurons
        this.ctx.strokeStyle = isSelected ? '#333' : '#999';
        this.ctx.lineWidth = isSelected ? 2 : 1;
        this.ctx.stroke();
        
        // Add activation indicator inside the neuron
        if (radius > 5) {
            const innerRadius = radius * 0.6 * activation;
            if (innerRadius > 1) {
                this.ctx.beginPath();
                this.ctx.arc(x, y, innerRadius, 0, 2 * Math.PI);
                this.ctx.fillStyle = isSelected ? '#333' : 'rgba(0,0,0,0.3)';
                this.ctx.fill();
            }
        }
    }

    drawNeuronDetails(layer, index) {
        const x = this.layerPositions[layer];
        const y = this.getNodeY(layer, index);
        
        // Draw detail popup
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        this.ctx.strokeStyle = '#666';
        this.ctx.lineWidth = 1;
        
        // Create detail box at appropriate position
        const boxWidth = 180;
        const boxHeight = 100;
        let boxX = x + 20;
        let boxY = y - boxHeight/2;
        
        // Keep box within canvas bounds
        if (boxX + boxWidth > this.canvas.width) boxX = x - boxWidth - 20;
        if (boxY < 10) boxY = 10;
        if (boxY + boxHeight > this.canvas.height - 10) boxY = this.canvas.height - boxHeight - 10;
        
        // Draw box with rounded corners
        this.ctx.beginPath();
        this.ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 8);
        this.ctx.fill();
        this.ctx.stroke();
        
        // Add detail content
        this.ctx.textAlign = 'left';
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 12px Arial';
        
        // Show neuron position info
        let layerName = "Input";
        if (layer === this.layerSizes.length - 1) layerName = "Output";
        else if (layer > 0) layerName = `Hidden ${layer}`;
        
        this.ctx.fillText(`${layerName} Neuron #${index}`, boxX + 10, boxY + 20);
        
        // Show activation info
        this.ctx.font = '12px Arial';
        let activationValue = 0;
        
        if (layer === 0 && this.neuronValues && index < this.neuronValues.length) {
            activationValue = this.neuronValues[index];
        } else if (this.activations && this.activations[layer]) {
            activationValue = this.activations[layer][index] || 0;
        }
        
        this.ctx.fillText(`Activation: ${activationValue.toFixed(4)}`, boxX + 10, boxY + 40);
        
        // Show function info based on layer
        let activationFunction = "Linear";
        if (layer > 0 && layer < this.layerSizes.length - 1) {
            activationFunction = "ReLU";
        } else if (layer === this.layerSizes.length - 1) {
            activationFunction = "Softmax";
        }
        
        this.ctx.fillText(`Function: ${activationFunction}`, boxX + 10, boxY + 60);
        
        // Simple formula visualization
        if (layer > 0) {
            this.ctx.fillText("Formula: Σ(weight × input) + bias", boxX + 10, boxY + 80);
        } else {
            this.ctx.fillText("Input value: normalized pixel", boxX + 10, boxY + 80);
        }
    }

    getNodeY(layer, index) {
        const layerHeight = this.canvas.height - 80; // Leave space at top and bottom
        const startY = 40;
        
        // If too many nodes, we need to adjust the visual spacing
        let totalNodes = this.layerSizes[layer];
        let visualIndex = index;
        
        const maxVisibleNodes = 15;
        if (totalNodes > maxVisibleNodes) {
            // Adjust mapping for representative nodes
            if (index < (maxVisibleNodes - 3)) {
                // Top nodes map directly
                visualIndex = index;
                totalNodes = maxVisibleNodes;
            } else if (index >= (this.layerSizes[layer] - 2)) {
                // Bottom few nodes
                visualIndex = maxVisibleNodes - 2 + (index - (this.layerSizes[layer] - 2));
                totalNodes = maxVisibleNodes;
            } else {
                // Middle nodes (hidden by ellipsis)
                return -100; // Off-screen
            }
        }
        
        // Calculate Y position based on layer size and node index
        return startY + (visualIndex * layerHeight) / (totalNodes - 1 || 1);
    }

    drawLabels() {
        this.ctx.font = 'bold 14px Arial';
        this.ctx.fillStyle = '#333';
        this.ctx.textAlign = 'center';
        
        // Layer labels
        const labels = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'];
        for (let i = 0; i < this.layerSizes.length; i++) {
            const labelText = i < labels.length ? labels[i] : `Layer ${i+1}`;
            this.ctx.fillText(labelText, this.layerPositions[i], 20);
        }

        // Size indicators
        this.ctx.font = '12px Arial';
        for (let i = 0; i < this.layerSizes.length; i++) {
            const sizeText = `${this.layerSizes[i]} neurons`;
            this.ctx.fillText(sizeText, this.layerPositions[i], this.canvas.height - 10);
            
            // Add activation function labels
            let fnText = i === 0 ? "Linear" : i === this.layerSizes.length - 1 ? "Softmax" : "ReLU";
            this.ctx.fillText(fnText, this.layerPositions[i], this.canvas.height - 25);
        }
        
        // Add legend
        this.drawLegend();
    }
    
    drawLegend() {
        const legendX = 15;
        const legendY = this.canvas.height - 60;
        const legendWidth = 120;
        const legendHeight = 50;
        
        // Draw legend background
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.strokeStyle = '#ccc';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.roundRect(legendX, legendY, legendWidth, legendHeight, 5);
        this.ctx.fill();
        this.ctx.stroke();
        
        // Draw legend title
        this.ctx.font = 'bold 12px Arial';
        this.ctx.fillStyle = '#333';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('Connection Weights:', legendX + 10, legendY + 15);
        
        // Draw positive weight example with more visible colors
        this.ctx.beginPath();
        this.ctx.moveTo(legendX + 10, legendY + 30);
        this.ctx.lineTo(legendX + 40, legendY + 30);
        this.ctx.strokeStyle = 'rgba(50, 100, 255, 1.0)'; // Match the color used in connections
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        this.ctx.fillStyle = '#333';
        this.ctx.fillText('Positive', legendX + 45, legendY + 33);
        
        // Draw negative weight example with more visible colors
        this.ctx.beginPath();
        this.ctx.moveTo(legendX + 10, legendY + 45);
        this.ctx.lineTo(legendX + 40, legendY + 45);
        this.ctx.strokeStyle = 'rgba(255, 70, 70, 1.0)'; // Match the color used in connections
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        this.ctx.fillStyle = '#333';
        this.ctx.fillText('Negative', legendX + 45, legendY + 48);
    }

    // Visualize the prediction process with the given input
    visualizePrediction(inputData, activations) {
        // Store input data for visualization
        this.neuronValues = inputData;
        
        // Create mock activations for visualization if not provided
        if (!activations) {
            this.createMockActivations();
        } else {
            this.activations = activations;
        }
        
        this.drawNetwork();
        
        // Start the animation if not already animating
        if (!this.animating) {
            this.animatePrediction();
        }
    }

    createMockActivations() {
        // Create placeholder activations for visualization purposes
        this.activations = [];
        
        // For each layer, generate some activation values
        for (let l = 0; l < this.layerSizes.length; l++) {
            const layerActivations = new Array(this.layerSizes[l]).fill(0);
            
            // For input layer, use the actual input values
            if (l === 0 && this.neuronValues) {
                for (let i = 0; i < Math.min(layerActivations.length, this.neuronValues.length); i++) {
                    layerActivations[i] = this.neuronValues[i];
                }
            } 
            // For other layers, generate some pattern based on layer position
            else {
                for (let i = 0; i < layerActivations.length; i++) {
                    // Generate a pseudo-random but consistent activation pattern
                    const seed = (i * l + i) % 100;
                    layerActivations[i] = (Math.sin(seed) + 1) / 2; // Range 0-1
                }
            }
            
            this.activations.push(layerActivations);
        }
    }
    
    // Simple animation for visualization
    animatePrediction() {
        this.animating = true;
        
        // Forward-pass animation
        let currentLayer = 0;
        const animateLayer = () => {
            // Highlight the current layer
            this.highlightLayer(currentLayer);
            
            currentLayer++;
            if (currentLayer < this.layerSizes.length) {
                setTimeout(animateLayer, 600);
            } else {
                setTimeout(() => {
                    this.drawNetwork(); // Redraw without highlights
                    this.animating = false;
                }, 800);
            }
        };
        
        animateLayer();
    }
    
    highlightLayer(layerIndex) {
        // Redraw network
        this.drawNetwork();
        
        const x = this.layerPositions[layerIndex];
        
        // Draw highlight effect
        const gradient = this.ctx.createLinearGradient(
            x - 30, 0, 
            x + 30, 0
        );
        gradient.addColorStop(0, 'rgba(255, 255, 100, 0)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 100, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 255, 100, 0)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x - 40, 30, 80, this.canvas.height - 60);
        
        // Add explanatory text
        this.ctx.font = 'bold 14px Arial';
        this.ctx.fillStyle = 'rgba(50, 50, 50, 0.8)';
        this.ctx.textAlign = 'center';
        
        let message = '';
        if (layerIndex === 0) {
            message = 'Input: pixels from drawing';
        } else if (layerIndex === this.layerSizes.length - 1) {
            message = 'Output: classification probabilities';
        } else {
            message = `Hidden layer ${layerIndex} processing`;
        }
        
        // Add "processing" text above the layer
        this.ctx.fillText(message, x, this.canvas.height - 45);
    }
}